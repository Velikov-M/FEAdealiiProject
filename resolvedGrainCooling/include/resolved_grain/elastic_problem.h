#ifndef RESOLVED_GRAIN_ELASTIC_PROBLEM_H
#define RESOLVED_GRAIN_ELASTIC_PROBLEM_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>
#include <deal.II/physics/notation.h>

#include <fstream>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <algorithm>
#include <limits>

#include <boost/math/tools/roots.hpp>

#include "resolved_grain/assembly_data.h"
#include "resolved_grain/material.h"
#include "resolved_grain/orientations.h"
#include "resolved_grain/output.h"
#include "resolved_grain/quadrature_data.h"

namespace StepCooling
{
  using namespace dealii;

  /** Boundary-condition families supported by the cooling problem. */
  enum class BoundaryConditionType
  {
    zeroDisplacement, ///< Clamp the complete boundary; macroscopic contraction is prohibited.
    freeSurface,      ///< Natural traction-free boundary with six minimal support constraints.
    periodic          ///< u = Ebar.x + u_periodic, with zero volume-averaged stress.
  };

  inline BoundaryConditionType parseBoundaryConditionType(const std::string &s)
  {
    if (s == "zeroDisplacement") return BoundaryConditionType::zeroDisplacement;
    if (s == "freeSurface")      return BoundaryConditionType::freeSurface;
    if (s == "periodic")         return BoundaryConditionType::periodic;
    throw std::runtime_error("unknown BoundaryConditions/Type '" + s +
                             "' (expected zeroDisplacement | freeSurface | periodic)");
  }

  inline std::string boundaryConditionTypeName(BoundaryConditionType t)
  {
    switch (t)
    {
      case BoundaryConditionType::zeroDisplacement: return "zeroDisplacement";
      case BoundaryConditionType::freeSurface:      return "freeSurface";
      case BoundaryConditionType::periodic:         return "periodic";
    }
    return "?";
  }

  /** Geometric boundary identifiers assigned to the cube before initial refinement. */
  namespace BoundaryIds
  {
    constexpr types::boundary_id allFaces = 0; ///< Mesh default before assignBoundaryIds().

    // Opposing faces share a direction index for periodic matching.
    constexpr types::boundary_id xMinus = 1;
    constexpr types::boundary_id xPlus  = 2;
    constexpr types::boundary_id yMinus = 3;
    constexpr types::boundary_id yPlus  = 4;
    constexpr types::boundary_id zMinus = 5;
    constexpr types::boundary_id zPlus  = 6;
  }

  /** Resolved-polycrystal, small-strain thermoelastic cooling problem with scalar basal damage.
   *
   * Owns the mesh, DoFs, per-grain orientations and material caches, quadrature-point damage,
   * constraints, and linear-system state. The referenced ParameterHandler must outlive this
   * object. All physical quantities use SI units: metres, pascals, kelvin, and dimensionless
   * strain and damage. Only dim == 3 is instantiated.
   *
   * Rotation matrices map global tensors to local crystal coordinates: local basis vectors are
   * matrix rows, and local axis 3 is the c-axis. Thus sigma_33, not global sigma_zz, drives damage.
   * Macroscopic vectors use deal.II Kelvin notation; recorded macroscopic strain components do not.
   */
  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem(ParameterHandler &prm);
    void run();
    void printProblemParametersToJson(const std::string filename);

  private:

    void setup_system();
    // Both cell loops implement identical equations; the serial path is the deterministic A/B
    // reference for the WorkStream path.
    void assemble_system(double curTemperature);
    void assemble_cells_serial(double curTemperature);
    void assemble_cells_workstream(double curTemperature);
    // The periodic solves share one matrix and therefore one factorization/preconditioner.
    // prepareLinearSolver() is called after assembly and before the first solve.
    void solve();
    void prepareLinearSolver();
    void solveLinearSystem(const Vector<double> &rhs, Vector<double> &dst);
    std::unique_ptr<SparseDirectUMFPACK> directSolver;
    std::unique_ptr<PreconditionSSOR<SparseMatrix<double>>> cgPreconditioner;
    // AMG requires a Trilinos copy of the deal.II system matrix.
    std::unique_ptr<TrilinosWrappers::SparseMatrix> trilinosMatrix;
    std::unique_ptr<TrilinosWrappers::PreconditionAMG> amgPreconditioner;
    mutable unsigned int lastCgIterations = 0;
    double lastCbarAsymmetry = 0.0; ///< Relative asymmetry of the homogenized stiffness.
    bool damageCeilingWarned = false;
    // Current temperature for const field-output stress reconstruction.
    double lastProcessedTemperature = 0.0;
    void calculateSolutionTemperatureStep(double curT, double dT, unsigned int refinementCycle, int temperatureStepNumber);
    void refine_grid(double fractionToRefine, double fractionToCoarse);
    void output_results(const unsigned int cycle = 1) const;
    void process_solution(unsigned int refinementCycle, const unsigned int tempStepNum, const unsigned int iterSolverStepNum, double curT);
    void parse_cellToMaterial_data(std::string fileName);

    // ----------------------------------------------------------------------------------------
    // Boundary conditions
    // ----------------------------------------------------------------------------------------
    BoundaryConditionType boundaryConditionType = BoundaryConditionType::zeroDisplacement;

    // Geometric face-pair labelling of the boundary. Needed only by "periodic", but harmless
    // (and cheap) for the others, so it runs unconditionally once after the mesh is read.
    void assignBoundaryIds();

    // Assigns every active cell a grain (material_id) from its centroid, via a
    // GrainsPerSide^3 lattice over the cube. Idempotent and a pure function of position, so it
    // is safe to re-run after refinement -- which is the point: the grain field then stays
    // fixed while the mesh under it gets finer.
    void assignGrains();
    unsigned int grainsPerSide = 1;

    /** Undamaged global material tensors, cached per grain at the current temperature.
     *
     * For global c-axis n, damage is applied at each quadrature point by exact rank-one
     * updates along n. Entries are indexed by material_id - 1, so refinement reuses the same
     * grain caches and only the scalar damage factor varies within a cell.
     */
    std::vector<SymmetricTensor<4, dim>> grainStiffnessZeroDamage;
    std::vector<SymmetricTensor<4, dim>> grainDamageDirection;
    // The CTE cache uses the analogous rank-two direction n (x) n.
    std::vector<SymmetricTensor<2, dim>> grainCteZeroDamage;
    std::vector<SymmetricTensor<2, dim>> grainCaxisOuter;        // n (x) n
    double grainCacheAlpha33 = 0.0;
    double grainCacheTemperature = std::numeric_limits<double>::quiet_NaN();
    double grainCacheC33 = 0.0;
    void refreshGrainStiffnessCache(double curTemperature);

    // Fills `constraints` for the selected BC family. Called from assemble_system() on every
    // iteration, because AffineConstraints must be rebuilt whenever it is re-closed.
    void applyBoundaryConstraints();

    // Minimal kinematic support for "freeSurface": exactly 6 scalar constraints removing the 3
    // translations and 3 rotations, and NOTHING else -- any extra constraint would apply a real
    // load and destroy the <sigma> = 0 property the free-surface case exists to have.
    void constrainRigidBodyModes();

    // Constrains all dim components of the DoFs sitting at `where` to zero. Used by the periodic
    // formulation to remove the three rigid translations.
    void constrainOneVertex(const Point<dim> &where);

    // The periodic problem is condensed into sparse displacement equations plus six Kelvin
    // macroscopic unknowns. Cbar is recovered from average stresses under six unit strains;
    // only the constrained coupling columns L are stored. See homogenization.md section 6.
    static constexpr unsigned int nMacroDofs = 6;

    // Assemble L with K and the thermal RHS in one pass so all use the same frozen damage state.
    bool assembleMacroCoupling = false;
    void distributeMacroCouplingColumns(
      const FullMatrix<double> &cell_L,
      const std::vector<types::global_dof_index> &local_dof_indices);

    // The 7-solve of section 6: U0 = K^-1 F_theta, Y_k = K^-1 L_k (six more, same matrix and
    // same factorization), Cbar by quadrature, Ebar = -Cbar^-1 sigma_bar_theta, and finally
    // U = U0 - sum_k Ebar_k Y_k with no extra solve. Leaves the answer in solDisplacement and
    // macroscopicStrain.
    void solve_with_zero_mean_stress(double curTemperature);

    // Builds Cbar and its inverse: six solves (Y_k = K^-1 L_k) plus a quadrature pass per column.
    // Shared by both paths -- the exact one calls it every pass, the quasi-Newton one only when
    // Cbar has gone stale.
    void refreshHomogenizedStiffness(double curTemperature);

    // The Ebar iteration with a frozen Cbar. Returns false if it did not converge within
    // MaxMacroscopicIterations, which is the caller's signal that Cbar is stale.
    bool iterateMacroscopicStrain(double curTemperature);

    std::vector<Vector<double>> macroUnitSolutions;   // Y_k = K^-1 L_k, kept for the exact path
    FullMatrix<double> homogenizedStiffnessInverse{nMacroDofs, nMacroDofs};
    bool cbarNeedsRefresh = true;
    unsigned int lastMacroIterations = 0;
    unsigned int cbarRefreshCount = 0;

    // Volume average of the stress over the RVE, by plain quadrature summation. Used as the
    // <sigma> = 0 residual check, and with unit macroscopic strains to build Cbar column by
    // column. includeThermalStrain is false for the unit-strain solves, whose whole point is the
    // purely mechanical response.
    SymmetricTensor<2, dim> computeAverageStress(
      double curTemperature,
      const Vector<double> &displacementField,
      const SymmetricTensor<2, dim> &macroStrain,
      bool includeThermalStrain) const;
    SymmetricTensor<2, dim> computeAverageStress(double curTemperature) const;

    // L: six right-hand-side vectors, one per Kelvin component of Ebar.
    std::vector<Vector<double>> macroCouplingColumns;
    // Homogenized stiffness Cbar, a by-product of the condensation and a result in its own
    // right (it gives the effective CTE, and its evolution is a macroscopic softening curve).
    FullMatrix<double> homogenizedStiffness{nMacroDofs, nMacroDofs};
    // The macroscopic strain itself: zero for every non-periodic BC family.
    SymmetricTensor<2, dim> macroscopicStrain;

    // Backward-Euler damage prediction at fixed displacement, followed by convergence checks.
    void predict_damage_tempInc_external_solver(double curTemperature, double tempInc);
    void predict_damage_cells_serial(double curTemperature, double tempInc,
                                      double stressThreshold, double A_param, double m_param,
                                      double residualTolerance);
    void predict_damage_cells_workstream(double curTemperature, double tempInc,
                                          double stressThreshold, double A_param, double m_param,
                                          double residualTolerance);
    // Per-quadrature-point damage update, shared verbatim by both paths above so the two cannot
    // drift apart. Writes only to this cell's own quadrature data, which is what makes the
    // threaded version safe without a copier.
    void update_damage_on_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      FEValues<dim> &fe_values,
      std::vector<Tensor<2, dim>> &global_displacement_gradients,
      double curTemperature, double tempInc,
      double stressThreshold, double A_param, double m_param, double residualTolerance);

    double kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc);

    // Independent scalar snapshots are required: copying CellDataStorage copies shared pointers,
    // not the DamageQData objects, and would alias the state being monitored.
    using DamageSnapshot = std::map<typename Triangulation<dim>::cell_iterator, std::vector<double>>;
    DamageSnapshot snapshotDamage();

    /** Irons-Tuck relaxation of the displacement-damage fixed point.
     *
     * Uses successive residuals without an extra linear solve. Relaxed damage is clamped to
     * [old_damage, 0.999] to preserve irreversibility and avoid the kinetic singularity.
     */
    void applyAitkenRelaxation(const DamageSnapshot &damageBeforePredict);
    void resetAitkenState();
    DamageSnapshot aitkenPreviousResidual;
    double aitkenMu = 1.0;
    bool aitkenHasPrevious = false;
    unsigned int aitkenRelaxedIterations = 0;
    bool checkConvergenceCriteria(double curT,
                                    const Vector<double> &oldSolDisplacement,
                                    const DamageSnapshot &oldDamageSnapshot);

    ElasticityTensor<dim> elasticityTensor;
    CteTensor<dim> cteTensor;

    Tensor<2,dim,double> getRotTensorGlobalToLocal(int cellId) const;
    SymmetricTensor<2,dim> getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature) const;

    std::istringstream parseMSHandPrepareStream(std::string mshFileName);

    ParameterHandler &prm;

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
  
    std::vector<Tensor<2, dim>> rotmatOrientationCell;
    std::vector<int> cellToMaterial;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                DamageQData> damageInQuadraturePoints;

    const FESystem<dim> fe;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solDisplacement;
    Vector<double> system_rhs;

    const FEValuesExtractors::Vector displacement{0};
    TimerOutput timer;

    ConvergenceTable convergence_table;

    /** Scalar history for one converged temperature step. */
    struct MacroscopicRecord
    {
      unsigned int refinementCycle, temperatureStep;
      double temperature, cooling;
      std::array<double, 6> Ebar;        // 11 22 33 23 13 12, TENSOR components (not Kelvin)
      std::array<double, 6> CbarDiag;    // Kelvin diagonal, Pa
      double meanDamage, maxDamage, meanStressNorm, CbarAsymmetry;
      unsigned int cgIterations;
    };
    std::vector<MacroscopicRecord> macroscopicHistory;
    void writeMacroscopicHistory(const std::string &fileName) const;

    // <OutputDirectory>/<OutputFileNamePrefix><suffix>, creating the directory if needed.
    std::string outputPath(const std::string &suffix) const;

    // (pseudo-time, filename) pairs for the ParaView .pvd time series written at the end of
    // run(). The time value is the accumulated cooling s = T_0 - T, which INCREASES over the run
    // -- temperature itself decreases, and ParaView orders a series by increasing time, so using
    // T directly would play the animation backwards.
    mutable std::vector<std::pair<double, std::string>> pvdTimeSeries;

  };
}

#endif

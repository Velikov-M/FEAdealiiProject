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
#include <deal.II/numerics/solution_transfer.h>
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
#include "resolved_grain/convergence.h"
#include "resolved_grain/damage_law.h"
#include "resolved_grain/material.h"
#include "resolved_grain/orientations.h"
#include "resolved_grain/output.h"
#include "resolved_grain/quadrature_data.h"
#include "resolved_grain/quadrature_transfer.h"

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

  /** Parse a BoundaryConditions/Type value.
   * @param[in] s Boundary-condition name.
   * @return Parsed boundary-condition family.
   * @throw std::runtime_error If @p s is not a supported name.
   */
  inline BoundaryConditionType parseBoundaryConditionType(const std::string &s)
  {
    if (s == "zeroDisplacement") return BoundaryConditionType::zeroDisplacement;
    if (s == "freeSurface")      return BoundaryConditionType::freeSurface;
    if (s == "periodic")         return BoundaryConditionType::periodic;
    throw std::runtime_error("unknown BoundaryConditions/Type '" + s +
                             "' (expected zeroDisplacement | freeSurface | periodic)");
  }

  /** Return the input-file name of a boundary-condition family.
   * @param[in] t Boundary-condition family.
   * @return Stable textual name used in parameters and logs.
   */
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
    /** Construct an uninitialized problem using a persistent parameter set.
     * @param[in] prm Parameter handler that must outlive this problem.
     */
    ElasticProblem(ParameterHandler &prm);

    /** Execute initialization, all refinement cycles, cooling histories, and final output. */
    void run();

    /** Write the declared parameter set as JSON.
     * @param[in] filename Destination file path.
     */
    void printProblemParametersToJson(const std::string filename);

  private:
    /** Configure threading, boundary conditions, material functions, and startup checks. */
    void initializeRun();

    /** Read the mesh and initialize grain identities, orientations, and boundary IDs.
     *
     * Boundary IDs are assigned on the coarse mesh before initial refinement. Random
     * orientations are generated once per grain and replace mesh orientation metadata.
     */
    void initializeMicrostructure();

    /** Run independent cooling histories on all requested refinement cycles. */
    void runRefinementCycles();

    /** Initialize or refine the mesh and all mesh-dependent state for one cycle.
     * @param[in] refinementCycle Zero-based cycle index.
     */
    void prepareRefinementCycle(unsigned int refinementCycle);

    /** Solve and selectively output the complete cooling history on the current mesh.
     * @param[in] refinementCycle Zero-based cycle index used in output numbering.
     */
    void runCoolingHistory(unsigned int refinementCycle);

    /** Write the convergence table, macroscopic CSV history, and ParaView time series. */
    void finalizeRunOutput();

    /** Distribute DoFs and initialize damage, constraints, sparsity, matrices, and vectors.
     *
     * This resets quadrature-point damage to zero; damage is not transferred between cycles.
     */
    void setup_system();

    /** Assemble the mechanical system at fixed damage using the selected cell-loop backend.
     * @param[in] curTemperature Current uniform temperature in K.
     */
    void assemble_system(double curTemperature);

    /** Serial reference implementation of cell assembly.
     * @param[in] curTemperature Current uniform temperature in K.
     */
    void assemble_cells_serial(double curTemperature);

    /** WorkStream implementation of cell assembly.
     * @param[in] curTemperature Current uniform temperature in K.
     */
    void assemble_cells_workstream(double curTemperature);

    /** Solve the non-periodic mechanical system and distribute constraints to the solution. */
    void solve();

    /** Prepare the selected direct solver or preconditioner for the assembled matrix. */
    void prepareLinearSolver();

    /** Solve one right-hand side without distributing constraints to the raw solution.
     * @param[in] rhs Right-hand-side vector.
     * @param[out] dst Solution vector overwritten by the solve.
     */
    void solveLinearSystem(const Vector<double> &rhs, Vector<double> &dst);
    std::unique_ptr<SparseDirectUMFPACK> directSolver;
    std::unique_ptr<PreconditionSSOR<SparseMatrix<double>>> cgPreconditioner;
    // AMG requires a Trilinos copy of the deal.II system matrix.
    std::unique_ptr<TrilinosWrappers::SparseMatrix> trilinosMatrix;
    std::unique_ptr<TrilinosWrappers::PreconditionAMG> amgPreconditioner;
    mutable unsigned int lastCgIterations = 0;
    unsigned long long totalCgIterations = 0; ///< Cumulative iterations over all solved right-hand sides.
    unsigned int totalLinearSolveCount = 0;   ///< Includes direct and exact zero-RHS solves.
    double lastCbarAsymmetry = 0.0; ///< Relative asymmetry of the homogenized stiffness.
    bool damageCeilingWarned = false;
    // Current temperature for const field-output stress reconstruction.
    double lastProcessedTemperature = 0.0;
    /** Solve one temperature step to coupled displacement-damage convergence.
     * @param[in] curT End-of-step temperature in K.
     * @param[in] dT Positive cooling increment for this step in K.
     * @param[in] refinementCycle Zero-based refinement-cycle index.
     * @param[in] temperatureStepNumber One-based temperature-step index.
     */
    void calculateSolutionTemperatureStep(double curT, double dT, unsigned int refinementCycle, int temperatureStepNumber);

    /** Append scalar results for a converged temperature step.
     * @param[in] refinementCycle Zero-based refinement-cycle index.
     * @param[in] temperatureStepNumber One-based temperature-step index.
     * @param[in] curT Converged step temperature in K.
     */
    void recordMacroscopicState(unsigned int refinementCycle,
                                unsigned int temperatureStepNumber,
                                double curT);

    /** Copy converged damage to the old-damage baseline used by the next temperature step. */
    void commitConvergedDamage();

    /** Estimate displacement-gradient jumps and adapt the mesh.
     *
     * KellyErrorEstimator is used as an indicator, not a rigorous anisotropic-composite error
     * bound. With transferState=true, displacement and both damage histories are transferred
     * before solving continues. With transferState=false, the existing independent-cycle
     * behavior reinitializes the solution and damage.
     * @param[in] fractionToRefine Fraction of cells selected for refinement.
     * @param[in] fractionToCoarse Fraction of cells selected for coarsening.
     * @param[in] transferState Whether to preserve the current mechanical and damage state.
     */
    void refine_grid(double fractionToRefine, double fractionToCoarse, bool transferState);

    /** Write one VTU field snapshot and append it to the PVD time series.
     * @param[in] cycle Output number used in the filename.
     */
    void output_results(const unsigned int cycle = 1) const;

    /** Add raw displacement, c-axis, and total displacement in the established field order.
     * @param[in,out] dataOut Destination that references the supplied fields.
     * @param[in] cAxisPostprocessor Orientation evaluator that must outlive DataOut evaluation.
     * @param[in] totalDisplacementPostprocessor Total-displacement evaluator with the same lifetime.
     */
    void addDisplacementAndOrientationOutput(
      DataOut<dim> &dataOut,
      CAxisPostprocessor<dim> &cAxisPostprocessor,
      TotalDisplacementPostprocessor<dim> &totalDisplacementPostprocessor) const;

    /** Compute all cell-centered mechanical output arrays in one quadrature pass.
     * @param[in] writeFullStress Whether to populate local/global stress arrays in Pa.
     * @param[in] writeFullStrain Whether to populate local/global total-strain arrays.
     * @return Owning field storage that must outlive DataOut evaluation.
     */
    CellOutputFields computeCellOutputFields(bool writeFullStress, bool writeFullStrain) const;

    /** Attach damage and scalar stress/strain fields in the established order.
     * @param[in,out] dataOut Destination that references @p fields.
     * @param[in] fields Computed cell-centered arrays.
     */
    void addScalarMechanicalOutput(DataOut<dim> &dataOut,
                                   const CellOutputFields &fields) const;

    /** Attach the uniform macroscopic-strain component arrays.
     * @param[in,out] dataOut Destination that references @p fields.
     * @param[in] fields Storage containing the six tensor components.
     */
    void addMacroscopicStrainOutput(DataOut<dim> &dataOut,
                                    const CellOutputFields &fields) const;

    /** Attach optional local and global full-stress component arrays.
     * @param[in,out] dataOut Destination that references @p fields.
     * @param[in] fields Computed stress arrays in Pa.
     */
    void addFullStressOutput(DataOut<dim> &dataOut, const CellOutputFields &fields) const;

    /** Attach optional local and global full-total-strain component arrays.
     * @param[in,out] dataOut Destination that references @p fields.
     * @param[in] fields Computed dimensionless strain arrays.
     */
    void addFullStrainOutput(DataOut<dim> &dataOut, const CellOutputFields &fields) const;

    /** Log stress/damage diagnostics and append convergence-table values for one iterate.
     * @param[in] refinementCycle Zero-based refinement-cycle index.
     * @param[in] tempStepNum One-based temperature-step index.
     * @param[in] iterSolverStepNum One-based displacement-damage iteration index.
     * @param[in] curT Current temperature in K.
     */
    void process_solution(unsigned int refinementCycle, const unsigned int tempStepNum, const unsigned int iterSolverStepNum, double curT);

    /** Read the legacy cell-to-material mapping into cellToMaterial.
     * @param[in] fileName Mapping file path.
     */
    void parse_cellToMaterial_data(std::string fileName);

    // ----------------------------------------------------------------------------------------
    // Boundary conditions
    // ----------------------------------------------------------------------------------------
    BoundaryConditionType boundaryConditionType = BoundaryConditionType::zeroDisplacement;

    /** Assign geometric face-pair IDs on the coarse cube for periodic matching. */
    void assignBoundaryIds();

    /** Assign a fixed 1-based grain material_id to every active cell by centroid position.
     *
     * The mapping is independent of mesh resolution and preserves grains under refinement.
     */
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
    /** Rebuild per-grain global stiffness/CTE caches if temperature or grain count changed.
     * @param[in] curTemperature Current uniform temperature in K.
     */
    void refreshGrainStiffnessCache(double curTemperature);

    /** Rebuild and close all hanging-node and selected boundary constraints. */
    void applyBoundaryConstraints();

    /** Apply exactly six free-surface constraints removing rigid translations and rotations. */
    void constrainRigidBodyModes();

    /** Constrain all displacement components at one mesh vertex.
     * @param[in] where Vertex position in m.
     */
    void constrainOneVertex(const Point<dim> &where);

    /** Number of macroscopic Kelvin strain/stress components in three dimensions. */
    static constexpr unsigned int nMacroDofs = 6;

    /** Assemble L with K and the thermal RHS in one pass at the same frozen damage state. */
    bool assembleMacroCoupling = false;
    /** Scatter one cell's six macroscopic coupling columns through homogeneous constraints.
     * @param[in] cell_L Local coupling matrix with one column per Kelvin strain component.
     * @param[in] local_dof_indices Global DoF indices corresponding to cell_L rows.
     */
    void distributeMacroCouplingColumns(
      const FullMatrix<double> &cell_L,
      const std::vector<types::global_dof_index> &local_dof_indices);

    /** Solve the periodic fluctuation and macroscopic strain at zero average stress.
     * @param[in] curTemperature Current uniform temperature in K.
     */
    void solve_with_zero_mean_stress(double curTemperature);

    /** Rebuild Cbar, its inverse, and six unit-strain fluctuation solutions at fixed damage.
     * @param[in] curTemperature Current uniform temperature in K.
     */
    void refreshHomogenizedStiffness(double curTemperature);

    /** Iterate macroscopic strain using a frozen Cbar and the exact average-stress residual.
     *
     * At fixed damage the residual is affine in Ebar. Every pass uses current damage and K;
     * stale Cbar changes only the correction direction, not the zero-stress target.
     * @param[in] curTemperature Current uniform temperature in K.
     * @return True when the residual converges within MaxMacroscopicIterations.
     */
    bool iterateMacroscopicStrain(double curTemperature);

    std::vector<Vector<double>> macroUnitSolutions;   // Y_k = K^-1 L_k, kept for the exact path
    FullMatrix<double> homogenizedStiffnessInverse{nMacroDofs, nMacroDofs};
    bool cbarNeedsRefresh = true;
    unsigned int lastMacroIterations = 0;
    unsigned int cbarRefreshCount = 0;

    /** Compute volume-averaged stress by quadrature for an arbitrary mechanical state.
     * @param[in] curTemperature Current uniform temperature in K.
     * @param[in] displacementField Fluctuation or total displacement field in m.
     * @param[in] macroStrain Uniform macroscopic strain, dimensionless.
     * @param[in] includeThermalStrain Whether to subtract the thermal eigenstrain.
     * @return Volume-averaged Cauchy stress in Pa in global coordinates.
     */
    SymmetricTensor<2, dim> computeAverageStress(
      double curTemperature,
      const Vector<double> &displacementField,
      const SymmetricTensor<2, dim> &macroStrain,
      bool includeThermalStrain) const;

    /** Compute the current state's volume-averaged stress including thermal strain.
     * @param[in] curTemperature Current uniform temperature in K.
     * @return Volume-averaged Cauchy stress in Pa in global coordinates.
     */
    SymmetricTensor<2, dim> computeAverageStress(double curTemperature) const;

    /** Six constrained right-hand sides, one per Kelvin component of macroscopic strain. */
    std::vector<Vector<double>> macroCouplingColumns;
    /** Homogenized stiffness in Pa, reconstructed from six unit-strain stress responses. */
    FullMatrix<double> homogenizedStiffness{nMacroDofs, nMacroDofs};
    /** Current macroscopic strain; identically zero for non-periodic boundary conditions. */
    SymmetricTensor<2, dim> macroscopicStrain;

    /** Predict damage over one cooling increment using the selected cell-loop backend.
     * @param[in] curTemperature Current uniform temperature in K.
     * @param[in] tempInc Positive cooling pseudo-time increment in K.
     */
    void predict_damage_tempInc_external_solver(double curTemperature, double tempInc);

    /** Serial reference implementation of damage prediction.
     * @param[in] curTemperature Current uniform temperature in K.
     * @param[in] tempInc Positive cooling pseudo-time increment in K.
     * @param[in] stressThreshold Tensile activation threshold in Pa.
     * @param[in] A_param Kinetic prefactor A.
     * @param[in] m_param Kinetic exponent m.
     * @param[in] residualTolerance Maximum accepted dimensionless kinetic residual.
     */
    void predict_damage_cells_serial(double curTemperature, double tempInc,
                                      double stressThreshold, double A_param, double m_param,
                                      double residualTolerance);

    /** WorkStream implementation of damage prediction.
     * @param[in] curTemperature Current uniform temperature in K.
     * @param[in] tempInc Positive cooling pseudo-time increment in K.
     * @param[in] stressThreshold Tensile activation threshold in Pa.
     * @param[in] A_param Kinetic prefactor A.
     * @param[in] m_param Kinetic exponent m.
     * @param[in] residualTolerance Maximum accepted dimensionless kinetic residual.
     */
    void predict_damage_cells_workstream(double curTemperature, double tempInc,
                                          double stressThreshold, double A_param, double m_param,
                                          double residualTolerance);

    /** Update all quadrature-point damage values owned by one cell.
     * @param[in] cell Active cell whose local damage storage is updated.
     * @param[in,out] fe_values Reinitialized FE values object supplied by the caller.
     * @param[out] global_displacement_gradients Reusable quadrature-point gradient storage.
     * @param[in] curTemperature Current uniform temperature in K.
     * @param[in] tempInc Positive cooling pseudo-time increment in K.
     * @param[in] stressThreshold Tensile activation threshold in Pa.
     * @param[in] A_param Kinetic prefactor A.
     * @param[in] m_param Kinetic exponent m.
     * @param[in] residualTolerance Maximum accepted dimensionless kinetic residual.
     */
    void update_damage_on_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      FEValues<dim> &fe_values,
      std::vector<Tensor<2, dim>> &global_displacement_gradients,
      double curTemperature, double tempInc,
      double stressThreshold, double A_param, double m_param, double residualTolerance);

    /** Independent damage values used for convergence and relaxation.
     *
     * Copying CellDataStorage would alias shared DamageQData pointers instead of creating a
     * snapshot, so values are copied explicitly by active cell and quadrature point.
     */
    using DamageSnapshot = std::map<typename Triangulation<dim>::cell_iterator, std::vector<double>>;

    /** Copy every active cell's current damage values into independent scalar storage.
     * @return Snapshot indexed by active cell and quadrature point.
     */
    DamageSnapshot snapshotDamage();

    /** Irons-Tuck relaxation of the displacement-damage fixed point.
     *
     * Uses successive residuals without an extra linear solve. Relaxed damage is clamped to
     * [old_damage, 0.999] to preserve irreversibility and avoid the kinetic singularity.
     * @param[in] damageBeforePredict Damage values before the current prediction pass.
     */
    void applyAitkenRelaxation(const DamageSnapshot &damageBeforePredict);

    /** Clear Aitken residual history at the start of a temperature step. */
    void resetAitkenState();
    DamageSnapshot aitkenPreviousResidual;
    double aitkenMu = 1.0;
    bool aitkenHasPrevious = false;
    unsigned int aitkenRelaxedIterations = 0;
    /** Test displacement, damage, and equilibrium convergence between fixed-point iterates.
     * @param[in] curT Current temperature in K.
     * @param[in] oldSolDisplacement Displacement at the previous iterate in m.
     * @param[in] oldDamageSnapshot Damage at the previous iterate.
     * @return True only when all three configured convergence criteria pass.
     */
    bool checkConvergenceCriteria(double curT,
                                    const Vector<double> &oldSolDisplacement,
                                    const DamageSnapshot &oldDamageSnapshot);

    ElasticityTensor<dim> elasticityTensor;
    CteTensor<dim> cteTensor;

    /** Return a grain's global-to-local rotation matrix.
     * @param[in] cellId One-based grain material_id.
     * @return Rotation whose rows are local basis vectors in global coordinates.
     */
    Tensor<2,dim,double> getRotTensorGlobalToLocal(int cellId) const;

    /** Evaluate local Cauchy stress from total local strain and thermal eigenstrain.
     * @param[in] localStrain Total strain in crystal coordinates, dimensionless.
     * @param[in] curOmega Current damage, dimensionless.
     * @param[in] curTemperature Current uniform temperature in K.
     * @return Local Cauchy stress in Pa.
     */
    SymmetricTensor<2,dim> getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature) const;

    /** Parse Neper mesh input and collect orientation metadata for GridIn.
     * @param[in] mshFileName Mesh file path.
     * @return Mesh stream with unsupported orientation metadata removed.
     */
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
    mutable TimerOutput timer; ///< Profiling sections, including const stress and output operations.

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
    /** Write the accumulated macroscopic history to CSV.
     * @param[in] fileName Destination CSV path.
     */
    void writeMacroscopicHistory(const std::string &fileName) const;

    /** Construct a prefixed output path and create its directory when needed.
     * @param[in] suffix Filename suffix appended to OutputFileNamePrefix.
     * @return Full output path.
     */
    std::string outputPath(const std::string &suffix) const;

    /** PVD (pseudo-time, filename) records using accumulated cooling T0 - T in K. */
    mutable std::vector<std::pair<double, std::string>> pvdTimeSeries;

  };
}

#endif

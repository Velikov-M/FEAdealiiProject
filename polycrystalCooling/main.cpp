// ============================================================================================
// POLYCRYSTAL COOLING MODEL
//
// Cooling of an RVE of nuclear graphite made of randomly oriented crystallite domains. Each
// original mesh cell is one domain, carrying its own c-axis orientation. Cooling alone -- with
// no applied load -- produces a non-trivial stress field, because neighbouring grains have
// mismatched thermal expansion (alpha_c >> alpha_a for graphite) and cannot contract freely
// past one another. That mismatch is the classic microcracking mechanism in polycrystalline
// graphite, and the point of the model is to locate where it drives damage.
//
// See README.md for the design rationale and homogenization.md for the full derivation of the
// periodic / <sigma> = 0 formulation.
//
// ---------------------------------------------------------------------------------------------
// BOUNDARY CONDITIONS -- implemented in increasing order of difficulty
//
// BoundaryConditions/Type in input.json selects one of:
//
//   "zeroDisplacement"  u = 0 on the whole boundary. Physically a rigidly clamped block, NOT
//                       free cooling: it forbids macroscopic contraction and therefore develops
//                       a large spurious triaxial tension. Included as the simplest possible
//                       well-posed case -- no rigid-body modes, best-conditioned matrix -- so
//                       the grain machinery (orientations, rotation of C and alpha, damage) can
//                       be exercised before boundary handling gets interesting. Do not read
//                       physics off it.
//
//   "freeSurface"       Traction-free on the whole boundary, plus a minimal kinematic support
//                       that removes the 6 rigid-body modes and nothing else. Because t = 0
//                       everywhere on the boundary, the average stress theorem gives
//                       <sigma> = (1/|Omega|) * integral of (t_i x_j) dS = 0 identically: this
//                       IS a free-cooling, zero-mean-stress state. What it is not is a
//                       *periodic* one -- boundary grains relax against a real free surface, so
//                       there is a surface layer with no RVE character, and no scale separation.
//                       Good physics near the centre, wrong physics near the faces.
//
//   "periodic"          The target formulation: u = Ebar.x + u_periodic with Ebar determined by
//                       <sigma> = 0. NOT YET IMPLEMENTED -- see assemble_macroscopic_blocks()
//                       and solve_with_zero_mean_stress() below, which carry the block structure
//                       and the 7-solve algorithm as a template to be filled in.
//
// The three share every other code path (assembly, damage, output), so results are directly
// comparable and the boundary treatment is the only thing that changes.
// ---------------------------------------------------------------------------------------------
//
// TEMPERATURE: uniform in space, stepped down from InitialTemperature to FinalTemperature. The
// driving force here is orientation mismatch between grains, not a thermal gradient, so unlike
// ../mmsSandbox (where a spatial gradient was needed to make the CTE observable) there is no
// reason for T to vary in space. The thermal strain is alpha(T)*(T-T_0), a secant/mean CTE --
// see homogenization.md section 2 if literature values turn out to be instantaneous.
//
// DAMAGE: the Rabotnov-type kinetic law verified in ../uniaxialDamageTest/, driven by the
// LOCAL-frame sigma_zz (the c-axis normal stress) and solved per quadrature point by a
// bracketed toms748 root search. Coupling to the displacement solve is staggered: damage is
// frozen while equilibrium is solved, and the two are iterated to convergence within each
// temperature step.
//
// NOT YET DONE (see README.md):
//   [ ] periodic boundary constraints + the macroscopic-strain unknowns (BC type "periodic")
//   [ ] several seeds / realization statistics (deliberately descoped for now)
//   [ ] sourced graphite CTE values -- input.json still carries flagged PLACEHOLDERS
//
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

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
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
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <algorithm>
#include <limits>

#include <boost/math/tools/roots.hpp>

//Right now refinement is not supported due to transfering damage is not realized, setup_system() must be used only once

namespace StepCooling
{
  using namespace dealii;

  struct DamageQData : public TransferableQuadraturePointData
  {
    double damage = 0.0;
    double old_damage = 0.0; //for iterative solution, for one step prediction it is not needed, but for iterative solution it is needed to calculate residual of kinetic equation

    unsigned int number_of_values() const override
    {
      return 2; // damage and old_damage
    }

    void pack_values(std::vector<double> &values) const override
    {
      AssertDimension(values.size(), 2);
      values[0] = damage;
      values[1] = old_damage;
    }

    void unpack_values(const std::vector<double> &values) override
    {
      AssertDimension(values.size(), 2);
      damage = values[0];
      old_damage = values[1];
    }
  };

  template <int dim>
  struct DamageAndOrientationQData : public TransferableQuadraturePointData
  {
    double damage = 0.0;
    double old_damage = 0.0; //for iterative solution, for one step prediction it is not needed, but for iterative solution it is needed to calculate residual of kinetic equation
    Tensor<2, 3> orientation = unit_symmetric_tensor<3, double>();

    unsigned int number_of_values() const override
    {
      return 2 + dim*dim; // 
    }

    void pack_values(std::vector<double> &values) const override
    {
      AssertDimension(values.size(), 2 + dim*dim);
      values[0] = damage;
      values[1] = old_damage;
      for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j) values[2+3*i+j] = orientation[i][j];
      } 
    }

    void unpack_values(const std::vector<double> &values) override
    {
      AssertDimension(values.size(), 2 + dim*dim);
      damage = values[0];
      old_damage = values[1];
      for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j) orientation[i][j] = values[2+3*i+j];
      }
      Assert(abs(determinant(orientation) - 1) < 0.05, ExcMessage("During qData transfer strong orientation matrix non-orthogonality appeared"));
      // Here orthogonalization must be considered and added lately
    }
  };

  // ------------------------------------------------------------------------------------------
  // Boundary conditions
  // ------------------------------------------------------------------------------------------
  // See the file header for what each one means physically and why they are staged in this
  // order. Parsed once from BoundaryConditions/Type; every other code path is shared.
  enum class BoundaryConditionType
  {
    zeroDisplacement, // u = 0 on the whole boundary (clamped; not free cooling)
    freeSurface,      // traction-free + minimal kinematic support (a valid <sigma> = 0 state)
    periodic          // u = Ebar.x + u_periodic with <sigma> = 0  -- NOT YET IMPLEMENTED
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

  // ------------------------------------------------------------------------------------------
  // Random crystallite orientations
  // ------------------------------------------------------------------------------------------
  // Only the c-axis DIRECTION matters: rotation about c is a symmetry of a transversely
  // isotropic crystal, so this is a point-on-the-sphere problem, not a uniform-SO(3) problem.
  //
  // The trap this avoids: sampling theta uniformly on [0, pi] does NOT give a uniform direction
  // -- it clusters near the poles, because the area element is sin(theta) dtheta dphi. Sampling
  // cos(theta) uniformly on [-1, 1] is the standard fix and is exactly uniform on the sphere.
  //
  // The two axes completing the frame are NOT random: they are built deterministically from c by
  // the branchless orthonormal-basis construction of Duff et al., "Building an Orthonormal
  // Basis, Revisited" (JCGT 2017) -- a published, numerically stable construction rather than
  // one improvised here (its `sign`/`a` algebra is specifically what keeps it well conditioned
  // when c is near -e_z, where the naive Hughes-Moller construction loses precision). Since
  // rotation about c is a material symmetry, which particular frame comes out cannot affect any
  // result; using a stable deterministic one just avoids spending randomness on a non-degree of
  // freedom.
  //
  // Returned matrix R maps GLOBAL -> LOCAL: its ROWS are the local basis vectors written in
  // global coordinates, so v_local = R * v_global and row 2 is the c-axis. That is the
  // convention getRotTensorGlobalToLocal() and every basis_transformation() call below assume,
  // and it is what makes localStress[2][2] the c-axis normal stress that drives damage.
  template <int dim>
  Tensor<2, dim> orientationFromCAxis(const Tensor<1, dim> &c)
  {
    static_assert(dim == 3, "crystallite orientations are only defined in 3D");
    const double sign = std::copysign(1.0, c[2]);
    const double a = -1.0 / (sign + c[2]);
    const double b = c[0] * c[1] * a;

    Tensor<1, dim> e1, e2;
    e1[0] = 1.0 + sign * c[0] * c[0] * a;  e1[1] = sign * b;                 e1[2] = -sign * c[0];
    e2[0] = b;                             e2[1] = sign + c[1] * c[1] * a;   e2[2] = -c[1];

    Tensor<2, dim> R;
    for (unsigned int d = 0; d < dim; ++d)
    {
      R[0][d] = e1[d];
      R[1][d] = e2[d];
      R[2][d] = c[d];
    }
    return R;
  }

  // One uniformly distributed c-axis direction, and the frame built on it.
  template <int dim>
  Tensor<2, dim> randomOrientationUniformOnSphere(std::mt19937_64 &rng)
  {
    static_assert(dim == 3, "crystallite orientations are only defined in 3D");
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    const double cosTheta = 2.0 * uniform01(rng) - 1.0;             // UNIFORM IN cos(theta)
    const double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
    const double phi      = 2.0 * numbers::PI * uniform01(rng);

    Tensor<1, dim> c;
    c[0] = sinTheta * std::cos(phi);
    c[1] = sinTheta * std::sin(phi);
    c[2] = cosTheta;
    return orientationFromCAxis<dim>(c);
  }


  // The .msh carries no surface elements at all, so every face arrives with the default id 0.
  // "zeroDisplacement" needs nothing finer than that. "periodic" will need the three face pairs
  // told apart, which is what assignBoundaryIds() (below) is for.
  namespace BoundaryIds
  {
    constexpr types::boundary_id allFaces = 0; // mesh default, before assignBoundaryIds() runs

    // Face-pair ids for periodic matching. xMinus/xPlus are the pair normal to e_x, etc.
    constexpr types::boundary_id xMinus = 1;
    constexpr types::boundary_id xPlus  = 2;
    constexpr types::boundary_id yMinus = 3;
    constexpr types::boundary_id yPlus  = 4;
    constexpr types::boundary_id zMinus = 5;
    constexpr types::boundary_id zPlus  = 6;
  }

  class ParameterReader : public EnableObserverPointer
    {
    public:
      ParameterReader(ParameterHandler &);
      void read_parameters(const std::string &);
      void declare_parameters();
  
    private:
      ParameterHandler &prm;
    };

  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler)
  {}

  void ParameterReader::declare_parameters()
  {
    prm.enter_subsection("ModelParameters");
    {
      prm.declare_entry("InitialTemperature",
                        "2000",
                        Patterns::Double(0),
                        "Initial temperature of the body");

      prm.declare_entry("TemperatureIncrement",
                        "100",
                        Patterns::Double(0),
                        "Temperature increment for each step");


      prm.declare_entry("LengthOfTheBody",
                        "1",
                        Patterns::Double(0),
                        "Length of the body in meters");
      prm.declare_entry("FinalTemperature",
                        "500",
                        Patterns::Double(0),
                        "Temperature at the end of cooling process");

    }
    prm.leave_subsection();
    prm.enter_subsection("BoundaryConditions");
    {
      // See the file header for the physical meaning of each and why they are staged this way.
      prm.declare_entry("Type",
                        "zeroDisplacement",
                        Patterns::Selection("zeroDisplacement|freeSurface|periodic"),
                        "Boundary condition family. 'zeroDisplacement' clamps the whole boundary "
                        "(simplest, well conditioned, but forbids macroscopic contraction so it "
                        "is NOT free cooling). 'freeSurface' is traction-free plus minimal "
                        "kinematic support: a genuine <sigma>=0 state, but with a surface layer "
                        "that has no RVE character. 'periodic' is the target formulation "
                        "(u = Ebar.x + u_periodic with <sigma>=0) and is NOT YET IMPLEMENTED.");

      prm.declare_entry("OrientationSeed",
                        "20260905",
                        Patterns::Integer(0),
                        "Seed for the per-grain random c-axis orientations. RECORD THIS with any "
                        "result: damage initiation is an extreme-value statistic and genuinely "
                        "realization-sensitive, so a result is not reproducible without it.");
    }
    prm.leave_subsection();
    prm.enter_subsection("MaterialParameters");
    {
      prm.enter_subsection("LinearElasticityParameters"); //For now only transversely isotropic material is supported, so only 5 independent parameters are needed
      {
        prm.declare_entry("C11_0",
                          "1060e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C11 at reference temperature");
        
        prm.declare_entry("C11_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C11");

        prm.declare_entry("C33_0",
                          "36.5e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C33 at reference temperature");

        prm.declare_entry("C33_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C33");

        prm.declare_entry("C44_0",
                          "0.25e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C44 at reference temperature");

        prm.declare_entry("C44_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C44");

        prm.declare_entry("C12_0",
                          "180e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C12 at reference temperature");
        
        prm.declare_entry("C12_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C12");

        prm.declare_entry("C13_0",
                          "15e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C13 at reference temperature");

        prm.declare_entry("C13_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C13");
      }
      prm.leave_subsection();
      prm.enter_subsection("DamageParameters");
      {
        prm.declare_entry("StressThreshold",
                          "200e6",
                          Patterns::Double(0),
                          "Stress threshold for damage activation");

        prm.declare_entry("a_kineticParam",
                          "1e-5",
                          Patterns::Double(0),
                          "Kinetic parameter a in the kinetic equation");

        prm.declare_entry("m_kineticParam",
                          "2",
                          Patterns::Double(0),
                          "Kinetic parameter m in the kinetic equation");
      }
      prm.leave_subsection();
      prm.enter_subsection("ThermoElasticParameters");
      {
        prm.declare_entry("alpha11_0", // equal to alpha_22 for transversely isotropic material
                          "1e-5",
                          Patterns::Double(0),
                          "Thermal expansion coefficient at a-axis direction");

        prm.declare_entry("alpha11_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for thermal expansion coefficient at a-axis direction");

        prm.declare_entry("alpha33_0",
                          "1e-8",
                          Patterns::Double(0),
                          "Thermal expansion coefficient at c-axis direction");

        prm.declare_entry("alpha33_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for thermal expansion coefficient at c-axis direction");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
    prm.enter_subsection("SolverParameters");
    {
      prm.declare_entry("ForceEqulibriumRelativeTolerance",
                        "1e-4",
                        Patterns::Double(0),
                        "Relative tolerance for force equilibrium residiual in displacement-damage iteration");

      prm.declare_entry("DisplacementIncrementRelativeTolerance",
                        "1e-5",
                        Patterns::Double(0),
                        "Relative tolerance for displacement increment in in displacement-damage iteration");

      prm.declare_entry("DamageIncrementRelativeTolerance",
                        "1e-5",
                        Patterns::Double(0),
                        "Relative tolerance for damage increment in in displacement-damage iteration");

      prm.declare_entry("MaxNumberOfIterations",
                        "100",
                        Patterns::Integer(1),
                        "Maximum number of iterations for displacement-damage iteration");

      prm.declare_entry("KineticAlgebraicSolverTolerance",
                        "1e-4",
                        Patterns::Double(0),
                        "Relative tolerance of iterative algebraic solver for kinetic equation");

      // Was hardcoded at 1e-6 in solve(). That is a fine engineering default, but it is far too
      // loose for verification: the algebraic stress-damage consistency check in
      // process_solution() bottomed out at ~1.4e-3*StressThreshold and was measuring the LINEAR
      // SOLVER, not the coupling. Tightening to 1e-12 drops that residual by eight orders of
      // magnitude, to ~2e-11*StressThreshold (i.e. round-off), which is what makes the check a
      // real verification of the coupling algebra. Left configurable because tight tolerances
      // cost CG iterations and will not be wanted for every production run -- and because the
      // upcoming refined/perturbed problems are more ill-conditioned, where demanding 1e-12
      // within the iteration cap may not be achievable.
      prm.declare_entry("LinearSolverTolerance",
                        "1e-6",
                        Patterns::Double(0),
                        "Relative tolerance (vs |rhs|) for the CG linear solve");

      prm.declare_entry("MaxLinearSolverIterations",
                        "1000",
                        Patterns::Integer(1),
                        "Iteration cap for the CG linear solve");
    }
    prm.leave_subsection();
    prm.enter_subsection("ParallelizationParameters");
    {
      // Shared-memory threading only (WorkStream/TBB). No MPI, no distributed triangulation, no
      // vector/matrix type changes -- that is a separate, later step.
      prm.declare_entry("UseWorkStream",
                        "true",
                        Patterns::Bool(),
                        "Assemble and predict damage via WorkStream::run (multi-threaded). When "
                        "false, the original single-threaded loops are used instead -- kept as a "
                        "reference implementation for A/B correctness and timing comparison.");

      prm.declare_entry("NumberOfThreads",
                        "0",
                        Patterns::Integer(0),
                        "Thread cap for WorkStream. 0 means let deal.II use all available cores.");
    }
    prm.leave_subsection();
    prm.enter_subsection("MeshRefinementParameters");
    {
      prm.declare_entry("NumberOfRefinementCycles",
                        "1",
                        Patterns::Integer(1),
                        "Number of mesh refinement cycles to run (1 = no refinement, just the "
                        "initial mesh). Each cycle after the first refines the mesh and re-solves "
                        "the full temperature range from scratch on it; damage does not carry over "
                        "between cycles (see refine_grid()/setup_system()), so this is intended for "
                        "mesh convergence studies on damage-free (or single-cycle) MMS tests, not "
                        "yet for refining mid-simulation on a run where damage has already "
                        "accumulated.");

      prm.declare_entry("RefinementStrategy",
                        "adaptive",
                        Patterns::Selection("adaptive|uniform"),
                        "'adaptive' refines via KellyErrorEstimator + refine_and_coarsen_fixed_number "
                        "(FractionToRefine/FractionToCoarsen). 'uniform' does a single global "
                        "refinement of every cell each cycle (triangulation.refine_global(1)) -- the "
                        "textbook mesh-convergence-study method, useful as a cross-check against "
                        "'adaptive' since global refinement isn't solution-dependent and has a clean "
                        "theoretical convergence rate to compare against.");

      prm.declare_entry("FractionToRefine",
                        "0.3",
                        Patterns::Double(0, 1),
                        "Fraction of cells (by estimated error) to flag for refinement each cycle");

      prm.declare_entry("FractionToCoarsen",
                        "0.03",
                        Patterns::Double(0, 1),
                        "Fraction of cells (by estimated error) to flag for coarsening each cycle");
    }
    prm.leave_subsection();
    prm.enter_subsection("OutputParameters");
    {
      prm.declare_entry("OutputDirectory",
                        "output",
                        Patterns::DirectoryName(),
                        "Directory to store output files");

      prm.declare_entry("OutputFileNamePrefix",
                        "solution",
                        Patterns::Anything(),
                        "Prefix for output file names");

      prm.declare_entry("WriteFullStressTensor",
                        "false",
                        Patterns::Bool(),
                        "Also write the six independent components of the GLOBAL-frame stress "
                        "tensor. Off by default: the frame-independent invariants plus the local "
                        "components already carry the physics, so this is clutter for a "
                        "demonstration and costs tens of MB per step on a refined mesh. Turn it "
                        "on for post-hoc analysis that needs the full tensor.");

      prm.declare_entry("OutputFrequency",
                        "10",
                        Patterns::Integer(1),
                        "Frequency of output (in terms of number of steps)");
    }
    prm.leave_subsection();
    prm.enter_subsection("InputFiles");
      {
        prm.declare_entry("MeshFile",
                          "mesh.msh",
                          Patterns::Anything(),
                          "Input mesh file in .msh format");
        prm.declare_entry("CellToMaterialFile",
                          "cell_to_material.txt",
                          Patterns::Anything(),
                          "Input file that maps each cell to a material ID");
        prm.declare_entry("OrientationFile",
                          "orientation.txt",
                          Patterns::Anything(),
                          "Input file that provides orientation matrices for each cell");
      }
    prm.leave_subsection();
    prm.print_parameters("tmp_sol.json", ParameterHandler::JSON);
  }

  void ParameterReader::read_parameters(const std::string &parameter_file)
    {
        declare_parameters();
        prm.parse_input(parameter_file);
    }

  template <int dim>
  class ElasticityTensor
  {
  public:
    ElasticityTensor() = default;
    void setFromPrm(const ParameterHandler &prm);
    SymmetricTensor<4,dim,double> getElasticityTensor(double curOmega, double curTemperature) const;

  private:
    std::unique_ptr<Function<1>> C11_func;
    std::unique_ptr<Function<1>> C33_func;
    std::unique_ptr<Function<1>> C44_func;
    std::unique_ptr<Function<1>> C12_func;
    std::unique_ptr<Function<1>> C13_func;

    double C11_0;
    double C33_0;
    double C44_0;
    double C12_0;
    double C13_0;

  };

  template <int dim>
  void ElasticityTensor<dim>::setFromPrm(const ParameterHandler &prm)
  {
    Assert(dim == 3, ExcNotImplemented());

    this->C11_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C11_0");
    this->C33_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C33_0");
    this->C44_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C44_0");
    this->C12_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C12_0");
    this->C13_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C13_0");

    auto fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C11_functionOfTemperature"), {});
    this->C11_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C33_functionOfTemperature"), {});
    this->C33_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C44_functionOfTemperature"), {});
    this->C44_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C12_functionOfTemperature"), {});
    this->C12_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C13_functionOfTemperature"), {});
    this->C13_func = std::move(fp);
  }

  template <int dim>
  SymmetricTensor<4,dim,double>  ElasticityTensor<dim>::getElasticityTensor(double curOmega, double curTemperature) const
  {
    Point<1> tempPoint(curTemperature);
    double C11 = this->C11_0 * this->C11_func->value(tempPoint);
    double C33 = this->C33_0 * this->C33_func->value(tempPoint) * (1 - curOmega);
    double C44 = this->C44_0 * this->C44_func->value(tempPoint);
    double C12 = this->C12_0 * this->C12_func->value(tempPoint);
    double C13 = this->C13_0 * this->C13_func->value(tempPoint);
    double C66 = (C11 - C12) / 2;

    FullMatrix<double> kelvin_matrix(6, 6);

    kelvin_matrix[0][0] = C11;

    kelvin_matrix[0][1] = C12;
    kelvin_matrix[1][0] = C12;

    kelvin_matrix[0][2] = C13;
    kelvin_matrix[2][0] = C13;

    kelvin_matrix[1][1] = C11;

    kelvin_matrix[1][2] = C13;
    kelvin_matrix[2][1] = C13;

    kelvin_matrix[2][2] = C33;

    kelvin_matrix[3][3] = C44;
    kelvin_matrix[4][4] = C44;

    kelvin_matrix[5][5] = C66;

    SymmetricTensor<4, dim, double> tmpTensor;
    Physics::Notation::Kelvin::to_tensor(kelvin_matrix, tmpTensor); //I should double-check notations (Kelvin, Voight, etc.) to make sure the conversion is correct, and the order of components in the matrix is correct
    return tmpTensor;
  }

  template <int dim>
  class CteTensor
  {
  public:
    CteTensor() = default;
    void setFromPrm(const ParameterHandler &prm);
    SymmetricTensor<2,dim,double> getcteTensor(double curTemperature) const;
  private:
    std::unique_ptr<Function<1>> alpha11_func;
    std::unique_ptr<Function<1>> alpha33_func;
    double alpha11_0;
    double alpha33_0;
  };

  template <int dim>
  void CteTensor<dim>::setFromPrm(const ParameterHandler &prm)
  {
    Assert(dim == 3, ExcNotImplemented());

    alpha11_0 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha11_0");
    alpha33_0 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha33_0");

    auto fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "ThermoElasticParameters"}, "alpha11_functionOfTemperature"), {});
    alpha11_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "ThermoElasticParameters"}, "alpha33_functionOfTemperature"), {});
    alpha33_func = std::move(fp);
  }

  template <int dim>
  SymmetricTensor<2,dim,double> CteTensor<dim>::getcteTensor(double curTemperature) const
  {
    SymmetricTensor<2, dim, double> cteTensor;
    Point<1> tempPoint(curTemperature);
    cteTensor[0][0] = this->alpha11_0 * this->alpha11_func->value(tempPoint);
    cteTensor[1][1] = this->alpha11_0 * this->alpha11_func->value(tempPoint);
    cteTensor[2][2] =this-> alpha33_0 * this->alpha33_func->value(tempPoint);
    return cteTensor; // inference of damage on thermal expansion is not considered in current implementation, but it can be added if needed
  }

  // WorkStream scratch/copy data. ScratchData is per-thread working memory (reused across the
  // cells that thread handles); CopyData carries one cell's result to the serialized copier.
  // Both need a copy constructor because WorkStream clones the prototype once per thread, and
  // FEValues is not copyable -- hence the reconstruct-from-fe/quadrature/flags idiom used here
  // (the same pattern deal.II's own step-32 uses).
  namespace AssemblyScratch
  {
    template <int dim>
    struct Scratch
    {
      Scratch(const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
              const UpdateFlags flags)
        : fe_values(fe, quadrature, flags)
        , globalElasticTensor_q(quadrature.size())
        , JxW_q(quadrature.size())
        , shapeGradSymm(fe.n_dofs_per_cell(),
                        std::vector<SymmetricTensor<2, dim>>(quadrature.size()))
      {}

      Scratch(const Scratch &s)
        : fe_values(s.fe_values.get_fe(), s.fe_values.get_quadrature(),
                    s.fe_values.get_update_flags())
        , globalElasticTensor_q(s.globalElasticTensor_q.size())
        , JxW_q(s.JxW_q.size())
        , shapeGradSymm(s.shapeGradSymm)
      {}

      FEValues<dim> fe_values;
      std::vector<SymmetricTensor<4, dim>> globalElasticTensor_q;
      std::vector<double> JxW_q;
      std::vector<std::vector<SymmetricTensor<2, dim>>> shapeGradSymm;
      // No per-q temperature and no body force any more: the RVE cools uniformly (so the
      // temperature rise is one scalar for the whole step) and the cooling model has no body
      // force at all -- the thermal load enters as an eigenstrain inside sigma. That second
      // point is not cosmetic: a body force would invalidate the average stress theorem that
      // <sigma> = 0 rests on (see homogenization.md section 13).
    };

    struct Copy
    {
      FullMatrix<double> cell_matrix;
      Vector<double> cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
    };
  } // namespace AssemblyScratch

  // Scratch/copy pair for the threaded damage update. Ported from ../uniaxialDamageTest/, where
  // the damage solver was verified against a closed-form reference.
  namespace DamageScratch
  {
    template <int dim>
    struct Scratch
    {
      Scratch(const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
              const UpdateFlags flags)
        : fe_values(fe, quadrature, flags)
        , global_displacement_gradients(quadrature.size())
      {}

      Scratch(const Scratch &s)
        : fe_values(s.fe_values.get_fe(), s.fe_values.get_quadrature(),
                    s.fe_values.get_update_flags())
        , global_displacement_gradients(s.global_displacement_gradients.size())
      {}

      FEValues<dim> fe_values;
      std::vector<Tensor<2, dim>> global_displacement_gradients;
    };

    // Empty: the damage update writes only into the current cell's own quadrature-point data,
    // which no other cell touches, so there is nothing to hand to a serialized copier.
    // WorkStream still requires the type to exist.
    struct Copy
    {};
  } // namespace DamageScratch

  // Emits the grain's c-axis as a genuine 3-vector so ParaView's Glyph filter can consume it
  // directly, with no Calculator step.
  //
  // Why a postprocessor rather than a plain cell array: DataOut's cell data is strictly ONE
  // component per array (data_out_dof_data.templates.h asserts names.size() == 1), so a cell
  // vector cannot be written at all. A DataPostprocessorVector is evaluated per cell -- the
  // current cell is available through get_cell(), which is how the per-grain orientation is
  // looked up -- and does produce a real vector field.
  //
  // NOTE FOR VISUALIZATION: the c-axis is a DIRECTOR, not a vector. c and -c describe the same
  // crystal, so the sign carried here is an artefact of how the direction was sampled. Use a
  // LINE glyph (double-ended), not an Arrow, or the picture will suggest a polarity the material
  // does not have. See README.md.
  template <int dim>
  class CAxisPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    CAxisPostprocessor(const std::vector<Tensor<2, dim>> &orientations)
      : DataPostprocessorVector<dim>("c_axis", update_default)
      , orientations(orientations)
    {}

    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      const auto cell = inputs.template get_cell<dim>();
      // Row 2 of the global->local rotation is the c-axis written in global coordinates.
      const Tensor<2, dim> &R = orientations[cell->material_id() - 1];
      for (auto &q : computed_quantities)
        for (unsigned int d = 0; d < dim; ++d)
          q(d) = R[dim - 1][d];
    }

  private:
    const std::vector<Tensor<2, dim>> &orientations;
  };

  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem(ParameterHandler &prm);
    void run();
    void meshTest();
    void elasticCalculationTest();
    void elasticAndDamageTest();
    void printProblemParametersToJson(const std::string filename);

  private:

    void setup_system();
    // Public entry points dispatch to a serial or a WorkStream implementation based on
    // ParallelizationParameters/UseWorkStream. The serial versions are kept deliberately (not
    // as dead code): they are the reference for A/B correctness checking and for measuring the
    // actual speedup, and a threading bug is much easier to localize with a known-good path to
    // diff against.
    // assemble_system() does the setup common to both paths (constraints, zeroing) and then
    // dispatches the CELL LOOP to one of these two.
    void assemble_system(double curTemperature);
    void assemble_cells_serial(double curTemperature);
    void assemble_cells_workstream(double curTemperature);
    void solve();
    mutable unsigned int lastCgIterations = 0;
    // output_results() is const and needs the current temperature to re-evaluate stress; the
    // temperature-step loop records it here rather than threading it through the signature.
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

    // Fills `constraints` for the selected BC family. Called from assemble_system() on every
    // iteration, because AffineConstraints must be rebuilt whenever it is re-closed.
    void applyBoundaryConstraints();

    // Minimal kinematic support for "freeSurface": exactly 6 scalar constraints removing the 3
    // translations and 3 rotations, and NOTHING else -- any extra constraint would apply a real
    // load and destroy the <sigma> = 0 property the free-surface case exists to have.
    void constrainRigidBodyModes();

    // ----------------------------------------------------------------------------------------
    // Macroscopic strain unknowns for the periodic formulation  -- TEMPLATE, NOT IMPLEMENTED
    // ----------------------------------------------------------------------------------------
    // See homogenization.md sections 6 and 8 for the derivation these must match. The (n+6)
    // system is
    //
    //     [ K   L  ] [ U    ]   [ F_theta ]        K = integral( B^T C B )       (n x n, sparse)
    //     [ L^T H  ] [ Ebar ] = [ P_theta ]        L = integral( B^T C   )       (n x 6, DENSE)
    //                                             H = integral( C       )       (6 x 6, dense)
    //
    // and is NEVER assembled as one matrix: the six dense rows/columns would wreck SSOR/AMG
    // (homogenization.md section 11). Instead K keeps exactly the sparsity, symmetry and
    // preconditioner it has today, L is six ordinary vectors and H is a 6x6 FullMatrix, and the
    // coupling is resolved by static condensation in solve_with_zero_mean_stress().
    //
    // Note L and the 6-row block are transposes of one another BY MAJOR SYMMETRY OF C. Assemble
    // them independently and check: if they do not come out transposed, something is wrong. That
    // same symmetry is what keeps the condensed system SPD and CG applicable.
    static constexpr unsigned int nMacroDofs = 6; // Kelvin 6-vector, matching Physics::Notation

    // Assembles the three coupling objects below alongside the usual K/F_theta. Must be called
    // with the same frozen damage state as assemble_system().
    void assemble_macroscopic_blocks(double curTemperature);

    // The 7-solve: z = K^-1 F_theta, Y = K^-1 L (6 columns, SAME matrix and preconditioner),
    // Cbar_times_volume = H - L^T Y, then Ebar = (H - L^T Y)^-1 (P_theta - L^T z) and finally
    // U = z - Y Ebar with no extra solve. Leaves the answer in solDisplacement/macroscopicStrain.
    void solve_with_zero_mean_stress();

    // Volume average of the stress over the RVE, by plain quadrature summation. Used both as the
    // <sigma> = 0 residual check and (with unit macroscopic strains) to build Cbar column by
    // column -- see homogenization.md section 8, which recommends this over forming L^T
    // explicitly, since the constrained-vs-unconstrained L distinction is an easy silent bug.
    SymmetricTensor<2, dim> computeAverageStress(double curTemperature) const;

    // L: six right-hand-side vectors, one per Kelvin component of Ebar.
    std::vector<Vector<double>> macroCouplingColumns;
    // H = integral(C) over the RVE, and the homogenized stiffness that condensation produces.
    FullMatrix<double> macroStiffnessH{nMacroDofs, nMacroDofs};
    FullMatrix<double> homogenizedStiffness{nMacroDofs, nMacroDofs};
    // The macroscopic strain itself: zero for every non-periodic BC family.
    SymmetricTensor<2, dim> macroscopicStrain;

    // ----------------------------------------------------------------------------------------
    // Damage  (ported from ../uniaxialDamageTest/, where it was verified against a closed form)
    // ----------------------------------------------------------------------------------------
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

    bool checkActivationCriteria(SymmetricTensor<2,dim> localStress);
    double kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc);
    double derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc);

    // Plain per-point value snapshot, NOT a copy of the CellDataStorage object itself:
    // CellDataStorage's implicit copy constructor/assignment only deep-copies its
    // std::map<CellId, std::vector<std::shared_ptr<DataType>>> -- the shared_ptrs inside are
    // copied (refcounted), not the DamageQData objects they point to. So
    // `CellDataStorage<...> a = b;` leaves `a` and `b` aliasing the SAME underlying damage
    // objects: mutating one through get_data() mutates both. Found via maxDamageInc always
    // printing exactly 0 in checkConvergenceCriteria's output -- confirmed against deal.II's
    // quadrature_point_data.h, which declares no custom copy ctor/assignment for
    // CellDataStorage, so the compiler-generated member-wise copy (shallow on the shared_ptrs)
    // is what actually runs. A real "previous iteration" snapshot needs the scalar values
    // copied out into independent storage instead -- see snapshotDamage().
    using DamageSnapshot = std::map<typename Triangulation<dim>::cell_iterator, std::vector<double>>;
    DamageSnapshot snapshotDamage();
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

    // (pseudo-time, filename) pairs for the ParaView .pvd time series written at the end of
    // run(). The time value is the accumulated cooling s = T_0 - T, which INCREASES over the run
    // -- temperature itself decreases, and ParaView orders a series by increasing time, so using
    // T directly would play the animation backwards.
    mutable std::vector<std::pair<double, std::string>> pvdTimeSeries;

  };

  template <int dim>
  void ElasticProblem<dim>::printProblemParametersToJson(const std::string filename)
  {
    prm.print_parameters(filename, ParameterHandler::JSON);
  }

  template <int dim>
  ElasticProblem<dim>::ElasticProblem(ParameterHandler &prm)
    : prm(prm)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
  {};

  template <int dim>
  std::istringstream ElasticProblem<dim>::parseMSHandPrepareStream(std::string mshFileName) //Designed to take output Neper .msh file, parse and clear additional data, so output stream wil be applyable for read_msh
  {
    std::ifstream msh_file(mshFileName);
    std::ofstream outFile("test.msh");
    std::ostringstream out;
    std::string line;

    while (std::getline(msh_file, line))
    {
      if(line == "$ElsetOrientations")
      {
        std::getline(msh_file, line);
        std::istringstream header_stream(line);
        unsigned int num_elsets;
        std::string orientation_type;
        if (!(header_stream >> num_elsets >> orientation_type)) throw std::runtime_error("Invalid $ElsetOrientations header format");
        for (unsigned int i = 0; i < num_elsets; ++i)
        {
          if (!std::getline(msh_file, line)) throw std::runtime_error("Premature EOF in $ElsetOrientations section");
          if (line.empty())
          {
              --i; // retry this iteration
              continue;
          }
          std::istringstream data_stream(line);
          unsigned int elset_id;
          Tensor<2,dim> m;
          if (!(data_stream >> elset_id >> m[0][0] >> m[0][1] >> m[0][2] >> m[1][0] >> m[1][1] >> m[1][2] >> m[2][0] >> m[2][1] >> m[2][2])) throw std::runtime_error("Invalid orientation data line: " + line);
          rotmatOrientationCell.push_back(m);
        }
        std::getline(msh_file, line);
        std::getline(msh_file, line);
      }
      out << line << '\n';
    }
  if (outFile.is_open())
  {
    outFile << out.str(); // Get content as a string and write
    outFile << "lol" << '\n';
    outFile.close();
  } else{
    cout << "well, it was not opened\n";
  }
  return std::istringstream(out.str());
  }
  
  template <int dim>
  void ElasticProblem<dim>::parse_cellToMaterial_data(std::string fileName)
  {
    std::ifstream msh_file(fileName);
    std::string line;

    while (std::getline(msh_file, line)) 
    {
      std::istringstream header_stream(line);
      unsigned int materialId;
      if (!(header_stream >> materialId)) throw std::runtime_error("bad cell_to_material file");
      cellToMaterial.push_back(materialId);
    }
  }

  template <int dim>
  SymmetricTensor<2,dim> ElasticProblem<dim>::getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature) const //provide localStrain!!!
  {
    double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    return elasticityTensor.getElasticityTensor(curOmega, curTemperature) * (localStrain - cteTensor.getcteTensor(curTemperature) * (curTemperature - T0)); 
  }

  template <int dim>
  Tensor<2,dim,double> ElasticProblem<dim>::getRotTensorGlobalToLocal(int cellId) const
  {
    Assert(dim == 3, ExcNotImplemented());
    return rotmatOrientationCell[cellId - 1];
  }

  template <int dim>
  void ElasticProblem<dim>::setup_system() // safe to call again after mesh refinement: clears and
                                            // freshly reinitializes damageInQuadraturePoints (damage
                                            // resets to 0 on the new mesh -- refinement does not
                                            // transfer quadrature-point data, by design for now)
  {
    elasticityTensor.setFromPrm(prm);
    cteTensor.setFromPrm(prm);
    TimerOutput::Scope timer_section(timer, "Setup of system");
    const QGauss<dim> quadrature_formula( fe.degree + 1);
    const unsigned int n_q_points    = quadrature_formula.size();
    // Without this, repeated setup_system() calls (once per refinement cycle) would leave
    // orphaned entries for cells that are no longer active (refined away or coarsened into
    // their parent) sitting in damageInQuadraturePoints alongside the fresh ones.
    damageInQuadraturePoints.clear();
    damageInQuadraturePoints.initialize(triangulation.begin_active(),
                                          triangulation.end(),
                                          n_q_points);

    for (auto &cell : dof_handler.active_cell_iterators())
    {
      damageInQuadraturePoints.initialize(cell, quadrature_formula.size());
    }

    dof_handler.distribute_dofs(fe);
    solDisplacement.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // The sparsity pattern must be built from a constraint set that already contains every
    // constraint that will ever couple two DoFs. Hanging nodes and Dirichlet conditions only
    // ever eliminate rows, but PERIODICITY genuinely couples DoFs on opposite faces: leaving it
    // out here would give a pattern with no entry for those couplings, and
    // distribute_local_to_global() would then silently drop them. So build the real constraint
    // set once here, purely for the pattern.
    applyBoundaryConstraints();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    // L's six columns live on the same DoF numbering as the solution vector.
    macroCouplingColumns.assign(nMacroDofs, Vector<double>(dof_handler.n_dofs()));
    macroscopicStrain = 0;
  }

  // Label the six faces of the cube geometrically. The .msh carries no surface elements at all,
  // so every face arrives with the default id 0 and there is nothing to inherit from the file.
  // Called once after the mesh is read: child faces inherit their parent's boundary id, so
  // refinement preserves this classification and it must NOT be redone per cycle.
  template <int dim>
  void ElasticProblem<dim>::assignBoundaryIds()
  {
    const double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    // Loose relative to L: the tolerance only has to separate a face from the cell layer behind
    // it, and mesh coordinates come from a text file with finite precision.
    const double tol = 1e-8 * L;

    std::array<unsigned int, 6> counts{};
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
      {
        if (!face->at_boundary())
          continue;
        const Point<dim> c = face->center();
        if      (std::abs(c[0])     < tol) { face->set_boundary_id(BoundaryIds::xMinus); ++counts[0]; }
        else if (std::abs(c[0] - L) < tol) { face->set_boundary_id(BoundaryIds::xPlus);  ++counts[1]; }
        else if (std::abs(c[1])     < tol) { face->set_boundary_id(BoundaryIds::yMinus); ++counts[2]; }
        else if (std::abs(c[1] - L) < tol) { face->set_boundary_id(BoundaryIds::yPlus);  ++counts[3]; }
        else if (std::abs(c[2])     < tol) { face->set_boundary_id(BoundaryIds::zMinus); ++counts[4]; }
        else if (std::abs(c[2] - L) < tol) { face->set_boundary_id(BoundaryIds::zPlus);  ++counts[5]; }
        else
          throw std::runtime_error(
            "boundary face at (" + std::to_string(c[0]) + "," + std::to_string(c[1]) + "," +
            std::to_string(c[2]) + ") lies on none of the six planes of the cube [0,L]^3 with "
            "L = " + std::to_string(L) + " -- LengthOfTheBody does not match the mesh, or the "
            "mesh is not the axis-aligned cube this model assumes.");
      }

    // Periodic matching needs the pairs to be the same size; catching it here beats a confusing
    // failure inside collect_periodic_faces later.
    for (unsigned int pair = 0; pair < 3; ++pair)
      if (counts[2 * pair] != counts[2 * pair + 1])
        throw std::runtime_error("boundary face counts differ across face pair " +
                                  std::to_string(pair) + " (" +
                                  std::to_string(counts[2 * pair]) + " vs " +
                                  std::to_string(counts[2 * pair + 1]) +
                                  ") -- the mesh is not periodic in that direction.");

    std::cout << "Boundary faces per pair: x=" << counts[0] << ", y=" << counts[2]
              << ", z=" << counts[4] << std::endl;
  }

  // Exactly 6 scalar constraints removing the 6 rigid-body modes of a traction-free body, and
  // nothing more. Any additional constraint would transmit a real reaction force and so destroy
  // the <sigma> = 0 property that makes the free-surface case a genuine free-cooling state.
  //
  // The standard 3-2-1 support, on three corners of the cube:
  //   corner at (0,0,0): u_x = u_y = u_z = 0   kills the 3 translations
  //   corner at (L,0,0): u_y = u_z = 0         kills rotation about e_z and about e_y
  //   corner at (0,L,0): u_z = 0               kills rotation about e_x
  // The three constrained points are not collinear, which is what makes the set independent.
  //
  // WARNING (conditioning): this leaves the stiffness matrix only barely positive definite --
  // the modes it removes are removed exactly, but nothing else is stiffened. Combined with the
  // C11/C44 ~ 4240 anisotropy of graphite this is the "floppy BC" regime measured earlier in
  // ../uniaxialDamageTest/, where CG needed roughly 40x the iterations of a well-supported
  // problem on the same mesh. Expect the free-surface runs to be much slower per solve than the
  // clamped ones, and watch the reported CG iteration counts.
  template <int dim>
  void ElasticProblem<dim>::constrainRigidBodyModes()
  {
    const double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    const double tol = 1e-8 * L;

    // Support point, and which displacement components are fixed there.
    struct Support { Point<dim> at; std::vector<unsigned int> components; };
    const std::array<Support, 3> supports{{
      { Point<dim>(0.0, 0.0, 0.0), {0, 1, 2} },  // all three translations
      { Point<dim>(L,   0.0, 0.0), {1, 2}    },  // rotations about e_z and e_y; u_x LEFT FREE
      { Point<dim>(0.0, L,   0.0), {2}       }   // rotation about e_x
    }};
    // u_x must stay free at the second support, and u_x and u_y at the third: fixing them would
    // stop the body contracting along those axes, which is a real applied load, not a support.

    std::vector<Point<dim>> supportLocations(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, supportLocations);

    std::array<std::array<bool, dim>, 3> constrained{};
    std::vector<types::global_dof_index> dofsOnCell(fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(dofsOnCell);
      for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k)
      {
        const unsigned int component = fe.system_to_component_index(k).first;
        const types::global_dof_index gi = dofsOnCell[k];
        for (unsigned int s = 0; s < supports.size(); ++s)
        {
          if (constrained[s][component] ||
              supportLocations[gi].distance(supports[s].at) > tol)
            continue;
          const auto &comps = supports[s].components;
          if (std::find(comps.begin(), comps.end(), component) == comps.end())
            continue;
          if (!constraints.is_constrained(gi))
            constraints.add_line(gi);   // homogeneous: the inhomogeneity defaults to 0
          constrained[s][component] = true;
        }
      }
    }

    unsigned int nConstraints = 0;
    for (const auto &perSupport : constrained)
      for (unsigned int d = 0; d < dim; ++d)
        if (perSupport[d]) ++nConstraints;
    if (nConstraints != 6)
      throw std::runtime_error(
        "minimal kinematic support produced " + std::to_string(nConstraints) +
        " constraints instead of 6 -- the three support corners were not all found as mesh "
        "vertices. Check LengthOfTheBody against the mesh.");
  }

  // Rebuilds `constraints` from scratch for the selected BC family. Must be called before every
  // assembly, because AffineConstraints cannot be added to once closed.
  template <int dim>
  void ElasticProblem<dim>::applyBoundaryConstraints()
  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    switch (boundaryConditionType)
    {
      case BoundaryConditionType::zeroDisplacement:
      {
        // u = 0 on the whole boundary. Kills every rigid-body mode as a side effect, so no
        // extra support is needed. Applied per face id rather than to id 0, because
        // assignBoundaryIds() has already relabelled everything.
        const Functions::ZeroFunction<dim> zero(dim);
        for (const types::boundary_id id : {BoundaryIds::xMinus, BoundaryIds::xPlus,
                                            BoundaryIds::yMinus, BoundaryIds::yPlus,
                                            BoundaryIds::zMinus, BoundaryIds::zPlus})
          VectorTools::interpolate_boundary_values(dof_handler, id, zero, constraints);
        break;
      }

      case BoundaryConditionType::freeSurface:
      {
        // Traction-free needs NO boundary term at all: zero traction is the natural boundary
        // condition of the weak form, so simply not integrating anything over the boundary is
        // exactly right. Only the rigid-body modes need removing.
        constrainRigidBodyModes();
        break;
      }

      case BoundaryConditionType::periodic:
      {
        // -----------------------------------------------------------------------------------
        // TEMPLATE -- NOT YET IMPLEMENTED. Intended shape:
        //
        //   std::vector<GridTools::PeriodicFacePair<
        //     typename DoFHandler<dim>::cell_iterator>> matchedPairs;
        //   GridTools::collect_periodic_faces(dof_handler, BoundaryIds::xMinus,
        //                                     BoundaryIds::xPlus, /*direction=*/0, matchedPairs);
        //   ... same for y (direction 1) and z (direction 2) ...
        //   DoFTools::make_periodicity_constraints<dim, dim>(matchedPairs, constraints);
        //
        // These are HOMOGENEOUS constraints on the fluctuation u_periodic. Ebar does NOT enter
        // here: it is carried as six separate unknowns and enters assembly as an eigenstrain
        // (homogenization.md section 8), which is why no inhomogeneity is needed and why the
        // existing thermal-load code path can be reused verbatim.
        //
        // Then pin ONE vertex (3 components) to remove the rigid translations. Rotations are
        // already excluded, because a rigid rotation field is not periodic -- so three
        // constraints is exactly enough and pinning more would apply a real load.
        // -----------------------------------------------------------------------------------
        throw std::runtime_error(
          "BoundaryConditions/Type = 'periodic' is not implemented yet. Use 'zeroDisplacement' "
          "or 'freeSurface'. See applyBoundaryConstraints(), assemble_macroscopic_blocks() and "
          "solve_with_zero_mean_stress() for the templates, and homogenization.md for the "
          "derivation they must implement.");
      }
    }

    constraints.close();
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system(double curTemperature)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    applyBoundaryConstraints();

    // AffineConstraints::distribute_local_to_global() ADDS local contributions into
    // system_matrix/system_rhs rather than overwriting them. assemble_system() is called
    // repeatedly (once per displacement-damage iteration, and again inside
    // checkConvergenceCriteria), so without resetting here every call keeps accumulating
    // on top of all previous calls -- system_matrix/system_rhs would grow unboundedly
    // over a temperature step instead of reflecting only the current state.
    system_matrix = 0;
    system_rhs    = 0;

    if (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream"))
      assemble_cells_workstream(curTemperature);
    else
      assemble_cells_serial(curTemperature);
  }

  // Reference single-threaded cell loop. Kept as the correctness/timing baseline for the
  // WorkStream version below -- the two must produce bit-identical matrices.
  template <int dim>
  void ElasticProblem<dim>::assemble_cells_serial(double curTemperature)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(fe, quadrature_formula,
      update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Per-(cell, q) data that does NOT depend on the (i, j) test/trial function indices:
    // hoisted out of the i,j loop and evaluated once per q instead of dofs_per_cell^2
    // (stiffness) / dofs_per_cell (rhs) times per q -- a rank-4 tensor rotation is expensive
    // enough that this was the dominant assembly cost before.
    std::vector<SymmetricTensor<4,dim>> globalElasticTensor_q(n_q_points);
    std::vector<double> JxW_q(n_q_points);
    std::vector<std::vector<SymmetricTensor<2,dim>>> shapeGradSymm(
      dofs_per_cell, std::vector<SymmetricTensor<2,dim>>(n_q_points));

    // The RVE cools uniformly, so the temperature rise is ONE scalar for the whole domain, not
    // a field. It is negative on cooling.
    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const double deltaT = curTemperature - T0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2,dim> rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      const Tensor<2,dim> rotTensorLocalToGlobal = transpose(rotTensorGlobalToLocal);

      cell_matrix = 0;
      cell_rhs    = 0;

      // Both material tensors are defined in the crystallite's own frame and rotated into the
      // global frame per cell. This is where the grain-to-grain mismatch actually enters: two
      // neighbouring cells with different orientations get different global C and different
      // global alpha from identical local data.
      const SymmetricTensor<2,dim> localcteTensor = cteTensor.getcteTensor(curTemperature);
      const SymmetricTensor<2,dim> globalcteTensor =
        Physics::Transformations::basis_transformation(localcteTensor, rotTensorLocalToGlobal);

      for (const unsigned int q : fe_values.quadrature_point_indices())
      {
        JxW_q[q] = fe_values.JxW(q);
        const SymmetricTensor<4,dim> localElasticTensor =
          elasticityTensor.getElasticityTensor(cell_damage[q]->damage, curTemperature);
        globalElasticTensor_q[q] =
          Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
      }

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int q : fe_values.quadrature_point_indices())
          shapeGradSymm[i][q] = symmetrize(fe_values[displacement].gradient(i, q));

      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            cell_matrix(i, j) +=
              (shapeGradSymm[i][q] * (globalElasticTensor_q[q] * shapeGradSymm[j][q])) * JxW_q[q];

        // Thermal load only -- there is NO body force in this model. Unlike the uniform-
        // temperature MMS case (where this term integrated to zero against every admissible
        // test function because C and alpha were spatially constant), here C and alpha jump
        // from grain to grain, so the integrand is discontinuous and the term does NOT vanish.
        // That discontinuity is precisely the physics: it is the mismatch load.
        for (const unsigned int q : fe_values.quadrature_point_indices())
          cell_rhs(i) +=
            ((shapeGradSymm[i][q] * (globalElasticTensor_q[q] * globalcteTensor)) * deltaT)
            * JxW_q[q];
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  }

  // Multi-threaded cell loop. The worker is pure per-cell work writing only into its own
  // CopyData; the copier -- which touches the shared system_matrix/system_rhs -- is run
  // serialized by WorkStream, so no explicit locking is needed.
  template <int dim>
  void ElasticProblem<dim>::assemble_cells_workstream(double curTemperature)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const UpdateFlags flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;

    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const double deltaT = curTemperature - T0; // uniform over the RVE; negative on cooling

    auto worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                      AssemblyScratch::Scratch<dim> &scratch,
                      AssemblyScratch::Copy &copy)
    {
      FEValues<dim> &fe_values = scratch.fe_values;
      fe_values.reinit(cell);
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2,dim> rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      const Tensor<2,dim> rotTensorLocalToGlobal = transpose(rotTensorGlobalToLocal);

      copy.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      copy.cell_rhs.reinit(dofs_per_cell);
      copy.local_dof_indices.resize(dofs_per_cell);
      copy.cell_matrix = 0;
      copy.cell_rhs    = 0;

      const SymmetricTensor<2,dim> localcteTensor = cteTensor.getcteTensor(curTemperature);
      const SymmetricTensor<2,dim> globalcteTensor =
        Physics::Transformations::basis_transformation(localcteTensor, rotTensorLocalToGlobal);

      for (const unsigned int q : fe_values.quadrature_point_indices())
      {
        scratch.JxW_q[q] = fe_values.JxW(q);
        const SymmetricTensor<4,dim> localElasticTensor =
          elasticityTensor.getElasticityTensor(cell_damage[q]->damage, curTemperature);
        scratch.globalElasticTensor_q[q] =
          Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
      }

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int q : fe_values.quadrature_point_indices())
          scratch.shapeGradSymm[i][q] = symmetrize(fe_values[displacement].gradient(i, q));

      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            copy.cell_matrix(i, j) +=
              (scratch.shapeGradSymm[i][q] *
               (scratch.globalElasticTensor_q[q] * scratch.shapeGradSymm[j][q])) * scratch.JxW_q[q];

        // Thermal load only -- no body force in this model (see the serial path).
        for (const unsigned int q : fe_values.quadrature_point_indices())
          copy.cell_rhs(i) +=
            ((scratch.shapeGradSymm[i][q] * (scratch.globalElasticTensor_q[q] * globalcteTensor))
             * deltaT) * scratch.JxW_q[q];
      }
      cell->get_dof_indices(copy.local_dof_indices);
    };

    auto copier = [&](const AssemblyScratch::Copy &copy)
    {
      constraints.distribute_local_to_global(copy.cell_matrix, copy.cell_rhs,
                                             copy.local_dof_indices, system_matrix, system_rhs);
    };

    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), worker, copier,
                    AssemblyScratch::Scratch<dim>(fe, quadrature_formula, flags),
                    AssemblyScratch::Copy());
  }

  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    TimerOutput::Scope timer_section(timer, "Solve system");
    const double linearSolverTolerance = prm.get_double({"SolverParameters"}, "LinearSolverTolerance");
    const unsigned int maxLinearIterations =
      prm.get_integer({"SolverParameters"}, "MaxLinearSolverIterations");
    SolverControl            solver_control(maxLinearIterations,
                                            linearSolverTolerance * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.2));

    cg.solve(system_matrix,  solDisplacement, system_rhs, preconditioner);

    // A plain runtime check rather than Assert(): Assert() compiles to a no-op in Release
    // builds (-DNDEBUG), so relying on it here would mean a genuinely non-converged CG
    // solve could silently pass through in Release with no indication anything went wrong.
    if (solver_control.last_check() != SolverControl::success)
      throw std::runtime_error("CG solver did not converge (last residual " +
                                std::to_string(solver_control.last_value()) + ")");

    lastCgIterations = solver_control.last_step();
    constraints.distribute(solDisplacement);
  }

  template <int dim>
  void ElasticProblem<dim>::refine_grid(double fractionToRefine, double fractionToCoarse)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    // KellyErrorEstimator estimates jumps in the solution gradient across cell *faces*,
    // hence a (dim-1)-dimensional face quadrature rule, not a dim-dimensional cell one.
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                        solDisplacement,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    fractionToRefine,
                                                    fractionToCoarse);

    // No quadrature-point data transfer across refinement/coarsening (yet -- see
    // setup_system()'s comment): damage is reset to 0 on the new mesh via the
    // setup_system() call that follows this function at every call site. This is
    // exact (not an approximation) for any state that starts a refinement cycle at
    // damage=0 everywhere, e.g. a fresh MMS mesh-convergence run; it would lose real
    // accumulated damage if refinement were ever triggered mid-simulation on a run
    // where damage is already nonzero, which is not yet supported.
    triangulation.execute_coarsening_and_refinement();
    setup_system();
  }

  template <int dim>
  bool ElasticProblem<dim>::checkActivationCriteria(SymmetricTensor<2,dim> localStress)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    return localStress[dim-1][dim-1] > stressThreshold;
  }

  // Backward-Euler residual of the Rabotnov-type kinetic equation
  //   domega/ds = A * (macaulay(sigma_zz - sigma_th)/sigma_th)^m * (1/(1-omega))^m
  // discretized as omega_new - omega_old = A*(...)^m * tempInc, where s = T_0 - T is the
  // (positive, increasing) pseudo-time and tempInc is its step magnitude. sigma_zz is the
  // LOCAL-frame c-axis normal stress -- which is why the per-grain orientation matters here and
  // not only in assembly.
  //
  // Two corrections carried over from the debugging in ../mmsSandbox, both easy to reintroduce:
  //  - actingStress is Macaulay-clamped rather than throwing: the root search evaluates the
  //    residual across a whole bracket, including omega values where the stress is subcritical,
  //    so a hard throw there would abort on perfectly legitimate probe evaluations. Below
  //    threshold the physical damage rate is genuinely zero, which the clamp expresses directly.
  //  - actingStress is normalized by stressThreshold, matching the kinetic law as written above.
  //    It previously was not, while derivativeKineticResidual's power term WAS -- so the
  //    residual and its own derivative disagreed by a factor of stressThreshold^m.
  template <int dim>
  double ElasticProblem<dim>::kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    double actingStress = std::max(localStress[dim-1][dim-1] - stressThreshold, 0.0);
    double a_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    return curOmega - oldOmega
           - a_kineticParam * std::pow(actingStress / stressThreshold / (1-curOmega), m_kineticParam) * tempInc;
  };

  template <int dim>
  double ElasticProblem<dim>::derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    double a_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    double actingStress = std::max(localStress[dim-1][dim-1] - stressThreshold, 0.0);
    return 1 - a_kineticParam * m_kineticParam * tempInc
               * std::pow(actingStress / stressThreshold, m_kineticParam)
               / std::pow(1 - curOmega, m_kineticParam + 1);
  }

  template <int dim>
  void ElasticProblem<dim>::predict_damage_tempInc_external_solver(double curTemperature, double tempInc)
  {
    TimerOutput::Scope timer_section(timer, "Predict damage increment");

    // Kinetic-equation constants, read once rather than per quadrature point (this loop is the
    // dominant cost in the timing report), then passed down to the per-cell worker.
    const double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    const double A_param = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    const double m_param = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    const double residualTolerance = prm.get_double({"SolverParameters"}, "KineticAlgebraicSolverTolerance");

    if (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream"))
      predict_damage_cells_workstream(curTemperature, tempInc, stressThreshold, A_param,
                                       m_param, residualTolerance);
    else
      predict_damage_cells_serial(curTemperature, tempInc, stressThreshold, A_param,
                                   m_param, residualTolerance);
  }

  // The per-cell damage update, shared verbatim by the serial and threaded paths so they cannot
  // drift apart. Writes only into this cell's own quadrature-point data -- no other cell reads or
  // writes it -- which is what makes the threaded version safe with no copier and no locking.
  template <int dim>
  void ElasticProblem<dim>::update_damage_on_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    FEValues<dim> &fe_values,
    std::vector<Tensor<2, dim>> &global_displacement_gradients,
    double curTemperature, double tempInc,
    double stressThreshold, double A_param, double m_param, double residualTolerance)
  {
    SymmetricTensor<2, dim> globalStrain;
    SymmetricTensor<2, dim> localStrain;
    SymmetricTensor<2, dim> localStress;

    auto cell_damage = damageInQuadraturePoints.get_data(cell);
    const Tensor<2, dim> rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
    fe_values.reinit(cell);
    const auto &qPoints = fe_values.get_quadrature_points();
    const unsigned int n_q_points = fe_values.get_quadrature().size();

    // One call fills the gradients at ALL quadrature points, rather than recomputing the whole
    // set once per point inside the q loop below.
    fe_values[displacement].get_function_gradients(solDisplacement, global_displacement_gradients);

    for (unsigned int q = 0; q < n_q_points; q++)
    {
        Point<dim> curQpoint = qPoints[q];
        double old_damage = cell_damage[q]->old_damage;
        double cur_damage = cell_damage[q]->damage;
        // Total strain = macroscopic + fluctuation. macroscopicStrain is identically zero for
        // every BC family except "periodic", so this reduces to sym(grad u) today and stays
        // correct once the macroscopic unknowns are switched on.
        globalStrain = dealii::symmetrize(global_displacement_gradients[q]) + macroscopicStrain;
        localStrain = Physics::Transformations::basis_transformation(globalStrain, rotTensorGlobalToLocal);
        localStress = getLocalStressTensor(localStrain, cur_damage, curTemperature);

        // The driving stress is the REAL stress computed from the current numerical damage, so
        // the negative-feedback loop is closed: growing damage lowers C33 -> lowers sigma_zz ->
        // slows further growth.
        const double lowerBound = old_damage; // damage is non-decreasing within a load step
        const double upperBound = 0.999;      // stay clear of the (1-omega)->0 singularity
        auto residualOnly = [&](double omega) -> double
        {
          // Stress is re-evaluated at each trial omega, so the root search sees the true
          // coupled equation rather than a stress frozen at the previous iterate.
          const SymmetricTensor<2,dim> trialStress = getLocalStressTensor(localStrain, omega, curTemperature);
          return kineticResidual(trialStress, omega, old_damage, tempInc);
        };

        // The damage term A*(ratio)^m*(1-omega)^-m is convex in omega whenever actingStress>0,
        // making the residual CONCAVE (single interior maximum, up to two roots) -- not safe for
        // raw Newton from an arbitrary guess. Locate that maximum analytically rather than
        // searching for it, then bracket-and-solve on whichever side actually contains a sign
        // change, preferring the smaller root (the branch continuous with old_damage).
        //
        // NOTE: the closed form below is specific to this Rabotnov-type power law -- it solves
        // d/domega[A*(ratio)^m*(1-omega)^-m * tempInc] = 1 analytically. A different kinetic law
        // would need its own derivation, or a law-agnostic fallback (numeric maximization, or a
        // coarse scan for sign changes). Worth revisiting if a second law is ever added.
        const double actingStressAtPoint =
          std::max(getLocalStressTensor(localStrain, old_damage, curTemperature)[dim-1][dim-1] - stressThreshold, 0.0);
        double newDamage, lo = lowerBound, hi = upperBound;
        bool rootBracketed = true;
        boost::uintmax_t bisectItersUsed = 0;

        if (actingStressAtPoint <= 0.0)
        {
          // Subcritical: no damage growth this step, and the residual reduces to
          // omega - old_damage whose root is exactly old_damage.
          newDamage = old_damage;
        }
        else
        {
          const double omega_max = std::clamp(
            1.0 - std::pow(m_param * A_param * std::pow(actingStressAtPoint / stressThreshold, m_param) * tempInc,
                            1.0 / (m_param + 1.0)),
            lowerBound, upperBound);
          const double residAtLower = residualOnly(lowerBound);
          const double residAtMax = residualOnly(omega_max);
          if (omega_max > lowerBound && (residAtLower < 0.0) != (residAtMax < 0.0))
          { lo = lowerBound; hi = omega_max; }
          else if (omega_max < upperBound && (residAtMax < 0.0) != (residualOnly(upperBound) < 0.0))
          { lo = omega_max; hi = upperBound; }
          else
            rootBracketed = false; // no sign change anywhere -- no root in [lowerBound, upperBound]

          if (rootBracketed)
          {
            // toms748_solve: bracket-guaranteed, superlinear. Tight x-tolerance here -- an
            // x-tolerance equal to residualTolerance isn't the same thing as a residual
            // tolerance near the (1-omega)^-m singularity, so the residual check below is the
            // real acceptance criterion.
            boost::uintmax_t bisectIters = 200;
            auto bracket = boost::math::tools::toms748_solve(residualOnly, lo, hi,
              [](double a, double b) { return std::abs(b - a) < 1e-12; }, bisectIters);
            bisectItersUsed = bisectIters;
            newDamage = 0.5 * (bracket.first + bracket.second);
          }
          else
            newDamage = lowerBound;
        }

        const double residualAtSolution = residualOnly(newDamage);
        if (!rootBracketed || std::abs(residualAtSolution) > residualTolerance)
          throw std::runtime_error("Kinetic equation root search did not converge (residual=" +
                                    std::to_string(residualAtSolution) +
                                    ", rootBracketed=" + std::to_string(rootBracketed) +
                                    ") at point " +
                                    std::to_string(curQpoint[0]) + "," + std::to_string(curQpoint[1]) +
                                    "," + std::to_string(curQpoint[2]) +
                                    " [old_damage=" + std::to_string(old_damage) +
                                    " newDamage=" + std::to_string(newDamage) +
                                    " T=" + std::to_string(curTemperature) +
                                    " tempInc=" + std::to_string(tempInc) +
                                    " actingStress=" + std::to_string(actingStressAtPoint) +
                                    " bracket=[" + std::to_string(lo) + "," + std::to_string(hi) + "]" +
                                    " residAtBracket=[" + std::to_string(residualOnly(lo)) + "," +
                                    std::to_string(residualOnly(hi)) + "]" +
                                    " bisectItersUsed=" + std::to_string(bisectItersUsed) + "]");
        cell_damage[q]->damage = newDamage;
    }
  }

  // Reference single-threaded loop, kept as the correctness/timing baseline.
  template <int dim>
  void ElasticProblem<dim>::predict_damage_cells_serial(
    double curTemperature, double tempInc,
    double stressThreshold, double A_param, double m_param, double residualTolerance)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points);
    std::vector<Tensor<2, dim>> global_displacement_gradients(quadrature_formula.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      update_damage_on_cell(cell, fe_values, global_displacement_gradients, curTemperature,
                            tempInc, stressThreshold, A_param, m_param, residualTolerance);
  }

  // Multi-threaded loop. No copier is needed (see update_damage_on_cell): each cell owns its
  // quadrature data outright. Note that update_damage_on_cell can throw when the root search
  // fails; WorkStream propagates exceptions out of the worker, so that diagnostic still surfaces
  // rather than being swallowed -- though with several cells in flight the failing cell is not
  // necessarily the first one that would have failed serially.
  template <int dim>
  void ElasticProblem<dim>::predict_damage_cells_workstream(
    double curTemperature, double tempInc,
    double stressThreshold, double A_param, double m_param, double residualTolerance)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const UpdateFlags flags = update_values | update_gradients | update_quadrature_points;

    auto worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                      DamageScratch::Scratch<dim> &scratch,
                      DamageScratch::Copy &)
    {
      update_damage_on_cell(cell, scratch.fe_values, scratch.global_displacement_gradients,
                            curTemperature, tempInc, stressThreshold, A_param, m_param,
                            residualTolerance);
    };
    auto copier = [](const DamageScratch::Copy &) {};

    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), worker, copier,
                    DamageScratch::Scratch<dim>(fe, quadrature_formula, flags),
                    DamageScratch::Copy());
  }

  // ============================================================================================
  // Macroscopic strain unknowns -- TEMPLATE for the periodic formulation
  // ============================================================================================

  // Volume average of the stress over the RVE. Plain quadrature summation of the same sigma the
  // damage update sees, so the two can never disagree about what the stress is.
  //
  // For "freeSurface" this should come out at ~0 to solver tolerance, since traction-free
  // boundaries give <sigma> = 0 identically (homogenization.md section 13). That makes it a
  // genuine correctness check and not just a diagnostic -- if it does not, either the boundary
  // treatment or the minimal support is wrong. For "zeroDisplacement" it will be large and
  // triaxial, which is the whole reason that case is not free cooling.
  template <int dim>
  SymmetricTensor<2, dim> ElasticProblem<dim>::computeAverageStress(double curTemperature) const
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_gradients | update_JxW_values);
    std::vector<Tensor<2, dim>> gradients(quadrature_formula.size());

    SymmetricTensor<2, dim> integratedStress;
    double volume = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[displacement].get_function_gradients(solDisplacement, gradients);
      const auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2, dim> rotGlobalToLocal =
        rotmatOrientationCell[cell->material_id() - 1];
      const Tensor<2, dim> rotLocalToGlobal = transpose(rotGlobalToLocal);

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
      {
        const SymmetricTensor<2, dim> globalStrain =
          dealii::symmetrize(gradients[q]) + macroscopicStrain;
        const SymmetricTensor<2, dim> localStrain =
          Physics::Transformations::basis_transformation(globalStrain, rotGlobalToLocal);
        const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
        const SymmetricTensor<2, dim> localStress =
          elasticityTensor.getElasticityTensor(cell_damage[q]->damage, curTemperature) *
          (localStrain - cteTensor.getcteTensor(curTemperature) * (curTemperature - T0));
        integratedStress +=
          Physics::Transformations::basis_transformation(localStress, rotLocalToGlobal) *
          fe_values.JxW(q);
        volume += fe_values.JxW(q);
      }
    }
    return integratedStress / volume;
  }

  // --------------------------------------------------------------------------------------------
  // NOT IMPLEMENTED. Fill in from homogenization.md section 6.
  //
  // Assembles, alongside the usual K and F_theta:
  //
  //   macroCouplingColumns[k]  =  L's k-th column  =  integral over Omega of B^T C I^(k) dV
  //                               i.e. exactly the existing thermal-load assembly with the
  //                               eigenstrain alpha*dT replaced by the unit Kelvin tensor I^(k).
  //                               Distribute through the SAME constraints as the RHS, using
  //                               constraints.distribute_local_to_global(cell_rhs, dofs, column).
  //
  //   macroStiffnessH          =  H  =  integral over Omega of C dV, a plain 6x6 volume integral
  //                               of the rotated global C in Kelvin notation. NOT touched by
  //                               constraints -- it involves no shape functions at all.
  //
  // Note the six columns are the eigenstrain path re-used six times, which is why splitting the
  // thermal load out of the (i,j) loop above was worth doing.
  // --------------------------------------------------------------------------------------------
  template <int dim>
  void ElasticProblem<dim>::assemble_macroscopic_blocks(double /*curTemperature*/)
  {
    throw std::runtime_error(
      "assemble_macroscopic_blocks() is a template and not implemented yet -- see "
      "homogenization.md section 6 for the blocks it must build.");
  }

  // --------------------------------------------------------------------------------------------
  // NOT IMPLEMENTED. Fill in from homogenization.md section 6, step by step:
  //
  //   1. z = K^-1 F_theta                                     1 CG solve
  //   2. Y_k = K^-1 macroCouplingColumns[k], k = 0..5         6 CG solves, SAME matrix and
  //                                                           SAME preconditioner -- build the
  //                                                           preconditioner once, outside.
  //   3. homogenizedStiffness = H - L^T Y                     6x6, and it must come out
  //                                                           SYMMETRIC and positive definite.
  //                                                           Check it: that is a free and very
  //                                                           sharp correctness test (section 4).
  //   4. Ebar = homogenizedStiffness^-1 (P_theta - L^T z)     6x6 dense solve, e.g.
  //                                                           FullMatrix::gauss_jordan()
  //   5. U = z - Y Ebar                                       NO extra solve
  //
  // Then set macroscopicStrain from Ebar (Kelvin 6-vector -> SymmetricTensor via
  // Physics::Notation::Kelvin::to_tensor) and solDisplacement from U.
  //
  // Validate with the uniform-orientation smoke test FIRST (homogenization.md section 9): give
  // every grain the same orientation and the answer is known in closed form -- Cbar = C exactly,
  // Ebar = alpha*dT, u_periodic identically 0, and sigma identically 0 pointwise. That single
  // test exercises periodicity, the pin, the eigenstrain path, all seven solves and the recovery
  // at once, against an answer that includes zero stress.
  // --------------------------------------------------------------------------------------------
  template <int dim>
  void ElasticProblem<dim>::solve_with_zero_mean_stress()
  {
    throw std::runtime_error(
      "solve_with_zero_mean_stress() is a template and not implemented yet -- see "
      "homogenization.md section 6 for the 7-solve algorithm it must implement.");
  }

  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    // WHAT GOES IN THE FILE, AND WHY
    //
    // The story this output has to tell is: c-axis tension builds up as the RVE cools ->
    // it crosses the threshold -> damage grows -> the grain softens and sheds load. So the
    // fields are chosen to make each link in that chain visible, and NOT to dump everything.
    //
    // On the local-vs-global question. sigma is one tensor; "local" and "global" are the same
    // tensor in different bases, so writing both in full would be redundant INFORMATION given
    // that the orientation is also written. But it is not redundant in PRACTICE, because
    // ParaView cannot apply a different rotation per cell -- anything wanted in the crystal
    // frame has to be computed here. So the file carries:
    //
    //   * frame-independent invariants (von Mises, pressure). No frame to get wrong, and they
    //     answer "how hard is this grain being worked" for any viewer.
    //   * the specific LOCAL components that carry physics: the c-axis normal stress (what the
    //     kinetic law actually reads) and the basal shear magnitude.
    //   * the full GLOBAL tensor only on request (OutputParameters/WriteFullStressTensor), for
    //     post-hoc analysis. Off by default: six more arrays per step is clutter for a
    //     demonstration, and at refine_global(3) it is tens of MB per step.
    //
    // Every array name states its frame, because "sigma_zz" alone is genuinely ambiguous here
    // and a viewer colouring by it has no way to tell which one they are looking at.
    //
    // IMPORTANT for reading the picture: the c-axis normal stress is NOT a component of one
    // fixed global direction -- each grain's c points somewhere different. It is well defined
    // (traction normal to that grain's basal planes) and it is exactly the damage driver, so
    // colouring by it is right; but the caption must say "c-axis normal stress", never
    // "sigma_zz".
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_gradients | update_JxW_values);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    // Displacement as a genuine vector, so ParaView's "Warp By Vector" works without setup.
    const std::vector<std::string> displacement_names(dim, "displacement");
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      displacement_interpretation(dim,
        DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(solDisplacement, displacement_names,
                             DataOut<dim>::type_dof_data, displacement_interpretation);

    // The grain c-axis, as a vector, for glyphing.
    CAxisPostprocessor<dim> cAxisPostprocessor(rotmatOrientationCell);
    data_out.add_data_vector(solDisplacement, cAxisPostprocessor);

    const bool writeFullTensor =
      prm.get_bool({"OutputParameters"}, "WriteFullStressTensor");
    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const double curT = lastProcessedTemperature;
    const double stressThreshold =
      prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");

    const unsigned int nCells = triangulation.n_active_cells();
    Vector<double> cell_damage(nCells);
    Vector<double> cell_sigma_caxis_normal(nCells);   // LOCAL sigma_33 -- the damage driver
    Vector<double> cell_sigma_basal_shear(nCells);    // LOCAL sqrt(s13^2 + s23^2)
    Vector<double> cell_von_mises(nCells);            // frame independent
    Vector<double> cell_pressure(nCells);             // frame independent
    Vector<double> cell_driving_ratio(nCells);        // sigma_caxis_normal / threshold
    std::array<Vector<double>, 6> cell_sigma_global;
    if (writeFullTensor)
      for (auto &v : cell_sigma_global) v.reinit(nCells);

    std::vector<Tensor<2, dim>> gradients(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[displacement].get_function_gradients(solDisplacement, gradients);
      const auto qp_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2, dim> rotGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      const Tensor<2, dim> rotLocalToGlobal = transpose(rotGlobalToLocal);

      double damage = 0.0, sigmaCaxis = 0.0, basalShear = 0.0, vonMises = 0.0, pressure = 0.0;
      SymmetricTensor<2, dim> globalStressAvg;

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const SymmetricTensor<2, dim> globalStrain =
          dealii::symmetrize(gradients[q]) + macroscopicStrain;
        const SymmetricTensor<2, dim> localStrain =
          Physics::Transformations::basis_transformation(globalStrain, rotGlobalToLocal);
        const SymmetricTensor<2, dim> localStress =
          getLocalStressTensor(localStrain, qp_damage[q]->damage, curT);

        damage     += qp_damage[q]->damage;
        sigmaCaxis += localStress[dim-1][dim-1];
        // Shear ON the basal planes: the components with one index along c and one in-plane.
        // The current kinetic law does NOT read this -- it is written so that its magnitude
        // relative to the c-axis normal stress can be judged, i.e. so the modelling choice to
        // ignore basal shear can be checked rather than assumed.
        basalShear += std::sqrt(localStress[0][dim-1] * localStress[0][dim-1] +
                                localStress[1][dim-1] * localStress[1][dim-1]);
        // Invariants: identical in either frame, so no ambiguity for the viewer.
        vonMises += std::sqrt(1.5 * (deviator(localStress) * deviator(localStress)));
        pressure += -trace(localStress) / dim;   // positive in compression

        if (writeFullTensor)
          globalStressAvg +=
            Physics::Transformations::basis_transformation(localStress, rotLocalToGlobal);
      }

      const unsigned int idx = cell->active_cell_index();
      cell_damage[idx]              = damage     / n_q_points;
      cell_sigma_caxis_normal[idx]  = sigmaCaxis / n_q_points;
      cell_sigma_basal_shear[idx]   = basalShear / n_q_points;
      cell_von_mises[idx]           = vonMises   / n_q_points;
      cell_pressure[idx]            = pressure   / n_q_points;
      // > 1 means this grain is above threshold and its damage is actively growing. The single
      // most legible scalar for the story: threshold it at 1 in ParaView and what remains is
      // exactly the set of grains driving the damage seen in the `damage` array.
      cell_driving_ratio[idx]       = cell_sigma_caxis_normal[idx] / stressThreshold;

      if (writeFullTensor)
      {
        globalStressAvg /= n_q_points;
        cell_sigma_global[0][idx] = globalStressAvg[0][0];
        cell_sigma_global[1][idx] = globalStressAvg[1][1];
        cell_sigma_global[2][idx] = globalStressAvg[2][2];
        cell_sigma_global[3][idx] = globalStressAvg[0][1];
        cell_sigma_global[4][idx] = globalStressAvg[0][2];
        cell_sigma_global[5][idx] = globalStressAvg[1][2];
      }
    }

    data_out.add_data_vector(cell_damage, "damage");
    data_out.add_data_vector(cell_sigma_caxis_normal, "sigma_caxis_normal_local");
    data_out.add_data_vector(cell_sigma_basal_shear, "sigma_basal_shear_local");
    data_out.add_data_vector(cell_von_mises, "von_mises");
    data_out.add_data_vector(cell_pressure, "pressure");
    data_out.add_data_vector(cell_driving_ratio, "damage_driving_ratio");
    if (writeFullTensor)
    {
      static const std::array<const char *, 6> names{{
        "sigma_global_xx", "sigma_global_yy", "sigma_global_zz",
        "sigma_global_xy", "sigma_global_xz", "sigma_global_yz"}};
      for (unsigned int k = 0; k < 6; ++k)
        data_out.add_data_vector(cell_sigma_global[k], names[k]);
    }

    data_out.build_patches();

    const std::string fileName = "polycrystalCooling-" + std::to_string(cycle) + ".vtu";
    std::ofstream output(fileName);
    data_out.write_vtu(output);

    // Accumulated cooling, which increases over the run -- see pvdTimeSeries.
    pvdTimeSeries.emplace_back(T0 - curT, fileName);
  }

  template <int dim>
  void ElasticProblem<dim>::run()
  {
    double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    double Tend =  prm.get_double({"ModelParameters"}, "FinalTemperature");
    double dT = prm.get_double({"ModelParameters"}, "TemperatureIncrement");

    // Thread cap for WorkStream. 0 keeps deal.II's default (all available cores); an explicit
    // value is mainly for scaling measurements (run the same problem at 1, 2, 4, ... threads).
    // Note this caps deal.II's task scheduler as a whole, not just our two loops.
    const unsigned int requestedThreads =
      prm.get_integer({"ParallelizationParameters"}, "NumberOfThreads");
    if (requestedThreads > 0)
      MultithreadInfo::set_thread_limit(requestedThreads);
    boundaryConditionType =
      parseBoundaryConditionType(prm.get({"BoundaryConditions"}, "Type"));

    // setFromPrm() must run before anything asks for a material tensor: the temperature-
    // dependence functors are only built there.
    elasticityTensor.setFromPrm(prm);
    cteTensor.setFromPrm(prm);

    std::cout << "Boundary conditions: " << boundaryConditionTypeName(boundaryConditionType)
              << std::endl;
    std::cout << "Threading: "
              << (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream")
                    ? "WorkStream" : "serial (reference path)")
              << ", n_threads=" << MultithreadInfo::n_threads()
              << " (hardware concurrency " << MultithreadInfo::n_cores() << ")" << std::endl;

    std::string mshFileName = prm.get({"InputFiles"}, "MeshFile");
    std::string cellToMaterialFileName = prm.get({"InputFiles"}, "CellToMaterialFile");
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName);
    auto mshStream = parseMSHandPrepareStream(mshFileName);
    grid_in.read_msh(mshStream);
    parse_cellToMaterial_data(cellToMaterialFileName);

    // --------------------------------------------------------------------------------------
    // One grain per mesh cell, each with its own random c-axis orientation.
    //
    // Grain identity is carried in material_id. Two reasons that is the right home for it
    // rather than a side table: deal.II PROPAGATES material_id to children on refinement, so
    // grain identity survives refine_global() with no bookkeeping at all; and every existing
    // lookup (getRotTensorGlobalToLocal) already goes through it.
    //
    // material_id is 1-based because getRotTensorGlobalToLocal() indexes
    // rotmatOrientationCell[material_id - 1].
    //
    // parseMSHandPrepareStream() has already pushed whatever orientation the mesh file's
    // $ElsetOrientations section carries. That is the right source once a real Neper morphology
    // is used (step 5 of the plan), but here the grains ARE the cells and the orientations are
    // generated, so the parsed list must be cleared first. Not clearing it was the mechanism of
    // a real bug in an earlier run: every cell had material_id 1, so all of them silently used
    // the mesh file's orientation at index 0 rather than the intended one.
    // --------------------------------------------------------------------------------------
    const unsigned int orientationSeed =
      prm.get_integer({"BoundaryConditions"}, "OrientationSeed");
    std::mt19937_64 rng(orientationSeed);

    rotmatOrientationCell.clear();
    rotmatOrientationCell.reserve(triangulation.n_active_cells());
    types::material_id cellCounter = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      cell->set_material_id(++cellCounter); // 1-based by the convention above
      rotmatOrientationCell.push_back(randomOrientationUniformOnSphere<dim>(rng));
    }

    // types::material_id is a build-time typedef (unsigned int in this deal.II build, but
    // historically unsigned char, which tops out at 255). One grain per cell needs a DISTINCT
    // id for every cell, so on a narrower build the counter would silently wrap and many grains
    // would share an orientation -- a wrong answer with no error anywhere. Check rather than
    // assume. If this ever does trip, the storage-free fix is to drop material_id entirely and
    // identify a grain by its level-0 ancestor: walk `while (c->level() > 0) c = c->parent();`
    // and use `c->index()`, which is unbounded and equally refinement-stable.
    if (cellCounter == numbers::invalid_material_id ||
        rotmatOrientationCell.size() != triangulation.n_active_cells())
      throw std::runtime_error(
        "could not assign a distinct material_id to each of " +
        std::to_string(triangulation.n_active_cells()) +
        " cells (assigned " + std::to_string(rotmatOrientationCell.size()) +
        "); types::material_id is too narrow on this deal.II build for one grain per cell.");

    // Sanity check on the sampling: the mean of a uniformly distributed direction is 0, and the
    // mean of c_z^2 is 1/3. A gross deviation means the sampler is biased -- the classic failure
    // being uniform theta instead of uniform cos(theta) -- which would quietly skew every stress
    // statistic downstream without ever looking like an error.
    {
      Tensor<1, dim> meanAxis;
      double meanCzSquared = 0.0;
      for (const auto &R : rotmatOrientationCell)
      {
        for (unsigned int d = 0; d < dim; ++d) meanAxis[d] += R[dim-1][d];
        meanCzSquared += R[dim-1][dim-1] * R[dim-1][dim-1];
      }
      meanAxis /= rotmatOrientationCell.size();
      meanCzSquared /= rotmatOrientationCell.size();
      std::cout << "Orientations: seed=" << orientationSeed
                << ", n=" << rotmatOrientationCell.size()
                << ", |mean c-axis|=" << meanAxis.norm()
                << " (expect ~0), mean c_z^2=" << meanCzSquared
                << " (expect ~0.3333)" << std::endl;
    }

    assignBoundaryIds();

    // Outer mesh-refinement-cycle loop, for mesh convergence studies: each cycle refines
    // (cycle 0 just does the initial setup) and then re-solves the *entire* temperature
    // range from scratch on the new mesh. Damage does not carry over between cycles (see
    // refine_grid()/setup_system()), so this is for damage-free/single-cycle MMS studies,
    // not yet for refining mid-simulation on a run with already-accumulated damage.
    const unsigned int nRefinementCycles = prm.get_integer({"MeshRefinementParameters"}, "NumberOfRefinementCycles");
    const std::string refinementStrategy = prm.get({"MeshRefinementParameters"}, "RefinementStrategy");
    const double fractionToRefine = prm.get_double({"MeshRefinementParameters"}, "FractionToRefine");
    const double fractionToCoarsen = prm.get_double({"MeshRefinementParameters"}, "FractionToCoarsen");
    const unsigned int outputFrequency = prm.get_integer({"OutputParameters"}, "OutputFrequency");

    for (unsigned int refinementCycle = 0; refinementCycle < nRefinementCycles; ++refinementCycle)
    {
      if (refinementCycle == 0)
        setup_system(); // was previously missing: without this call, dof_handler has zero DoFs,
                         // damageInQuadraturePoints/material property functors are never initialized,
                         // and system_matrix/rhs/solution vectors stay unsized (see cubeTest.cpp's
                         // meshTest()/elasticCalculationTest() for the pattern this followed)
      else if (refinementStrategy == "uniform")
      {
        triangulation.refine_global(1);
        setup_system();
      }
      else
        refine_grid(fractionToRefine, fractionToCoarsen);

      // Step count computed up front, and curT derived as T0 - n*dT rather than accumulated by
      // repeated subtraction. The previous `do { curT -= dT; } while (curT > Tend)` form had two
      // problems: (a) rounding accumulated in curT, and (b) that accumulated error routinely left
      // curT a few ulp above Tend after the nominal final step, triggering one EXTRA step that
      // ran past the end temperature. Both are real: a step-size sweep over
      // dT = 0.1/0.05/0.025/0.0125/0.00625 produced 11/20/40/81/161 steps instead of
      // 10/20/40/80/160, so different step sizes silently simulated to different final states
      // (dT=0.1 ended at pseudo-time 1.1, not 1.0) -- which invalidates any convergence study
      // comparing them.
      //
      // A range not divisible by dT shortens the final step to land exactly on Tend rather than
      // overshooting, and that shortened increment is what gets passed to the kinetic equation.
      const double temperatureRange = T0 - Tend;
      const unsigned int nTemperatureSteps =
        static_cast<unsigned int>(std::ceil(temperatureRange / dT - 1e-9));
      for (unsigned int temperatureStepNumber = 1;
           temperatureStepNumber <= nTemperatureSteps;
           ++temperatureStepNumber)
      {
        // curT is the END-of-step temperature: the step is solved at the new state, not the old
        // one, matching genuine implicit (backward-Euler) semantics -- the kinetic law's RHS
        // belongs at the new state. T0 itself is the known initial condition and is never solved
        // for. (Getting this backwards previously meant step 1 always evaluated everything at
        // t=0, which was masked for a long time by two other bugs that kept the kinetic-equation
        // path from running at all.)
        const double prevT = std::max(T0 - (temperatureStepNumber - 1) * dT, Tend);
        const double curT  = std::max(T0 - temperatureStepNumber * dT, Tend);
        const double actualTempInc = prevT - curT;

        lastProcessedTemperature = curT;
        calculateSolutionTemperatureStep(curT, actualTempInc, refinementCycle, temperatureStepNumber);

        // OutputFrequency was declared and documented but never actually consulted -- every step
        // wrote a full field dump regardless. That is a real cost for fine-step runs (e.g. the
        // sweep above writes hundreds of ~450KB files nobody looks at). The final step is always
        // written, whatever the frequency, so the end state is never missing.
        if (temperatureStepNumber % outputFrequency == 0 ||
            temperatureStepNumber == nTemperatureSteps)
          output_results(refinementCycle * 1000 + temperatureStepNumber);
      }
    }

    convergence_table.set_precision("meanDamage", 3);
    convergence_table.set_precision("maxDamage", 3);
    convergence_table.set_precision("maxSigmaZZ", 3);
    convergence_table.set_precision("minSigmaZZ", 3);
    convergence_table.set_precision("tensionFrac", 3);
    convergence_table.set_precision("meanStressNorm", 3);
    convergence_table.set_scientific("maxSigmaZZ", true);
    convergence_table.set_scientific("minSigmaZZ", true);
    convergence_table.set_scientific("meanStressNorm", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");

    convergence_table.write_text(std::cout);

    // ParaView time series. Without this the .vtu files are an unordered pile; with it ParaView
    // opens ONE entry, scrubs through the cooling history, and shows accumulated cooling in K on
    // the animation clock -- which is what makes "stress builds up, then damage grows" legible
    // as a sequence rather than as ten separate pictures.
    if (!pvdTimeSeries.empty())
    {
      std::ofstream pvd("polycrystalCooling.pvd");
      DataOutBase::write_pvd_record(pvd, pvdTimeSeries);
      std::cout << "Wrote ParaView time series polycrystalCooling.pvd ("
                << pvdTimeSeries.size() << " steps). Open THIS, not the individual .vtu files."
                << std::endl;
    }
  }

  template <int dim>
  void ElasticProblem<dim>::calculateSolutionTemperatureStep(double curT, double dT, unsigned int refinementCycle, int temperatureStepNumber)
  {
    // "prev*" tracks the *previous inner iteration's* state, so the increment-based
    // convergence checks measure iteration-to-iteration change (as their tolerances
    // assume) rather than the total change since the start of this temperature step.
    // On iteration 1, prev* is still the end-of-previous-temperature-step state, which
    // is the correct reference point for that first increment.
    Vector<double> prevSolDisplacement = solDisplacement;
    DamageSnapshot prevDamageSnapshot = snapshotDamage();
    const unsigned int maxIterations = prm.get_integer({"SolverParameters"}, "MaxNumberOfIterations");
    unsigned int dispDamageIterationNumber = 0;
    bool converged = false;

    // Staggered displacement-damage iteration. Within one pass damage is FROZEN while
    // equilibrium is solved, which is what keeps the mechanical problem linear and the tangent
    // symmetric (so CG stays applicable). The outer loop then re-solves with the updated damage
    // until neither the displacement nor the damage moves.
    //
    // This is also exactly why the periodic <sigma> = 0 machinery does not need a nonlinear
    // solver: with damage frozen, each pass is a linear problem in (u_periodic, Ebar), so the
    // 7-solve of homogenization.md section 6 is EXACT per pass, not an approximation. Only the
    // number of times it runs changes. Once solve_with_zero_mean_stress() is implemented, it
    // replaces the assemble/solve pair below and nothing else here changes.
    do
    {
      dispDamageIterationNumber++;

      assemble_system(curT);
      solve();
      predict_damage_tempInc_external_solver(curT, dT);
      process_solution(refinementCycle, temperatureStepNumber, dispDamageIterationNumber, curT);

      converged = checkConvergenceCriteria(curT, prevSolDisplacement, prevDamageSnapshot);

      prevSolDisplacement = solDisplacement;
      prevDamageSnapshot = snapshotDamage();

      if (!converged && dispDamageIterationNumber >= maxIterations)
        throw std::runtime_error("Displacement-damage iteration did not converge within "
                                  "MaxNumberOfIterations = " + std::to_string(maxIterations) +
                                  " iterations at temperature step " + std::to_string(temperatureStepNumber));
    } while (!converged);

    // The kinetic equation's residual (see mmsKineticResidual/predict_damage_tempInc_external_solver)
    // integrates the damage increment against old_damage, i.e. the damage at the *start* of the
    // current temperature step. That baseline must be advanced once this step has converged,
    // otherwise every subsequent temperature step keeps integrating from the previous old_damage
    // (stuck at 0.0 for the whole run, since nothing else ever assigns to it).
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q)
        cell_damage[q]->old_damage = cell_damage[q]->damage;
    }

  }

  template <int dim>
  typename ElasticProblem<dim>::DamageSnapshot ElasticProblem<dim>::snapshotDamage()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    DamageSnapshot snapshot;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      std::vector<double> values(quadrature_formula.size());
      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        values[q] = cell_damage[q]->damage;
      snapshot[cell] = std::move(values);
    }
    return snapshot;
  }

  template <int dim>
  bool ElasticProblem<dim>::checkConvergenceCriteria(double curT,
                                    const Vector<double> &oldSolDisplacement,
                                    const DamageSnapshot &oldDamageSnapshot)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    Vector<double> solutionIncrement = solDisplacement;
    solutionIncrement -= oldSolDisplacement;
    double solutionIncrementNorm = solutionIncrement.l2_norm();
    double solutionNorm = solDisplacement.l2_norm();
    double relativeSolutionIncrementNorm = solutionIncrementNorm / solutionNorm;

    double maxDamageIncrement = 0.0; // Damage is self-limited to [0,1], so max norm is more informative than L2 norm
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const auto &old_cell_damage = oldDamageSnapshot.at(cell);
      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
      {
        double damageIncrement = cell_damage[q]->damage - old_cell_damage[q];
        maxDamageIncrement = std::max(maxDamageIncrement, std::abs(damageIncrement));
      }
    }

    assemble_system(curT);
    Vector<double> systemResidual(system_matrix.m());
    system_matrix.residual(systemResidual, solDisplacement, system_rhs);
    // Constrained-DOF rows carry a placeholder diagonal that distribute_local_to_global()
    // writes purely to keep the matrix well-conditioned for the preconditioner -- with no
    // matching right-hand-side entry, since those rows are never meant to be interpreted as
    // real equilibrium equations (their value is fixed directly via constraints.distribute(),
    // not solved for). Left in, a handful of O(1e10) placeholder-diagonal rows swamp the
    // residual norm and it plateaus at a large, meaningless constant regardless of how well
    // the free-DOF system actually solved. Zero them out before taking the norm, matching the
    // standard deal.II idiom (e.g. step-15's Newton residual) for excluding constrained DOFs
    // from a residual/error norm.
    constraints.set_zero(systemResidual);
    double systemResidualNorm = systemResidual.l2_norm();
    double systemRhsNorm = system_rhs.l2_norm(); // should check, that this norm will not be too small;
    double relativeSystemResidualNorm  = systemResidualNorm / systemRhsNorm;

    double solIncrementEps = prm.get_double({"SolverParameters"}, "DisplacementIncrementRelativeTolerance");
    double damageIncrementEps = prm.get_double({"SolverParameters"}, "DamageIncrementRelativeTolerance");
    double forseResidualEps = prm.get_double({"SolverParameters"}, "ForceEqulibriumRelativeTolerance");

    std::cout << "    convergence check: relSolInc=" << relativeSolutionIncrementNorm
              << " (tol " << solIncrementEps << "), maxDamageInc=" << maxDamageIncrement
              << " (tol " << damageIncrementEps << "), relResidual=" << relativeSystemResidualNorm
              << " (tol " << forseResidualEps << ")" << std::endl;

    if (relativeSolutionIncrementNorm < solIncrementEps and maxDamageIncrement < damageIncrementEps and relativeSystemResidualNorm < forseResidualEps){return true;}
    else {return false;}
  }

  template <int dim>
  void ElasticProblem<dim>::process_solution(unsigned int refinementCycle, const unsigned int tempStepNum, const unsigned int iterSolverStepNum, double curT)
  {
    // There is no manufactured solution to compare against here -- this is the model, not a
    // verification test -- so what gets reported is the physics the study is about: how much of
    // the RVE is in c-axis tension, how extreme it gets, and how far damage has run.
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_gradients | update_JxW_values);
    std::vector<Tensor<2, dim>> gradients(n_q_points);

    const double stressThreshold =
      prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");

    double maxSigmaZZ = -std::numeric_limits<double>::max();
    double minSigmaZZ =  std::numeric_limits<double>::max();
    double tensionVolume = 0.0, overThresholdVolume = 0.0, totalVolume = 0.0;
    double meanDamage = 0.0, maxDamage = 0.0, damageWeight = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[displacement].get_function_gradients(solDisplacement, gradients);
      const auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2, dim> rotGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double w = fe_values.JxW(q);
        const SymmetricTensor<2, dim> globalStrain =
          dealii::symmetrize(gradients[q]) + macroscopicStrain;
        const SymmetricTensor<2, dim> localStrain =
          Physics::Transformations::basis_transformation(globalStrain, rotGlobalToLocal);
        const SymmetricTensor<2, dim> localStress =
          getLocalStressTensor(localStrain, cell_damage[q]->damage, curT);

        // sigma_zz in the LOCAL frame: the c-axis normal stress, which is what drives the
        // kinetic law. A global-frame sigma_zz would mean nothing here, since every grain has a
        // different c-axis.
        const double sZZ = localStress[dim-1][dim-1];
        maxSigmaZZ = std::max(maxSigmaZZ, sZZ);
        minSigmaZZ = std::min(minSigmaZZ, sZZ);
        totalVolume += w;
        if (sZZ > 0.0)             tensionVolume       += w;
        if (sZZ > stressThreshold) overThresholdVolume += w;

        meanDamage += cell_damage[q]->damage * w;
        maxDamage = std::max(maxDamage, cell_damage[q]->damage);
        damageWeight += w;
      }
    }
    meanDamage /= damageWeight;
    const double tensionFraction   = tensionVolume / totalVolume;
    const double thresholdFraction = overThresholdVolume / totalVolume;

    // <sigma>: identically zero for a genuinely free-cooling state (traction-free boundaries
    // give it by the average stress theorem), and large and triaxial for the clamped case. So
    // this doubles as a correctness check on "freeSurface" and as the quantity the periodic
    // formulation will drive to zero by construction.
    const SymmetricTensor<2, dim> meanStress = computeAverageStress(curT);

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();

    std::cout << "Refinement cycle " << refinementCycle << ", temperature step " << tempStepNum
              << ", iteration " << iterSolverStepNum << " (T = " << curT << "):" << std::endl
              << "   Cells: " << n_active_cells << ", DoFs: " << n_dofs
              << ", CG iterations = " << lastCgIterations << std::endl
              << "   local sigma_zz: min=" << minSigmaZZ << ", max=" << maxSigmaZZ
              << ", tension fraction=" << tensionFraction
              << ", above threshold=" << thresholdFraction << std::endl
              << "   damage: mean=" << meanDamage << ", max=" << maxDamage << std::endl
              << "   <sigma> norm=" << meanStress.norm()
              << " (trace/3 = " << trace(meanStress) / dim << ")" << std::endl;

    convergence_table.add_value("cycle", refinementCycle);
    convergence_table.add_value("tempStep", tempStepNum);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("meanDamage", meanDamage);
    convergence_table.add_value("maxDamage", maxDamage);
    convergence_table.add_value("maxSigmaZZ", maxSigmaZZ);
    convergence_table.add_value("minSigmaZZ", minSigmaZZ);
    convergence_table.add_value("tensionFrac", tensionFraction);
    convergence_table.add_value("meanStressNorm", meanStress.norm());
  }

} // namespace StepCooling

int main()
{
  try
    {
      StepCooling::ParameterHandler prm;
      StepCooling::ParameterReader myReader(prm);
      myReader.read_parameters("input.json");
      StepCooling::ElasticProblem<3> myProblem(prm);
      myProblem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
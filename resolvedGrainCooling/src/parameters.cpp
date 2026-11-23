#include "resolved_grain/parameters.h"

namespace StepCooling
{
  using namespace dealii;

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
      prm.declare_entry("Type",
                        "zeroDisplacement",
                        Patterns::Selection("zeroDisplacement|freeSurface|periodic"),
                        "Boundary condition family. 'zeroDisplacement' clamps the whole boundary "
                        "and forbids macroscopic contraction. 'freeSurface' is traction-free "
                        "with minimal kinematic support. 'periodic' uses u = Ebar.x + u_periodic "
                        "and enforces zero volume-averaged stress.");

      prm.declare_entry("GrainsPerSide",
                        "10",
                        Patterns::Integer(1),
                        "Grains along each cube edge, so GrainsPerSide^3 grains in total. Grain "
                        "identity is assigned from a cell's CENTROID, not from its index, which "
                        "decouples grains from the mesh: refine and the grain count is unchanged "
                        "while each grain gains elements. With the 10x10x10 mesh, "
                        "GrainsPerSide=10 gives 1 element per grain, and uniform refinement then "
                        "gives 8 and 64. That ladder is 8x per level because hexahedra split "
                        "eightfold -- 4 or 16 elements per grain would need non-cubic grain "
                        "boxes, i.e. deliberately anisotropic grain shapes.");

      prm.declare_entry("UseOrientationFile",
                        "false",
                        Patterns::Bool(),
                        "Read one global-to-local rotation matrix per grain from "
                        "InputFiles/OrientationFile instead of random sampling. Intended for "
                        "reproducible microstructures and analytical verification cases.");

      prm.declare_entry("UniformOrientation",
                        "false",
                        Patterns::Bool(),
                        "Give every grain the same orientation for the homogeneous-crystal "
                        "verification case: Cbar = C, Ebar = alpha*dT, and the fluctuation "
                        "and stress vanish pointwise.");

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
      // Five independent constants define the transversely isotropic local stiffness.
      prm.enter_subsection("LinearElasticityParameters");
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
                        "Absolute dimensionless residual tolerance for the kinetic equation");

      // All periodic right-hand sides reuse one prepared direct factorization or preconditioner.
      prm.declare_entry("LinearSolverType",
                        "cg",
                        Patterns::Selection("direct|cg|amg"),
                        "Linear solver for the shared system matrix: 'direct' uses UMFPACK, "
                        "'cg' uses SSOR-preconditioned CG, and 'amg' uses CG with Trilinos ML. "
                        "Periodic right-hand sides reuse the same factorization or preconditioner.");

      prm.declare_entry("DamageWarningThreshold",
                        "0.9",
                        Patterns::Double(0, 1),
                        "Warn once per run when the maximum damage anywhere exceeds this. The "
                        "kinetic law carries a (1-omega)^-m term and the root search brackets at "
                        "omega = 0.999, so as damage approaches 1 the equation stiffens sharply "
                        "and the bracket narrows: results past this point are not trustworthy "
                        "even when the solver still converges. This is a loud warning, not a "
                        "stop, because a few quadrature points reaching it does not invalidate "
                        "the rest of the field.");

      prm.declare_entry("FrozenCbarQuasiNewton",
                        "false",
                        Patterns::Bool(),
                        "Use a frozen-Cbar quasi-Newton iteration for Ebar instead of rebuilding "
                        "the exact condensation on every pass. Each pass uses the current K and "
                        "the exact quadrature-averaged stress residual; Cbar is rebuilt when the "
                        "iteration does not converge within MaxMacroscopicIterations.");

      prm.declare_entry("MacroscopicStressTolerance",
                        "1e-8",
                        Patterns::Double(0),
                        "Convergence tolerance for the Ebar iteration: stop when ||<sigma>|| falls "
                        "below this times ||Cbar:Ebar||, the natural macroscopic stress scale. "
                        "The exact path reaches ~1e-10 relative, so this matches it.");

      prm.declare_entry("MaxMacroscopicIterations",
                        "4",
                        Patterns::Integer(1),
                        "Ebar passes allowed before Cbar is declared stale and rebuilt. With a "
                        "fresh Cbar the iteration converges in one pass, so exceeding this is the "
                        "signal to refresh -- which makes the refresh policy self-tuning rather "
                        "than a damage threshold someone has to calibrate. Note the floor is TWO "
                        "passes, not one: the first solve only evaluates the residual at the "
                        "Ebar carried over from the previous pass, and the second confirms the "
                        "corrected one.");

      prm.declare_entry("ReportedCbarIsFrozen",
                        "true",
                        Patterns::Bool(),
                        "Documentation-only reminder: with FrozenCbarQuasiNewton, reported Cbar "
                        "is the frozen matrix used for the last update. Rebuild it at the "
                        "converged state, or use the exact path, when quoting Cbar itself.");

      prm.declare_entry("CouplingAcceleration",
                        "none",
                        Patterns::Selection("none|aitken"),
                        "Acceleration of the staggered displacement-damage fixed point. 'none' "
                        "uses successive substitution; 'aitken' applies Irons-Tuck relaxation "
                        "from successive damage residuals without an extra linear solve.");

      prm.declare_entry("AmgSmootherSweeps",
                        "2",
                        Patterns::Integer(1),
                        "Smoother sweeps per AMG level; larger values cost more per iteration.");

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
      // WorkStream threading controls assembly and damage prediction, not linear-algebra layout.
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
      prm.declare_entry("InitialGlobalRefinements",
                        "0",
                        Patterns::Integer(0),
                        "Uniform refinements applied ONCE after reading the mesh, before anything "
                        "else. With GrainsPerSide matching the base mesh this sets the grain "
                        "resolution directly: 0 -> 1 element per grain, 1 -> 8, 2 -> 64. Prefer "
                        "this over NumberOfRefinementCycles for a sweep: cycles re-run the whole "
                        "temperature range per cycle and write all of them into one output series "
                        "with repeating time values, which makes a .pvd that ParaView cannot "
                        "order.");

      prm.declare_entry("NumberOfRefinementCycles",
                        "1",
                        Patterns::Integer(1),
                        "Number of mesh refinement cycles to run (1 = no refinement, just the "
                        "initial mesh). Each cycle after the first refines the mesh and re-solves "
                        "the full temperature range from scratch on it; damage does not carry over "
                        "between cycles (see refine_grid()/setup_system()), so this is intended for "
                        "mesh-convergence studies that restart each cycle from the initial "
                        "temperature and zero damage, not for refining an accumulated damage state.");

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

      prm.declare_entry("WriteFullStrainTensor",
                        "false",
                        Patterns::Bool(),
                        "Also write the complete TOTAL strain tensor in both frames: eps_11, "
                        "eps_22, eps_12, eps_13, eps_23 (local; eps_33 is always written) and "
                        "eps_xx ... eps_yz (global). Total strain means eps = Ebar + sym(grad u~), "
                        "whose volume average is exactly Ebar. The ELASTIC strain that actually "
                        "generates stress is eps minus the thermal eigenstrain, and that "
                        "eigenstrain is diag(alpha_a, alpha_a, alpha_c)*(T-T_0) in the LOCAL frame "
                        "for every grain -- so elastic strain is one known subtraction away and is "
                        "not written separately.");

      prm.declare_entry("WriteFullStressTensor",
                        "false",
                        Patterns::Bool(),
                        "Also write the complete stress tensor in BOTH frames: the remaining "
                        "five local components (sigma_11, sigma_22, sigma_12, sigma_13, "
                        "sigma_23 -- sigma_33 is always written) and all six global ones "
                        "(sigma_xx ... sigma_yz). Off by default: sigma_33, tau_basal and the "
                        "frame-independent invariants already carry the physics, so this is "
                        "clutter for a demonstration and costs tens of MB per step on a refined "
                        "mesh. Turn it on for post-hoc analysis needing the full tensor.");

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
}

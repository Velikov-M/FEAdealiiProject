#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  void ElasticProblem<dim>::prepareLinearSolver()
  {
    TimerOutput::Scope timer_section(timer, "Linear solver setup");
    directSolver.reset();
    cgPreconditioner.reset();
    amgPreconditioner.reset();
    trilinosMatrix.reset();

    const std::string type = prm.get({"SolverParameters"}, "LinearSolverType");

    if (type == "direct")
    {
      directSolver = std::make_unique<SparseDirectUMFPACK>();
      directSolver->initialize(system_matrix);
    }
    else if (type == "amg")
    {
      trilinosMatrix = std::make_unique<TrilinosWrappers::SparseMatrix>();
      trilinosMatrix->reinit(system_matrix);

      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      // Elasticity is a SYSTEMS problem: three displacement components live at every node, and
      // the near-null space of the operator is the three rigid translations. Handing ML those
      // modes is what lets it build a coarse hierarchy that respects the vector structure --
      // without them it coarsens as if the problem were scalar and the iteration count barely
      // improves on SSOR. This is the single most important AMG setting for elasticity.
      data.constant_modes = DoFTools::extract_constant_modes(dof_handler, ComponentMask());
      data.n_cycles = 1;
      data.elliptic = true;
      data.higher_order_elements = (fe.degree > 1);
      data.aggregation_threshold = 0.02;
      data.smoother_sweeps = prm.get_integer({"SolverParameters"}, "AmgSmootherSweeps");

      amgPreconditioner = std::make_unique<TrilinosWrappers::PreconditionAMG>();
      amgPreconditioner->initialize(*trilinosMatrix, data);
    }
    else
    {
      cgPreconditioner = std::make_unique<PreconditionSSOR<SparseMatrix<double>>>();
      cgPreconditioner->initialize(
        system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.2));
    }
  }

  // Solve against the prepared matrix. Constraint distribution is deferred because the periodic
  // path combines raw unit and thermal solutions before applying it.
  template <int dim>
  void ElasticProblem<dim>::solveLinearSystem(const Vector<double> &rhs, Vector<double> &dst)
  {
    TimerOutput::Scope timer_section(timer, "Solve system");
    ++totalLinearSolveCount;

    if (directSolver)
    {
      directSolver->vmult(dst, rhs);
      lastCgIterations = 0;   // direct solve: no iteration count to report
      return;
    }

    const double linearSolverTolerance = prm.get_double({"SolverParameters"}, "LinearSolverTolerance");
    const unsigned int maxLinearIterations =
      prm.get_integer({"SolverParameters"}, "MaxLinearSolverIterations");
    // An all-zero right-hand side would make a RELATIVE criterion demand a zero tolerance, which
    // CG can never reach. The answer there is exactly zero. (This does occur: a unit-strain
    // column of L can vanish for a symmetric enough microstructure.)
    const double rhsNorm = rhs.l2_norm();
    if (rhsNorm == 0.0) { dst = 0; lastCgIterations = 0; return; }

    SolverControl solver_control(maxLinearIterations, linearSolverTolerance * rhsNorm);

    // Both preconditioners operate on deal.II vectors and the deal.II system matrix.
    SolverCG<Vector<double>> cg(solver_control);
    if (amgPreconditioner)
      cg.solve(system_matrix, dst, rhs, *amgPreconditioner);
    else
      cg.solve(system_matrix, dst, rhs, *cgPreconditioner);

    lastCgIterations = solver_control.last_step();
    totalCgIterations += lastCgIterations;
    // A plain runtime check rather than Assert(): Assert() compiles to a no-op in Release
    // builds (-DNDEBUG), so relying on it here would mean a genuinely non-converged CG
    // solve could silently pass through in Release with no indication anything went wrong.
    if (solver_control.last_check() != SolverControl::success)
      throw std::runtime_error("CG solver did not converge (last residual " +
                                std::to_string(solver_control.last_value()) + ")");
  }

  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    prepareLinearSolver();
    solveLinearSystem(system_rhs, solDisplacement);
    constraints.distribute(solDisplacement);
  }

  template class ElasticProblem<3>;
}

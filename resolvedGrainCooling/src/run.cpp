#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  void ElasticProblem<dim>::run()
  {
    initializeRun();
    initializeMicrostructure();
    runRefinementCycles();
    finalizeRunOutput();
  }

  template <int dim>
  void ElasticProblem<dim>::initializeRun()
  {
    const double temperatureIncrement =
      prm.get_double({"ModelParameters"}, "TemperatureIncrement");
    const double length = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    const double stressThreshold =
      prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    const double kineticPrefactor =
      prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    const double kineticExponent =
      prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    if (temperatureIncrement <= 0.0 || length <= 0.0 || stressThreshold <= 0.0 ||
        kineticPrefactor < 0.0 || kineticExponent <= 0.0)
      throw std::runtime_error(
        "TemperatureIncrement, LengthOfTheBody, StressThreshold, and m_kineticParam must be "
        "positive; a_kineticParam must be nonnegative");

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

    // Physical self-check of the stiffness at both ends of the temperature range -- the constants
    // are temperature dependent, so checking one temperature would not cover the run.
    for (const double Tcheck : {prm.get_double({"ModelParameters"}, "InitialTemperature"),
                                prm.get_double({"ModelParameters"}, "FinalTemperature")})
    {
      const auto c = elasticityTensor.voigtConstants(Tcheck);
      checkTransverseIsotropy<dim>(elasticityTensor.getElasticityTensor(0.0, Tcheck),
                                   c[0], c[1], c[2], c[3], c[4]);
      checkPositiveDefiniteness(c, 0.0);
    }
    std::cout << "Stiffness self-check passed (Voigt components and c-axis rotation invariance)."
              << std::endl;

    assembleMacroCoupling = (boundaryConditionType == BoundaryConditionType::periodic);

    std::cout << "Boundary conditions: " << boundaryConditionTypeName(boundaryConditionType)
              << std::endl;
    std::cout << "Threading: "
              << (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream")
                    ? "WorkStream" : "serial (reference path)")
              << ", n_threads=" << MultithreadInfo::n_threads()
              << " (hardware concurrency " << MultithreadInfo::n_cores() << ")" << std::endl;
  }

  template <int dim>
  void ElasticProblem<dim>::runRefinementCycles()
  {
    // Each refinement cycle restarts the full cooling range with freshly initialized damage.
    const unsigned int nRefinementCycles =
      prm.get_integer({"MeshRefinementParameters"}, "NumberOfRefinementCycles");
    for (unsigned int refinementCycle = 0; refinementCycle < nRefinementCycles; ++refinementCycle)
    {
      prepareRefinementCycle(refinementCycle);
      runCoolingHistory(refinementCycle);
    }
  }

  template <int dim>
  void ElasticProblem<dim>::runCoolingHistory(const unsigned int refinementCycle)
  {
    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const double Tend = prm.get_double({"ModelParameters"}, "FinalTemperature");
    const double dT = prm.get_double({"ModelParameters"}, "TemperatureIncrement");
    const unsigned int outputFrequency = prm.get_integer({"OutputParameters"}, "OutputFrequency");

    // Derive temperatures from the integer step index and shorten the last increment if needed
    // so every run lands exactly on Tend without accumulating subtraction error.
    const double temperatureRange = T0 - Tend;

    // Cooling only: the kinetic law integrates s = T0 - T with non-decreasing damage, and the
    // bilateral damage model has no heating/crack-closure branch.
    if (temperatureRange <= 0.0)
      throw std::runtime_error(
        "FinalTemperature (" + std::to_string(Tend) + " K) must be BELOW InitialTemperature (" +
        std::to_string(T0) + " K): only the cooling branch is implemented. Heating needs a "
        "kinetic law for dT > 0 and a unilateral (crack-closure) damage response -- see the "
        "comment at this check.");

    const unsigned int nTemperatureSteps =
      static_cast<unsigned int>(std::ceil(temperatureRange / dT - 1e-9));
    for (unsigned int temperatureStepNumber = 1;
         temperatureStepNumber <= nTemperatureSteps;
         ++temperatureStepNumber)
    {
      // Solve at the end-of-step temperature for backward-Euler damage integration.
      const double prevT = std::max(T0 - (temperatureStepNumber - 1) * dT, Tend);
      const double curT  = std::max(T0 - temperatureStepNumber * dT, Tend);
      const double actualTempInc = prevT - curT;

      lastProcessedTemperature = curT;
      calculateSolutionTemperatureStep(curT, actualTempInc, refinementCycle, temperatureStepNumber);

      // Always write the final state, independent of OutputFrequency.
      if (temperatureStepNumber % outputFrequency == 0 ||
          temperatureStepNumber == nTemperatureSteps)
        output_results(refinementCycle * 1000 + temperatureStepNumber);
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
    // With frozen damage, each periodic pass is linear in (u_periodic, Ebar). The convergence
    // check assembles at updated damage and leaves the matrix/RHS ready for the next solve.
    const bool useAitken =
      prm.get({"SolverParameters"}, "CouplingAcceleration") == "aitken";
    resetAitkenState();   // the relaxation history is per temperature step, not per run

    assemble_system(curT);
    do
    {
      dispDamageIterationNumber++;

      if (boundaryConditionType == BoundaryConditionType::periodic)
        solve_with_zero_mean_stress(curT);
      else
        solve();

      DamageSnapshot damageBeforePredict;
      if (useAitken)
        damageBeforePredict = snapshotDamage();
      predict_damage_tempInc_external_solver(curT, dT);
      if (useAitken)
        applyAitkenRelaxation(damageBeforePredict);
      process_solution(refinementCycle, temperatureStepNumber, dispDamageIterationNumber, curT);

      // Re-assembles at the updated damage, and leaves system_matrix/system_rhs ready for the
      // next pass.
      converged = checkConvergenceCriteria(curT, prevSolDisplacement, prevDamageSnapshot);

      prevSolDisplacement = solDisplacement;
      prevDamageSnapshot = snapshotDamage();

      if (!converged && dispDamageIterationNumber >= maxIterations)
        throw std::runtime_error("Displacement-damage iteration did not converge within "
                                  "MaxNumberOfIterations = " + std::to_string(maxIterations) +
                                  " iterations at temperature step " + std::to_string(temperatureStepNumber));
    } while (!converged);

    recordMacroscopicState(refinementCycle, temperatureStepNumber, curT);
    commitConvergedDamage();
  }

  template class ElasticProblem<3>;
}

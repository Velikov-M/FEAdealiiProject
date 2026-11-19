#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

  /** Backward-Euler residual of the cooling-only basal damage law.
   *
   * The pseudo-time increment is @p tempInc = -Delta T. Local sigma_33 is evaluated at the
   * trial damage, and the Macaulay clamp gives zero damage rate below the tensile threshold.
   */
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

  /** Advance damage over one cooling increment using the serial or WorkStream cell loop. */
  template <int dim>
  void ElasticProblem<dim>::predict_damage_tempInc_external_solver(double curTemperature, double tempInc)
  {
    TimerOutput::Scope timer_section(timer, "Predict damage increment");

    // Read kinetic parameters once and pass them to every cell worker.
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
        // Total strain = macroscopic strain + fluctuation strain; Ebar is zero for other BCs.
        globalStrain = dealii::symmetrize(global_displacement_gradients[q]) + macroscopicStrain;
        localStrain = Physics::Transformations::basis_transformation(globalStrain, rotTensorGlobalToLocal);
        localStress = getLocalStressTensor(localStrain, cur_damage, curTemperature);

        // The driving stress is the REAL stress computed from the current numerical damage, so
        // the negative-feedback loop is closed: growing damage lowers C33 -> lowers sigma_33 ->
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
  void ElasticProblem<dim>::resetAitkenState()
  {
    aitkenPreviousResidual.clear();
    aitkenMu = 1.0;
    aitkenHasPrevious = false;
  }

  template <int dim>
  void ElasticProblem<dim>::applyAitkenRelaxation(const DamageSnapshot &damageBeforePredict)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();

    // r_k = F(omega_k) - omega_k, with F(omega_k) sitting in damageInQuadraturePoints right now.
    DamageSnapshot residual;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const auto &before = damageBeforePredict.at(cell);
      std::vector<double> r(n_q_points);
      for (unsigned int q = 0; q < n_q_points; ++q)
        r[q] = cell_damage[q]->damage - before[q];
      residual[cell] = std::move(r);
    }

    if (aitkenHasPrevious)
    {
      double numerator = 0.0, denominator = 0.0;
      for (const auto &entry : residual)
      {
        const auto &rPrev = aitkenPreviousResidual.at(entry.first);
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const double dr = entry.second[q] - rPrev[q];
          numerator   += rPrev[q] * dr;
          denominator += dr * dr;
        }
      }
      // A vanishing denominator means the residual did not move between passes: either already
      // converged, or stuck. Either way there is nothing to extrapolate from, so fall back to
      // plain substitution rather than dividing by ~0.
      if (denominator > 1e-300)
        aitkenMu = -aitkenMu * numerator / denominator;
      else
        aitkenMu = 1.0;

      // Guard rails. Aitken is derived for a linear fixed point; on a nonlinear one the factor
      // can come out negative (which would move damage backwards, unphysical here) or very large
      // (which would overshoot far past any admissible state). Both are clamped rather than
      // trusted.
      aitkenMu = std::clamp(aitkenMu, 0.1, 5.0);

      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        auto cell_damage = damageInQuadraturePoints.get_data(cell);
        const auto &before = damageBeforePredict.at(cell);
        const auto &r = residual.at(cell);
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // Clamp to the same admissible window the kinetic solver uses: damage is
          // non-decreasing within a temperature step, and must stay clear of (1-omega)^-m.
          cell_damage[q]->damage =
            std::clamp(before[q] + aitkenMu * r[q], cell_damage[q]->old_damage, 0.999);
        }
      }
      ++aitkenRelaxedIterations;
    }

    aitkenPreviousResidual = std::move(residual);
    aitkenHasPrevious = true;
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
    // residual() returns system_rhs - K U, i.e. F_theta - K U.
    system_matrix.residual(systemResidual, solDisplacement, system_rhs);
    // Periodic equilibrium is K U + L Ebar = F_theta, so include the coupling term.
    if (boundaryConditionType == BoundaryConditionType::periodic)
    {
      if (prm.get_bool({"SolverParameters"}, "FrozenCbarQuasiNewton"))
        std::cout << "   Ebar iterations this pass = " << lastMacroIterations
                  << ", Cbar rebuilds so far = " << cbarRefreshCount << std::endl;
      const Vector<double> EbarKelvin =
        Physics::Notation::Kelvin::to_vector(macroscopicStrain);
      for (unsigned int k = 0; k < nMacroDofs; ++k)
        systemResidual.add(-EbarKelvin[k], macroCouplingColumns[k]);
    }
    // Constrained-DOF rows carry a placeholder diagonal that distribute_local_to_global()
    // writes purely to keep the matrix well-conditioned for the preconditioner -- with no
    // matching right-hand-side entry, since those rows are never meant to be interpreted as
    // real equilibrium equations. Exclude their placeholder diagonal entries from the norm.
    constraints.set_zero(systemResidual);
    double systemResidualNorm = systemResidual.l2_norm();
    double systemRhsNorm = system_rhs.l2_norm();
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

  template class ElasticProblem<3>;
}

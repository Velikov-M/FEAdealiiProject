#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

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

    // Physical self-check of the stiffness at both ends of the temperature range -- the constants
    // are temperature dependent, so checking one temperature would not cover the run.
    for (const double Tcheck : {prm.get_double({"ModelParameters"}, "InitialTemperature"),
                                prm.get_double({"ModelParameters"}, "FinalTemperature")})
    {
      const auto c = elasticityTensor.voigtConstants(Tcheck);
      checkTransverseIsotropy<dim>(elasticityTensor.getElasticityTensor(0.0, Tcheck),
                                   c[0], c[1], c[2], c[3], c[4]);
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

    std::string mshFileName = prm.get({"InputFiles"}, "MeshFile");
    std::string cellToMaterialFileName = prm.get({"InputFiles"}, "CellToMaterialFile");
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName);
    auto mshStream = parseMSHandPrepareStream(mshFileName);
    grid_in.read_msh(mshStream);
    parse_cellToMaterial_data(cellToMaterialFileName);

    // Grain identity is a 1-based material_id inherited by children on refinement. Orientations
    // are generated per grain below, replacing any metadata read from the mesh.
    const bool uniformOrientation = prm.get_bool({"BoundaryConditions"}, "UniformOrientation");
    const unsigned int orientationSeed =
      prm.get_integer({"BoundaryConditions"}, "OrientationSeed");
    // ORDER IS LOAD-BEARING; see the assignBoundaryIds() call below, which must run on the
    // COARSE mesh and therefore BEFORE this refinement.
    const unsigned int initialRefinements =
      prm.get_integer({"MeshRefinementParameters"}, "InitialGlobalRefinements");
    assignBoundaryIds();
    if (initialRefinements > 0)
      triangulation.refine_global(initialRefinements);

    grainsPerSide = prm.get_integer({"BoundaryConditions"}, "GrainsPerSide");
    const unsigned int nGrains = grainsPerSide * grainsPerSide * grainsPerSide;
    if (uniformOrientation)
      std::cout << "*** UniformOrientation smoke test: every grain identically oriented. "
                   "Expect Cbar = C, u~ = 0, sigma = 0 pointwise. ***" << std::endl;

    // Generate one orientation per grain; positional grain assignment preserves them on refinement.
    std::mt19937_64 rng(orientationSeed);
    rotmatOrientationCell.clear();
    rotmatOrientationCell.reserve(nGrains);
    for (unsigned int g = 0; g < nGrains; ++g)
      rotmatOrientationCell.push_back(
        uniformOrientation ? Tensor<2, dim>(unit_symmetric_tensor<dim>())
                           : randomOrientationUniformOnSphere<dim>(rng));

    assignGrains();

    // types::material_id is a build-time typedef (unsigned int in this deal.II build, but
    // historically unsigned char, which tops out at 255). Check rather than assume: on a
    // narrower build the ids would wrap silently and distinct grains would share an orientation.
    if (nGrains >= numbers::invalid_material_id)
      throw std::runtime_error(
        "GrainsPerSide^3 = " + std::to_string(nGrains) + " grains exceeds what "
        "types::material_id can represent on this deal.II build.");

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
                << ", |mean c-axis|=" << meanAxis.norm()
                << " (expect ~0), mean c_z^2=" << meanCzSquared
                << " (expect ~0.3333)" << std::endl;
    }

    // Each refinement cycle restarts the full cooling range with freshly initialized damage.
    const unsigned int nRefinementCycles = prm.get_integer({"MeshRefinementParameters"}, "NumberOfRefinementCycles");
    const std::string refinementStrategy = prm.get({"MeshRefinementParameters"}, "RefinementStrategy");
    const double fractionToRefine = prm.get_double({"MeshRefinementParameters"}, "FractionToRefine");
    const double fractionToCoarsen = prm.get_double({"MeshRefinementParameters"}, "FractionToCoarsen");
    const unsigned int outputFrequency = prm.get_integer({"OutputParameters"}, "OutputFrequency");

    for (unsigned int refinementCycle = 0; refinementCycle < nRefinementCycles; ++refinementCycle)
    {
      if (refinementCycle == 0)
        setup_system();
      else if (refinementStrategy == "uniform")
      {
        triangulation.refine_global(1);
        assignGrains();   // grain count fixed; each grain simply gains elements
        setup_system();
      }
      else
        refine_grid(fractionToRefine, fractionToCoarsen);

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
    writeMacroscopicHistory(outputPath("_macroscopic_history.csv"));

    if (!pvdTimeSeries.empty())
    {
      std::ofstream pvd(outputPath(".pvd"));
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

    // One row per CONVERGED temperature step -- not per inner iteration, which would record
    // states that are not equilibrium solutions.
    {
      MacroscopicRecord record;
      record.refinementCycle  = refinementCycle;
      record.temperatureStep  = temperatureStepNumber;
      record.temperature      = curT;
      record.cooling          = prm.get_double({"ModelParameters"}, "InitialTemperature") - curT;
      record.Ebar = {{macroscopicStrain[0][0], macroscopicStrain[1][1], macroscopicStrain[2][2],
                      macroscopicStrain[1][2], macroscopicStrain[0][2], macroscopicStrain[0][1]}};
      for (unsigned int k = 0; k < nMacroDofs; ++k)
        record.CbarDiag[k] = homogenizedStiffness(k, k);
      record.CbarAsymmetry = lastCbarAsymmetry;
      record.cgIterations  = lastCgIterations;
      record.meanStressNorm = computeAverageStress(curT).norm();

      double meanDamage = 0.0, maxDamage = 0.0, weight = 0.0;
      const QGauss<dim> quad(fe.degree + 1);
      FEValues<dim> fev(fe, quad, update_JxW_values);
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fev.reinit(cell);
        const auto qp = damageInQuadraturePoints.get_data(cell);
        for (unsigned int q = 0; q < quad.size(); ++q)
        {
          meanDamage += qp[q]->damage * fev.JxW(q);
          maxDamage = std::max(maxDamage, qp[q]->damage);
          weight += fev.JxW(q);
        }
      }
      record.meanDamage = meanDamage / weight;
      record.maxDamage  = maxDamage;
      macroscopicHistory.push_back(record);
    }

    // Advance the converged start-of-step damage baseline only after convergence.
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q)
        cell_damage[q]->old_damage = cell_damage[q]->damage;
    }

  }

  template class ElasticProblem<3>;
}

#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  std::string ElasticProblem<dim>::outputPath(const std::string &suffix) const
  {
    const std::string dir = prm.get({"OutputParameters"}, "OutputDirectory");
    const std::string prefix = prm.get({"OutputParameters"}, "OutputFileNamePrefix");
    if (dir.empty() || dir == ".")
      return prefix + suffix;
    std::filesystem::create_directories(dir);
    return (std::filesystem::path(dir) / (prefix + suffix)).string();
  }

  template <int dim>
  void ElasticProblem<dim>::writeMacroscopicHistory(const std::string &fileName) const
  {
    if (macroscopicHistory.empty())
      return;

    std::ofstream out(fileName);
    out << "# Macroscopic history, one row per converged temperature step.\n"
        << "# Ebar and alphaBar are TENSOR components (11 22 33 23 13 12), not Kelvin -- no\n"
        << "# sqrt(2) factors here. alphaBar = Ebar / (T - T0) is the effective CTE of the\n"
        << "# polycrystal, defined only where the cooling is nonzero. CbarDiag is the Kelvin\n"
        << "# diagonal of the homogenized stiffness in Pa. CbarAsymmetry should stay at solver\n"
        << "# precision: Hill-Mandel guarantees symmetry, nothing in the code enforces it.\n";
    out << "cycle,step,T,cooling,"
        << "Ebar_11,Ebar_22,Ebar_33,Ebar_23,Ebar_13,Ebar_12,"
        << "alphaBar_11,alphaBar_22,alphaBar_33,alphaBar_23,alphaBar_13,alphaBar_12,"
        << "CbarDiag_1,CbarDiag_2,CbarDiag_3,CbarDiag_4,CbarDiag_5,CbarDiag_6,"
        << "meanDamage,maxDamage,meanStressNorm,CbarAsymmetry,cgIterations\n";
    out.precision(10);
    out << std::scientific;
    for (const auto &r : macroscopicHistory)
    {
      out << r.refinementCycle << ',' << r.temperatureStep << ','
          << r.temperature << ',' << r.cooling;
      for (double v : r.Ebar) out << ',' << v;
      // Effective CTE. The cooling is nonzero for every recorded step, but guard anyway rather
      // than emit inf if a zero-length step is ever recorded.
      for (double v : r.Ebar) out << ',' << (r.cooling != 0.0 ? v / (-r.cooling) : 0.0);
      for (double v : r.CbarDiag) out << ',' << v;
      out << ',' << r.meanDamage << ',' << r.maxDamage << ',' << r.meanStressNorm
          << ',' << r.CbarAsymmetry << ',' << r.cgIterations << '\n';
    }
    std::cout << "Wrote " << fileName << " (" << macroscopicHistory.size() << " steps)."
              << std::endl;
  }

  CellOutputFields::CellOutputFields(unsigned int cellCount,
                                     bool writeFullStress,
                                     bool writeFullStrain)
    : damage(cellCount)
    , sigma33(cellCount)
    , eps33(cellCount)
    , basalShear(cellCount)
    , vonMises(cellCount)
    , pressure(cellCount)
    , damageDrivingRatio(cellCount)
    , writeFullStress(writeFullStress)
    , writeFullStrain(writeFullStrain)
  {
    for (auto &component : macroscopicStrain)
      component.reinit(cellCount);
    if (writeFullStress)
      {
        for (auto &component : localStress)
          component.reinit(cellCount);
        for (auto &component : globalStress)
          component.reinit(cellCount);
      }
    if (writeFullStrain)
      {
        for (auto &component : localStrain)
          component.reinit(cellCount);
        for (auto &component : globalStrain)
          component.reinit(cellCount);
      }
  }

  template <int dim>
  void ElasticProblem<dim>::addDisplacementAndOrientationOutput(
    DataOut<dim> &dataOut,
    CAxisPostprocessor<dim> &cAxisPostprocessor,
    TotalDisplacementPostprocessor<dim> &totalDisplacementPostprocessor) const
  {
    const std::vector<std::string> displacementNames(dim, "displacement");
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      displacementInterpretation(dim,
        DataComponentInterpretation::component_is_part_of_vector);
    dataOut.add_data_vector(solDisplacement, displacementNames,
                            DataOut<dim>::type_dof_data, displacementInterpretation);
    dataOut.add_data_vector(solDisplacement, cAxisPostprocessor);
    dataOut.add_data_vector(solDisplacement, totalDisplacementPostprocessor);
  }

  template <int dim>
  CellOutputFields ElasticProblem<dim>::computeCellOutputFields(
    bool writeFullStress, bool writeFullStrain) const
  {
    CellOutputFields fields(triangulation.n_active_cells(), writeFullStress, writeFullStrain);
    const std::array<double, 6> ebarValues{{
      macroscopicStrain[0][0], macroscopicStrain[1][1], macroscopicStrain[2][2],
      macroscopicStrain[1][2], macroscopicStrain[0][2], macroscopicStrain[0][1]}};
    for (unsigned int k = 0; k < 6; ++k)
      fields.macroscopicStrain[k] = ebarValues[k];

    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> feValues(fe, quadrature, update_gradients | update_JxW_values);
    std::vector<Tensor<2, dim>> gradients(quadrature.size());
    const double curT = lastProcessedTemperature;
    const double stressThreshold =
      prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        feValues.reinit(cell);
        feValues[displacement].get_function_gradients(solDisplacement, gradients);
        const auto qpDamage = damageInQuadraturePoints.get_data(cell);
        const Tensor<2, dim> rotGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
        const Tensor<2, dim> rotLocalToGlobal = transpose(rotGlobalToLocal);

        double damage = 0.0;
        SymmetricTensor<2, dim> localStressAvg, localStrainAvg;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const SymmetricTensor<2, dim> globalStrain =
              dealii::symmetrize(gradients[q]) + macroscopicStrain;
            const SymmetricTensor<2, dim> localStrain =
              Physics::Transformations::basis_transformation(globalStrain, rotGlobalToLocal);
            damage += qpDamage[q]->damage;
            localStrainAvg += localStrain;
            localStressAvg += getLocalStressTensor(localStrain, qpDamage[q]->damage, curT);
          }
        damage /= quadrature.size();
        localStrainAvg /= quadrature.size();
        localStressAvg /= quadrature.size();

        const unsigned int idx = cell->active_cell_index();
        fields.damage[idx] = damage;
        fields.sigma33[idx] = localStressAvg[2][2];
        fields.eps33[idx] = localStrainAvg[2][2];
        fields.basalShear[idx] = std::sqrt(localStressAvg[0][2] * localStressAvg[0][2] +
                                           localStressAvg[1][2] * localStressAvg[1][2]);
        fields.vonMises[idx] =
          std::sqrt(1.5 * (deviator(localStressAvg) * deviator(localStressAvg)));
        fields.pressure[idx] = -trace(localStressAvg) / dim;
        fields.damageDrivingRatio[idx] = fields.sigma33[idx] / stressThreshold;

        if (writeFullStrain)
          {
            const SymmetricTensor<2, dim> globalStrainAvg =
              Physics::Transformations::basis_transformation(localStrainAvg, rotLocalToGlobal);
            fields.localStrain[0][idx] = localStrainAvg[0][0];
            fields.localStrain[1][idx] = localStrainAvg[1][1];
            fields.localStrain[2][idx] = localStrainAvg[0][1];
            fields.localStrain[3][idx] = localStrainAvg[0][2];
            fields.localStrain[4][idx] = localStrainAvg[1][2];
            fields.globalStrain[0][idx] = globalStrainAvg[0][0];
            fields.globalStrain[1][idx] = globalStrainAvg[1][1];
            fields.globalStrain[2][idx] = globalStrainAvg[2][2];
            fields.globalStrain[3][idx] = globalStrainAvg[0][1];
            fields.globalStrain[4][idx] = globalStrainAvg[0][2];
            fields.globalStrain[5][idx] = globalStrainAvg[1][2];
          }

        if (writeFullStress)
          {
            const SymmetricTensor<2, dim> globalStressAvg =
              Physics::Transformations::basis_transformation(localStressAvg, rotLocalToGlobal);
            fields.localStress[0][idx] = localStressAvg[0][0];
            fields.localStress[1][idx] = localStressAvg[1][1];
            fields.localStress[2][idx] = localStressAvg[0][1];
            fields.localStress[3][idx] = localStressAvg[0][2];
            fields.localStress[4][idx] = localStressAvg[1][2];
            fields.globalStress[0][idx] = globalStressAvg[0][0];
            fields.globalStress[1][idx] = globalStressAvg[1][1];
            fields.globalStress[2][idx] = globalStressAvg[2][2];
            fields.globalStress[3][idx] = globalStressAvg[0][1];
            fields.globalStress[4][idx] = globalStressAvg[0][2];
            fields.globalStress[5][idx] = globalStressAvg[1][2];
          }
      }
    return fields;
  }

  template <int dim>
  void ElasticProblem<dim>::addScalarMechanicalOutput(
    DataOut<dim> &dataOut, const CellOutputFields &fields) const
  {
    dataOut.add_data_vector(fields.damage, "damage");
    dataOut.add_data_vector(fields.sigma33, "sigma_33");
    dataOut.add_data_vector(fields.eps33, "eps_33");
    dataOut.add_data_vector(fields.basalShear, "tau_basal");
    dataOut.add_data_vector(fields.vonMises, "von_mises");
    dataOut.add_data_vector(fields.pressure, "pressure");
    dataOut.add_data_vector(fields.damageDrivingRatio, "damage_driving_ratio");
  }

  template <int dim>
  void ElasticProblem<dim>::addMacroscopicStrainOutput(
    DataOut<dim> &dataOut, const CellOutputFields &fields) const
  {
    static const std::array<const char *, 6> names{{
      "Ebar_11", "Ebar_22", "Ebar_33", "Ebar_23", "Ebar_13", "Ebar_12"}};
    for (unsigned int k = 0; k < names.size(); ++k)
      dataOut.add_data_vector(fields.macroscopicStrain[k], names[k]);
  }

  template <int dim>
  void ElasticProblem<dim>::addFullStressOutput(
    DataOut<dim> &dataOut, const CellOutputFields &fields) const
  {
    if (!fields.writeFullStress)
      return;
    static const std::array<const char *, 5> localNames{{
      "sigma_11", "sigma_22", "sigma_12", "sigma_13", "sigma_23"}};
    static const std::array<const char *, 6> globalNames{{
      "sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"}};
    for (unsigned int k = 0; k < localNames.size(); ++k)
      dataOut.add_data_vector(fields.localStress[k], localNames[k]);
    for (unsigned int k = 0; k < globalNames.size(); ++k)
      dataOut.add_data_vector(fields.globalStress[k], globalNames[k]);
  }

  template <int dim>
  void ElasticProblem<dim>::addFullStrainOutput(
    DataOut<dim> &dataOut, const CellOutputFields &fields) const
  {
    if (!fields.writeFullStrain)
      return;
    static const std::array<const char *, 5> localNames{{
      "eps_11", "eps_22", "eps_12", "eps_13", "eps_23"}};
    static const std::array<const char *, 6> globalNames{{
      "eps_xx", "eps_yy", "eps_zz", "eps_xy", "eps_xz", "eps_yz"}};
    for (unsigned int k = 0; k < localNames.size(); ++k)
      dataOut.add_data_vector(fields.localStrain[k], localNames[k]);
    for (unsigned int k = 0; k < globalNames.size(); ++k)
      dataOut.add_data_vector(fields.globalStrain[k], globalNames[k]);
  }

  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    TimerOutput::Scope timer_section(timer, "Output results");
    DataOut<dim> dataOut;
    dataOut.attach_dof_handler(dof_handler);

    CAxisPostprocessor<dim> cAxisPostprocessor(rotmatOrientationCell);
    TotalDisplacementPostprocessor<dim> totalDisplacementPostprocessor(macroscopicStrain);
    addDisplacementAndOrientationOutput(dataOut, cAxisPostprocessor,
                                        totalDisplacementPostprocessor);

    const bool writeFullStress = prm.get_bool({"OutputParameters"}, "WriteFullStressTensor");
    const bool writeFullStrain = prm.get_bool({"OutputParameters"}, "WriteFullStrainTensor");
    const CellOutputFields fields = computeCellOutputFields(writeFullStress, writeFullStrain);
    addScalarMechanicalOutput(dataOut, fields);
    addMacroscopicStrainOutput(dataOut, fields);
    addFullStressOutput(dataOut, fields);
    addFullStrainOutput(dataOut, fields);

    dataOut.build_patches();
    const std::string fileName = outputPath("-" + std::to_string(cycle) + ".vtu");
    std::ofstream output(fileName);
    dataOut.write_vtu(output);

    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    pvdTimeSeries.emplace_back(T0 - lastProcessedTemperature,
                               std::filesystem::path(fileName).filename().string());
  }
  template <int dim>
  void ElasticProblem<dim>::process_solution(unsigned int refinementCycle, const unsigned int tempStepNum, const unsigned int iterSolverStepNum, double curT)
  {
    TimerOutput::Scope timer_section(timer, "Process solution");
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

        // sigma_33 -- LOCAL frame, i.e. normal to THIS grain's basal plane, which is what
        // drives the kinetic law. sigma_zz (global) would mean something quite different and is
        // not what the damage law reads.
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

    // Damage approaching 1 is where the kinetic law stops being trustworthy: the (1-omega)^-m
    // term stiffens sharply and the root bracket [old_damage, 0.999] narrows towards nothing.
    // Warn loudly ONCE rather than per step, and do not stop -- a handful of quadrature points at
    // the ceiling does not invalidate the rest of the field, but it does mean the peak values
    // should not be quoted.
    const double damageWarningThreshold =
      prm.get_double({"SolverParameters"}, "DamageWarningThreshold");
    if (maxDamage > damageWarningThreshold && !damageCeilingWarned)
    {
      damageCeilingWarned = true;
      std::cout << "\n*** WARNING: maximum damage " << maxDamage << " exceeded "
                << damageWarningThreshold << " at temperature step " << tempStepNum
                << " (T = " << curT << ").\n"
                << "    The kinetic law's (1-omega)^-m term is stiff here and the root search\n"
                << "    brackets at 0.999, so peak damage values past this point are not\n"
                << "    trustworthy even where the solver reports convergence. Consider a\n"
                << "    smaller temperature step, or treat this run as reaching its validity\n"
                << "    limit. ***\n" << std::endl;
    }

    // <sigma>: identically zero for a genuinely free-cooling state (traction-free boundaries
    // give it by the average stress theorem), and large and triaxial for the clamped case. So
    // this doubles as a correctness check on "freeSurface" and as the periodic residual.
    const SymmetricTensor<2, dim> meanStress = computeAverageStress(curT);

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();

    std::cout << "Refinement cycle " << refinementCycle << ", temperature step " << tempStepNum
              << ", iteration " << iterSolverStepNum << " (T = " << curT << "):" << std::endl
              << "   Cells: " << n_active_cells << ", DoFs: " << n_dofs
              << ", CG iterations = " << lastCgIterations << std::endl
              << "   sigma_33 (local): min=" << minSigmaZZ << ", max=" << maxSigmaZZ
              << ", tension fraction=" << tensionFraction
              << ", above threshold=" << thresholdFraction << std::endl
              << "   damage: mean=" << meanDamage << ", max=" << maxDamage << std::endl
              << "   <sigma> norm=" << meanStress.norm()
              << " (trace/3 = " << trace(meanStress) / dim << ")" << std::endl;
    if (boundaryConditionType == BoundaryConditionType::periodic)
    {
      // Ebar and the effective CTE are results, not diagnostics: alphaBar is the homogenized
      // thermal expansion of the polycrystal, which falls out of this formulation for free.
      const double dT = curT - prm.get_double({"ModelParameters"}, "InitialTemperature");
      std::cout << "   Ebar (diag) = " << macroscopicStrain[0][0] << ", "
                << macroscopicStrain[1][1] << ", " << macroscopicStrain[2][2]
                << "   alphaBar (diag) = ";
      if (dT != 0.0)
        std::cout << macroscopicStrain[0][0]/dT << ", " << macroscopicStrain[1][1]/dT
                  << ", " << macroscopicStrain[2][2]/dT;
      std::cout << "\n   Cbar diag (GPa) = ";
      for (unsigned int i = 0; i < nMacroDofs; ++i)
        std::cout << homogenizedStiffness(i, i) / 1e9 << " ";
      std::cout << "  (rel. asymmetry " << lastCbarAsymmetry << ")" << std::endl;
    }

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

  template <int dim>
  void ElasticProblem<dim>::recordMacroscopicState(const unsigned int refinementCycle,
                                                   const unsigned int temperatureStepNumber,
                                                   const double curT)
  {
    TimerOutput::Scope timer_section(timer, "Record macroscopic state");
    // Record only the converged state, not intermediate fixed-point iterates.
    MacroscopicRecord record;
    record.refinementCycle = refinementCycle;
    record.temperatureStep = temperatureStepNumber;
    record.temperature = curT;
    record.cooling = prm.get_double({"ModelParameters"}, "InitialTemperature") - curT;
    record.Ebar = {{macroscopicStrain[0][0], macroscopicStrain[1][1], macroscopicStrain[2][2],
                    macroscopicStrain[1][2], macroscopicStrain[0][2], macroscopicStrain[0][1]}};
    for (unsigned int k = 0; k < nMacroDofs; ++k)
      record.CbarDiag[k] = homogenizedStiffness(k, k);
    record.CbarAsymmetry = lastCbarAsymmetry;
    record.cgIterations = lastCgIterations;
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
    record.maxDamage = maxDamage;
    macroscopicHistory.push_back(record);
  }

  template <int dim>
  void ElasticProblem<dim>::finalizeRunOutput()
  {
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
    writeMacroscopicHistory(outputPath("_macroscopic_history.csv"));
    std::cout << "Linear solver totals: solves=" << totalLinearSolveCount
              << ", CG iterations=" << totalCgIterations
              << ", Cbar rebuilds=" << cbarRefreshCount << std::endl;

    if (!pvdTimeSeries.empty())
    {
      std::ofstream pvd(outputPath(".pvd"));
      DataOutBase::write_pvd_record(pvd, pvdTimeSeries);
      std::cout << "Wrote ParaView time series polycrystalCooling.pvd ("
                << pvdTimeSeries.size() << " steps). Open THIS, not the individual .vtu files."
                << std::endl;
    }
  }

  template class ElasticProblem<3>;
}

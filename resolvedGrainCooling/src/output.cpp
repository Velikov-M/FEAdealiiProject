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

  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    // Local 11/22/33 and global xx/yy/zz names identify tensor frames explicitly. sigma_33 is
    // the per-grain c-axis normal stress that drives damage, not global sigma_zz.
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

    // u = Ebar.x + u~. Warp By Vector on THIS shows the RVE actually contracting; warping by
    // `displacement` alone would show only the fluctuation. See the postprocessor's comment.
    TotalDisplacementPostprocessor<dim> totalDisplacementPostprocessor(macroscopicStrain);
    data_out.add_data_vector(solDisplacement, totalDisplacementPostprocessor);

    const bool writeFullTensor =
      prm.get_bool({"OutputParameters"}, "WriteFullStressTensor");
    const bool writeFullStrain =
      prm.get_bool({"OutputParameters"}, "WriteFullStrainTensor");
    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const double curT = lastProcessedTemperature;
    const double stressThreshold =
      prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");

    const unsigned int nCells = triangulation.n_active_cells();
    Vector<double> cell_damage(nCells);
    Vector<double> cell_sigma_33(nCells);          // LOCAL: normal to the basal plane -- the driver
    Vector<double> cell_eps_33(nCells);            // LOCAL: its kinematic partner, total strain
    Vector<double> cell_tau_basal(nCells);         // LOCAL: sqrt(sigma_13^2 + sigma_23^2)
    Vector<double> cell_von_mises(nCells);         // frame independent
    Vector<double> cell_pressure(nCells);          // frame independent
    Vector<double> cell_driving_ratio(nCells);     // sigma_33 / threshold

    // Complete tensors, only when asked for. Local: 11, 22, 12, 13, 23 (33 is always written
    // above, so it is not repeated here). Global: all six.
    static const std::array<const char *, 5> localNames{{
      "sigma_11", "sigma_22", "sigma_12", "sigma_13", "sigma_23"}};
    static const std::array<const char *, 6> globalNames{{
      "sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"}};
    static const std::array<const char *, 5> localStrainNames{{
      "eps_11", "eps_22", "eps_12", "eps_13", "eps_23"}};
    static const std::array<const char *, 6> globalStrainNames{{
      "eps_xx", "eps_yy", "eps_zz", "eps_xy", "eps_xz", "eps_yz"}};
    std::array<Vector<double>, 5> cell_sigma_local, cell_eps_local;
    std::array<Vector<double>, 6> cell_sigma_global, cell_eps_global;
    if (writeFullTensor)
    {
      for (auto &v : cell_sigma_local)  v.reinit(nCells);
      for (auto &v : cell_sigma_global) v.reinit(nCells);
    }
    if (writeFullStrain)
    {
      for (auto &v : cell_eps_local)  v.reinit(nCells);
      for (auto &v : cell_eps_global) v.reinit(nCells);
    }

    std::vector<Tensor<2, dim>> gradients(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[displacement].get_function_gradients(solDisplacement, gradients);
      const auto qp_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2, dim> rotGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      const Tensor<2, dim> rotLocalToGlobal = transpose(rotGlobalToLocal);

      // Average tensors first, then derive cell scalars so nonlinear fields such as tau_basal
      // remain consistent with the reported tensor components.
      double damage = 0.0;
      SymmetricTensor<2, dim> localStressAvg, localStrainAvg;

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Total strain: macroscopic plus fluctuation. macroscopicStrain is identically zero for
        // every BC family except "periodic".
        const SymmetricTensor<2, dim> globalStrain =
          dealii::symmetrize(gradients[q]) + macroscopicStrain;
        const SymmetricTensor<2, dim> localStrain =
          Physics::Transformations::basis_transformation(globalStrain, rotGlobalToLocal);
        damage         += qp_damage[q]->damage;
        localStrainAvg += localStrain;
        localStressAvg += getLocalStressTensor(localStrain, qp_damage[q]->damage, curT);
      }
      damage         /= n_q_points;
      localStrainAvg /= n_q_points;
      localStressAvg /= n_q_points;

      const unsigned int idx = cell->active_cell_index();
      cell_damage[idx]   = damage;
      cell_sigma_33[idx] = localStressAvg[2][2];
      // The kinematic partner of sigma_33: TOTAL strain normal to the basal plane. Elastic strain
      // is this minus alpha_c*(T-T_0); the thermal part is the same for every grain in the local
      // frame, which is why it is not written separately.
      cell_eps_33[idx] = localStrainAvg[2][2];
      // Basal shear magnitude; the current kinetic law itself reads only sigma_33.
      cell_tau_basal[idx] = std::sqrt(localStressAvg[0][2] * localStressAvg[0][2] +
                                      localStressAvg[1][2] * localStressAvg[1][2]);
      // Invariants: identical in either frame, so there is no frame for a viewer to get wrong.
      cell_von_mises[idx] =
        std::sqrt(1.5 * (deviator(localStressAvg) * deviator(localStressAvg)));
      cell_pressure[idx] = -trace(localStressAvg) / dim;   // positive in compression
      // > 1 means this grain is above threshold and its damage is actively growing. The single
      // most legible scalar for the story: threshold it at 1 in ParaView and what remains is
      // exactly the set of grains driving the damage seen in the `damage` array.
      cell_driving_ratio[idx] = cell_sigma_33[idx] / stressThreshold;

      if (writeFullStrain)
      {
        const SymmetricTensor<2, dim> globalStrainAvg =
          Physics::Transformations::basis_transformation(localStrainAvg, rotLocalToGlobal);
        // Order must match localStrainNames / globalStrainNames above.
        cell_eps_local[0][idx] = localStrainAvg[0][0];
        cell_eps_local[1][idx] = localStrainAvg[1][1];
        cell_eps_local[2][idx] = localStrainAvg[0][1];
        cell_eps_local[3][idx] = localStrainAvg[0][2];
        cell_eps_local[4][idx] = localStrainAvg[1][2];
        cell_eps_global[0][idx] = globalStrainAvg[0][0];
        cell_eps_global[1][idx] = globalStrainAvg[1][1];
        cell_eps_global[2][idx] = globalStrainAvg[2][2];
        cell_eps_global[3][idx] = globalStrainAvg[0][1];
        cell_eps_global[4][idx] = globalStrainAvg[0][2];
        cell_eps_global[5][idx] = globalStrainAvg[1][2];
      }

      if (writeFullTensor)
      {
        const SymmetricTensor<2, dim> globalStressAvg =
          Physics::Transformations::basis_transformation(localStressAvg, rotLocalToGlobal);
        // Order must match localNames / globalNames above.
        cell_sigma_local[0][idx] = localStressAvg[0][0];
        cell_sigma_local[1][idx] = localStressAvg[1][1];
        cell_sigma_local[2][idx] = localStressAvg[0][1];
        cell_sigma_local[3][idx] = localStressAvg[0][2];
        cell_sigma_local[4][idx] = localStressAvg[1][2];
        cell_sigma_global[0][idx] = globalStressAvg[0][0];
        cell_sigma_global[1][idx] = globalStressAvg[1][1];
        cell_sigma_global[2][idx] = globalStressAvg[2][2];
        cell_sigma_global[3][idx] = globalStressAvg[0][1];
        cell_sigma_global[4][idx] = globalStressAvg[0][2];
        cell_sigma_global[5][idx] = globalStressAvg[1][2];
      }
    }

    data_out.add_data_vector(cell_damage, "damage");
    data_out.add_data_vector(cell_sigma_33, "sigma_33");
    data_out.add_data_vector(cell_eps_33, "eps_33");
    data_out.add_data_vector(cell_tau_basal, "tau_basal");
    data_out.add_data_vector(cell_von_mises, "von_mises");
    data_out.add_data_vector(cell_pressure, "pressure");
    data_out.add_data_vector(cell_driving_ratio, "damage_driving_ratio");

    // The macroscopic strain. It is ONE tensor for the whole RVE, not a field -- deal.II's
    // DataOut has no VTK FieldData support, so it is written as uniform cell arrays purely so it
    // travels with the picture and can be read off in ParaView's Information panel. The
    // per-step history in macroscopic_history.csv is the better vehicle for plotting it.
    static const std::array<const char *, 6> ebarNames{{
      "Ebar_11", "Ebar_22", "Ebar_33", "Ebar_23", "Ebar_13", "Ebar_12"}};
    const std::array<double, 6> ebarValues{{
      macroscopicStrain[0][0], macroscopicStrain[1][1], macroscopicStrain[2][2],
      macroscopicStrain[1][2], macroscopicStrain[0][2], macroscopicStrain[0][1]}};
    std::array<Vector<double>, 6> cell_ebar;
    for (unsigned int k = 0; k < 6; ++k)
    {
      cell_ebar[k].reinit(nCells);
      cell_ebar[k] = ebarValues[k];
      data_out.add_data_vector(cell_ebar[k], ebarNames[k]);
    }
    if (writeFullTensor)
    {
      for (unsigned int k = 0; k < localNames.size(); ++k)
        data_out.add_data_vector(cell_sigma_local[k], localNames[k]);
      for (unsigned int k = 0; k < globalNames.size(); ++k)
        data_out.add_data_vector(cell_sigma_global[k], globalNames[k]);
    }
    if (writeFullStrain)
    {
      for (unsigned int k = 0; k < localStrainNames.size(); ++k)
        data_out.add_data_vector(cell_eps_local[k], localStrainNames[k]);
      for (unsigned int k = 0; k < globalStrainNames.size(); ++k)
        data_out.add_data_vector(cell_eps_global[k], globalStrainNames[k]);
    }

    data_out.build_patches();

    const std::string fileName = outputPath("-" + std::to_string(cycle) + ".vtu");
    std::ofstream output(fileName);
    data_out.write_vtu(output);

    // The .pvd resolves dataset paths RELATIVE to itself, and both sit in the same directory,
    // so the record must carry the bare filename, not the path used to open the stream.
    pvdTimeSeries.emplace_back(T0 - curT,
                               std::filesystem::path(fileName).filename().string());
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

  template class ElasticProblem<3>;
}

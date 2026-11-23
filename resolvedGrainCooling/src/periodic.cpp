#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

  // Periodic homogenization and macroscopic strain unknowns.

  // Volume average of the stress over the RVE. Plain quadrature summation of the same sigma the
  // damage update sees, so the two can never disagree about what the stress is.
  //
  // For "freeSurface" this should come out at ~0 to solver tolerance, since traction-free
  // boundaries give <sigma> = 0 identically (homogenization.md section 13). That makes it a
  // genuine correctness check and not just a diagnostic -- if it does not, either the boundary
  // treatment or the minimal support is wrong. For "zeroDisplacement" it will be large and
  // triaxial, which is the whole reason that case is not free cooling.
  template <int dim>
  SymmetricTensor<2, dim> ElasticProblem<dim>::computeAverageStress(
    double curTemperature,
    const Vector<double> &displacementField,
    const SymmetricTensor<2, dim> &macroStrain,
    bool includeThermalStrain) const
  {
    TimerOutput::Scope timer_section(timer, "Compute average stress");
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula, update_gradients | update_JxW_values);
    std::vector<Tensor<2, dim>> gradients(quadrature_formula.size());

    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const double deltaT = includeThermalStrain ? (curTemperature - T0) : 0.0;

    SymmetricTensor<2, dim> integratedStress;
    double volume = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[displacement].get_function_gradients(displacementField, gradients);
      const auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2, dim> rotGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      const Tensor<2, dim> rotLocalToGlobal = transpose(rotGlobalToLocal);
      // Damage-dependent, so it moves inside the quadrature loop below.

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
      {
        // Total strain = macroscopic + fluctuation, per the u = Ebar.x + u~ split.
        const SymmetricTensor<2, dim> globalStrain =
          dealii::symmetrize(gradients[q]) + macroStrain;
        const SymmetricTensor<2, dim> localStrain =
          Physics::Transformations::basis_transformation(globalStrain, rotGlobalToLocal);
        const SymmetricTensor<2, dim> localStress =
          elasticityTensor.getElasticityTensor(cell_damage[q]->damage, curTemperature) *
          (localStrain - cteTensor.getcteTensor(cell_damage[q]->damage, curTemperature) * deltaT);
        integratedStress +=
          Physics::Transformations::basis_transformation(localStress, rotLocalToGlobal) *
          fe_values.JxW(q);
        volume += fe_values.JxW(q);
      }
    }
    return integratedStress / volume;
  }

  // The state actually being carried by the solver.
  template <int dim>
  SymmetricTensor<2, dim>
  ElasticProblem<dim>::computeAverageStress(double curTemperature) const
  {
    return computeAverageStress(curTemperature, solDisplacement, macroscopicStrain, true);
  }

  // Scatters one cell's six L columns into the six global right-hand sides, through the SAME
  // constraints as the ordinary RHS. That gives L_hat = P^T L, which is exactly what the solves
  // in solve_with_zero_mean_stress() need.
  //
  // Every constraint here is homogeneous -- periodicity relates fluctuation DoFs to each other
  // with no offset, and the pin is u~ = 0 -- so the three-argument distribute_local_to_global is
  // correct and no constraint matrix needs to be passed.
  template <int dim>
  void ElasticProblem<dim>::distributeMacroCouplingColumns(
    const FullMatrix<double> &cell_L,
    const std::vector<types::global_dof_index> &local_dof_indices)
  {
    Vector<double> column(local_dof_indices.size());
    for (unsigned int k = 0; k < nMacroDofs; ++k)
    {
      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        column(i) = cell_L(i, k);
      constraints.distribute_local_to_global(column, local_dof_indices,
                                             macroCouplingColumns[k]);
    }
  }

  // The 7-solve of homogenization.md section 6.
  //
  // With damage frozen (which the staggered scheme guarantees within one pass), the problem is
  // LINEAR in (u~, Ebar), so this is exact rather than a linearization:
  //
  //     K U + L Ebar = F_theta          n equations
  //     <sigma> = 0                     6 equations, the natural condition conjugate to Ebar
  //
  // Eliminating U costs one solve for the thermal load and six for L's columns -- all against
  // the SAME matrix, hence one factorization (see prepareLinearSolver / LinearSolverType).
  template <int dim>
  void ElasticProblem<dim>::refreshHomogenizedStiffness(double curTemperature)
  {
    const unsigned int n = dof_handler.n_dofs();
    macroUnitSolutions.assign(nMacroDofs, Vector<double>(n));

    // Six unit-macroscopic-strain solves. K U = F_theta - L Ebar, so the fluctuation belonging to
    // Ebar = I^(k) is -Y_k; the minus sign is carried at the point of use rather than folded in,
    // to keep Y meaning K^-1 L.
    for (unsigned int k = 0; k < nMacroDofs; ++k)
    {
      solveLinearSystem(macroCouplingColumns[k], macroUnitSolutions[k]);
      constraints.distribute(macroUnitSolutions[k]);
    }

    // Cbar column by column from the averaged stress. Column k is <sigma> for the state
    // (Ebar = I^(k), u~ = -Y_k) with NO thermal strain. I^(k) is the tensor whose Kelvin vector is
    // the k-th unit vector -- for the shear entries that is the tensor with 1/sqrt(2) off-diagonal,
    // which to_tensor() produces correctly. This is the one place the sqrt(2) convention could be
    // got wrong by hand, so it is not done by hand.
    FullMatrix<double> Cbar(nMacroDofs, nMacroDofs);
    for (unsigned int k = 0; k < nMacroDofs; ++k)
    {
      Vector<double> unitKelvin(nMacroDofs);
      unitKelvin[k] = 1.0;
      SymmetricTensor<2, dim> unitStrain;
      Physics::Notation::Kelvin::to_tensor(unitKelvin, unitStrain);

      Vector<double> fluctuation(macroUnitSolutions[k]);
      fluctuation *= -1.0;

      const SymmetricTensor<2, dim> columnStress =
        computeAverageStress(curTemperature, fluctuation, unitStrain, /*includeThermal=*/false);
      const Vector<double> columnKelvin = Physics::Notation::Kelvin::to_vector(columnStress);
      for (unsigned int i = 0; i < nMacroDofs; ++i)
        Cbar(i, k) = columnKelvin[i];
    }
    homogenizedStiffness = Cbar;

    // Cbar must come out symmetric -- that follows from Hill-Mandel (section 4), NOT from how it
    // was built, so it is a genuinely independent check on periodicity, the pin, the eigenstrain
    // path and the six solves all at once. Nothing above enforces it.
    double asymmetry = 0.0, magnitude = 0.0;
    for (unsigned int i = 0; i < nMacroDofs; ++i)
      for (unsigned int j = 0; j < nMacroDofs; ++j)
      {
        asymmetry = std::max(asymmetry, std::abs(Cbar(i, j) - Cbar(j, i)));
        magnitude = std::max(magnitude, std::abs(Cbar(i, j)));
      }
    lastCbarAsymmetry = (magnitude > 0.0) ? asymmetry / magnitude : 0.0;
    if (lastCbarAsymmetry > 1e-6)
      throw std::runtime_error(
        "homogenized stiffness Cbar is not symmetric (relative asymmetry " +
        std::to_string(lastCbarAsymmetry) + "). Hill-Mandel guarantees symmetry, so this means "
        "the periodic constraints, the pin, or the L assembly is wrong -- not that the material "
        "is unusual.");

    homogenizedStiffnessInverse = Cbar;
    homogenizedStiffnessInverse.gauss_jordan();
    ++cbarRefreshCount;
  }

  template <int dim>
  bool ElasticProblem<dim>::iterateMacroscopicStrain(double curTemperature)
  {
    const double tolerance =
      prm.get_double({"SolverParameters"}, "MacroscopicStressTolerance");
    const unsigned int maxIterations =
      prm.get_integer({"SolverParameters"}, "MaxMacroscopicIterations");

    Vector<double> Ebar = Physics::Notation::Kelvin::to_vector(macroscopicStrain);
    Vector<double> rhs(dof_handler.n_dofs()), CbarEbar(nMacroDofs), correction(nMacroDofs);

    for (unsigned int m = 1; m <= maxIterations; ++m)
    {
      Physics::Notation::Kelvin::to_tensor(Ebar, macroscopicStrain);

      // Solve K U = F_theta - L Ebar against the CURRENT matrix. This is the one solve per pass.
      rhs = system_rhs;
      for (unsigned int k = 0; k < nMacroDofs; ++k)
        rhs.add(-Ebar[k], macroCouplingColumns[k]);
      solveLinearSystem(rhs, solDisplacement);
      constraints.distribute(solDisplacement);

      // Exact macroscopic residual: quadrature only, no solve.
      const SymmetricTensor<2, dim> meanStress = computeAverageStress(curTemperature);
      const Vector<double> residual = Physics::Notation::Kelvin::to_vector(meanStress);

      // Scale against ||Cbar:Ebar||, the macroscopic stress the current strain would produce.
      // The floor matters only on the very first pass, where Ebar is still zero.
      homogenizedStiffness.vmult(CbarEbar, Ebar);
      const double scale = std::max(CbarEbar.l2_norm(), 1.0);
      if (residual.l2_norm() < tolerance * scale)
      {
        lastMacroIterations = m;
        return true;
      }

      homogenizedStiffnessInverse.vmult(correction, residual);
      Ebar -= correction;
    }
    lastMacroIterations = maxIterations;
    return false;
  }

  // The 7-solve of homogenization.md section 6, or its quasi-Newton replacement.
  //
  // With damage frozen (which the staggered scheme guarantees within one pass), the problem is
  // LINEAR in (u~, Ebar), so the direct path is exact rather than a linearization:
  //
  //     K U + L Ebar = F_theta          n equations
  //     <sigma> = 0                     6 equations, the natural condition conjugate to Ebar
  //
  // Eliminating U costs one solve for the thermal load and six for L's columns -- all against the
  // SAME matrix, hence one factorization (see prepareLinearSolver / LinearSolverType).
  template <int dim>
  void ElasticProblem<dim>::solve_with_zero_mean_stress(double curTemperature)
  {
    prepareLinearSolver();

    if (prm.get_bool({"SolverParameters"}, "FrozenCbarQuasiNewton"))
    {
      // Rebuild Cbar once if the frozen-Jacobian iteration reaches its pass limit.
      for (unsigned int attempt = 0; attempt < 2; ++attempt)
      {
        if (cbarNeedsRefresh)
        {
          refreshHomogenizedStiffness(curTemperature);
          cbarNeedsRefresh = false;
        }
        if (iterateMacroscopicStrain(curTemperature))
          return;
        cbarNeedsRefresh = true;
      }
      throw std::runtime_error(
        "the Ebar iteration failed to converge even with a freshly rebuilt Cbar. That is not "
        "staleness -- suspect the macroscopic residual or the coupling columns.");
    }

    // ---- exact path: rebuild Cbar every pass, then condense directly -------------------
    refreshHomogenizedStiffness(curTemperature);

    Vector<double> U0(dof_handler.n_dofs());
    solveLinearSystem(system_rhs, U0);
    constraints.distribute(U0);

    const SymmetricTensor<2, dim> sigmaTheta =
      computeAverageStress(curTemperature, U0, SymmetricTensor<2, dim>(), /*includeThermal=*/true);
    Vector<double> rhs6 = Physics::Notation::Kelvin::to_vector(sigmaTheta);
    rhs6 *= -1.0;

    Vector<double> Ebar(nMacroDofs);
    homogenizedStiffnessInverse.vmult(Ebar, rhs6);
    Physics::Notation::Kelvin::to_tensor(Ebar, macroscopicStrain);

    solDisplacement = U0;
    for (unsigned int k = 0; k < nMacroDofs; ++k)
      solDisplacement.add(-Ebar[k], macroUnitSolutions[k]);
    lastMacroIterations = 1;
  }

  template class ElasticProblem<3>;
}

#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  void ElasticProblem<dim>::assemble_system(double curTemperature)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    refreshGrainStiffnessCache(curTemperature);
    applyBoundaryConstraints();

    // AffineConstraints::distribute_local_to_global() ADDS local contributions into
    // system_matrix/system_rhs rather than overwriting them. assemble_system() is called
    // repeatedly (once per displacement-damage iteration, and again inside
    // checkConvergenceCriteria), so without resetting here every call keeps accumulating
    // on top of all previous calls -- system_matrix/system_rhs would grow unboundedly
    // over a temperature step instead of reflecting only the current state.
    system_matrix = 0;
    system_rhs    = 0;
    if (assembleMacroCoupling)
      for (auto &column : macroCouplingColumns)
        column = 0;

    if (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream"))
      assemble_cells_workstream(curTemperature);
    else
      assemble_cells_serial(curTemperature);
  }

  // Serial cell loop and deterministic reference for the WorkStream implementation.
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
    FullMatrix<double> cell_L(dofs_per_cell, nMacroDofs);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Cache values that are independent of test/trial function indices.
    std::vector<SymmetricTensor<4,dim>> globalElasticTensor_q(n_q_points);
    std::vector<double> JxW_q(n_q_points);
    std::vector<std::vector<SymmetricTensor<2,dim>>> shapeGradSymm(
      dofs_per_cell, std::vector<SymmetricTensor<2,dim>>(n_q_points));
    std::vector<SymmetricTensor<2,dim>> globalCte_q(n_q_points);

    // The RVE cools uniformly, so the temperature rise is ONE scalar for the whole domain, not
    // a field. It is negative on cooling.
    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const double deltaT = curTemperature - T0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      // The grain cache already contains tensors rotated to the global frame.

      cell_matrix = 0;
      cell_rhs    = 0;

      // Apply rank-one damage updates to global tensors cached per grain.
      const unsigned int grain = cell->material_id() - 1;
      const SymmetricTensor<4,dim> &C0 = grainStiffnessZeroDamage[grain];
      const SymmetricTensor<4,dim> &nnnn = grainDamageDirection[grain];
      const SymmetricTensor<2,dim> &a0 = grainCteZeroDamage[grain];
      const SymmetricTensor<2,dim> &nn = grainCaxisOuter[grain];
      for (const unsigned int q : fe_values.quadrature_point_indices())
      {
        JxW_q[q] = fe_values.JxW(q);
        const double w = cell_damage[q]->damage;
        globalElasticTensor_q[q] = C0 - (w * grainCacheC33) * nnnn;
        globalCte_q[q]           = a0 - (w * grainCacheAlpha33) * nn;
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

        // L's k-th column, only when the macroscopic strain is an unknown. Accumulated as a
        // TENSOR over the quadrature points and converted to Kelvin once, after the q loop.
        if (assembleMacroCoupling)
        {
          SymmetricTensor<2,dim> coupling;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            coupling += (shapeGradSymm[i][q] * globalElasticTensor_q[q]) * JxW_q[q];
          const Vector<double> kelvin = Physics::Notation::Kelvin::to_vector(coupling);
          for (unsigned int k = 0; k < nMacroDofs; ++k)
            cell_L(i, k) = kelvin[k];
        }

        // Thermal mismatch load; this model has no body force.
        for (const unsigned int q : fe_values.quadrature_point_indices())
          cell_rhs(i) +=
            ((shapeGradSymm[i][q] * (globalElasticTensor_q[q] * globalCte_q[q])) * deltaT)
            * JxW_q[q];
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      if (assembleMacroCoupling)
        distributeMacroCouplingColumns(cell_L, local_dof_indices);
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
      // The grain cache already contains tensors rotated to the global frame.

      copy.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      copy.cell_rhs.reinit(dofs_per_cell);
      copy.local_dof_indices.resize(dofs_per_cell);
      copy.cell_matrix = 0;
      copy.cell_rhs    = 0;
      if (assembleMacroCoupling)
      {
        copy.cell_L.reinit(dofs_per_cell, nMacroDofs);
        copy.cell_L = 0;
      }

      // Rank-one damage updates against the per-grain cache -- see the serial path.
      const unsigned int grain = cell->material_id() - 1;
      const SymmetricTensor<4,dim> &C0 = grainStiffnessZeroDamage[grain];
      const SymmetricTensor<4,dim> &nnnn = grainDamageDirection[grain];
      const SymmetricTensor<2,dim> &a0 = grainCteZeroDamage[grain];
      const SymmetricTensor<2,dim> &nn = grainCaxisOuter[grain];
      for (const unsigned int q : fe_values.quadrature_point_indices())
      {
        scratch.JxW_q[q] = fe_values.JxW(q);
        const double w = cell_damage[q]->damage;
        scratch.globalElasticTensor_q[q] = C0 - (w * grainCacheC33) * nnnn;
        scratch.globalCte_q[q]           = a0 - (w * grainCacheAlpha33) * nn;
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

        // L's k-th column -- see the serial path for the reasoning.
        if (assembleMacroCoupling)
        {
          scratch.macroCoupling_i[i] = 0;
          for (const unsigned int q : fe_values.quadrature_point_indices())
            scratch.macroCoupling_i[i] +=
              (scratch.shapeGradSymm[i][q] * scratch.globalElasticTensor_q[q]) * scratch.JxW_q[q];
          const Vector<double> kelvin =
            Physics::Notation::Kelvin::to_vector(scratch.macroCoupling_i[i]);
          for (unsigned int k = 0; k < nMacroDofs; ++k)
            copy.cell_L(i, k) = kelvin[k];
        }

        // Thermal load only -- no body force in this model (see the serial path).
        for (const unsigned int q : fe_values.quadrature_point_indices())
          copy.cell_rhs(i) +=
            ((scratch.shapeGradSymm[i][q] * (scratch.globalElasticTensor_q[q] * scratch.globalCte_q[q]))
             * deltaT) * scratch.JxW_q[q];
      }
      cell->get_dof_indices(copy.local_dof_indices);
    };

    // Runs serialized by WorkStream, which is what makes writing the shared macroCouplingColumns
    // safe here without any locking of our own.
    auto copier = [&](const AssemblyScratch::Copy &copy)
    {
      constraints.distribute_local_to_global(copy.cell_matrix, copy.cell_rhs,
                                             copy.local_dof_indices, system_matrix, system_rhs);
      if (assembleMacroCoupling)
        distributeMacroCouplingColumns(copy.cell_L, copy.local_dof_indices);
    };

    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), worker, copier,
                    AssemblyScratch::Scratch<dim>(fe, quadrature_formula, flags),
                    AssemblyScratch::Copy());
  }

  template class ElasticProblem<3>;
}

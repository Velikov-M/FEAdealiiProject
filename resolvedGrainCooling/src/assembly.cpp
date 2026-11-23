#include "resolved_grain/elastic_problem.h"
#include "resolved_grain/cell_assembly.h"

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  void ElasticProblem<dim>::assemble_system(double curTemperature)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    refreshGrainStiffnessCache(curTemperature);
    applyBoundaryConstraints();

    system_matrix = 0;
    system_rhs = 0;
    if (assembleMacroCoupling)
      for (auto &column : macroCouplingColumns)
        column = 0;

    if (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream"))
      assemble_cells_workstream(curTemperature);
    else
      assemble_cells_serial(curTemperature);
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_cells_serial(double curTemperature)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(fe, quadrature_formula,
      update_values | update_gradients | update_quadrature_points | update_JxW_values);

    AssemblyScratch::Copy copy;
    copy.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    copy.cell_rhs.reinit(dofs_per_cell);
    copy.cell_L.reinit(dofs_per_cell, nMacroDofs);
    copy.local_dof_indices.resize(dofs_per_cell);

    std::vector<SymmetricTensor<4, dim>> globalElasticTensor_q(n_q_points);
    std::vector<SymmetricTensor<2, dim>> globalCte_q(n_q_points);
    std::vector<std::vector<SymmetricTensor<2, dim>>> shapeGradSymm(
      dofs_per_cell, std::vector<SymmetricTensor<2, dim>>(n_q_points));
    std::vector<SymmetricTensor<2, dim>> macroCoupling(dofs_per_cell);

    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const std::vector<double> temperatureChange(n_q_points, curTemperature - T0);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        auto cell_damage = damageInQuadraturePoints.get_data(cell);
        const unsigned int grain = cell->material_id() - 1;
        const SymmetricTensor<4, dim> &C0 = grainStiffnessZeroDamage[grain];
        const SymmetricTensor<4, dim> &nnnn = grainDamageDirection[grain];
        const SymmetricTensor<2, dim> &a0 = grainCteZeroDamage[grain];
        const SymmetricTensor<2, dim> &nn = grainCaxisOuter[grain];
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const double w = cell_damage[q]->damage;
            globalElasticTensor_q[q] = C0 - (w * grainCacheC33) * nnnn;
            globalCte_q[q] = a0 - (w * grainCacheAlpha33) * nn;
          }

        assembleCell<dim>(fe_values, displacement, globalElasticTensor_q, globalCte_q,
                     temperatureChange, nullptr, assembleMacroCoupling,
                     shapeGradSymm, macroCoupling, copy);
        cell->get_dof_indices(copy.local_dof_indices);
        constraints.distribute_local_to_global(copy.cell_matrix, copy.cell_rhs,
                                               copy.local_dof_indices, system_matrix, system_rhs);
        if (assembleMacroCoupling)
          distributeMacroCouplingColumns(copy.cell_L, copy.local_dof_indices);
      }
  }

  template <int dim>
  void ElasticProblem<dim>::assemble_cells_workstream(double curTemperature)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const UpdateFlags flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;

    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const std::vector<double> temperatureChange(quadrature_formula.size(), curTemperature - T0);

    auto worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                      AssemblyScratch::Scratch<dim> &scratch,
                      AssemblyScratch::Copy &copy)
    {
      FEValues<dim> &fe_values = scratch.fe_values;
      fe_values.reinit(cell);
      auto cell_damage = damageInQuadraturePoints.get_data(cell);

      copy.cell_matrix.reinit(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
      copy.cell_rhs.reinit(fe.n_dofs_per_cell());
      copy.local_dof_indices.resize(fe.n_dofs_per_cell());
      if (assembleMacroCoupling)
        copy.cell_L.reinit(fe.n_dofs_per_cell(), nMacroDofs);

      const unsigned int grain = cell->material_id() - 1;
      const SymmetricTensor<4, dim> &C0 = grainStiffnessZeroDamage[grain];
      const SymmetricTensor<4, dim> &nnnn = grainDamageDirection[grain];
      const SymmetricTensor<2, dim> &a0 = grainCteZeroDamage[grain];
      const SymmetricTensor<2, dim> &nn = grainCaxisOuter[grain];
      for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          const double w = cell_damage[q]->damage;
          scratch.globalElasticTensor_q[q] = C0 - (w * grainCacheC33) * nnnn;
          scratch.globalCte_q[q] = a0 - (w * grainCacheAlpha33) * nn;
        }

      assembleCell<dim>(fe_values, displacement, scratch.globalElasticTensor_q, scratch.globalCte_q,
                   temperatureChange, nullptr, assembleMacroCoupling,
                   scratch.shapeGradSymm, scratch.macroCoupling_i, copy);
      cell->get_dof_indices(copy.local_dof_indices);
    };

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

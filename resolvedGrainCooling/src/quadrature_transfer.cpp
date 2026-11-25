#include "resolved_grain/quadrature_transfer.h"

#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace StepCooling
{
  template <int dim>
  DamageQuadratureTransfer<dim>::DamageQuadratureTransfer(Triangulation<dim> &triangulation,
                                                           const Storage &source)
    : transferFe(QGauss<1>(2))
    , transferDofs(triangulation)
    , solutionTransfer(transferDofs)
  {
    transferDofs.distribute_dofs(transferFe);
    packedFields.assign(2, Vector<double>(transferDofs.n_dofs()));

    std::vector<types::global_dof_index> localDofs(transferFe.n_dofs_per_cell());
    for (const auto &cell : transferDofs.active_cell_iterators())
      {
        const auto data = source.get_data(cell);
        if (data.size() != transferFe.n_dofs_per_cell())
          throw std::runtime_error("Damage transfer requires two Gauss points per direction");
        cell->get_dof_indices(localDofs);
        for (unsigned int q = 0; q < data.size(); ++q)
          {
            if (!std::isfinite(data[q]->damage) || !std::isfinite(data[q]->old_damage))
              throw std::runtime_error("Cannot transfer non-finite quadrature damage");
            packedFields[0][localDofs[q]] = data[q]->damage;
            packedFields[1][localDofs[q]] = data[q]->old_damage;
          }
      }

    solutionTransfer.prepare_for_coarsening_and_refinement(packedFields);
  }

  template <int dim>
  void DamageQuadratureTransfer<dim>::interpolate(Storage &destination)
  {
    transferDofs.distribute_dofs(transferFe);
    std::vector<Vector<double>> transferred(2);
    for (auto &field : transferred)
      field.reinit(transferDofs.n_dofs());
    solutionTransfer.interpolate(transferred);

    const QGauss<dim> quadrature(2);
    FEValues<dim> feValues(transferFe, quadrature, update_values);
    std::vector<double> currentValues(quadrature.size());
    std::vector<double> oldValues(quadrature.size());
    for (const auto &cell : transferDofs.active_cell_iterators())
      {
        auto data = destination.get_data(cell);
        if (data.size() != quadrature.size())
          throw std::runtime_error("Adapted damage storage has the wrong quadrature size");
        feValues.reinit(cell);
        feValues.get_function_values(transferred[0], currentValues);
        feValues.get_function_values(transferred[1], oldValues);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            if (!std::isfinite(currentValues[q]) || !std::isfinite(oldValues[q]))
              throw std::runtime_error(
                "Quadrature damage transfer produced a non-finite value at cell " +
                std::to_string(cell->active_cell_index()) + ", point " + std::to_string(q) +
                ": current=" + std::to_string(currentValues[q]) +
                ", old=" + std::to_string(oldValues[q]));
            data[q]->damage = std::clamp(currentValues[q], 0.0, 0.999);
            data[q]->old_damage = std::clamp(oldValues[q], 0.0, data[q]->damage);
          }
      }
  }

  template class DamageQuadratureTransfer<3>;
}

#include "resolved_grain/cell_assembly.h"

#include <deal.II/physics/notation.h>

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  void assembleCell(
    FEValues<dim> &fe_values,
    const FEValuesExtractors::Vector &displacement,
    const std::vector<SymmetricTensor<4, dim>> &elasticity,
    const std::vector<SymmetricTensor<2, dim>> &cte,
    const std::vector<double> &temperatureChange,
    const std::vector<Tensor<1, dim>> *bodyForce,
    const bool assembleMacroCoupling,
    std::vector<std::vector<SymmetricTensor<2, dim>>> &shapeGradSymm,
    std::vector<SymmetricTensor<2, dim>> &macroCoupling,
    AssemblyScratch::Copy &copy)
  {
    copy.cell_matrix = 0;
    copy.cell_rhs = 0;
    if (assembleMacroCoupling)
      copy.cell_L = 0;

    for (const unsigned int i : fe_values.dof_indices())
      for (const unsigned int q : fe_values.quadrature_point_indices())
        shapeGradSymm[i][q] = symmetrize(fe_values[displacement].gradient(i, q));

    for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            copy.cell_matrix(i, j) +=
              (shapeGradSymm[i][q] * (elasticity[q] * shapeGradSymm[j][q])) * fe_values.JxW(q);

        if (assembleMacroCoupling)
          {
            macroCoupling[i] = 0;
            for (const unsigned int q : fe_values.quadrature_point_indices())
              macroCoupling[i] +=
                (shapeGradSymm[i][q] * elasticity[q]) * fe_values.JxW(q);
            const Vector<double> kelvin = Physics::Notation::Kelvin::to_vector(macroCoupling[i]);
            for (unsigned int k = 0; k < kelvin.size(); ++k)
              copy.cell_L(i, k) = kelvin[k];
          }

        for (const unsigned int q : fe_values.quadrature_point_indices())
          copy.cell_rhs(i) +=
            ((shapeGradSymm[i][q] * (elasticity[q] * cte[q])) * temperatureChange[q])
            * fe_values.JxW(q);

        if (bodyForce != nullptr)
          for (const unsigned int q : fe_values.quadrature_point_indices())
            copy.cell_rhs(i) += fe_values[displacement].value(i, q) * bodyForce->at(q)
                                * fe_values.JxW(q);
      }
  }

  template void assembleCell<3>(
    FEValues<3> &, const FEValuesExtractors::Vector &,
    const std::vector<SymmetricTensor<4, 3>> &,
    const std::vector<SymmetricTensor<2, 3>> &,
    const std::vector<double> &, const std::vector<Tensor<1, 3>> *, bool,
    std::vector<std::vector<SymmetricTensor<2, 3>>> &,
    std::vector<SymmetricTensor<2, 3>> &, AssemblyScratch::Copy &);
}

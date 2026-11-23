#ifndef RESOLVED_GRAIN_CELL_ASSEMBLY_H
#define RESOLVED_GRAIN_CELL_ASSEMBLY_H

#include "resolved_grain/assembly_data.h"

#include <deal.II/fe/fe_values.h>

#include <vector>

namespace StepCooling
{
  /** Assemble one mechanical cell at fixed quadrature-point constitutive data.
   *
   * The weak form is int eps(phi_i):C:(eps(phi_j)-cte*deltaT) = int phi_i.bodyForce.
   * Temperature and constitutive tensors may vary by quadrature point. Production cooling uses
   * uniform deltaT and a null body-force pointer; manufactured-solution tests use all inputs.
   *
   * @param[in] fe_values Reinitialized cell values with gradients, values, and JxW available.
   * @param[in] displacement Vector extractor for the displacement components.
   * @param[in] elasticity Quadrature-point global stiffness tensors in Pa.
   * @param[in] cte Quadrature-point global secant CTE tensors in 1/K.
   * @param[in] temperatureChange Quadrature-point T-T0 values in K.
   * @param[in] bodyForce Optional quadrature-point body forces in N/m^3; null means zero.
   * @param[in] assembleMacroCoupling Whether to assemble six Kelvin macroscopic columns.
   * @param[out] shapeGradSymm Reusable symmetric displacement gradients indexed by DoF and point.
   * @param[out] macroCoupling Reusable rank-two coupling storage indexed by DoF.
   * @param[out] copy Cell matrix, thermal/body-force RHS, and optional coupling columns.
   */
  template <int dim>
  void assembleCell(
    FEValues<dim> &fe_values,
    const FEValuesExtractors::Vector &displacement,
    const std::vector<SymmetricTensor<4, dim>> &elasticity,
    const std::vector<SymmetricTensor<2, dim>> &cte,
    const std::vector<double> &temperatureChange,
    const std::vector<Tensor<1, dim>> *bodyForce,
    bool assembleMacroCoupling,
    std::vector<std::vector<SymmetricTensor<2, dim>>> &shapeGradSymm,
    std::vector<SymmetricTensor<2, dim>> &macroCoupling,
    AssemblyScratch::Copy &copy);
}

#endif

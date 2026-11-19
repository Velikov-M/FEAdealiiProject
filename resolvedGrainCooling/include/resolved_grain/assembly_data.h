#ifndef RESOLVED_GRAIN_ASSEMBLY_DATA_H
#define RESOLVED_GRAIN_ASSEMBLY_DATA_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <vector>

namespace StepCooling
{
  using namespace dealii;

  /** Per-thread assembly scratch and per-cell copy data for WorkStream.
   *
   * FEValues is reconstructed by each scratch copy because it is not copyable. Only the copier
   * writes shared global containers; workers own their scratch and cell-local results.
   */
  namespace AssemblyScratch
  {
    template <int dim>
    struct Scratch
    {
      Scratch(const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
              const UpdateFlags flags)
        : fe_values(fe, quadrature, flags)
        , globalElasticTensor_q(quadrature.size())
        , JxW_q(quadrature.size())
        , shapeGradSymm(fe.n_dofs_per_cell(),
                        std::vector<SymmetricTensor<2, dim>>(quadrature.size()))
        , macroCoupling_i(fe.n_dofs_per_cell())
        , globalCte_q(quadrature.size())
      {}

      Scratch(const Scratch &s)
        : fe_values(s.fe_values.get_fe(), s.fe_values.get_quadrature(),
                    s.fe_values.get_update_flags())
        , globalElasticTensor_q(s.globalElasticTensor_q.size())
        , JxW_q(s.JxW_q.size())
        , shapeGradSymm(s.shapeGradSymm)
        , macroCoupling_i(s.macroCoupling_i.size())
        , globalCte_q(s.globalCte_q.size())
      {}

      FEValues<dim> fe_values;
      std::vector<SymmetricTensor<4, dim>> globalElasticTensor_q;
      std::vector<double> JxW_q;
      std::vector<std::vector<SymmetricTensor<2, dim>>> shapeGradSymm;
      // Integral eps(phi_i):C per shape function, converted to Kelvin notation once per cell.
      std::vector<SymmetricTensor<2, dim>> macroCoupling_i;
      // Global CTE varies with quadrature-point damage. Temperature is uniform over the RVE.
      std::vector<SymmetricTensor<2, dim>> globalCte_q;
    };

    struct Copy
    {
      FullMatrix<double> cell_matrix;
      Vector<double> cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
      // dofs_per_cell x 6: column k is this cell's contribution to L's k-th Kelvin column.
      // Left empty (and untouched) unless the periodic formulation is active.
      FullMatrix<double> cell_L;
    };
  } // namespace AssemblyScratch

  /** Per-thread damage-update scratch; cell damage itself lives in CellDataStorage. */
  namespace DamageScratch
  {
    template <int dim>
    struct Scratch
    {
      Scratch(const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
              const UpdateFlags flags)
        : fe_values(fe, quadrature, flags)
        , global_displacement_gradients(quadrature.size())
      {}

      Scratch(const Scratch &s)
        : fe_values(s.fe_values.get_fe(), s.fe_values.get_quadrature(),
                    s.fe_values.get_update_flags())
        , global_displacement_gradients(s.global_displacement_gradients.size())
      {}

      FEValues<dim> fe_values;
      std::vector<Tensor<2, dim>> global_displacement_gradients;
    };

    // Empty: the damage update writes only into the current cell's own quadrature-point data,
    // which no other cell touches, so there is nothing to hand to a serialized copier.
    // WorkStream still requires the type to exist.
    struct Copy
    {};
  } // namespace DamageScratch
}

#endif

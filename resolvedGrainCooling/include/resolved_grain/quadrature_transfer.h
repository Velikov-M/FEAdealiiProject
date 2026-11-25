#ifndef RESOLVED_GRAIN_QUADRATURE_TRANSFER_H
#define RESOLVED_GRAIN_QUADRATURE_TRANSFER_H

#include "resolved_grain/quadrature_data.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/solution_transfer.h>

#include <vector>

namespace StepCooling
{
  using namespace dealii;

  /** Transfer damage at two-point Gauss quadrature through one mesh adaptation.
   *
   * Each scalar history is represented by a cell-wise trilinear discontinuous field whose
   * support points are the two-point Gauss points in each direction. Refinement therefore
   * interpolates within the parent cell without coupling neighboring materials. Coarsening uses
   * deal.II's finite-element restriction and L2 projection. Both fields are clamped to
   * [0, 0.999], and old_damage is limited by damage after transfer.
   *
   * Construct after Triangulation::prepare_coarsening_and_refinement(), execute the mesh change,
   * distribute any other production DoFs, initialize DamageQData on every new active cell, and
   * then call interpolate(). This object must remain alive until interpolate() completes.
   */
  template <int dim>
  class DamageQuadratureTransfer
  {
  public:
    using CellIterator = typename Triangulation<dim>::cell_iterator;
    using Storage = CellDataStorage<CellIterator, DamageQData>;

    /** Pack current and old damage and register them for the pending mesh change.
     * @param[in,out] triangulation Mesh with refinement/coarsening flags already prepared.
     * @param[in] source Quadrature damage initialized at eight Gauss points on every active cell.
     */
    DamageQuadratureTransfer(Triangulation<dim> &triangulation, const Storage &source);

    /** Redistribute and evaluate transferred damage on the adapted mesh.
     * @param[in,out] destination Initialized quadrature storage on every new active cell.
     */
    void interpolate(Storage &destination);

  private:
    FE_DGQArbitraryNodes<dim> transferFe;
    DoFHandler<dim> transferDofs;
    SolutionTransfer<dim, Vector<double>> solutionTransfer;
    std::vector<Vector<double>> packedFields;
  };
}

#endif

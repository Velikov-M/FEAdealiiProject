#include "resolved_grain/quadrature_transfer.h"

#include <gtest/gtest.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>

#include <algorithm>
#include <cmath>

using namespace dealii;
using StepCooling::DamageQData;
using StepCooling::DamageQuadratureTransfer;

namespace
{
  using Storage = CellDataStorage<Triangulation<3>::cell_iterator, DamageQData>;

  void initialize(Triangulation<3> &triangulation, Storage &storage)
  {
    storage.clear();
    storage.initialize(triangulation.begin_active(), triangulation.end(), 8);
  }

  double analyticDamage(const Point<3> &point, const unsigned int material)
  {
    return 0.08 * material + 0.03 * point[0] + 0.02 * point[1] - 0.01 * point[2]
           + 0.005 * point[0] * point[1];
  }

  void fillAnalytic(Triangulation<3> &triangulation, Storage &storage)
  {
    const FE_DGQArbitraryNodes<3> fe(QGauss<1>(2));
    const QGauss<3> quadrature(2);
    FEValues<3> feValues(fe, quadrature, update_quadrature_points);
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        feValues.reinit(cell);
        const auto data = storage.get_data(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            data[q]->damage = analyticDamage(feValues.quadrature_point(q), cell->material_id());
            data[q]->old_damage = data[q]->damage - 0.02;
          }
      }
  }

  void adapt(Triangulation<3> &triangulation, Storage &storage)
  {
    triangulation.prepare_coarsening_and_refinement();
    DamageQuadratureTransfer<3> transfer(triangulation, storage);
    triangulation.execute_coarsening_and_refinement();
    initialize(triangulation, storage);
    transfer.interpolate(storage);
  }

  void expectAnalytic(Triangulation<3> &triangulation, const Storage &storage)
  {
    const FE_DGQArbitraryNodes<3> fe(QGauss<1>(2));
    const QGauss<3> quadrature(2);
    FEValues<3> feValues(fe, quadrature, update_quadrature_points);
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        feValues.reinit(cell);
        const auto data = storage.get_data(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const double expected = analyticDamage(feValues.quadrature_point(q), cell->material_id());
            EXPECT_NEAR(data[q]->damage, expected, 2e-14);
            EXPECT_NEAR(data[q]->old_damage, expected - 0.02, 2e-14);
          }
      }
  }
}

TEST(QuadratureTransfer, RefinesTrilinearDamageWithoutMixingDiscontinuousMaterials)
{
  Triangulation<3> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 2, 0.0, 1.0);
  for (const auto &cell : triangulation.active_cell_iterators())
    cell->set_material_id(cell->center()[0] < 0.5 ? 1 : 2);
  Storage storage;
  initialize(triangulation, storage);
  fillAnalytic(triangulation, storage);

  triangulation.begin_active()->set_refine_flag();
  adapt(triangulation, storage);

  EXPECT_EQ(triangulation.n_active_cells(), 15);
  expectAnalytic(triangulation, storage);
}

TEST(QuadratureTransfer, CoarseningRecoversRepresentableTrilinearHistory)
{
  Triangulation<3> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.begin_active()->set_material_id(1);
  triangulation.refine_global(1);
  Storage storage;
  initialize(triangulation, storage);
  fillAnalytic(triangulation, storage);

  for (const auto &cell : triangulation.active_cell_iterators())
    cell->set_coarsen_flag();
  adapt(triangulation, storage);

  ASSERT_EQ(triangulation.n_active_cells(), 1);
  expectAnalytic(triangulation, storage);
}

TEST(QuadratureTransfer, CoarseningEnforcesDamageBoundsAndHistoryOrdering)
{
  Triangulation<3> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  Storage storage;
  initialize(triangulation, storage);
  unsigned int index = 0;
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      const auto data = storage.get_data(cell);
      for (unsigned int q = 0; q < data.size(); ++q)
        {
          data[q]->damage = ((index + q) % 3 == 0) ? 0.95 : 0.05;
          data[q]->old_damage = data[q]->damage - 0.04;
        }
      ++index;
    }

  for (const auto &cell : triangulation.active_cell_iterators())
    cell->set_coarsen_flag();
  adapt(triangulation, storage);

  ASSERT_EQ(triangulation.n_active_cells(), 1);
  const auto data = storage.get_data(triangulation.begin_active());
  for (const auto &point : data)
    {
      EXPECT_GE(point->damage, 0.0);
      EXPECT_LE(point->damage, 0.999);
      EXPECT_GE(point->old_damage, 0.0);
      EXPECT_LE(point->old_damage, point->damage);
    }
}

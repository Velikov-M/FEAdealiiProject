#ifndef RESOLVED_GRAIN_QUADRATURE_DATA_H
#define RESOLVED_GRAIN_QUADRATURE_DATA_H

#include <deal.II/base/quadrature_point_data.h>

namespace StepCooling
{
  using namespace dealii;

  /** Dimensionless damage state owned by one quadrature point.
   *
   * @p damage is the current iterate. @p old_damage is the converged value at the start of the
   * temperature step and anchors the backward-Euler residual and irreversibility bound.
   */
  struct DamageQData : public TransferableQuadraturePointData
  {
    double damage = 0.0;
    double old_damage = 0.0;

    unsigned int number_of_values() const override
    {
      return 2;
    }

    void pack_values(std::vector<double> &values) const override
    {
      AssertDimension(values.size(), 2);
      values[0] = damage;
      values[1] = old_damage;
    }

    void unpack_values(const std::vector<double> &values) override
    {
      AssertDimension(values.size(), 2);
      damage = values[0];
      old_damage = values[1];
    }
  };
}

#endif

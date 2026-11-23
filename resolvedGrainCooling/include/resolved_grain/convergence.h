#ifndef RESOLVED_GRAIN_CONVERGENCE_H
#define RESOLVED_GRAIN_CONVERGENCE_H

#include <cmath>
#include <limits>

namespace StepCooling
{
  /** Convert an absolute increment to a relative convergence measure safely at zero scale.
   * @param[in] increment Nonnegative absolute increment norm.
   * @param[in] scale Nonnegative norm of the reference state.
   * @return Zero for a zero increment and scale, infinity for nonzero increment at zero scale,
   *         and increment/scale otherwise. Non-finite inputs produce infinity.
   */
  inline double scaledConvergenceMeasure(double increment, double scale)
  {
    if (!std::isfinite(increment) || !std::isfinite(scale) || increment < 0.0 || scale < 0.0)
      return std::numeric_limits<double>::infinity();
    if (scale == 0.0)
      return increment == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return increment / scale;
  }
}

#endif

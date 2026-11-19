#ifndef RESOLVED_GRAIN_ORIENTATIONS_H
#define RESOLVED_GRAIN_ORIENTATIONS_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/numbers.h>
#include <algorithm>
#include <cmath>
#include <random>

namespace StepCooling
{
  using namespace dealii;

  /** Build a crystal frame from a unit c-axis using Duff et al.'s stable orthonormal basis.
   *
   * Only the c-axis direction is physically relevant for transverse isotropy. The remaining
   * axes are deterministic because rotation about c is a material symmetry.
   *
   * @return R mapping GLOBAL to LOCAL coordinates: v_local = R * v_global. Local basis vectors
   *         are matrix rows, and row 2 is the c-axis expressed in global coordinates.
   */
  template <int dim>
  Tensor<2, dim> orientationFromCAxis(const Tensor<1, dim> &c)
  {
    static_assert(dim == 3, "crystallite orientations are only defined in 3D");
    const double sign = std::copysign(1.0, c[2]);
    const double a = -1.0 / (sign + c[2]);
    const double b = c[0] * c[1] * a;

    Tensor<1, dim> e1, e2;
    e1[0] = 1.0 + sign * c[0] * c[0] * a;  e1[1] = sign * b;                 e1[2] = -sign * c[0];
    e2[0] = b;                             e2[1] = sign + c[1] * c[1] * a;   e2[2] = -c[1];

    Tensor<2, dim> R;
    for (unsigned int d = 0; d < dim; ++d)
    {
      R[0][d] = e1[d];
      R[1][d] = e2[d];
      R[2][d] = c[d];
    }
    return R;
  }

  /** Sample a c-axis uniformly on the sphere and return its global-to-local frame.
   *
   * Uniform cos(theta), rather than uniform theta, gives the required spherical area measure.
   */
  template <int dim>
  Tensor<2, dim> randomOrientationUniformOnSphere(std::mt19937_64 &rng)
  {
    static_assert(dim == 3, "crystallite orientations are only defined in 3D");
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    const double cosTheta = 2.0 * uniform01(rng) - 1.0;             // UNIFORM IN cos(theta)
    const double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
    const double phi      = 2.0 * numbers::PI * uniform01(rng);

    Tensor<1, dim> c;
    c[0] = sinTheta * std::cos(phi);
    c[1] = sinTheta * std::sin(phi);
    c[2] = cosTheta;
    return orientationFromCAxis<dim>(c);
  }
}

#endif

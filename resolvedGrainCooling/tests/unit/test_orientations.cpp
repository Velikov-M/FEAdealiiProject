#include "resolved_grain/orientations.h"

#include <gtest/gtest.h>

#include <random>

using namespace dealii;
using StepCooling::orientationFromCAxis;
using StepCooling::randomOrientationUniformOnSphere;

namespace
{
  void checkFrame(const Tensor<1, 3> &c)
  {
    const Tensor<2, 3> R = orientationFromCAxis(c);
    Tensor<2, 3> identity;
    for (unsigned int d = 0; d < 3; ++d)
      identity[d][d] = 1.0;
    EXPECT_NEAR((R * transpose(R) - identity).norm(), 0.0, 2e-14);
    EXPECT_NEAR(determinant(R), 1.0, 2e-14);
    for (unsigned int d = 0; d < 3; ++d)
      EXPECT_NEAR(R[2][d], c[d], 2e-15);
  }
}

TEST(Orientations, BasisIsOrthonormalIncludingCoordinateAxes)
{
  for (const auto &c : {Tensor<1, 3>({1.0, 0.0, 0.0}),
                        Tensor<1, 3>({0.0, 1.0, 0.0}),
                        Tensor<1, 3>({0.0, 0.0, 1.0}),
                        Tensor<1, 3>({0.0, 0.0, -1.0}),
                        Tensor<1, 3>({0.6, 0.48, 0.64})})
    checkFrame(c);
}

TEST(Orientations, SamplingIsReproducibleAndReturnsUnitAxes)
{
  std::mt19937_64 first(12345), second(12345);
  for (unsigned int i = 0; i < 32; ++i)
    {
      const auto R1 = randomOrientationUniformOnSphere<3>(first);
      const auto R2 = randomOrientationUniformOnSphere<3>(second);
      EXPECT_NEAR((R1 - R2).norm(), 0.0, 0.0);
      Tensor<1, 3> c;
      for (unsigned int d = 0; d < 3; ++d)
        c[d] = R1[2][d];
      EXPECT_NEAR(c.norm(), 1.0, 2e-15);
      checkFrame(c);
    }
}

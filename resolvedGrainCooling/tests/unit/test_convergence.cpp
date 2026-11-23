#include "resolved_grain/convergence.h"

#include <gtest/gtest.h>

#include <limits>

using StepCooling::scaledConvergenceMeasure;

TEST(ConvergenceMeasure, AcceptsExactZeroResponse)
{
  EXPECT_DOUBLE_EQ(scaledConvergenceMeasure(0.0, 0.0), 0.0);
}

TEST(ConvergenceMeasure, RejectsNonzeroIncrementAtZeroScale)
{
  EXPECT_TRUE(std::isinf(scaledConvergenceMeasure(1e-12, 0.0)));
}

TEST(ConvergenceMeasure, UsesRelativeScaleAwayFromZero)
{
  EXPECT_DOUBLE_EQ(scaledConvergenceMeasure(2.0, 4.0), 0.5);
}

TEST(ConvergenceMeasure, RejectsNonFiniteAndNegativeInputs)
{
  EXPECT_TRUE(std::isinf(scaledConvergenceMeasure(
    std::numeric_limits<double>::quiet_NaN(), 1.0)));
  EXPECT_TRUE(std::isinf(scaledConvergenceMeasure(-1.0, 1.0)));
  EXPECT_TRUE(std::isinf(scaledConvergenceMeasure(1.0, -1.0)));
}

#include "resolved_grain/damage_law.h"

#include "symbolic_oracles.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

using StepCooling::DamageLawParameters;
using StepCooling::damageKineticResidual;
using StepCooling::solveBackwardEulerDamage;

TEST(DamageLaw, CoupledStressCounterexampleFindsSmallerRoot)
{
  const DamageLawParameters law{1.0, 2.450539712794255, 2.0, 0.999};
  const auto stress = [](double omega) { return 1.3 - 0.2 * omega; };
  const double root = solveBackwardEulerDamage(stress, 0.0, 1.0, law, 1e-12);
  EXPECT_NEAR(root, SymbolicOracles::coupledStressRoot, 2e-12);
  EXPECT_LT(root, SymbolicOracles::coupledStressSecondRoot);
  EXPECT_NEAR(damageKineticResidual(stress(root), root, 0.0, 1.0, law), 0.0, 1e-11);
}

TEST(DamageLaw, ConstantStressLinearExponentMatchesSymbolicRoot)
{
  const DamageLawParameters law{2.0, 0.2, 1.0, 0.999};
  const auto stress = [](double) { return 3.0; };
  const double root = solveBackwardEulerDamage(stress, 0.1, 1.0, law, 1e-12);
  EXPECT_NEAR(root, SymbolicOracles::constantStressM1Root, 2e-12);
}

TEST(DamageLaw, SubcriticalAndCompressiveStressDoNotEvolveDamage)
{
  const DamageLawParameters law{10.0, 4.0, 2.0, 0.999};
  EXPECT_DOUBLE_EQ(solveBackwardEulerDamage([](double) { return 10.0; },
                                            0.2, 5.0, law, 1e-10), 0.2);
  EXPECT_DOUBLE_EQ(solveBackwardEulerDamage([](double) { return -100.0; },
                                            0.2, 5.0, law, 1e-10), 0.2);
}

TEST(DamageLaw, ZeroPrefactorIsAnExactNoEvolutionCase)
{
  const DamageLawParameters law{1.0, 0.0, 2.0, 0.999};
  EXPECT_DOUBLE_EQ(solveBackwardEulerDamage([](double) { return 100.0; },
                                            0.3, 5.0, law, 1e-10), 0.3);
}

TEST(DamageLaw, RejectsInvalidParametersAndUnbracketedSteps)
{
  DamageLawParameters law{1.0, 4.0, 2.0, 0.999};
  EXPECT_THROW(solveBackwardEulerDamage([](double) { return 2.0; },
                                        0.0, 0.0, law, 1e-10), std::runtime_error);
  law.kineticExponent = 0.0;
  EXPECT_THROW(solveBackwardEulerDamage([](double) { return 2.0; },
                                        0.0, 1.0, law, 1e-10), std::runtime_error);
  law = DamageLawParameters{1.0, 1e12, 2.0, 0.999};
  EXPECT_THROW(solveBackwardEulerDamage([](double) { return 2.0; },
                                        0.0, 1.0, law, 1e-10), std::runtime_error);
}

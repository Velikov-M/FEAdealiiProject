#include "resolved_grain/material.h"

#include "symbolic_oracles.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <stdexcept>

using namespace dealii;
using StepCooling::CteTensor;
using StepCooling::ElasticityTensor;
using StepCooling::checkPositiveDefiniteness;
using StepCooling::checkTransverseIsotropy;
using StepCooling::criticalDamageForPositiveDefiniteness;
using StepCooling::isPositiveDefiniteTransverselyIsotropic;

namespace
{
  struct MaterialFixture
  {
    MaterialFixture()
    {
      prm.enter_subsection("MaterialParameters");
      prm.enter_subsection("LinearElasticityParameters");
      prm.declare_entry("C11_0", "760e9");
      prm.declare_entry("C12_0", "80e9");
      prm.declare_entry("C13_0", "13e9");
      prm.declare_entry("C33_0", "12e9");
      prm.declare_entry("C44_0", "0.48e9");
      prm.declare_entry("C11_functionOfTemperature", "1 + 1e-4*(T-2000)");
      prm.declare_entry("C12_functionOfTemperature", "1 - 2e-4*(T-2000)");
      prm.declare_entry("C13_functionOfTemperature", "1 + 3e-4*(T-2000)");
      prm.declare_entry("C33_functionOfTemperature", "1 - 4e-4*(T-2000)");
      prm.declare_entry("C44_functionOfTemperature", "1 + 5e-4*(T-2000)");
      prm.leave_subsection();
      prm.enter_subsection("ThermoElasticParameters");
      prm.declare_entry("alpha11_0", "1.2e-6");
      prm.declare_entry("alpha33_0", "32e-6");
      prm.declare_entry("alpha11_functionOfTemperature", "1 + 1e-3*(T-2000)");
      prm.declare_entry("alpha33_functionOfTemperature", "1 - 1e-3*(T-2000)");
      prm.leave_subsection();
      prm.leave_subsection();
      elasticity.setFromPrm(prm);
      cte.setFromPrm(prm);
    }

    ParameterHandler prm;
    ElasticityTensor<3> elasticity;
    CteTensor<3> cte;
  };
}

TEST(Material, ThermoelasticStressMatchesIndependentSymPyIndexNotation)
{
  MaterialFixture fixture;
  SymmetricTensor<2, 3> strain;
  strain[0][0] = 1e-4;
  strain[1][1] = -2e-4;
  strain[2][2] = 3e-4;
  strain[1][2] = 4e-4;
  strain[0][2] = -5e-4;
  strain[0][1] = 6e-4;
  const double damage = 0.2;
  const double temperature = 1800.0;
  const auto stress = fixture.elasticity.getElasticityTensor(damage, temperature) *
                      (strain - fixture.cte.getcteTensor(damage, temperature) * -200.0);
  const std::array<double, 6> actual{{stress[0][0], stress[1][1], stress[2][2],
                                      stress[1][2], stress[0][2], stress[0][1]}};
  for (unsigned int i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], SymbolicOracles::stress[i],
                2e-13 * std::max(1.0, std::abs(SymbolicOracles::stress[i]))) << "component " << i;
}

TEST(Material, KelvinTensorPassesPhysicalComponentAndSymmetryChecks)
{
  MaterialFixture fixture;
  const auto constants = fixture.elasticity.voigtConstants(1800.0);
  const auto C = fixture.elasticity.getElasticityTensor(0.0, 1800.0);
  EXPECT_NO_THROW(checkTransverseIsotropy<3>(C, constants[0], constants[1], constants[2],
                                             constants[3], constants[4]));
  EXPECT_NEAR(C[1][2][1][2], constants[4], 1e-12 * constants[4]);
  EXPECT_NEAR(C[0][1][0][1], 0.5 * (constants[0] - constants[1]),
              1e-12 * constants[0]);
}

TEST(Material, PositiveDefinitenessLimitMatchesSymbolicDeterminant)
{
  const std::array<double, 5> baseline{{7.593581e11, 8.324534e10, 1.305714e10,
                                        1.261083e10, 4.8e8}};
  EXPECT_NEAR(criticalDamageForPositiveDefiniteness(baseline),
              SymbolicOracles::baselineCriticalDamage, 2e-15);
  EXPECT_TRUE(isPositiveDefiniteTransverselyIsotropic(baseline, 0.0));
  EXPECT_TRUE(isPositiveDefiniteTransverselyIsotropic(baseline, 0.9));
  EXPECT_FALSE(isPositiveDefiniteTransverselyIsotropic(baseline, 0.999));
  EXPECT_NO_THROW(checkPositiveDefiniteness(baseline, 0.9));
  EXPECT_THROW(checkPositiveDefiniteness(baseline, 0.999), std::runtime_error);
}

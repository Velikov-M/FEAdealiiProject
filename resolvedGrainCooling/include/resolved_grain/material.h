#ifndef RESOLVED_GRAIN_MATERIAL_H
#define RESOLVED_GRAIN_MATERIAL_H

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/physics/notation.h>
#include <deal.II/physics/transformations.h>
#include <array>
#include <memory>

namespace StepCooling
{
  using namespace dealii;

  /** Temperature-dependent transversely isotropic stiffness in the local crystal frame.
   *
   * Components are in Pa. Damage scales only C33 by (1 - omega); the returned rank-four tensor
   * is constructed through Kelvin notation, whose shear diagonal entries are 2*C44 and 2*C66.
   * Call setFromPrm() before evaluating any component.
   */
  template <int dim>
  class ElasticityTensor
  {
  public:
    ElasticityTensor() = default;
    void setFromPrm(const ParameterHandler &prm);
    SymmetricTensor<4,dim,double> getElasticityTensor(double curOmega, double curTemperature) const;

    // C33 at zero damage, i.e. the coefficient of the rank-one damage term. Damage scales ONLY
    // this entry (getElasticityTensor multiplies it by (1-omega) and touches nothing else), which
    // is what makes the rank-one identity below exact.
    double getC33AtZeroDamage(double curTemperature) const
    {
      Point<1> tempPoint(curTemperature);
      return this->C33_0 * this->C33_func->value(tempPoint);
    }

    // Undamaged Voigt constants at T, in the order C11, C12, C13, C33, C44. Used by the startup
    // self-check, which compares the constructed tensor against these.
    std::array<double, 5> voigtConstants(double curTemperature) const
    {
      Point<1> p(curTemperature);
      return {{C11_0 * C11_func->value(p), C12_0 * C12_func->value(p), C13_0 * C13_func->value(p),
               C33_0 * C33_func->value(p), C44_0 * C44_func->value(p)}};
    }

  private:
    std::unique_ptr<Function<1>> C11_func;
    std::unique_ptr<Function<1>> C33_func;
    std::unique_ptr<Function<1>> C44_func;
    std::unique_ptr<Function<1>> C12_func;
    std::unique_ptr<Function<1>> C13_func;

    double C11_0;
    double C33_0;
    double C44_0;
    double C12_0;
    double C13_0;

  };

  /** Secant thermal-expansion tensor in the local crystal frame, in 1/K.
   *
   * The coefficients are referenced to the model's stress-free initial temperature. Damage
   * scales only alpha33 by (1 - omega). Call setFromPrm() before evaluating any component.
   */
  template <int dim>
  class CteTensor
  {
  public:
    CteTensor() = default;
    void setFromPrm(const ParameterHandler &prm);
    SymmetricTensor<2,dim,double> getcteTensor(double curOmega, double curTemperature) const;

    // alpha_33 at zero damage: the coefficient of the rank-one damage term, exactly analogous to
    // ElasticityTensor::getC33AtZeroDamage().
    double getAlpha33AtZeroDamage(double curTemperature) const
    {
      Point<1> tempPoint(curTemperature);
      return this->alpha33_0 * this->alpha33_func->value(tempPoint);
    }
  private:
    std::unique_ptr<Function<1>> alpha11_func;
    std::unique_ptr<Function<1>> alpha33_func;
    double alpha11_0;
    double alpha33_0;
  };

  template <int dim>
  void checkTransverseIsotropy(const SymmetricTensor<4, dim> &C,
                               double C11, double C12, double C13, double C33, double C44);
}

#endif

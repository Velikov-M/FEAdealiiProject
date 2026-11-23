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
    /** Construct an empty tensor wrapper; call setFromPrm() before evaluation. */
    ElasticityTensor() = default;

    /** Load stiffness constants and temperature functions from the parameter schema.
     * @param[in] prm Parsed model parameters.
     */
    void setFromPrm(const ParameterHandler &prm);

    /** Construct the damaged local stiffness at a given temperature.
     * @param[in] curOmega Current damage, dimensionless.
     * @param[in] curTemperature Current temperature in K.
     * @return Rank-four stiffness tensor in Pa in local crystal coordinates.
     */
    SymmetricTensor<4,dim,double> getElasticityTensor(double curOmega, double curTemperature) const;

    /** Return undamaged C33, the coefficient of the rank-one damage update.
     * @param[in] curTemperature Current temperature in K.
     * @return Undamaged C33 in Pa.
     */
    double getC33AtZeroDamage(double curTemperature) const
    {
      Point<1> tempPoint(curTemperature);
      return this->C33_0 * this->C33_func->value(tempPoint);
    }

    /** Return undamaged constants ordered as C11, C12, C13, C33, C44.
     * @param[in] curTemperature Current temperature in K.
     * @return Five undamaged Voigt constants in Pa.
     */
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
    /** Construct an empty CTE wrapper; call setFromPrm() before evaluation. */
    CteTensor() = default;

    /** Load secant CTE constants and temperature functions from the parameter schema.
     * @param[in] prm Parsed model parameters.
     */
    void setFromPrm(const ParameterHandler &prm);

    /** Construct the damaged local secant CTE tensor at a given temperature.
     * @param[in] curOmega Current damage, dimensionless.
     * @param[in] curTemperature Current temperature in K.
     * @return Secant CTE tensor in 1/K in local crystal coordinates.
     */
    SymmetricTensor<2,dim,double> getcteTensor(double curOmega, double curTemperature) const;

    /** Return undamaged alpha33, the coefficient of the rank-one damage update.
     * @param[in] curTemperature Current temperature in K.
     * @return Undamaged secant alpha33 in 1/K.
     */
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

  /** Check stiffness components and invariance under rotation about the local c-axis.
   * @param[in] C Constructed rank-four stiffness tensor in Pa.
   * @param[in] C11 Expected undamaged C11 in Pa.
   * @param[in] C12 Expected undamaged C12 in Pa.
   * @param[in] C13 Expected undamaged C13 in Pa.
   * @param[in] C33 Expected undamaged C33 in Pa.
   * @param[in] C44 Expected undamaged C44 in Pa.
   */
  template <int dim>
  void checkTransverseIsotropy(const SymmetricTensor<4, dim> &C,
                               double C11, double C12, double C13, double C33, double C44);

  /** Test positive definiteness of a damaged transversely isotropic stiffness.
   * @param[in] constants Undamaged constants ordered C11, C12, C13, C33, C44 in Pa.
   * @param[in] omega Damage, dimensionless; only C33 is scaled by (1-omega).
   * @return True when all normal and shear eigenvalue conditions are strictly positive.
   */
  bool isPositiveDefiniteTransverselyIsotropic(const std::array<double, 5> &constants,
                                               double omega);

  /** Return the damage at which the normal-stiffness determinant reaches zero.
   * @param[in] constants Undamaged constants ordered C11, C12, C13, C33, C44 in Pa.
   * @return Critical damage, dimensionless. Valid undamaged materials return a value in (0,1).
   */
  double criticalDamageForPositiveDefiniteness(const std::array<double, 5> &constants);

  /** Require a damaged stiffness to be positive definite.
   * @param[in] constants Undamaged constants ordered C11, C12, C13, C33, C44 in Pa.
   * @param[in] omega Damage, dimensionless.
   * @throw std::runtime_error If any stiffness eigenvalue condition is non-positive.
   */
  void checkPositiveDefiniteness(const std::array<double, 5> &constants, double omega);
}

#endif

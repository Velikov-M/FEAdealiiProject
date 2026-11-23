#include "resolved_grain/material.h"

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  void ElasticityTensor<dim>::setFromPrm(const ParameterHandler &prm)
  {
    Assert(dim == 3, ExcNotImplemented());

    this->C11_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C11_0");
    this->C33_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C33_0");
    this->C44_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C44_0");
    this->C12_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C12_0");
    this->C13_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C13_0");

    auto fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C11_functionOfTemperature"), {});
    this->C11_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C33_functionOfTemperature"), {});
    this->C33_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C44_functionOfTemperature"), {});
    this->C44_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C12_functionOfTemperature"), {});
    this->C12_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "LinearElasticityParameters"}, "C13_functionOfTemperature"), {});
    this->C13_func = std::move(fp);
  }

  template <int dim>
  SymmetricTensor<4,dim,double>  ElasticityTensor<dim>::getElasticityTensor(double curOmega, double curTemperature) const
  {
    Point<1> tempPoint(curTemperature);
    double C11 = this->C11_0 * this->C11_func->value(tempPoint);
    double C33 = this->C33_0 * this->C33_func->value(tempPoint) * (1 - curOmega);
    double C44 = this->C44_0 * this->C44_func->value(tempPoint);
    double C12 = this->C12_0 * this->C12_func->value(tempPoint);
    double C13 = this->C13_0 * this->C13_func->value(tempPoint);
    double C66 = (C11 - C12) / 2;

    FullMatrix<double> kelvin_matrix(6, 6);

    kelvin_matrix[0][0] = C11;

    kelvin_matrix[0][1] = C12;
    kelvin_matrix[1][0] = C12;

    kelvin_matrix[0][2] = C13;
    kelvin_matrix[2][0] = C13;

    kelvin_matrix[1][1] = C11;

    kelvin_matrix[1][2] = C13;
    kelvin_matrix[2][1] = C13;

    kelvin_matrix[2][2] = C33;

    // Kelvin shear diagonals are twice the Voigt moduli; C66 = (C11-C12)/2.
    kelvin_matrix[3][3] = 2.0 * C44;
    kelvin_matrix[4][4] = 2.0 * C44;

    kelvin_matrix[5][5] = 2.0 * C66;   // = C11 - C12

    SymmetricTensor<4, dim, double> tmpTensor;
    Physics::Notation::Kelvin::to_tensor(kelvin_matrix, tmpTensor);
    return tmpTensor;
  }

  template <int dim>
  void checkTransverseIsotropy(const SymmetricTensor<4, dim> &C,
                               double C11, double C12, double C13, double C33, double C44)
  {
    const double scale = C.norm();
    const double tol = 1e-12 * scale;
    const std::array<std::pair<const char *, std::pair<double, double>>, 6> components{{
      {"C_1111 vs C11", {C[0][0][0][0], C11}},
      {"C_1122 vs C12", {C[0][0][1][1], C12}},
      {"C_1133 vs C13", {C[0][0][2][2], C13}},
      {"C_3333 vs C33", {C[2][2][2][2], C33}},
      {"C_2323 vs C44", {C[1][2][1][2], C44}},
      {"C_1212 vs (C11-C12)/2", {C[0][1][0][1], 0.5 * (C11 - C12)}}}};
    for (const auto &c : components)
      if (std::abs(c.second.first - c.second.second) > tol)
        throw std::runtime_error(std::string("stiffness tensor fails its physical self-check: ") +
                                 c.first + " = " + std::to_string(c.second.first) + " vs " +
                                 std::to_string(c.second.second) +
                                 ". Suspect the Kelvin/Voigt convention in getElasticityTensor.");

    // Invariance under rotation about the c-axis (e3 in the local frame), at an arbitrary angle.
    const double a = 0.6;
    Tensor<2, dim> R;
    R[0][0] = std::cos(a); R[0][1] = -std::sin(a);
    R[1][0] = std::sin(a); R[1][1] =  std::cos(a);
    R[2][2] = 1.0;
    const double drift = (Physics::Transformations::basis_transformation(C, R) - C).norm();
    if (drift > 1e-10 * scale)
      throw std::runtime_error(
        "stiffness tensor is NOT transversely isotropic: rotating it about the c-axis changes it "
        "by " + std::to_string(drift / scale) + " relative. The in-plane frame of each grain is "
        "arbitrary, so this would make results depend on it.");
  }

  template <int dim>
  void CteTensor<dim>::setFromPrm(const ParameterHandler &prm)
  {
    Assert(dim == 3, ExcNotImplemented());

    alpha11_0 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha11_0");
    alpha33_0 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha33_0");

    auto fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "ThermoElasticParameters"}, "alpha11_functionOfTemperature"), {});
    alpha11_func = std::move(fp);

    fp = std::make_unique<FunctionParser<1>>();
    fp->initialize("T", prm.get({"MaterialParameters", "ThermoElasticParameters"}, "alpha33_functionOfTemperature"), {});
    alpha33_func = std::move(fp);
  }

  // Damage reduces only the secant c-axis CTE, using the same (1-omega) factor as C33.
  template <int dim>
  SymmetricTensor<2,dim,double> CteTensor<dim>::getcteTensor(double curOmega, double curTemperature) const
  {
    SymmetricTensor<2, dim, double> cteTensor;
    Point<1> tempPoint(curTemperature);
    cteTensor[0][0] = this->alpha11_0 * this->alpha11_func->value(tempPoint);
    cteTensor[1][1] = this->alpha11_0 * this->alpha11_func->value(tempPoint);
    cteTensor[2][2] = this->alpha33_0 * this->alpha33_func->value(tempPoint) * (1 - curOmega);
    return cteTensor;
  }

  bool isPositiveDefiniteTransverselyIsotropic(const std::array<double, 5> &constants,
                                               const double omega)
  {
    const double C11 = constants[0];
    const double C12 = constants[1];
    const double C13 = constants[2];
    const double C33 = constants[3] * (1.0 - omega);
    const double C44 = constants[4];
    return C11 - C12 > 0.0 && C11 + C12 > 0.0 && C33 > 0.0 && C44 > 0.0 &&
           C33 * (C11 + C12) - 2.0 * C13 * C13 > 0.0;
  }

  double criticalDamageForPositiveDefiniteness(const std::array<double, 5> &constants)
  {
    return 1.0 - 2.0 * constants[2] * constants[2] /
                  (constants[3] * (constants[0] + constants[1]));
  }

  void checkPositiveDefiniteness(const std::array<double, 5> &constants, const double omega)
  {
    if (!isPositiveDefiniteTransverselyIsotropic(constants, omega))
      {
        const double critical = criticalDamageForPositiveDefiniteness(constants);
        throw std::runtime_error(
          "damaged transversely isotropic stiffness is not positive definite at damage " +
          std::to_string(omega) + "; normal-subspace critical damage is " +
          std::to_string(critical));
      }
  }

  template class ElasticityTensor<3>;
  template class CteTensor<3>;
  template void checkTransverseIsotropy<3>(const SymmetricTensor<4, 3> &C,
                                           double C11, double C12, double C13,
                                           double C33, double C44);
}

#ifndef RESOLVED_GRAIN_DAMAGE_LAW_H
#define RESOLVED_GRAIN_DAMAGE_LAW_H

#include <functional>

namespace StepCooling
{
  /** Parameters of the local cooling-driven basal damage law. */
  struct DamageLawParameters
  {
    double stressThreshold;   ///< Tensile activation threshold in Pa.
    double kineticPrefactor;  ///< Prefactor A in 1/K.
    double kineticExponent;   ///< Dimensionless exponent m, greater than zero.
    double maximumDamage = 0.999; ///< Numerical upper bound, dimensionless.
  };

  /** Evaluate the implicit backward-Euler damage residual.
   * @param[in] stress33 Trial local c-axis normal stress in Pa.
   * @param[in] omega Trial damage, dimensionless.
   * @param[in] oldOmega Converged damage at the start of the step, dimensionless.
   * @param[in] coolingIncrement Positive cooling increment in K.
   * @param[in] parameters Kinetic-law parameters.
   * @return Dimensionless kinetic residual.
   */
  double damageKineticResidual(double stress33,
                               double omega,
                               double oldOmega,
                               double coolingIncrement,
                               const DamageLawParameters &parameters);

  /** Solve one backward-Euler damage update at fixed strain and temperature.
   *
   * The stress callback is reevaluated at trial damage and returns local sigma_33 in Pa.
   * The smaller admissible root continuous with oldOmega is preferred. The historical
   * analytic partition is used when valid; a law-agnostic bracketing scan handles states
   * where damage-dependent stress invalidates that partition.
   *
   * @param[in] stressAtDamage Callback returning sigma_33(omega) in Pa.
   * @param[in] oldOmega Converged damage at the start of the step, dimensionless.
   * @param[in] coolingIncrement Positive cooling increment in K.
   * @param[in] parameters Kinetic-law parameters.
   * @param[in] residualTolerance Maximum accepted absolute dimensionless residual.
   * @return Updated damage in [oldOmega, maximumDamage], dimensionless.
   * @throw std::runtime_error If parameters are invalid or no converged root is found.
   */
  double solveBackwardEulerDamage(
    const std::function<double(double)> &stressAtDamage,
    double oldOmega,
    double coolingIncrement,
    const DamageLawParameters &parameters,
    double residualTolerance);
}

#endif

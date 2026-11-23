#include "resolved_grain/damage_law.h"

#include <boost/math/tools/roots.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace StepCooling
{
  namespace
  {
    std::pair<double, double> solveBracket(const std::function<double(double)> &residual,
                                           double lower,
                                           double upper)
    {
      boost::uintmax_t iterations = 200;
      const auto bracket = boost::math::tools::toms748_solve(
        residual, lower, upper,
        [](double a, double b) { return std::abs(b - a) < 1e-12; }, iterations);
      return {bracket.first, bracket.second};
    }
  }

  double damageKineticResidual(const double stress33,
                               const double omega,
                               const double oldOmega,
                               const double coolingIncrement,
                               const DamageLawParameters &parameters)
  {
    const double actingStress = std::max(stress33 - parameters.stressThreshold, 0.0);
    return omega - oldOmega
           - parameters.kineticPrefactor
             * std::pow(actingStress / parameters.stressThreshold / (1.0 - omega),
                        parameters.kineticExponent) * coolingIncrement;
  }

  double solveBackwardEulerDamage(
    const std::function<double(double)> &stressAtDamage,
    const double oldOmega,
    const double coolingIncrement,
    const DamageLawParameters &parameters,
    const double residualTolerance)
  {
    if (!std::isfinite(oldOmega) || oldOmega < 0.0 || oldOmega > parameters.maximumDamage ||
        !std::isfinite(coolingIncrement) || coolingIncrement <= 0.0 ||
        !std::isfinite(parameters.stressThreshold) || parameters.stressThreshold <= 0.0 ||
        !std::isfinite(parameters.kineticPrefactor) || parameters.kineticPrefactor < 0.0 ||
        !std::isfinite(parameters.kineticExponent) || parameters.kineticExponent <= 0.0 ||
        !std::isfinite(parameters.maximumDamage) || parameters.maximumDamage <= 0.0 ||
        parameters.maximumDamage >= 1.0 || !std::isfinite(residualTolerance) ||
        residualTolerance <= 0.0)
      throw std::runtime_error("invalid backward-Euler damage-law parameters");

    if (parameters.kineticPrefactor == 0.0 ||
        stressAtDamage(oldOmega) <= parameters.stressThreshold)
      return oldOmega;

    const auto residual = [&](double omega) {
      return damageKineticResidual(stressAtDamage(omega), omega, oldOmega,
                                   coolingIncrement, parameters);
    };
    const auto acceptable = [&](double omega) {
      return std::isfinite(omega) && omega >= oldOmega &&
             omega <= parameters.maximumDamage &&
             std::abs(residual(omega)) <= residualTolerance;
    };

    const double actingStressAtOld =
      std::max(stressAtDamage(oldOmega) - parameters.stressThreshold, 0.0);
    const double split = std::clamp(
      1.0 - std::pow(parameters.kineticExponent * parameters.kineticPrefactor *
                       std::pow(actingStressAtOld / parameters.stressThreshold,
                                parameters.kineticExponent) * coolingIncrement,
                     1.0 / (parameters.kineticExponent + 1.0)),
      oldOmega, parameters.maximumDamage);

    const double residualAtOld = residual(oldOmega);
    const double residualAtSplit = residual(split);
    const double residualAtMaximum = residual(parameters.maximumDamage);
    double lower = oldOmega;
    double upper = parameters.maximumDamage;
    bool bracketed = false;
    if (split > oldOmega && (residualAtOld < 0.0) != (residualAtSplit < 0.0))
      {
        lower = oldOmega;
        upper = split;
        bracketed = true;
      }
    else if (split < parameters.maximumDamage &&
             (residualAtSplit < 0.0) != (residualAtMaximum < 0.0))
      {
        lower = split;
        upper = parameters.maximumDamage;
        bracketed = true;
      }

    if (bracketed)
      {
        const auto bracket = solveBracket(residual, lower, upper);
        const double candidate = 0.5 * (bracket.first + bracket.second);
        if (acceptable(candidate))
          return candidate;
      }

    // The analytic split assumes stress frozen at oldOmega. If that assumption fails,
    // locate the first sign-changing interval using only the actual coupled residual.
    constexpr unsigned int scanIntervals = 512;
    std::vector<double> points;
    points.reserve(scanIntervals + 2);
    for (unsigned int i = 0; i <= scanIntervals; ++i)
      points.push_back(oldOmega + (parameters.maximumDamage - oldOmega) *
                                  static_cast<double>(i) / scanIntervals);
    points.push_back(split);
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());

    for (unsigned int i = 0; i + 1 < points.size(); ++i)
      {
        if (points[i + 1] <= points[i])
          continue;
        const double leftResidual = residual(points[i]);
        const double rightResidual = residual(points[i + 1]);
        if (std::isfinite(leftResidual) && std::isfinite(rightResidual) &&
            (leftResidual < 0.0) != (rightResidual < 0.0))
          {
            const auto bracket = solveBracket(residual, points[i], points[i + 1]);
            const double candidate = 0.5 * (bracket.first + bracket.second);
            if (acceptable(candidate))
              return candidate;
          }
      }

    std::ostringstream message;
    message.precision(16);
    message << "backward-Euler damage root was not found; old_damage=" << oldOmega
            << ", cooling_increment=" << coolingIncrement
            << ", split=" << split << ", residual(old)=" << residualAtOld
            << ", residual(split)=" << residualAtSplit
            << ", residual(maximum)=" << residualAtMaximum;
    throw std::runtime_error(message.str());
  }
}

#ifndef RESOLVED_GRAIN_PARAMETERS_H
#define RESOLVED_GRAIN_PARAMETERS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/observer_pointer.h>
#include <string>

namespace StepCooling
{
  using namespace dealii;

  /** Declares and parses the resolved-grain parameter schema.
   *
   * The referenced ParameterHandler must outlive this reader. Parsing accepts JSON or other
   * formats supported by deal.II; default values and valid selections are declared centrally.
   */
  class ParameterReader : public EnableObserverPointer
  {
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);
    void declare_parameters();

  private:
    ParameterHandler &prm;
  };
}

#endif

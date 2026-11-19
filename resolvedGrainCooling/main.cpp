#include "resolved_grain/elastic_problem.h"
#include "resolved_grain/parameters.h"

int main(int argc, char *argv[])
{
  // Trilinos calls MPI_Comm_size() during construction of any of its objects, so MPI must be up
  // before the AMG preconditioner is built -- even though this program is serial and runs on one
  // rank. The thread count is left as invalid_unsigned_int so that MPI_InitFinalize does NOT cap
  // threading: WorkStream still gets every core, and ParallelizationParameters/NumberOfThreads
  // remains the single place that controls it.
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, dealii::numbers::invalid_unsigned_int);

  try
    {
      StepCooling::ParameterHandler prm;
      StepCooling::ParameterReader myReader(prm);
      // Use an explicit run-specific parameter file when supplied; otherwise use ./input.json.
      const std::string parameterFile = (argc > 1) ? argv[1] : "input.json";
      std::cout << "Parameter file: " << parameterFile << std::endl;
      myReader.read_parameters(parameterFile);
      StepCooling::ElasticProblem<3> myProblem(prm);
      myProblem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

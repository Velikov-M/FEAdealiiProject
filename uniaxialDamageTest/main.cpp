// Uniaxial displacement-controlled damage test.
//
// PURPOSE: verify the damage kinetic-equation (ODE) solver and the stress-damage coupling --
// the two things the MMS test in ../mmsSandbox/ did NOT actually verify. That test validated the
// PDE/elasticity solver well (textbook convergence orders), but its kinetic-equation forcing term
// was built from the exact/manufactured stress, which does not respond to the numerically-computed
// damage. With no negative feedback, small per-step errors amplified geometrically (~3x/step) and
// the test could never converge -- see the damage-mms-kinetic-debugging notes.
//
// SETUP: a single-material unit cube under uniform axial displacement.
//   - z = 0    face: u_z = 0                        (BoundaryIds::fixedZFace)
//   - z = L    face: u_z = strainRate * L * t       (BoundaryIds::loadedZFace)
//   - x/y      faces: traction-free, unconstrained  (BoundaryIds::lateralFaces)
//   - one corner additionally pinned in x/y to remove rigid-body modes
//   - body force = 0, CTE = 0 (set in input.json)
//
// WHY THIS WORKS: for a homogeneous material this makes the stress state exactly uniform, so
// div(sigma) = 0 holds identically for any constant sigma -- equilibrium is trivially satisfied
// and the FE discretization is exact (constant-strain field, a patch-test case). What remains
// under test is purely the ODE integration and the coupling. Because only C33 carries the damage
// factor (C33 = C33_0*(1-omega); the rest are undamaged), the transverse response is
// damage-independent and the closed-form axial relation is:
//
//   eps_transverse = -C13_0 * eps_zz / (C11_0 + C12_0)                      [damage-independent]
//   sigma_zz(omega, eps_zz) = eps_zz * [ C33_0*(1-omega) - 2*C13_0^2/(C11_0+C12_0) ]
//
// Under DISPLACEMENT control this coupling is genuinely self-limiting: growing damage lowers
// sigma_zz, which slows further growth. That is the real negative feedback the MMS construction
// was missing, and it is the regime where the implicit scheme's stability guarantee actually
// applies. (Force/stress control would instead be self-accelerating -- classic creep rupture --
// and is deliberately NOT what this first test does.)
//
// "Temperature" is reused as a dimensionless pseudo-time loading parameter ONLY. With CTE = 0 it
// has no thermal meaning here; it just drives the applied displacement ramp and lets this test
// reuse the existing stepping/convergence machinery. See AppliedAxialDisplacement.
//
// System International (SI) units are used in the code, so all the parameters should be provided in SI units


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>
#include <deal.II/physics/notation.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>

#include <boost/math/tools/roots.hpp>

//Right now refinement is not supported due to transfering damage is not realized, setup_system() must be used only once

namespace StepCooling
{
  using namespace dealii;

  struct DamageQData : public TransferableQuadraturePointData
  {
    double damage = 0.0;
    double old_damage = 0.0; //for iterative solution, for one step prediction it is not needed, but for iterative solution it is needed to calculate residual of kinetic equation

    unsigned int number_of_values() const override
    {
      return 2; // damage and old_damage
    }

    void pack_values(std::vector<double> &values) const override
    {
      AssertDimension(values.size(), 2);
      values[0] = damage;
      values[1] = old_damage;
    }

    void unpack_values(const std::vector<double> &values) override
    {
      AssertDimension(values.size(), 2);
      damage = values[0];
      old_damage = values[1];
    }
  };

  template <int dim>
  struct DamageAndOrientationQData : public TransferableQuadraturePointData
  {
    double damage = 0.0;
    double old_damage = 0.0; //for iterative solution, for one step prediction it is not needed, but for iterative solution it is needed to calculate residual of kinetic equation
    Tensor<2, 3> orientation = unit_symmetric_tensor<3, double>();

    unsigned int number_of_values() const override
    {
      return 2 + dim*dim; // 
    }

    void pack_values(std::vector<double> &values) const override
    {
      AssertDimension(values.size(), 2 + dim*dim);
      values[0] = damage;
      values[1] = old_damage;
      for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j) values[2+3*i+j] = orientation[i][j];
      } 
    }

    void unpack_values(const std::vector<double> &values) override
    {
      AssertDimension(values.size(), 2 + dim*dim);
      damage = values[0];
      old_damage = values[1];
      for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j) orientation[i][j] = values[2+3*i+j];
      }
      Assert(abs(determinant(orientation) - 1) < 0.05, ExcMessage("During qData transfer strong orientation matrix non-orthogonality appeared"));
      // Here orthogonalization must be considered and added lately
    }
  };

  // Applied axial displacement on the loaded (z = L) face, ramped linearly in pseudo-time.
  //
  // "Pseudo-time" here is the existing temperature-stepping variable, reused as a dimensionless
  // loading parameter: t = (T_0 - curT)/(T_0 - T_end) runs 0 -> 1 over the run. This test sets
  // the CTE tensor to zero in input.json, so temperature has NO thermal effect whatsoever -- it
  // is purely a loop counter in disguise. Reusing it avoids duplicating the whole stepping/
  // convergence infrastructure just to drive a monotone load parameter.
  //
  // Only the z-component is prescribed (u_z = strainRate * L * t, i.e. a uniform axial strain
  // eps_zz = strainRate * t). The x/y components are left to the ComponentMask at the call site,
  // so the loaded face is free to contract laterally -- prescribing them would impose a triaxial
  // state and break the closed-form solution this test checks against.
  template <int dim>
  class AppliedAxialDisplacement : public Function<dim>
  {
    public:
      AppliedAxialDisplacement() : Function<dim>(dim) {}

      void set_prm_consts(const ParameterHandler &prm)
      {
        L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
        T_0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
        T_end = prm.get_double({"ModelParameters"}, "FinalTemperature");
        strainRate = prm.get_double({"ModelParameters"}, "AppliedAxialStrainRate");
      }
      void set_current_temperature(const double cT) { curT = cT; }

      // Axial strain currently applied, eps_zz(t). The analytical reference solution needs this
      // too, so it lives here rather than being recomputed from displacement at the call site.
      double applied_axial_strain() const
      {
        const double t = (T_0 - curT) / (T_0 - T_end);
        return strainRate * t;
      }

      virtual void vector_value(const Point<dim> & /*p*/,
                                Vector<double> &values) const override
      {
        Assert(dim == 3, ExcNotImplemented());
        AssertDimension(values.size(), dim);
        values = 0.0;
        values(dim - 1) = applied_axial_strain() * L;
      }

    private:
      double L = 0.0, T_0 = 0.0, T_end = 0.0, strainRate = 0.0, curT = 0.0;
  };

  // Boundary ids assigned geometrically in run() (the mesh file carries no surface markers at
  // all -- every face defaults to 0, verified directly).
  namespace BoundaryIds
  {
    constexpr types::boundary_id fixedZFace   = 1; // z = 0,  u_z = 0
    constexpr types::boundary_id loadedZFace  = 2; // z = L,  u_z = applied
    constexpr types::boundary_id lateralFaces = 3; // x/y faces: traction-free, unconstrained
  }

  class ParameterReader : public EnableObserverPointer
    {
    public:
      ParameterReader(ParameterHandler &);
      void read_parameters(const std::string &);
      void declare_parameters();
  
    private:
      ParameterHandler &prm;
    };

  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler)
  {}

  void ParameterReader::declare_parameters()
  {
    prm.enter_subsection("ModelParameters");
    {
      prm.declare_entry("InitialTemperature",
                        "2000",
                        Patterns::Double(0),
                        "Initial temperature of the body");

      prm.declare_entry("TemperatureIncrement",
                        "100",
                        Patterns::Double(0),
                        "Temperature increment for each step");

      prm.declare_entry("LengthOfTheBody",
                        "1",
                        Patterns::Double(0),
                        "Length of the body in meters");
      prm.declare_entry("FinalTemperature",
                        "500",
                        Patterns::Double(0),
                        "Temperature at the end of cooling process");

      // "Temperature" in this test is a dimensionless pseudo-time/loading parameter, not a real
      // temperature -- CTE is set to 0 in input.json so it has no thermal effect at all; it only
      // drives the applied boundary displacement below (existing temperature-stepping loop
      // infrastructure reused as-is). See run()'s comment for the full rationale.
      prm.declare_entry("AppliedAxialStrainRate",
                        "0.02",
                        Patterns::Double(0),
                        "dEpsilonZZ/d(pseudo-time): applied axial strain accumulated per unit of "
                        "InitialTemperature-FinalTemperature range, ramped linearly. Boundary "
                        "displacement on the loaded face is this times LengthOfTheBody times "
                        "elapsed pseudo-time.");
    }
    prm.leave_subsection();
    prm.enter_subsection("MaterialParameters");
    {
      prm.enter_subsection("LinearElasticityParameters"); //For now only transversely isotropic material is supported, so only 5 independent parameters are needed
      {
        prm.declare_entry("C11_0",
                          "1060e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C11 at reference temperature");
        
        prm.declare_entry("C11_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C11");

        prm.declare_entry("C33_0",
                          "36.5e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C33 at reference temperature");

        prm.declare_entry("C33_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C33");

        prm.declare_entry("C44_0",
                          "0.25e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C44 at reference temperature");

        prm.declare_entry("C44_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C44");

        prm.declare_entry("C12_0",
                          "180e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C12 at reference temperature");
        
        prm.declare_entry("C12_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C12");

        prm.declare_entry("C13_0",
                          "15e9",
                          Patterns::Double(0),
                          "Elasticity tensor component C13 at reference temperature");

        prm.declare_entry("C13_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for elasticity tensor component C13");
      }
      prm.leave_subsection();
      prm.enter_subsection("DamageParameters");
      {
        prm.declare_entry("StressThreshold",
                          "200e6",
                          Patterns::Double(0),
                          "Stress threshold for damage activation");

        prm.declare_entry("a_kineticParam",
                          "1e-5",
                          Patterns::Double(0),
                          "Kinetic parameter a in the kinetic equation");

        prm.declare_entry("m_kineticParam",
                          "2",
                          Patterns::Double(0),
                          "Kinetic parameter m in the kinetic equation");
      }
      prm.leave_subsection();
      prm.enter_subsection("ThermoElasticParameters");
      {
        prm.declare_entry("alpha11_0", // equal to alpha_22 for transversely isotropic material
                          "1e-5",
                          Patterns::Double(0),
                          "Thermal expansion coefficient at a-axis direction");

        prm.declare_entry("alpha11_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for thermal expansion coefficient at a-axis direction");

        prm.declare_entry("alpha33_0",
                          "1e-8",
                          Patterns::Double(0),
                          "Thermal expansion coefficient at c-axis direction");

        prm.declare_entry("alpha33_functionOfTemperature",
                          "1.0",
                          Patterns::Anything(),
                          "Function of temperature for thermal expansion coefficient at c-axis direction");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
    prm.enter_subsection("SolverParameters");
    {
      prm.declare_entry("ForceEqulibriumRelativeTolerance",
                        "1e-4",
                        Patterns::Double(0),
                        "Relative tolerance for force equilibrium residiual in displacement-damage iteration");

      prm.declare_entry("DisplacementIncrementRelativeTolerance",
                        "1e-5",
                        Patterns::Double(0),
                        "Relative tolerance for displacement increment in in displacement-damage iteration");

      prm.declare_entry("DamageIncrementRelativeTolerance",
                        "1e-5",
                        Patterns::Double(0),
                        "Relative tolerance for damage increment in in displacement-damage iteration");

      prm.declare_entry("MaxNumberOfIterations",
                        "100",
                        Patterns::Integer(1),
                        "Maximum number of iterations for displacement-damage iteration");

      prm.declare_entry("KineticAlgebraicSolverTolerance",
                        "1e-4",
                        Patterns::Double(0),
                        "Relative tolerance of iterative algebraic solver for kinetic equation");

      // Was hardcoded at 1e-6 in solve(). That is a fine engineering default, but it is far too
      // loose for verification: the algebraic stress-damage consistency check in
      // process_solution() bottomed out at ~1.4e-3*StressThreshold and was measuring the LINEAR
      // SOLVER, not the coupling. Tightening to 1e-12 drops that residual by eight orders of
      // magnitude, to ~2e-11*StressThreshold (i.e. round-off), which is what makes the check a
      // real verification of the coupling algebra. Left configurable because tight tolerances
      // cost CG iterations and will not be wanted for every production run -- and because the
      // upcoming refined/perturbed problems are more ill-conditioned, where demanding 1e-12
      // within the iteration cap may not be achievable.
      prm.declare_entry("LinearSolverTolerance",
                        "1e-6",
                        Patterns::Double(0),
                        "Relative tolerance (vs |rhs|) for the CG linear solve");
    }
    prm.leave_subsection();
    prm.enter_subsection("ParallelizationParameters");
    {
      // Shared-memory threading only (WorkStream/TBB). No MPI, no distributed triangulation, no
      // vector/matrix type changes -- that is a separate, later step.
      prm.declare_entry("UseWorkStream",
                        "true",
                        Patterns::Bool(),
                        "Assemble and predict damage via WorkStream::run (multi-threaded). When "
                        "false, the original single-threaded loops are used instead -- kept as a "
                        "reference implementation for A/B correctness and timing comparison.");

      prm.declare_entry("NumberOfThreads",
                        "0",
                        Patterns::Integer(0),
                        "Thread cap for WorkStream. 0 means let deal.II use all available cores.");
    }
    prm.leave_subsection();
    prm.enter_subsection("MeshRefinementParameters");
    {
      prm.declare_entry("NumberOfRefinementCycles",
                        "1",
                        Patterns::Integer(1),
                        "Number of mesh refinement cycles to run (1 = no refinement, just the "
                        "initial mesh). Each cycle after the first refines the mesh and re-solves "
                        "the full temperature range from scratch on it; damage does not carry over "
                        "between cycles (see refine_grid()/setup_system()), so this is intended for "
                        "mesh convergence studies on damage-free (or single-cycle) MMS tests, not "
                        "yet for refining mid-simulation on a run where damage has already "
                        "accumulated.");

      prm.declare_entry("RefinementStrategy",
                        "adaptive",
                        Patterns::Selection("adaptive|uniform"),
                        "'adaptive' refines via KellyErrorEstimator + refine_and_coarsen_fixed_number "
                        "(FractionToRefine/FractionToCoarsen). 'uniform' does a single global "
                        "refinement of every cell each cycle (triangulation.refine_global(1)) -- the "
                        "textbook mesh-convergence-study method, useful as a cross-check against "
                        "'adaptive' since global refinement isn't solution-dependent and has a clean "
                        "theoretical convergence rate to compare against.");

      prm.declare_entry("FractionToRefine",
                        "0.3",
                        Patterns::Double(0, 1),
                        "Fraction of cells (by estimated error) to flag for refinement each cycle");

      prm.declare_entry("FractionToCoarsen",
                        "0.03",
                        Patterns::Double(0, 1),
                        "Fraction of cells (by estimated error) to flag for coarsening each cycle");
    }
    prm.leave_subsection();
    prm.enter_subsection("OutputParameters");
    {
      prm.declare_entry("OutputDirectory",
                        "output",
                        Patterns::DirectoryName(),
                        "Directory to store output files");

      prm.declare_entry("OutputFileNamePrefix",
                        "solution",
                        Patterns::Anything(),
                        "Prefix for output file names");

      prm.declare_entry("OutputFrequency",
                        "10",
                        Patterns::Integer(1),
                        "Frequency of output (in terms of number of steps)");
    }
    prm.leave_subsection();
    prm.enter_subsection("InputFiles");
      {
        prm.declare_entry("MeshFile",
                          "mesh.msh",
                          Patterns::Anything(),
                          "Input mesh file in .msh format");
        prm.declare_entry("CellToMaterialFile",
                          "cell_to_material.txt",
                          Patterns::Anything(),
                          "Input file that maps each cell to a material ID");
        prm.declare_entry("OrientationFile",
                          "orientation.txt",
                          Patterns::Anything(),
                          "Input file that provides orientation matrices for each cell");
      }
    prm.leave_subsection();
    prm.print_parameters("tmp_sol.json", ParameterHandler::JSON);
  }

  void ParameterReader::read_parameters(const std::string &parameter_file)
    {
        declare_parameters();
        prm.parse_input(parameter_file);
    }

  template <int dim>
  class ElasticityTensor
  {
  public:
    ElasticityTensor() = default;
    void setFromPrm(const ParameterHandler &prm);
    SymmetricTensor<4,dim,double> getElasticityTensor(double curOmega, double curTemperature) const;

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

    kelvin_matrix[3][3] = C44;
    kelvin_matrix[4][4] = C44;

    kelvin_matrix[5][5] = C66;

    SymmetricTensor<4, dim, double> tmpTensor;
    Physics::Notation::Kelvin::to_tensor(kelvin_matrix, tmpTensor); //I should double-check notations (Kelvin, Voight, etc.) to make sure the conversion is correct, and the order of components in the matrix is correct
    return tmpTensor;
  }

  template <int dim>
  class CteTensor
  {
  public:
    CteTensor() = default;
    void setFromPrm(const ParameterHandler &prm);
    SymmetricTensor<2,dim,double> getcteTensor(double curTemperature);
  private:
    std::unique_ptr<Function<1>> alpha11_func;
    std::unique_ptr<Function<1>> alpha33_func;
    double alpha11_0;
    double alpha33_0;
  };

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

  template <int dim>
  SymmetricTensor<2,dim,double> CteTensor<dim>::getcteTensor(double curTemperature)
  {
    SymmetricTensor<2, dim, double> cteTensor;
    Point<1> tempPoint(curTemperature);
    cteTensor[0][0] = this->alpha11_0 * this->alpha11_func->value(tempPoint);
    cteTensor[1][1] = this->alpha11_0 * this->alpha11_func->value(tempPoint);
    cteTensor[2][2] =this-> alpha33_0 * this->alpha33_func->value(tempPoint);
    return cteTensor; // inference of damage on thermal expansion is not considered in current implementation, but it can be added if needed
  }

  // WorkStream scratch/copy data. ScratchData is per-thread working memory (reused across the
  // cells that thread handles); CopyData carries one cell's result to the serialized copier.
  // Both need a copy constructor because WorkStream clones the prototype once per thread, and
  // FEValues is not copyable -- hence the reconstruct-from-fe/quadrature/flags idiom used here
  // (the same pattern deal.II's own step-32 uses).
  namespace AssemblyScratch
  {
    template <int dim>
    struct Scratch
    {
      Scratch(const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
              const UpdateFlags flags)
        : fe_values(fe, quadrature, flags)
        , globalElasticTensor_q(quadrature.size())
        , JxW_q(quadrature.size())
        , shapeGradSymm(fe.n_dofs_per_cell(),
                        std::vector<SymmetricTensor<2, dim>>(quadrature.size()))
      {}

      Scratch(const Scratch &s)
        : fe_values(s.fe_values.get_fe(), s.fe_values.get_quadrature(),
                    s.fe_values.get_update_flags())
        , globalElasticTensor_q(s.globalElasticTensor_q.size())
        , JxW_q(s.JxW_q.size())
        , shapeGradSymm(s.shapeGradSymm)
      {}

      FEValues<dim> fe_values;
      std::vector<SymmetricTensor<4, dim>> globalElasticTensor_q;
      std::vector<double> JxW_q;
      std::vector<std::vector<SymmetricTensor<2, dim>>> shapeGradSymm;
    };

    struct Copy
    {
      FullMatrix<double> cell_matrix;
      Vector<double> cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
    };
  } // namespace AssemblyScratch

  namespace DamageScratch
  {
    template <int dim>
    struct Scratch
    {
      Scratch(const FiniteElement<dim> &fe, const Quadrature<dim> &quadrature,
              const UpdateFlags flags)
        : fe_values(fe, quadrature, flags)
        , global_displacement_gradients(quadrature.size())
      {}

      Scratch(const Scratch &s)
        : fe_values(s.fe_values.get_fe(), s.fe_values.get_quadrature(),
                    s.fe_values.get_update_flags())
        , global_displacement_gradients(s.global_displacement_gradients.size())
      {}

      FEValues<dim> fe_values;
      std::vector<Tensor<2, dim>> global_displacement_gradients;
    };

    // Empty: the damage update writes only into the current cell's own quadrature-point data,
    // which no other cell touches, so there is nothing to hand to a serialized copier.
    // WorkStream still requires the type to exist.
    struct Copy
    {};
  } // namespace DamageScratch

  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem(ParameterHandler &prm);
    void run();
    void meshTest();
    void elasticCalculationTest();
    void elasticAndDamageTest();
    void printProblemParametersToJson(const std::string filename);

  private:

    void setup_system();
    // Public entry points dispatch to a serial or a WorkStream implementation based on
    // ParallelizationParameters/UseWorkStream. The serial versions are kept deliberately (not
    // as dead code): they are the reference for A/B correctness checking and for measuring the
    // actual speedup, and a threading bug is much easier to localize with a known-good path to
    // diff against.
    // assemble_system() does the setup common to both paths (constraints, zeroing) and then
    // dispatches the CELL LOOP to one of these two.
    void assemble_system(double curTemperature);
    void assemble_cells_serial(double curTemperature);
    void assemble_cells_workstream(double curTemperature);
    void solve();
    //void predict_damage_tempInc(double curTemperature, double tempInc);
    void predict_damage_tempInc_external_solver(double curTemperature, double tempInc);
    void predict_damage_cells_serial(double curTemperature, double tempInc,
                                      double stressThreshold, double A_param, double m_param,
                                      double residualTolerance);
    void predict_damage_cells_workstream(double curTemperature, double tempInc,
                                          double stressThreshold, double A_param, double m_param,
                                          double residualTolerance);
    // Per-quadrature-point damage update, shared verbatim by both paths above so the two cannot
    // drift apart. Writes only to this cell's own quadrature data, which is what makes the
    // threaded version safe without a copier.
    void update_damage_on_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      FEValues<dim> &fe_values,
      std::vector<Tensor<2, dim>> &global_displacement_gradients,
      double curTemperature, double tempInc,
      double stressThreshold, double A_param, double m_param, double residualTolerance);
    void calculateSolutionTemperatureStep(double curT, double dT, unsigned int refinementCycle, int temperatureStepNumber);
    void refine_grid(double fractionToRefine, double fractionToCoarse);
    void output_results(const unsigned int cycle = 1) const;
    void process_solution(unsigned int refinementCycle, const unsigned int tempStepNum, const unsigned int iterSolverStepNum, double curT);
    void parse_cellToMaterial_data(std::string fileName);

    bool checkActivationCriteria(SymmetricTensor<2,dim> localStress);
    // Plain per-point value snapshot, NOT a copy of the CellDataStorage object itself:
    // CellDataStorage's implicit copy constructor/assignment only deep-copies its
    // std::map<CellId, std::vector<std::shared_ptr<DataType>>> -- the shared_ptrs inside are
    // copied (refcounted), not the DamageQData objects they point to. So
    // `CellDataStorage<...> a = b;` leaves `a` and `b` aliasing the SAME underlying damage
    // objects: mutating one through get_data() mutates both. Found via maxDamageInc always
    // printing exactly 0 in checkConvergenceCriteria's output -- confirmed against deal.II's
    // quadrature_point_data.h, which declares no custom copy ctor/assignment for
    // CellDataStorage, so the compiler-generated member-wise copy (shallow on the shared_ptrs)
    // is what actually runs. A real "previous iteration" snapshot needs the scalar values
    // copied out into independent storage instead -- see snapshotDamage().
    using DamageSnapshot = std::map<typename Triangulation<dim>::cell_iterator, std::vector<double>>;
    DamageSnapshot snapshotDamage();
    bool checkConvergenceCriteria(double curT,
                                    const Vector<double> &oldSolDisplacement,
                                    const DamageSnapshot &oldDamageSnapshot);

    //double solve_kinetic_equation(SymmetricTensor<2,dim> localStress, double old_damage, double tempInc);
    double kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc);
    double derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc);

    void assignBoundaryIds();
    void constrainRigidBodyModes(AffineConstraints<double> &constraintsToAddTo);

    ElasticityTensor<dim> elasticityTensor;
    CteTensor<dim> cteTensor;

    Tensor<2,dim,double> getRotTensorGlobalToLocal(int cellId);
    SymmetricTensor<2,dim> getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature);

    std::istringstream parseMSHandPrepareStream(std::string mshFileName);

    ParameterHandler &prm;

    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
  
    std::vector<Tensor<2, dim>> rotmatOrientationCell;
    std::vector<int> cellToMaterial;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                DamageQData> damageInQuadraturePoints;

    const FESystem<dim> fe;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solDisplacement;
    Vector<double> system_rhs;

    const FEValuesExtractors::Vector displacement{0};
    TimerOutput timer;

    AppliedAxialDisplacement<dim> appliedDisplacement;
    ConvergenceTable convergence_table;

  };

  template <int dim>
  void ElasticProblem<dim>::printProblemParametersToJson(const std::string filename)
  {
    prm.print_parameters(filename, ParameterHandler::JSON);
  }

  template <int dim>
  ElasticProblem<dim>::ElasticProblem(ParameterHandler &prm)
    : prm(prm)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
  {};

  template <int dim>
  std::istringstream ElasticProblem<dim>::parseMSHandPrepareStream(std::string mshFileName) //Designed to take output Neper .msh file, parse and clear additional data, so output stream wil be applyable for read_msh
  {
    std::ifstream msh_file(mshFileName);
    std::ofstream outFile("test.msh");
    std::ostringstream out;
    std::string line;

    while (std::getline(msh_file, line))
    {
      if(line == "$ElsetOrientations")
      {
        std::getline(msh_file, line);
        std::istringstream header_stream(line);
        unsigned int num_elsets;
        std::string orientation_type;
        if (!(header_stream >> num_elsets >> orientation_type)) throw std::runtime_error("Invalid $ElsetOrientations header format");
        for (unsigned int i = 0; i < num_elsets; ++i)
        {
          if (!std::getline(msh_file, line)) throw std::runtime_error("Premature EOF in $ElsetOrientations section");
          if (line.empty())
          {
              --i; // retry this iteration
              continue;
          }
          std::istringstream data_stream(line);
          unsigned int elset_id;
          Tensor<2,dim> m;
          if (!(data_stream >> elset_id >> m[0][0] >> m[0][1] >> m[0][2] >> m[1][0] >> m[1][1] >> m[1][2] >> m[2][0] >> m[2][1] >> m[2][2])) throw std::runtime_error("Invalid orientation data line: " + line);
          rotmatOrientationCell.push_back(m);
        }
        std::getline(msh_file, line);
        std::getline(msh_file, line);
      }
      out << line << '\n';
    }
  if (outFile.is_open())
  {
    outFile << out.str(); // Get content as a string and write
    outFile << "lol" << '\n';
    outFile.close();
  } else{
    cout << "well, it was not opened\n";
  }
  return std::istringstream(out.str());
  }
  
  template <int dim>
  void ElasticProblem<dim>::parse_cellToMaterial_data(std::string fileName)
  {
    std::ifstream msh_file(fileName);
    std::string line;

    while (std::getline(msh_file, line)) 
    {
      std::istringstream header_stream(line);
      unsigned int materialId;
      if (!(header_stream >> materialId)) throw std::runtime_error("bad cell_to_material file");
      cellToMaterial.push_back(materialId);
    }
  }

  template <int dim>
  SymmetricTensor<2,dim> ElasticProblem<dim>::getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature) //provide localStrain!!!
  {
    double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    return elasticityTensor.getElasticityTensor(curOmega, curTemperature) * (localStrain - cteTensor.getcteTensor(curTemperature) * (curTemperature - T0)); 
  }

  template <int dim>
  Tensor<2,dim,double> ElasticProblem<dim>::getRotTensorGlobalToLocal(int cellId)
  {
    Assert(dim == 3, ExcNotImplemented());
    return rotmatOrientationCell[cellId - 1];
  }

  template <int dim>
  bool ElasticProblem<dim>::checkActivationCriteria(SymmetricTensor<2,dim> localStress)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    return localStress[dim-1][dim-1] > stressThreshold;
  }

  // Backward-Euler residual of the Rabotnov-type kinetic equation
  //   domega/ds = A * (macaulay(sigma_zz - sigma_th)/sigma_th)^m * (1/(1-omega))^m
  // discretized as omega_new - omega_old = A*(...)^m * tempInc, where s = T_0 - T is the
  // (positive, increasing) pseudo-time and tempInc is its step magnitude.
  //
  // Two corrections relative to the version this file was branched from, both found while
  // debugging the MMS test (see ../mmsSandbox):
  //  - actingStress is Macaulay-clamped rather than throwing: the root search evaluates the
  //    residual across a whole bracket, including omega values where the stress is subcritical,
  //    so a hard throw there would abort on perfectly legitimate probe evaluations. Below
  //    threshold the physical damage rate is genuinely zero, which the clamp expresses directly.
  //  - actingStress is normalized by stressThreshold, matching the kinetic law as written above
  //    (and as derived in the notebook). It previously was not here, while
  //    derivativeKineticResidual's power term WAS normalized -- so the residual and its own
  //    derivative disagreed by a factor of stressThreshold^m. Invisible until damage actually
  //    activated, which never happened before this round of testing.
  template <int dim>
  double ElasticProblem<dim>::kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    double actingStress = std::max(localStress[dim-1][dim-1] - stressThreshold, 0.0);
    double a_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    return curOmega - oldOmega
           - a_kineticParam * std::pow(actingStress / stressThreshold / (1-curOmega), m_kineticParam) * tempInc;
  };

  template <int dim>
  double ElasticProblem<dim>::derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    double a_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    double actingStress = std::max(localStress[dim-1][dim-1] - stressThreshold, 0.0);
    return 1 - a_kineticParam * m_kineticParam * tempInc
               * std::pow(actingStress / stressThreshold, m_kineticParam)
               / std::pow(1 - curOmega, m_kineticParam + 1);
  }

  template <int dim>
  void ElasticProblem<dim>::assignBoundaryIds()
  {
    const double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    // Scaled to the mesh, not the domain: the requirement is "much closer to the boundary plane
    // than to the next parallel face plane", which is a cell-spacing property. A domain-scaled
    // tolerance would misclassify faces once cells get small (this runs again after each
    // refinement cycle, where cells shrink but L does not).
    const double tol = 1e-6 * GridTools::minimal_cell_diameter(triangulation);

    unsigned int nFixed = 0, nLoaded = 0, nLateral = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
      {
        if (!face->at_boundary()) continue;
        const double zc = face->center()[dim-1];
        if (std::abs(zc) < tol)          { face->set_boundary_id(BoundaryIds::fixedZFace);   ++nFixed; }
        else if (std::abs(zc - L) < tol) { face->set_boundary_id(BoundaryIds::loadedZFace);  ++nLoaded; }
        else                             { face->set_boundary_id(BoundaryIds::lateralFaces); ++nLateral; }
      }

    // Assert rather than trust: a mesh that is not the axis-aligned cube spanning [0,L]^3 this
    // test assumes would otherwise silently produce a nonsense load case (e.g. everything
    // classified lateral => no Dirichlet data at all => singular system).
    if (nFixed == 0 || nLoaded == 0 || nFixed != nLoaded)
      throw std::runtime_error("unexpected z-face counts (fixed=" + std::to_string(nFixed) +
                                ", loaded=" + std::to_string(nLoaded) +
                                ") -- is the mesh an axis-aligned cube spanning [0,L]^3 with "
                                "LengthOfTheBody=" + std::to_string(L) + "?");
    std::cout << "Boundary faces: fixed(z=0)=" << nFixed << ", loaded(z=L)=" << nLoaded
              << ", lateral=" << nLateral << std::endl;
  }

  // Removes the rigid-body modes that the z-only Dirichlet data leaves free: translation in x
  // and y, and rotation about z. Deliberately pins individual VERTICES rather than whole faces --
  // constraining a full face in x/y would suppress the lateral contraction that the closed-form
  // solution (and the traction-free lateral boundary) depends on.
  //
  // Pinning u_x and u_y at one vertex kills both translations; pinning u_x at a second vertex
  // offset along y kills the remaining rotation about z. Both vertices are on the z=0 plane so
  // they do not fight the applied axial displacement.
  template <int dim>
  void ElasticProblem<dim>::constrainRigidBodyModes(AffineConstraints<double> &constraintsToAddTo)
  {
    const double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    const double tol = 1e-6 * GridTools::minimal_cell_diameter(triangulation);

    const Point<dim> originVertex(0.0, 0.0, 0.0);   // pin u_x, u_y here
    const Point<dim> yAxisVertex(0.0, L, 0.0);      // pin u_x here as well

    bool foundOrigin = false, foundYAxis = false;
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto v : cell->vertex_indices())
      {
        const Point<dim> &vertex = cell->vertex(v);
        const bool isOrigin = (vertex.distance(originVertex) < tol);
        const bool isYAxis  = (vertex.distance(yAxisVertex) < tol);
        if (!isOrigin && !isYAxis) continue;

        // Components to pin: x (and y only at the origin vertex).
        const unsigned int nComponentsToPin = isOrigin ? 2 : 1;
        for (unsigned int d = 0; d < nComponentsToPin; ++d)
        {
          const types::global_dof_index dofIndex = cell->vertex_dof_index(v, d);
          // A DoF already constrained (e.g. by hanging nodes) must not be re-added.
          if (!constraintsToAddTo.is_constrained(dofIndex))
          {
            constraintsToAddTo.add_line(dofIndex);
            constraintsToAddTo.set_inhomogeneity(dofIndex, 0.0);
          }
        }
        if (isOrigin) foundOrigin = true;
        if (isYAxis)  foundYAxis = true;
      }

    if (!foundOrigin || !foundYAxis)
      throw std::runtime_error("could not locate the rigid-body-constraint vertices (origin found="
                                + std::to_string(foundOrigin) + ", y-axis found="
                                + std::to_string(foundYAxis) + ") -- mesh geometry does not match "
                                "the assumed [0,L]^3 cube");
  }


  template <int dim>
  void ElasticProblem<dim>::setup_system() // safe to call again after mesh refinement: clears and
                                            // freshly reinitializes damageInQuadraturePoints (damage
                                            // resets to 0 on the new mesh -- refinement does not
                                            // transfer quadrature-point data, by design for now)
  {
    elasticityTensor.setFromPrm(prm);
    cteTensor.setFromPrm(prm);
    appliedDisplacement.set_prm_consts(prm);
    TimerOutput::Scope timer_section(timer, "Setup of system");
    const QGauss<dim> quadrature_formula( fe.degree + 1);
    const unsigned int n_q_points    = quadrature_formula.size();
    // Without this, repeated setup_system() calls (once per refinement cycle) would leave
    // orphaned entries for cells that are no longer active (refined away or coarsened into
    // their parent) sitting in damageInQuadraturePoints alongside the fresh ones.
    damageInQuadraturePoints.clear();
    damageInQuadraturePoints.initialize(triangulation.begin_active(),
                                          triangulation.end(),
                                          n_q_points);

    for (auto &cell : dof_handler.active_cell_iterators())
    {
      damageInQuadraturePoints.initialize(cell, quadrature_formula.size());
    }

    dof_handler.distribute_dofs(fe);
    solDisplacement.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

  }

  template <int dim>
  void ElasticProblem<dim>::assemble_system(double curTemperature)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");
    // No body force in this test: the stress state is uniform, so div(sigma)=0 is satisfied with
    // b=0 identically (that is the whole point of the uniaxial setup -- see the file header).

    double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    appliedDisplacement.set_current_temperature(curTemperature);

    // Only u_z is constrained on the two z-faces; x/y are left free so the specimen can contract
    // laterally (a fully-constrained face would impose a triaxial state and invalidate the
    // closed-form reference this test checks against). The lateral faces get no Dirichlet data at
    // all -- they are naturally traction-free.
    const FEValuesExtractors::Scalar zComponent(dim - 1);
    const ComponentMask zMask = fe.component_mask(zComponent);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             BoundaryIds::fixedZFace,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             zMask);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             BoundaryIds::loadedZFace,
                                             appliedDisplacement,
                                             constraints,
                                             zMask);

    // Rigid-body modes: constraining only u_z above leaves x/y translation and rotation about z
    // free, so the stiffness matrix would be singular. Pin the in-plane motion of a single
    // vertex (not a whole face, which would over-constrain the lateral contraction the
    // closed-form solution depends on). One vertex kills both translations; a second vertex's
    // u_y kills the remaining rotation about z.
    constrainRigidBodyModes(constraints);

    constraints.close();

    // AffineConstraints::distribute_local_to_global() ADDS local contributions into
    // system_matrix/system_rhs rather than overwriting them. assemble_system() is called
    // repeatedly (once per displacement-damage iteration, and again inside
    // checkConvergenceCriteria), so without resetting here every call keeps accumulating
    // on top of all previous calls -- system_matrix/system_rhs would grow unboundedly
    // over a temperature step instead of reflecting only the current state.
    system_matrix = 0;
    system_rhs    = 0;

    if (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream"))
      assemble_cells_workstream(curTemperature);
    else
      assemble_cells_serial(curTemperature);
  }

  // Reference single-threaded cell loop. Kept as the correctness/timing baseline for the
  // WorkStream version below -- the two must produce bit-identical matrices.
  template <int dim>
  void ElasticProblem<dim>::assemble_cells_serial(double curTemperature)
  {
    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(fe, quadrature_formula,
      update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Per-(cell, q) data that does NOT depend on the (i, j) test/trial function indices:
    // hoisted out of the i,j loop and evaluated once per q instead of dofs_per_cell^2
    // (stiffness) / dofs_per_cell (rhs) times per q -- a rank-4 tensor rotation is expensive
    // enough that this was the dominant assembly cost before.
    std::vector<SymmetricTensor<4,dim>> globalElasticTensor_q(n_q_points);
    std::vector<double> JxW_q(n_q_points);
    std::vector<std::vector<SymmetricTensor<2,dim>>> shapeGradSymm(
      dofs_per_cell, std::vector<SymmetricTensor<2,dim>>(n_q_points));

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2,dim> rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      const Tensor<2,dim> rotTensorLocalToGlobal = transpose(rotTensorGlobalToLocal);

      cell_matrix = 0;
      cell_rhs    = 0;

      const SymmetricTensor<2,dim> localcteTensor = cteTensor.getcteTensor(curTemperature);
      const SymmetricTensor<2,dim> globalcteTensor =
        Physics::Transformations::basis_transformation(localcteTensor, rotTensorLocalToGlobal);

      for (const unsigned int q : fe_values.quadrature_point_indices())
      {
        JxW_q[q] = fe_values.JxW(q);
        const SymmetricTensor<4,dim> localElasticTensor =
          elasticityTensor.getElasticityTensor(cell_damage[q]->damage, curTemperature);
        globalElasticTensor_q[q] =
          Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
      }

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int q : fe_values.quadrature_point_indices())
          shapeGradSymm[i][q] = symmetrize(fe_values[displacement].gradient(i, q));

      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            cell_matrix(i, j) +=
              (shapeGradSymm[i][q] * (globalElasticTensor_q[q] * shapeGradSymm[j][q])) * JxW_q[q];

        // Thermal term retained for structural compatibility; CTE is 0 in this test's
        // input.json, so it contributes nothing. No body-force term (see assemble_system).
        for (const unsigned int q : fe_values.quadrature_point_indices())
          cell_rhs(i) +=
            ((shapeGradSymm[i][q] * (globalElasticTensor_q[q] * globalcteTensor)) * (curTemperature - T0)) * JxW_q[q];
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  }

  // Multi-threaded cell loop. The worker is pure per-cell work writing only into its own
  // CopyData; the copier -- which touches the shared system_matrix/system_rhs -- is run
  // serialized by WorkStream, so no explicit locking is needed.
  template <int dim>
  void ElasticProblem<dim>::assemble_cells_workstream(double curTemperature)
  {
    const double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const UpdateFlags flags =
      update_values | update_gradients | update_quadrature_points | update_JxW_values;

    auto worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                      AssemblyScratch::Scratch<dim> &scratch,
                      AssemblyScratch::Copy &copy)
    {
      FEValues<dim> &fe_values = scratch.fe_values;
      fe_values.reinit(cell);
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const Tensor<2,dim> rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      const Tensor<2,dim> rotTensorLocalToGlobal = transpose(rotTensorGlobalToLocal);

      copy.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      copy.cell_rhs.reinit(dofs_per_cell);
      copy.local_dof_indices.resize(dofs_per_cell);
      copy.cell_matrix = 0;
      copy.cell_rhs    = 0;

      const SymmetricTensor<2,dim> localcteTensor = cteTensor.getcteTensor(curTemperature);
      const SymmetricTensor<2,dim> globalcteTensor =
        Physics::Transformations::basis_transformation(localcteTensor, rotTensorLocalToGlobal);

      for (const unsigned int q : fe_values.quadrature_point_indices())
      {
        scratch.JxW_q[q] = fe_values.JxW(q);
        const SymmetricTensor<4,dim> localElasticTensor =
          elasticityTensor.getElasticityTensor(cell_damage[q]->damage, curTemperature);
        scratch.globalElasticTensor_q[q] =
          Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
      }

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int q : fe_values.quadrature_point_indices())
          scratch.shapeGradSymm[i][q] = symmetrize(fe_values[displacement].gradient(i, q));

      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
          for (const unsigned int q : fe_values.quadrature_point_indices())
            copy.cell_matrix(i, j) +=
              (scratch.shapeGradSymm[i][q] *
               (scratch.globalElasticTensor_q[q] * scratch.shapeGradSymm[j][q])) * scratch.JxW_q[q];

        for (const unsigned int q : fe_values.quadrature_point_indices())
          copy.cell_rhs(i) +=
            ((scratch.shapeGradSymm[i][q] * (scratch.globalElasticTensor_q[q] * globalcteTensor)) *
             (curTemperature - T0)) * scratch.JxW_q[q];
      }
      cell->get_dof_indices(copy.local_dof_indices);
    };

    auto copier = [&](const AssemblyScratch::Copy &copy)
    {
      constraints.distribute_local_to_global(copy.cell_matrix, copy.cell_rhs,
                                             copy.local_dof_indices, system_matrix, system_rhs);
    };

    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), worker, copier,
                    AssemblyScratch::Scratch<dim>(fe, quadrature_formula, flags),
                    AssemblyScratch::Copy());
  }

  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    TimerOutput::Scope timer_section(timer, "Solve system");
    const double linearSolverTolerance = prm.get_double({"SolverParameters"}, "LinearSolverTolerance");
    SolverControl            solver_control(1000, linearSolverTolerance * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.2));

    cg.solve(system_matrix,  solDisplacement, system_rhs, preconditioner);

    // A plain runtime check rather than Assert(): Assert() compiles to a no-op in Release
    // builds (-DNDEBUG), so relying on it here would mean a genuinely non-converged CG
    // solve could silently pass through in Release with no indication anything went wrong.
    if (solver_control.last_check() != SolverControl::success)
      throw std::runtime_error("CG solver did not converge (last residual " +
                                std::to_string(solver_control.last_value()) + ")");

    constraints.distribute(solDisplacement);
  }

  template <int dim>
  void ElasticProblem<dim>::refine_grid(double fractionToRefine, double fractionToCoarse)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    // KellyErrorEstimator estimates jumps in the solution gradient across cell *faces*,
    // hence a (dim-1)-dimensional face quadrature rule, not a dim-dimensional cell one.
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                        solDisplacement,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    fractionToRefine,
                                                    fractionToCoarse);

    // No quadrature-point data transfer across refinement/coarsening (yet -- see
    // setup_system()'s comment): damage is reset to 0 on the new mesh via the
    // setup_system() call that follows this function at every call site. This is
    // exact (not an approximation) for any state that starts a refinement cycle at
    // damage=0 everywhere, e.g. a fresh MMS mesh-convergence run; it would lose real
    // accumulated damage if refinement were ever triggered mid-simulation on a run
    // where damage is already nonzero, which is not yet supported.
    triangulation.execute_coarsening_and_refinement();
    setup_system();
  }

  // template <int dim>
  // void ElasticProblem<dim>::predict_damage_tempInc(double curTemperature, double tempInc)
  // {
  //   SymmetricTensor<2, dim> globalStrain;
  //   SymmetricTensor<2, dim> localStrain;
  //   SymmetricTensor<2, dim> localStress;

  //   QGauss<dim> quadrature_formula(fe.degree + 1);
  //   FEValues<dim> fe_values(fe, quadrature_formula,
  //                           update_values | update_gradients | update_quadrature_points);
  //   const unsigned int n_q_points = quadrature_formula.size();
  //   Tensor<2,dim> rotTensorGlobalToLocal;
  //   for (auto &cell : dof_handler.active_cell_iterators())
  //   {
  //     auto cell_damage = damageInQuadraturePoints.get_data(cell);
  //     rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());

  //     fe_values.reinit(cell);
  //     std::vector<Tensor<2, dim>> global_displacement_gradients(quadrature_formula.size());
  //     global_displacement_gradients.resize(quadrature_formula.size());
  //     fe_values[displacement].get_function_gradients(solDisplacement, global_displacement_gradients); //check if it is calculated right, maybe specific test for this function?
  //     for (unsigned int q = 0; q < n_q_points; q++)
  //     {

  //       globalStrain = dealii::symmetrize(global_displacement_gradients[q]); //is it right? double-check!
  //       localStrain = Physics::Transformations::basis_transformation(globalStrain, rotTensorGlobalToLocal);
  //       double old_damage = cell_damage[q]->old_damage;
  //       localStress = getLocalStressTensor(localStrain, old_damage, curTemperature);
  //       if (!checkActivationCriteria(localStress)) continue;
  //       cell_damage[q]->damage = solve_kinetic_equation(localStress, old_damage, tempInc);
  //     } 
  //   }
  // }

  template <int dim>
  void ElasticProblem<dim>::predict_damage_tempInc_external_solver(double curTemperature, double tempInc)
  {
    TimerOutput::Scope timer_section(timer, "Predict damage increment");

    // Kinetic-equation constants, read once rather than per quadrature point (this loop is the
    // dominant cost in the timing report), then passed down to the per-cell worker.
    const double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    const double A_param = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    const double m_param = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    const double residualTolerance = prm.get_double({"SolverParameters"}, "KineticAlgebraicSolverTolerance");

    if (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream"))
      predict_damage_cells_workstream(curTemperature, tempInc, stressThreshold, A_param,
                                       m_param, residualTolerance);
    else
      predict_damage_cells_serial(curTemperature, tempInc, stressThreshold, A_param,
                                   m_param, residualTolerance);
    // Damage-vs-reference deviation is tracked in process_solution() and reported as
    // convergence_table columns each iteration.
  }

  // The per-cell damage update, shared verbatim by the serial and threaded paths so they cannot
  // drift apart. Writes only into this cell's own quadrature-point data -- no other cell reads or
  // writes it -- which is what makes the threaded version safe with no copier and no locking.
  // (damageInQuadraturePoints::get_data() is a const map lookup here; concurrent lookups are safe
  // because nothing inserts during the loop.)
  template <int dim>
  void ElasticProblem<dim>::update_damage_on_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    FEValues<dim> &fe_values,
    std::vector<Tensor<2, dim>> &global_displacement_gradients,
    double curTemperature, double tempInc,
    double stressThreshold, double A_param, double m_param, double residualTolerance)
  {
    SymmetricTensor<2, dim> globalStrain;
    SymmetricTensor<2, dim> localStrain;
    SymmetricTensor<2, dim> localStress;

    auto cell_damage = damageInQuadraturePoints.get_data(cell);
    const Tensor<2, dim> rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
    fe_values.reinit(cell);
    const auto &qPoints = fe_values.get_quadrature_points();
    const unsigned int n_q_points = fe_values.get_quadrature().size();

    // One call fills the gradients at ALL quadrature points. This used to sit inside the q loop
    // below, so every point's gradients were recomputed n_q_points (=8) times over -- pure
    // redundant work, identical results.
    fe_values[displacement].get_function_gradients(solDisplacement, global_displacement_gradients);

    for (unsigned int q = 0; q < n_q_points; q++)
    {
        Point<dim> curQpoint = qPoints[q];
        double old_damage = cell_damage[q]->old_damage;
        double cur_damage = cell_damage[q]->damage;
        globalStrain = dealii::symmetrize(global_displacement_gradients[q]);
        localStrain = Physics::Transformations::basis_transformation(globalStrain, rotTensorGlobalToLocal);
        localStress = getLocalStressTensor(localStrain, cur_damage, curTemperature);

        // The real, self-consistent kinetic equation: the driving stress is the REAL stress
        // computed from the current numerical damage, not a prescribed field. That closes the
        // negative-feedback loop this whole test exists to exercise (growing damage lowers
        // C33 -> lowers sigma_zz -> slows further growth), which the MMS construction lacked.
        const double lowerBound = old_damage; // damage is non-decreasing within a load step
        const double upperBound = 0.999;      // stay clear of the (1-omega)->0 singularity
        auto residualOnly = [&](double omega) -> double
        {
          // Stress is re-evaluated at each trial omega, so the root search sees the true
          // coupled equation rather than a stress frozen at the previous iterate.
          const SymmetricTensor<2,dim> trialStress = getLocalStressTensor(localStrain, omega, curTemperature);
          return kineticResidual(trialStress, omega, old_damage, tempInc);
        };

        // The damage term A*(ratio)^m*(1-omega)^-m is convex in omega whenever actingStress>0,
        // making the residual CONCAVE (single interior maximum, up to two roots) -- not safe for
        // raw Newton from an arbitrary guess. Locate that maximum analytically rather than
        // searching for it, then bracket-and-solve on whichever side actually contains a sign
        // change, preferring the smaller root (the branch continuous with old_damage).
        //
        // NOTE: the closed form below is specific to this Rabotnov-type power law -- it solves
        // d/domega[A*(ratio)^m*(1-omega)^-m * tempInc] = 1 analytically. A different kinetic law
        // would need its own derivation, or a law-agnostic fallback (numeric maximization, or a
        // coarse scan for sign changes). Worth revisiting if a second law is ever added.
        //
        // Here `ratio` uses the stress at old_damage purely to locate the bracket; the residual
        // itself always re-evaluates stress at the trial omega (above). Since sigma_zz decreases
        // monotonically in omega under displacement control, this over-estimates the damage
        // term's growth and so places omega_max conservatively -- fine for bracketing, and the
        // sign-change checks below validate the bracket regardless.
        const double actingStressAtPoint =
          std::max(getLocalStressTensor(localStrain, old_damage, curTemperature)[dim-1][dim-1] - stressThreshold, 0.0);
        double newDamage, lo = lowerBound, hi = upperBound;
        bool rootBracketed = true;
        boost::uintmax_t bisectItersUsed = 0;

        if (actingStressAtPoint <= 0.0)
        {
          // Subcritical: no damage growth this step. Unlike the MMS version there is no
          // manufactured forcing term to balance, so the residual reduces to omega - old_damage
          // and the root is exactly old_damage. (Stress only decreases as omega grows under
          // displacement control, so subcritical at old_damage means subcritical throughout.)
          newDamage = old_damage;
        }
        else
        {
          const double omega_max = std::clamp(
            1.0 - std::pow(m_param * A_param * std::pow(actingStressAtPoint / stressThreshold, m_param) * tempInc,
                            1.0 / (m_param + 1.0)),
            lowerBound, upperBound);
          const double residAtLower = residualOnly(lowerBound);
          const double residAtMax = residualOnly(omega_max);
          if (omega_max > lowerBound && (residAtLower < 0.0) != (residAtMax < 0.0))
          { lo = lowerBound; hi = omega_max; }
          else if (omega_max < upperBound && (residAtMax < 0.0) != (residualOnly(upperBound) < 0.0))
          { lo = omega_max; hi = upperBound; }
          else
            rootBracketed = false; // no sign change anywhere -- no root exists in [lowerBound, upperBound]

          if (rootBracketed)
          {
            // toms748_solve: bracket-guaranteed, superlinear. Tight x-tolerance here -- an
            // x-tolerance equal to residualTolerance isn't the same thing as a residual
            // tolerance near the (1-omega)^-m singularity, so the residual check below is the
            // real acceptance criterion.
            boost::uintmax_t bisectIters = 200;
            auto bracket = boost::math::tools::toms748_solve(residualOnly, lo, hi,
              [](double a, double b) { return std::abs(b - a) < 1e-12; }, bisectIters);
            bisectItersUsed = bisectIters;
            newDamage = 0.5 * (bracket.first + bracket.second);
          }
          else
            newDamage = lowerBound;
        }

        const double residualAtSolution = residualOnly(newDamage);
        if (!rootBracketed || std::abs(residualAtSolution) > residualTolerance)
          throw std::runtime_error("Kinetic equation root search did not converge (residual=" +
                                    std::to_string(residualAtSolution) +
                                    ", rootBracketed=" + std::to_string(rootBracketed) +
                                    ") at point " +
                                    std::to_string(curQpoint[0]) + "," + std::to_string(curQpoint[1]) +
                                    "," + std::to_string(curQpoint[2]) +
                                    " [old_damage=" + std::to_string(old_damage) +
                                    " newDamage=" + std::to_string(newDamage) +
                                    " T=" + std::to_string(curTemperature) +
                                    " tempInc=" + std::to_string(tempInc) +
                                    " actingStress=" + std::to_string(actingStressAtPoint) +
                                    " bracket=[" + std::to_string(lo) + "," + std::to_string(hi) + "]" +
                                    " residAtBracket=[" + std::to_string(residualOnly(lo)) + "," +
                                    std::to_string(residualOnly(hi)) + "]" +
                                    " bisectItersUsed=" + std::to_string(bisectItersUsed) + "]");
        cell_damage[q]->damage = newDamage;
    }
  }

  // Reference single-threaded loop, kept as the correctness/timing baseline.
  template <int dim>
  void ElasticProblem<dim>::predict_damage_cells_serial(
    double curTemperature, double tempInc,
    double stressThreshold, double A_param, double m_param, double residualTolerance)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points);
    std::vector<Tensor<2, dim>> global_displacement_gradients(quadrature_formula.size());

    for (const auto &cell : dof_handler.active_cell_iterators())
      update_damage_on_cell(cell, fe_values, global_displacement_gradients, curTemperature,
                            tempInc, stressThreshold, A_param, m_param, residualTolerance);
  }

  // Multi-threaded loop. No copier is needed (see update_damage_on_cell): each cell owns its
  // quadrature data outright. Note that update_damage_on_cell can throw when the root search
  // fails; WorkStream propagates exceptions out of the worker, so that diagnostic still surfaces
  // rather than being swallowed -- though with several cells in flight the failing cell is not
  // necessarily the first one that would have failed serially.
  template <int dim>
  void ElasticProblem<dim>::predict_damage_cells_workstream(
    double curTemperature, double tempInc,
    double stressThreshold, double A_param, double m_param, double residualTolerance)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const UpdateFlags flags = update_values | update_gradients | update_quadrature_points;

    auto worker = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                      DamageScratch::Scratch<dim> &scratch,
                      DamageScratch::Copy &)
    {
      update_damage_on_cell(cell, scratch.fe_values, scratch.global_displacement_gradients,
                            curTemperature, tempInc, stressThreshold, A_param, m_param,
                            residualTolerance);
    };
    auto copier = [](const DamageScratch::Copy &) {};

    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), worker, copier,
                    DamageScratch::Scratch<dim>(fe, quadrature_formula, flags),
                    DamageScratch::Copy());
  }

  // template <int dim>
  // double ElasticProblem<dim>::solve_kinetic_equation(SymmetricTensor<2,dim> localStress, double old_damage, double tempInc)
  // {
  //   double omegaPrevIter = old_damage;
  //   double dOmega = -1.0 * kineticResidual(localStress, omegaPrevIter, old_damage, tempInc) /  derivativeKineticResidual(localStress, omegaPrevIter, tempInc);
  //   double omegaNextIter = omegaPrevIter + dOmega;
  //   double resOmega = abs(omegaPrevIter - omegaNextIter);
  //   int nIter = 0;
  //   while (resOmega > epsilonDamageIntegration)
  //   {
  //     nIter++;
  //     if (nIter > maxIterationDamageIntegration) throw std::runtime_error("max number of iterations for kinetic damage equation reached!");
  //     omegaPrevIter = omegaNextIter;
  //     dOmega = -1.0 * kineticResidual(localStress, omegaPrevIter, old_damage, tempInc) /  derivativeKineticResidual(localStress, omegaPrevIter, tempInc);
  //     omegaNextIter = omegaPrevIter + dOmega;
  //     resOmega = abs(omegaPrevIter - omegaNextIter);
  //   }
  //   return omegaNextIter;
  // }

  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FEValues<dim> fe_values(fe, quadrature_formula,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string>  solDisplacement_names;
    switch (dim)
      {
        case 1:
           solDisplacement_names.emplace_back("displacement");
          break;
        case 2:
           solDisplacement_names.emplace_back("x_displacement");
           solDisplacement_names.emplace_back("y_displacement");
          break;
        case 3:
           solDisplacement_names.emplace_back("x_displacement");
           solDisplacement_names.emplace_back("y_displacement");
           solDisplacement_names.emplace_back("z_displacement");
          break;
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

    data_out.add_data_vector( solDisplacement,  solDisplacement_names);

    Vector<double> cell_damage(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;                     // essential in parallel computations

        fe_values.reinit(cell);

        // Retrieve the per‑quadrature‑point history
        const auto qp_damage =
            damageInQuadraturePoints.get_data(cell);

        double average = 0.0;
        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
            average += qp_damage[q]->damage;   // dereference the shared_ptr
        average /= quadrature_formula.size();

        // Index by active cell index is mandatory for DataOut
        cell_damage[cell->active_cell_index()] = average;
    }
    data_out.add_data_vector(cell_damage, "damage");

    data_out.build_patches();

    std::ofstream output("solDisplacement-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }

  template <int dim>
  void ElasticProblem<dim>::run()
  {
    double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    double Tend =  prm.get_double({"ModelParameters"}, "FinalTemperature");
    double dT = prm.get_double({"ModelParameters"}, "TemperatureIncrement");

    // Thread cap for WorkStream. 0 keeps deal.II's default (all available cores); an explicit
    // value is mainly for scaling measurements (run the same problem at 1, 2, 4, ... threads).
    // Note this caps deal.II's task scheduler as a whole, not just our two loops.
    const unsigned int requestedThreads =
      prm.get_integer({"ParallelizationParameters"}, "NumberOfThreads");
    if (requestedThreads > 0)
      MultithreadInfo::set_thread_limit(requestedThreads);
    std::cout << "Threading: "
              << (prm.get_bool({"ParallelizationParameters"}, "UseWorkStream")
                    ? "WorkStream" : "serial (reference path)")
              << ", n_threads=" << MultithreadInfo::n_threads()
              << " (hardware concurrency " << MultithreadInfo::n_cores() << ")" << std::endl;

    std::string mshFileName = prm.get({"InputFiles"}, "MeshFile");
    std::string cellToMaterialFileName = prm.get({"InputFiles"}, "CellToMaterialFile");
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName);
    auto mshStream = parseMSHandPrepareStream(mshFileName);
    grid_in.read_msh(mshStream); //it will read only one phsycial tag, related to id of cell (so u need to use special map material[cellId] for material, rotation have direct relation to cell id)
    parse_cellToMaterial_data(cellToMaterialFileName);

    //here manually setting one main orientation for all cells, in general case it should be read from msh file and pushed in
    //it is done for test purposes, to check if damage is predicted correctly for one orientation
    Tensor<1, 3> axis;
    axis[0] = 0.0;
    axis[1] = 1.0;
    axis[2] = 0.0;

    const double angle = numbers::PI / 2.0 * 0;

    auto rotation_matrix = Physics::Transformations::Rotations::rotation_matrix_3d(axis, angle);

    // parseMSHandPrepareStream() (above) already pushed whatever real crystallite orientation
    // the mesh file's $ElsetOrientations section carries. Since every cell's material_id is 1,
    // getRotTensorGlobalToLocal(1) resolves to rotmatOrientationCell[0] -- so without clearing
    // here, that mesh-file orientation (not the identity this test intends) is what actually
    // gets used, silently. Clear so the manual override below lands at index 0 as intended.
    rotmatOrientationCell.clear();
    rotmatOrientationCell.push_back(rotation_matrix);

    // getRotTensorGlobalToLocal indexes rotmatOrientationCell[material_id - 1] with no bounds
    // checking of its own (a plain std::vector::operator[]) -- an out-of-range material_id is
    // silent undefined behavior in a Release build, not a caught error. This is exactly the
    // mechanism behind a real bug found and fixed in this run (parseMSHandPrepareStream() had
    // already pushed the mesh file's real crystallite orientation at index 0 before the manual
    // override above; every cell's material_id happens to be 1, so that mesh-file orientation
    // -- not the intended identity -- was silently what got used). Check eagerly here so any
    // future mismatch (e.g. a multi-material mesh not matched by rotmatOrientationCell's size)
    // fails loudly instead of reading garbage.
    for (const auto &cell : triangulation.active_cell_iterators())
    {
      const auto materialId = cell->material_id();
      if (materialId < 1 || materialId > rotmatOrientationCell.size())
        throw std::runtime_error("cell material_id " + std::to_string(materialId) +
                                  " is out of range for rotmatOrientationCell (size " +
                                  std::to_string(rotmatOrientationCell.size()) + ")");
    }

    // Boundary ids, assigned geometrically: the mesh file carries no surface elements at all, so
    // every face arrives with the default id 0 (verified by inspecting the .msh directly).
    // Called once here rather than per refinement cycle -- child faces inherit their parent's
    // boundary id, so refinement preserves this classification.
    assignBoundaryIds();

    // Outer mesh-refinement-cycle loop, for mesh convergence studies: each cycle refines
    // (cycle 0 just does the initial setup) and then re-solves the *entire* temperature
    // range from scratch on the new mesh. Damage does not carry over between cycles (see
    // refine_grid()/setup_system()), so this is for damage-free/single-cycle MMS studies,
    // not yet for refining mid-simulation on a run with already-accumulated damage.
    const unsigned int nRefinementCycles = prm.get_integer({"MeshRefinementParameters"}, "NumberOfRefinementCycles");
    const std::string refinementStrategy = prm.get({"MeshRefinementParameters"}, "RefinementStrategy");
    const double fractionToRefine = prm.get_double({"MeshRefinementParameters"}, "FractionToRefine");
    const double fractionToCoarsen = prm.get_double({"MeshRefinementParameters"}, "FractionToCoarsen");
    const unsigned int outputFrequency = prm.get_integer({"OutputParameters"}, "OutputFrequency");

    for (unsigned int refinementCycle = 0; refinementCycle < nRefinementCycles; ++refinementCycle)
    {
      if (refinementCycle == 0)
        setup_system(); // was previously missing: without this call, dof_handler has zero DoFs,
                         // damageInQuadraturePoints/material property functors are never initialized,
                         // and system_matrix/rhs/solution vectors stay unsized (see cubeTest.cpp's
                         // meshTest()/elasticCalculationTest() for the pattern this followed)
      else if (refinementStrategy == "uniform")
      {
        triangulation.refine_global(1);
        setup_system();
      }
      else
        refine_grid(fractionToRefine, fractionToCoarsen);

      // Step count computed up front, and curT derived as T0 - n*dT rather than accumulated by
      // repeated subtraction. The previous `do { curT -= dT; } while (curT > Tend)` form had two
      // problems: (a) rounding accumulated in curT, and (b) that accumulated error routinely left
      // curT a few ulp above Tend after the nominal final step, triggering one EXTRA step that
      // ran past the end temperature. Both are real: a step-size sweep over
      // dT = 0.1/0.05/0.025/0.0125/0.00625 produced 11/20/40/81/161 steps instead of
      // 10/20/40/80/160, so different step sizes silently simulated to different final states
      // (dT=0.1 ended at pseudo-time 1.1, not 1.0) -- which invalidates any convergence study
      // comparing them.
      //
      // A range not divisible by dT shortens the final step to land exactly on Tend rather than
      // overshooting, and that shortened increment is what gets passed to the kinetic equation.
      const double temperatureRange = T0 - Tend;
      const unsigned int nTemperatureSteps =
        static_cast<unsigned int>(std::ceil(temperatureRange / dT - 1e-9));
      for (unsigned int temperatureStepNumber = 1;
           temperatureStepNumber <= nTemperatureSteps;
           ++temperatureStepNumber)
      {
        // curT is the END-of-step temperature: the step is solved at the new state, not the old
        // one, matching genuine implicit (backward-Euler) semantics -- the kinetic law's RHS
        // belongs at the new state. T0 itself is the known initial condition and is never solved
        // for. (Getting this backwards previously meant step 1 always evaluated everything at
        // t=0, which was masked for a long time by two other bugs that kept the kinetic-equation
        // path from running at all.)
        const double prevT = std::max(T0 - (temperatureStepNumber - 1) * dT, Tend);
        const double curT  = std::max(T0 - temperatureStepNumber * dT, Tend);
        const double actualTempInc = prevT - curT;

        calculateSolutionTemperatureStep(curT, actualTempInc, refinementCycle, temperatureStepNumber);

        // OutputFrequency was declared and documented but never actually consulted -- every step
        // wrote a full field dump regardless. That is a real cost for fine-step runs (e.g. the
        // sweep above writes hundreds of ~450KB files nobody looks at). The final step is always
        // written, whatever the frequency, so the end state is never missing.
        if (temperatureStepNumber % outputFrequency == 0 ||
            temperatureStepNumber == nTemperatureSteps)
          output_results(refinementCycle * 1000 + temperatureStepNumber);
      }
    }

    convergence_table.set_precision("epsZZ", 5);
    convergence_table.set_precision("damage", 6);
    convergence_table.set_precision("damageSpread", 3);
    convergence_table.set_precision("sigmaZZ", 4);
    convergence_table.set_precision("sigmaZZerr", 3);

    convergence_table.set_scientific("damageSpread", true);
    convergence_table.set_scientific("sigmaZZ", true);
    convergence_table.set_scientific("sigmaZZerr", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("epsZZ", "@f$\\varepsilon_{zz}@f$ applied");
    convergence_table.set_tex_caption("damage", "@f$\\omega@f$ (mean)");
    convergence_table.set_tex_caption("damageSpread", "@f$\\omega@f$ spread");
    convergence_table.set_tex_caption("sigmaZZ", "@f$\\sigma_{zz}@f$");
    convergence_table.set_tex_caption("sigmaZZerr", "@f$|\\sigma_{zz}^{FE}-\\sigma_{zz}^{exact}|@f$");

    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");
 
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
  }

  template <int dim>
  void ElasticProblem<dim>::calculateSolutionTemperatureStep(double curT, double dT, unsigned int refinementCycle, int temperatureStepNumber)
  {
    // "prev*" tracks the *previous inner iteration's* state, so the increment-based
    // convergence checks measure iteration-to-iteration change (as their tolerances
    // assume) rather than the total change since the start of this temperature step.
    // On iteration 1, prev* is still the end-of-previous-temperature-step state, which
    // is the correct reference point for that first increment.
    Vector<double> prevSolDisplacement = solDisplacement;
    DamageSnapshot prevDamageSnapshot = snapshotDamage();
    const unsigned int maxIterations = prm.get_integer({"SolverParameters"}, "MaxNumberOfIterations");
    unsigned int dispDamageIterationNumber = 0;
    bool converged = false;
    do
    {
      dispDamageIterationNumber++;
      assemble_system(curT);
      solve();
      predict_damage_tempInc_external_solver(curT, dT);
      process_solution(refinementCycle, temperatureStepNumber, dispDamageIterationNumber, curT);

      converged = checkConvergenceCriteria(curT, prevSolDisplacement, prevDamageSnapshot);

      prevSolDisplacement = solDisplacement;
      prevDamageSnapshot = snapshotDamage();

      if (!converged && dispDamageIterationNumber >= maxIterations)
        throw std::runtime_error("Displacement-damage iteration did not converge within "
                                  "MaxNumberOfIterations = " + std::to_string(maxIterations) +
                                  " iterations at temperature step " + std::to_string(temperatureStepNumber));
    } while (!converged);

    // The kinetic equation's residual (see mmsKineticResidual/predict_damage_tempInc_external_solver)
    // integrates the damage increment against old_damage, i.e. the damage at the *start* of the
    // current temperature step. That baseline must be advanced once this step has converged,
    // otherwise every subsequent temperature step keeps integrating from the previous old_damage
    // (stuck at 0.0 for the whole run, since nothing else ever assigns to it).
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; ++q)
        cell_damage[q]->old_damage = cell_damage[q]->damage;
    }

  }

  template <int dim>
  typename ElasticProblem<dim>::DamageSnapshot ElasticProblem<dim>::snapshotDamage()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    DamageSnapshot snapshot;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      std::vector<double> values(quadrature_formula.size());
      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        values[q] = cell_damage[q]->damage;
      snapshot[cell] = std::move(values);
    }
    return snapshot;
  }

  template <int dim>
  bool ElasticProblem<dim>::checkConvergenceCriteria(double curT,
                                    const Vector<double> &oldSolDisplacement,
                                    const DamageSnapshot &oldDamageSnapshot)
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);
    Vector<double> solutionIncrement = solDisplacement;
    solutionIncrement -= oldSolDisplacement;
    double solutionIncrementNorm = solutionIncrement.l2_norm();
    double solutionNorm = solDisplacement.l2_norm();
    double relativeSolutionIncrementNorm = solutionIncrementNorm / solutionNorm;

    double maxDamageIncrement = 0.0; // Damage is self-limited to [0,1], so max norm is more informative than L2 norm
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      const auto &old_cell_damage = oldDamageSnapshot.at(cell);
      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
      {
        double damageIncrement = cell_damage[q]->damage - old_cell_damage[q];
        maxDamageIncrement = std::max(maxDamageIncrement, std::abs(damageIncrement));
      }
    }

    assemble_system(curT);
    Vector<double> systemResidual(system_matrix.m());
    system_matrix.residual(systemResidual, solDisplacement, system_rhs);
    // Constrained-DOF rows carry a placeholder diagonal that distribute_local_to_global()
    // writes purely to keep the matrix well-conditioned for the preconditioner -- with no
    // matching right-hand-side entry, since those rows are never meant to be interpreted as
    // real equilibrium equations (their value is fixed directly via constraints.distribute(),
    // not solved for). Left in, a handful of O(1e10) placeholder-diagonal rows swamp the
    // residual norm and it plateaus at a large, meaningless constant regardless of how well
    // the free-DOF system actually solved. Zero them out before taking the norm, matching the
    // standard deal.II idiom (e.g. step-15's Newton residual) for excluding constrained DOFs
    // from a residual/error norm.
    constraints.set_zero(systemResidual);
    double systemResidualNorm = systemResidual.l2_norm();
    double systemRhsNorm = system_rhs.l2_norm(); // should check, that this norm will not be too small;
    double relativeSystemResidualNorm  = systemResidualNorm / systemRhsNorm;

    double solIncrementEps = prm.get_double({"SolverParameters"}, "DisplacementIncrementRelativeTolerance");
    double damageIncrementEps = prm.get_double({"SolverParameters"}, "DamageIncrementRelativeTolerance");
    double forseResidualEps = prm.get_double({"SolverParameters"}, "ForceEqulibriumRelativeTolerance");

    std::cout << "    convergence check: relSolInc=" << relativeSolutionIncrementNorm
              << " (tol " << solIncrementEps << "), maxDamageInc=" << maxDamageIncrement
              << " (tol " << damageIncrementEps << "), relResidual=" << relativeSystemResidualNorm
              << " (tol " << forseResidualEps << ")" << std::endl;

    if (relativeSolutionIncrementNorm < solIncrementEps and maxDamageIncrement < damageIncrementEps and relativeSystemResidualNorm < forseResidualEps){return true;}
    else {return false;}
  }

  template <int dim>
  void ElasticProblem<dim>::process_solution(unsigned int refinementCycle, const unsigned int tempStepNum, const unsigned int iterSolverStepNum, double curT)
  {
    // Reference solution for this uniaxial test. Because the stress state is uniform and only
    // C33 carries the damage factor, damage and stress are spatially CONSTANT here (up to
    // solver error), so the checks below report spread across quadrature points (which should
    // stay ~0 -- any real spatial variation means something is wrong) alongside the value
    // itself. Displacement is not compared against a manufactured solution -- there is none in
    // this test; the FE solution of a constant-strain field is exact by construction, and the
    // PDE solver was already verified by the MMS test in ../mmsSandbox.
    const double C11_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C11_0");
    const double C12_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C12_0");
    const double C13_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C13_0");
    const double C33_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C33_0");

    appliedDisplacement.set_current_temperature(curT);
    const double epsZZApplied = appliedDisplacement.applied_axial_strain();

    // sigma_zz(omega, eps_zz) = eps_zz * [C33_0*(1-omega) - 2*C13_0^2/(C11_0+C12_0)],
    // derived from sigma_xx = sigma_yy = 0 (traction-free lateral faces) with eps_xx = eps_yy.
    auto sigmaZZClosedForm = [&](double omega) {
      return epsZZApplied * (C33_0*(1.0 - omega) - 2.0*C13_0*C13_0/(C11_0 + C12_0));
    };

    double damageMin = std::numeric_limits<double>::max();
    double damageMax = std::numeric_limits<double>::lowest();
    double damageMean = 0.0, totalVolume = 0.0;
    double sigmaZZNumMin = std::numeric_limits<double>::max();
    double sigmaZZNumMax = std::numeric_limits<double>::lowest();
    double maxStressVsClosedForm = 0.0;
    {
      const QGauss<dim> quadrature_formula(fe.degree + 1);
      FEValues<dim> fe_values(fe, quadrature_formula,
                              update_gradients | update_quadrature_points | update_JxW_values);
      std::vector<Tensor<2,dim>> displacementGradients(quadrature_formula.size());
      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        auto cell_damage = damageInQuadraturePoints.get_data(cell);
        fe_values[displacement].get_function_gradients(solDisplacement, displacementGradients);
        const Tensor<2,dim> rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        {
          const double omega = cell_damage[q]->damage;
          damageMin = std::min(damageMin, omega);
          damageMax = std::max(damageMax, omega);
          damageMean += omega * fe_values.JxW(q);
          totalVolume += fe_values.JxW(q);

          const SymmetricTensor<2,dim> localStrain = Physics::Transformations::basis_transformation(
            dealii::symmetrize(displacementGradients[q]), rotTensorGlobalToLocal);
          const double sigmaZZNumeric = getLocalStressTensor(localStrain, omega, curT)[dim-1][dim-1];
          sigmaZZNumMin = std::min(sigmaZZNumMin, sigmaZZNumeric);
          sigmaZZNumMax = std::max(sigmaZZNumMax, sigmaZZNumeric);
          // Consistency of the FE stress with the closed-form relation at the SAME omega. This
          // isolates the elastic/coupling algebra from the ODE integration error: it should hold
          // to solver tolerance regardless of whether omega itself is accurate.
          maxStressVsClosedForm =
            std::max(maxStressVsClosedForm, std::abs(sigmaZZNumeric - sigmaZZClosedForm(omega)));
        }
      }
    }
    damageMean /= totalVolume;
    const double damageSpread = damageMax - damageMin;
    const double sigmaZZSpread = sigmaZZNumMax - sigmaZZNumMin;
    const double sigmaZZAtMeanDamage = sigmaZZClosedForm(damageMean);

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();

    std::cout << "Refinement cycle " << refinementCycle << ", pseudo-time step " << tempStepNum
              << ", displacement-damage iteration " << iterSolverStepNum << ':' << std::endl
              << "   Cells: " << n_active_cells << ", DoFs: " << n_dofs << std::endl
              << "   eps_zz applied = " << epsZZApplied << std::endl
              << "   damage: mean=" << damageMean << " spread=" << damageSpread << std::endl
              << "   sigma_zz: numeric mean~" << 0.5*(sigmaZZNumMin + sigmaZZNumMax)
              << " spread=" << sigmaZZSpread
              << ", closed-form(mean damage)=" << sigmaZZAtMeanDamage << std::endl
              << "   max|sigma_zz_FE - sigma_zz_closedform| = " << maxStressVsClosedForm << std::endl;

    convergence_table.add_value("cycle", refinementCycle);
    convergence_table.add_value("tempStep", tempStepNum);
    convergence_table.add_value("iter", iterSolverStepNum);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("epsZZ", epsZZApplied);
    convergence_table.add_value("damage", damageMean);
    convergence_table.add_value("damageSpread", damageSpread);
    convergence_table.add_value("sigmaZZ", 0.5*(sigmaZZNumMin + sigmaZZNumMax));
    convergence_table.add_value("sigmaZZerr", maxStressVsClosedForm);
  }

} // namespace StepCooling

int main()
{
  try
    {
      StepCooling::ParameterHandler prm;
      StepCooling::ParameterReader myReader(prm);
      myReader.read_parameters("input.json");
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
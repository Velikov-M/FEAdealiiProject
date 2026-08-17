// Main features to add:
// - realize the parallel computation using dealii library native parallelization features (MPI, multithreading)
// - realize the one-source-of-truth approach for the data management (e.g., using a single HDF5 file to store all the data)
// - rewrite operations with material data, prepare it for parallelization and one-source-of-truth approach
// - add adaptive mesh refinement

// Unfortunatly, the current implementation of the code does not support automatic MMS testing,
// so boundary and initial conditions, additional body force and kinetic eqation additional term should be changed manually for each specific MMS test
// the MMS routine should be done in order:
// 0) pick and derive analytical MMS formulas via SageMath .ipynb file
// 1) change precise solution function (boundary conditions will be updated automatically) in the code
// 2) change body force
// 3) change kinetic equation additional term
// 4) run the code and check if the solution is close to the exact solution, if not, go back to step 0 and change the MMS formulas
// initial conditions for displacement are not needed, because the problem is quasi-static

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

  template <int dim>
  class PreciseDisplacementSolution : public Function<dim>
  {
    public:
      PreciseDisplacementSolution();
      virtual void vector_value(const Point<dim> &p,
                                Vector<double>   &values) const override;
      virtual void vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>>   &value_list) const override;

      void set_prm_consts(const ParameterHandler &prm);
      void set_current_temperature(const double curT);

    private:  
      double L, T_0, T_end, curT;

  };

  template <int dim>
  PreciseDisplacementSolution<dim>::PreciseDisplacementSolution()
    : Function<dim>(dim)
  {
    this->L = 0.0;
    this->T_0 = 0.0;
    this->T_end = 0.0;
    this->curT = 0.0;
  }

  template <int dim>
  void PreciseDisplacementSolution<dim>::vector_value(const Point<dim> & p,
                                           Vector<double> &values) const
  {
    Assert(dim == 3, ExcNotImplemented());
    AssertDimension(values.size(), dim);

    double t = (T_0 - curT) / (T_0 - T_end);
    values(0) = cos(M_PI*p[0]/L) * cos(M_PI*p[1]/L) * cos(M_PI*p[2]/L) * exp(-t);
    values(1) = sin(M_PI*p[0]/L) * cos(M_PI*p[1]/L) * cos(M_PI*p[2]/L) * exp(-t);
    values(2) = sin(M_PI*p[0]/L) * sin(M_PI*p[1]/L) * cos(M_PI*p[2]/L) * exp(-t);
  }

  template <int dim>
  void PreciseDisplacementSolution<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>>   &value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      PreciseDisplacementSolution<dim>::vector_value(points[p], value_list[p]);
  }

  template <int dim>
  void PreciseDisplacementSolution<dim>::set_prm_consts(const ParameterHandler &prm)
  {
    this->L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    this->T_0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    this->T_end = prm.get_double({"ModelParameters"}, "FinalTemperature");
  }

  template <int dim>
  void PreciseDisplacementSolution<dim>::set_current_temperature(const double curT)
  {
    this->curT = curT;
  }

  template <int dim>
  class MmsBodyForce : public Function<dim>
  {
  public:
    MmsBodyForce();
    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override;
    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override;
    void set_prm_consts(const ParameterHandler &prm);
    void set_current_temperature(const double curT);
  private:
    double L, T_0, T_end, C11_0, C33_0, C44_0, C12_0, C13_0, alpha11, alpha33, curT;
  };
 
 
  template <int dim>
  MmsBodyForce<dim>::MmsBodyForce()
    : Function<dim>(dim)
  {}

  template <int dim>
  inline void MmsBodyForce<dim>::vector_value(const Point<dim> & p,
                                           Vector<double> &values) const
  {
    Assert(dim == 3, ExcNotImplemented());
    AssertDimension(values.size(), dim);
    double t = (T_0 - curT) / (T_0 - T_end); 
    const double bx = (1.0/4.0)*(2*(2*pow(M_PI, 2)*C13_0 + pow(M_PI, 2)*C44_0)*sin(M_PI*p[1]/L)*sin(M_PI*p[2]/L)*cos(M_PI*p[0]/L) + ((pow(M_PI, 2)*C11_0 + 3*pow(M_PI, 2)*C12_0)*sin(M_PI*p[1]/L)*cos(M_PI*p[0]/L) + (5*pow(M_PI, 2)*C11_0 - pow(M_PI, 2)*C12_0 + 2*pow(M_PI, 2)*C44_0)*cos(M_PI*p[0]/L)*cos(M_PI*p[1]/L))*cos(M_PI*p[2]/L))*exp(-t)/pow(L, 2);
    const double by = (1.0/4.0)*(2*(2*pow(M_PI, 2)*C13_0 + pow(M_PI, 2)*C44_0)*sin(M_PI*p[0]/L)*sin(M_PI*p[2]/L)*cos(M_PI*p[1]/L) + (-(pow(M_PI, 2)*C11_0 + 3*pow(M_PI, 2)*C12_0)*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L) + (5*pow(M_PI, 2)*C11_0 - pow(M_PI, 2)*C12_0 + 2*pow(M_PI, 2)*C44_0)*sin(M_PI*p[0]/L)*cos(M_PI*p[1]/L))*cos(M_PI*p[2]/L))*exp(-t)/pow(L, 2);
    const double bz = -1.0/2.0*(1.0*pow(M_PI, 2)*C33_0*sqrt(t)*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L)*cos(M_PI*p[2]/L) - 2*(pow(M_PI, 2)*C33_0 + pow(M_PI, 2)*C44_0)*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L)*cos(M_PI*p[2]/L) + ((2*pow(M_PI, 2)*C13_0 + pow(M_PI, 2)*C44_0)*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L) + (2*pow(M_PI, 2)*C13_0 + pow(M_PI, 2)*C44_0)*sin(M_PI*p[0]/L)*cos(M_PI*p[1]/L))*sin(M_PI*p[2]/L))*exp(-t)/pow(L, 2);
    values(0) = bx;
    values(1) = by;
    values(2) = bz;
  }
 
  template <int dim>
  void MmsBodyForce<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>>   &value_list) const
  {
    const unsigned int n_points = points.size();
 
    AssertDimension(value_list.size(), n_points);
 
    for (unsigned int p = 0; p < n_points; ++p)
      MmsBodyForce<dim>::vector_value(points[p], value_list[p]);
  }

  template <int dim>
  void MmsBodyForce<dim>::set_prm_consts(const ParameterHandler &prm)
  {
    this->L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    this->T_0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    this->T_end = prm.get_double({"ModelParameters"}, "FinalTemperature");
    this->C11_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C11_0");
    this->C33_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C33_0");
    this->C44_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C44_0");
    this->C12_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C12_0");
    this->C13_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C13_0");
    this->alpha11 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha11_0");
    this->alpha33 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha33_0");
  }

  template <int dim>
  void MmsBodyForce<dim>::set_current_temperature(const double curT)
  {
    this->curT = curT;
  }

  template <int dim>
  class boundaryDisplacementFunction : public Function<dim> //need to add actual temperature dependence
  {
    public:
      boundaryDisplacementFunction(): Function<dim>(dim),
      currentTemperature(2000.0),
      initialTemperature(2000.0)
      {}

      void setCurrentTemperature(const double cT){currentTemperature = cT;}
      void setInitialTemperarure(const double t0){initialTemperature = t0;}

      virtual void vector_value(const Point<dim> &p,
                         Vector<double>   &values) const override
      {
        Assert(dim == 3, ExcNotImplemented());
        const Tensor<2, dim, double> cteEff = 4.0e-6 * unit_symmetric_tensor<dim, double>();
        Tensor<1,dim> coordVec;
        for (unsigned int d=0; d<dim; ++d) coordVec[d] = p[d];
        const Tensor<1,dim> u = (cteEff *(currentTemperature - initialTemperature))* coordVec;
        for (unsigned int d=0; d<dim; ++d) values(d) = u[d];
      }
    private:
      double currentTemperature;
      double initialTemperature;

  };

  template <int dim>
  class MmsBoundaryDisplacementFunction : public Function<dim> // boundary displacement function for method of manufactured solutions (MMS) testing
  {
    public:
      MmsBoundaryDisplacementFunction(const PreciseDisplacementSolution<dim>& ps): Function<dim>(dim),
      preciseSolution(ps)
      {}

      virtual void vector_value(const Point<dim> &p,
                         Vector<double>   &values) const override
      { 
        Vector<double> u_exact(dim);
        preciseSolution.vector_value(p, u_exact);
        for (unsigned int d=0; d<dim; ++d) values(d) = u_exact(d);
      }
    private:
      const PreciseDisplacementSolution<dim>& preciseSolution;

  };

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
    void assemble_system(double curTemperature);
    void solve();
    //void predict_damage_tempInc(double curTemperature, double tempInc);
    void predict_damage_tempInc_external_solver(double curTemperature, double tempInc);
    void calculateSolutionTemperatureStep(double curT, double dT, int temperatureStepNumber);
    void refine_grid();
    void output_results(const unsigned int cycle = 1) const;
    void process_solution(const unsigned int tempStepNum, const unsigned int iterSolverStepNum);
    void parse_cellToMaterial_data(std::string fileName);

    bool checkActivationCriteria(SymmetricTensor<2,dim> localStress);
    bool checkConvergenceCriteria(double curT,
                                    const Vector<double> &oldSolDisplacement,
                                    const CellDataStorage<typename Triangulation<dim>::cell_iterator, DamageQData> &oldDamageInQuadraturePoints);

    //double solve_kinetic_equation(SymmetricTensor<2,dim> localStress, double old_damage, double tempInc);
    double kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc);
    double derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc);

    double mmsKineticResidual(const Point<dim> &p, const SymmetricTensor<2,dim> &localStress, double curOmega, double oldOmega, double curT, double tempInc);
    //double mmsDerivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc);

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

    PreciseDisplacementSolution<dim> preciseSolution;
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

  template <int dim>
  double ElasticProblem<dim>::kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc)
  {   
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    double actingStress = localStress[dim-1][dim-1] - stressThreshold;
    if(actingStress < 0) throw std::runtime_error("acting stress is negative, this function should've not be called");
    double a_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    return curOmega - oldOmega - a_kineticParam * std::pow((actingStress /(1-curOmega)), m_kineticParam) * tempInc;
  };

  template <int dim>
  double ElasticProblem<dim>::derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    double a_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m_kineticParam = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    double actingStress = localStress[dim-1][dim-1] - stressThreshold;
    if (actingStress < 0) throw std::runtime_error("acting stress is negative, this function should've not be called");
    return 1 - a_kineticParam * m_kineticParam * tempInc * std::pow(actingStress / (stressThreshold * (1-curOmega)), m_kineticParam) / (1-curOmega);
  }

  template <int dim>
  double ElasticProblem<dim>::mmsKineticResidual(const Point<dim> &p, const SymmetricTensor<2,dim> &localStress, double curOmega, double oldOmega, double curT, double tempInc)
  {
    double stressThreshold = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold"); //may be use external class approach
    double A = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    double C11_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C11_0");
    double C33_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C33_0");
    double C44_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C44_0");
    double C12_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C12_0");
    double C13_0 = prm.get_double({"MaterialParameters", "LinearElasticityParameters"}, "C13_0");
    double alpha_11 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha11_0");
    double alpha_33 = prm.get_double({"MaterialParameters", "ThermoElasticParameters"}, "alpha33_0");
    double T_0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    double T_end = prm.get_double({"ModelParameters"}, "FinalTemperature");
    double sigma_th = prm.get_double({"MaterialParameters", "DamageParameters"}, "StressThreshold");
    double T = curT;
    double t = (T_0 - curT) / (T_0 - T_end);

    double actingStress = localStress[dim-1][dim-1] - stressThreshold;
    Assert(actingStress > 0, ExcMessage("acting stress is negative, this function should've not be called"));
    double defaultPart = curOmega - oldOmega - A * std::pow((actingStress /(1-curOmega)), m) * tempInc;

    // should be changed for every specific MMS test, so that the exact solution will satisfy the kinetic equation with this additional part, so it can be used for testing the kinetic equation solver
    auto kineticAddTerm = [&](const Point<dim> &p, const double t) -> double
    {
      const double fadd = A*pow(-1/(0.5*sqrt(t) - 1), m)*pow(-1.0/2.0*(M_PI*C33_0*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L)*sin(M_PI*p[2]/L)*fabs(L) + L*sigma_th*exp(t)*fabs(L) - L*fabs(M_PI*C33_0*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L)*sin(M_PI*p[2]/L) + L*sigma_th*exp(t) + 2*alpha_11*(C13_0*L*T*exp(t) - C13_0*L*T_0*exp(t)) + alpha_33*(C33_0*L*T*exp(t) - C33_0*L*T_0*exp(t)) - sqrt(t)*(0.5*M_PI*C33_0*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L)*sin(M_PI*p[2]/L) + alpha_33*(0.5*C33_0*L*T*exp(t) - 0.5*C33_0*L*T_0*exp(t))) + (M_PI*C13_0*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L) + M_PI*C13_0*sin(M_PI*p[0]/L)*cos(M_PI*p[1]/L))*cos(M_PI*p[2]/L)) + 2*alpha_11*(C13_0*L*T*exp(t)*fabs(L) - C13_0*L*T_0*exp(t)*fabs(L)) + alpha_33*(C33_0*L*T*exp(t)*fabs(L) - C33_0*L*T_0*exp(t)*fabs(L)) - sqrt(t)*(0.5*M_PI*C33_0*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L)*sin(M_PI*p[2]/L)*fabs(L) + alpha_33*(0.5*C33_0*L*T*exp(t)*fabs(L) - 0.5*C33_0*L*T_0*exp(t)*fabs(L))) + (M_PI*C13_0*sin(M_PI*p[0]/L)*sin(M_PI*p[1]/L)*fabs(L) + M_PI*C13_0*sin(M_PI*p[0]/L)*cos(M_PI*p[1]/L)*fabs(L))*cos(M_PI*p[2]/L))*exp(-t)/(L*sigma_th*fabs(L)), m);
      return fadd;
    };

    double mmsResidualPart = kineticAddTerm(p, t) * tempInc; //toDO
    return mmsResidualPart + defaultPart; //mmsResidualPart should be defined in a way, that the exact solution will satisfy the kinetic equation with this additional part, so it can be used for testing the kinetic equation solver
  }

  template <int dim>
  void ElasticProblem<dim>::setup_system() //never before mesh refinement or damageInQuadraturePoints must be reinitialized
  { 
    elasticityTensor.setFromPrm(prm);
    cteTensor.setFromPrm(prm);
    preciseSolution.set_prm_consts(prm);
    TimerOutput::Scope timer_section(timer, "Setup of system");
    const QGauss<dim> quadrature_formula( fe.degree + 1);
    const unsigned int n_q_points    = quadrature_formula.size();
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
    MmsBodyForce<dim> mmsBodyForce = MmsBodyForce<dim>();
    mmsBodyForce.set_prm_consts(prm);
    mmsBodyForce.set_current_temperature(curTemperature);
    Vector<double> bodyForceValue(dim);
    Tensor<1,dim> bodyForceTensor;

    double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");

    const QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FEValues<dim> fe_values(fe, quadrature_formula,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    SymmetricTensor<4,dim> localElasticTensor;
    SymmetricTensor<4,dim> globalElasticTensor;
    SymmetricTensor<2,dim> localcteTensor;
    SymmetricTensor<2,dim> globalcteTensor;
    SymmetricTensor<2,dim> iShapeGradSymm;
    SymmetricTensor<2,dim> jShapeGradSymm;
    Tensor<2,dim> rotTensorGlobalToLocal;
    Tensor<2,dim> rotTensorLocalToGlobal;
    double damageInQpoint;
    double JxW;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    preciseSolution.set_current_temperature(curTemperature);
    auto prescribed_displacement = MmsBoundaryDisplacementFunction<dim>(preciseSolution);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             types::boundary_id(0),
                                             prescribed_displacement,
                                             constraints);
    constraints.close();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      timer.enter_subsection ("fe_values reinit");
      fe_values.reinit(cell);
      timer.leave_subsection();
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      rotTensorLocalToGlobal = transpose(rotTensorGlobalToLocal);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
        {
          for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            iShapeGradSymm = symmetrize(fe_values[displacement].gradient(i, q));
            jShapeGradSymm = symmetrize(fe_values[displacement].gradient(j, q));
            JxW = fe_values.JxW(q);
            damageInQpoint = cell_damage[q]->damage;
            timer.enter_subsection ("geting elastic tensor");
            localElasticTensor = elasticityTensor.getElasticityTensor(damageInQpoint, curTemperature);
            timer.leave_subsection();
            timer.enter_subsection ("calculating cell matrix via 4th order tensor multiplication");
            globalElasticTensor = Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
            cell_matrix(i, j) += (iShapeGradSymm * (globalElasticTensor * jShapeGradSymm))*JxW;  
            timer.leave_subsection();            
          }
        }
        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          timer.enter_subsection ("rhs calculation");
          iShapeGradSymm = symmetrize(fe_values[displacement].gradient(i, q));
          JxW = fe_values.JxW(q);
          damageInQpoint = cell_damage[q]->damage;
          localcteTensor = cteTensor.getcteTensor(curTemperature);
          globalcteTensor = Physics::Transformations::basis_transformation(localcteTensor, rotTensorLocalToGlobal);
          localElasticTensor = elasticityTensor.getElasticityTensor(damageInQpoint, curTemperature);
          globalElasticTensor = Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
          cell_rhs(i) += ((iShapeGradSymm * (globalElasticTensor * globalcteTensor)) * (curTemperature - T0)) * JxW;
          mmsBodyForce.vector_value(fe_values.quadrature_point(q), bodyForceValue);
          for (unsigned int d = 0; d < dim; ++d) bodyForceTensor[d] = bodyForceValue[d]; 
          cell_rhs(i) +=  bodyForceTensor * fe_values[displacement].value(i, q) * JxW; 
          timer.leave_subsection();      
        }
      }
      cell->get_dof_indices(local_dof_indices);
      timer.enter_subsection ("destribution to global");
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      timer.leave_subsection();  
    }
  }

  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    TimerOutput::Scope timer_section(timer, "Solve system");
    SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.2));

    cg.solve(system_matrix,  solDisplacement, system_rhs, preconditioner);

    Assert(solver_control.last_check() == SolverControl::success, ExcMessage("Solver did not converge."));

    constraints.distribute(solDisplacement);
  }

  template <int dim>
  void ElasticProblem<dim>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                        solDisplacement,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);

    triangulation.execute_coarsening_and_refinement();
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
    boost::math::tools::eps_tolerance<double> kineticAlgebraicSolverEps(prm.get_double({"SolverParameters"}, "KineticAlgebraicSolverTolerance"));

    SymmetricTensor<2, dim> globalStrain;
    SymmetricTensor<2, dim> localStrain;
    SymmetricTensor<2, dim> localStress;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points);
    const unsigned int n_q_points = quadrature_formula.size();
    Tensor<2,dim> rotTensorGlobalToLocal;
    std::vector<Tensor<2, dim>> global_displacement_gradients(quadrature_formula.size());
    global_displacement_gradients.resize(quadrature_formula.size());

    // very specific for rabotnov function, should not be considered as general approach
    double A = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");
    double m = prm.get_double({"MaterialParameters", "DamageParameters"}, "m_kineticParam");
    double omegaMax = 1 - pow(A * m *tempInc, 1 / ( m + 1) );

    for (auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());
      fe_values.reinit(cell);
      auto qPoints = fe_values.get_quadrature_points();
      for (unsigned int q = 0; q < n_q_points; q++)
      { 
        Point<dim> curQpoint = qPoints[q];
        double old_damage = cell_damage[q]->old_damage;
        double cur_damage = cell_damage[q]->damage;
        fe_values[displacement].get_function_gradients(solDisplacement, global_displacement_gradients);
        globalStrain = dealii::symmetrize(global_displacement_gradients[q]);
        localStrain = Physics::Transformations::basis_transformation(globalStrain, rotTensorGlobalToLocal);
        localStress = getLocalStressTensor(localStrain, cur_damage, curTemperature);
        if (!checkActivationCriteria(localStress)) continue;
        auto f = [&](double omega)-> double 
        {
          return mmsKineticResidual(curQpoint, localStress, omega, old_damage, curTemperature, tempInc);   
        };
        
        Assert(omegaMax > old_damage, ExcMessage("temperature increment is too big, can't use analytical omega maximum to locate root"));
        boost::uintmax_t max_iter = 100;
        std::pair<double, double> r = boost::math::tools::toms748_solve(f, omegaMax, 0.99, kineticAlgebraicSolverEps, max_iter);
        cell_damage[q]->damage = (r.first + r.second) / 2.0;
      } 
    }
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
    double curT = prm.get_double({"ModelParameters"}, "InitialTemperature");
    double Tend =  prm.get_double({"ModelParameters"}, "FinalTemperature");
    double dT = prm.get_double({"ModelParameters"}, "TemperatureIncrement");

    double testVar = prm.get_double({"MaterialParameters", "DamageParameters"}, "a_kineticParam");

    unsigned int temperatureStepNumber = 0;

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

    rotmatOrientationCell.push_back(rotation_matrix);

    do 
    {
      temperatureStepNumber++;
      calculateSolutionTemperatureStep(curT, dT, temperatureStepNumber);
      output_results(temperatureStepNumber);
      curT = curT - dT;
    } while (curT < Tend);

    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("Linfty", 3);
 
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);
 
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
    convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
 
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");
 
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
  }

  template <int dim>
  void ElasticProblem<dim>::calculateSolutionTemperatureStep(double curT, double dT, int temperatureStepNumber)
  {
    Vector<double> oldSolDisplacement = solDisplacement;
    Vector<double> solutionIncrement;
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                DamageQData> oldDamageInQuadraturePoints = damageInQuadraturePoints;
    unsigned int dispDamageIterationNumber = 0;
    do
    {
      dispDamageIterationNumber++; 
      assemble_system(curT);
      solve();
      predict_damage_tempInc_external_solver(curT, dT);
      process_solution(temperatureStepNumber, dispDamageIterationNumber);
    } while (!checkConvergenceCriteria(curT, oldSolDisplacement, oldDamageInQuadraturePoints));
  }

  template <int dim>
  bool ElasticProblem<dim>::checkConvergenceCriteria(double curT,
                                    const Vector<double> &oldSolDisplacement,
                                    const CellDataStorage<typename Triangulation<dim>::cell_iterator, DamageQData> &oldDamageInQuadraturePoints)
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
      auto old_cell_damage = oldDamageInQuadraturePoints.get_data(cell);
      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
      {
        double damageIncrement = cell_damage[q]->damage - old_cell_damage[q]->damage;
        maxDamageIncrement = std::max(maxDamageIncrement, std::abs(damageIncrement));
      }
    }

    assemble_system(curT);
    Vector<double> systemResidual(system_matrix.m());
    double systemResidualNorm = system_matrix.residual(systemResidual, solDisplacement, system_rhs);
    double systemRhsNorm = system_rhs.l2_norm(); // should check, that this norm will not be too small;
    double relativeSystemResidualNorm  = systemResidualNorm / systemRhsNorm;

    double solIncrementEps = prm.get_double({"SolverParameters"}, "DisplacementIncrementRelativeTolerance");
    double damageIncrementEps = prm.get_double({"SolverParameters"}, "DamageIncrementRelativeTolerance");
    double forseResidualEps = prm.get_double({"SolverParameters"}, "ForceEqulibriumRelativeTolerance");

    if (relativeSolutionIncrementNorm < solIncrementEps and maxDamageIncrement < damageIncrementEps and relativeSystemResidualNorm < forseResidualEps){return true;}
    else {return false;}
  }

  template <int dim>
  void ElasticProblem<dim>::process_solution(const unsigned int tempStepNum, const unsigned int iterSolverStepNum)
  {
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solDisplacement,
                                      preciseSolution,
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::L2_norm);
    const double L2_error =
    VectorTools::compute_global_error(triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
 
    VectorTools::integrate_difference(dof_handler,
                                      solDisplacement,
                                      preciseSolution,
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::H1_seminorm);
    const double H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);
 
    const QTrapezoid<1>  q_trapez;
    const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
    VectorTools::integrate_difference(dof_handler,
                                      solDisplacement,
                                      preciseSolution,
                                      difference_per_cell,
                                      q_iterated,
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::Linfty_norm);
 
    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();
 
    std::cout << "Temperature step number " << tempStepNum << "Displacement-iterative solution step number"<< iterSolverStepNum << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl;
 
    convergence_table.add_value("cycle", tempStepNum);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
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
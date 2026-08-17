
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>

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
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>
#include <deal.II/physics/notation.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>

//Right now refinement is not supported due to transfering damage is not realized, setup_system() must be used only once

namespace StepCooling
{
  using namespace dealii;

  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem();
    //void run();
    void meshTest();
    void elasticCalculationTest();
    void elasticAndDamageTest();
    void elasticDamageSelfConsistentTest(); //checking if solution will converge if we will update damage and solve elasticity with new damage prediction

  private:

    class StressPostprocessor;
    friend class StressPostprocessor;
    class DamagePostprocessor;
    friend class DamagePostprocessor;

    class StressPostprocessor : public DataPostprocessorTensor<dim>
    {
      public:
        StressPostprocessor(
          ElasticProblem<dim> &problem,const double curTemperature):
          problem(problem), curTemperature(curTemperature), 
          DataPostprocessorTensor<dim>("stress", update_gradients){}

        virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                                  std::vector<Vector<double>> &computed_quantities) const override
        {
          const auto cell = input_data.template get_cell<dim>();
          const auto cell_damage = problem.damageInQuadraturePoints.get_data(cell);

          const unsigned int n_q_points = input_data.solution_gradients.size();

          Tensor<2,dim> rotTensorGlobalToLocal;
          Tensor<2,dim> globalStrain;
          SymmetricTensor<2,dim> globalStrainSymm;
          SymmetricTensor<2,dim> localStrain;
          SymmetricTensor<2,dim> localStress;

          double damageInQpoint = 0.0;

          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Assert(input_data.solution_gradients.size() == computed_quantities.size(), ExcDimensionMismatch(input_data.solution_gradients.size(), computed_quantities.size()));
            rotTensorGlobalToLocal = problem.getRotTensorGlobalToLocal(cell->material_id());
            for (unsigned int i = 0; i <dim; ++i)
            { 
              for (unsigned int j=0; j<dim; ++j)
              {
                globalStrain[i][j]= input_data.solution_gradients[q][i][j];
              }
            }
            globalStrainSymm = dealii::symmetrize(globalStrain);
            localStrain = Physics::Transformations::basis_transformation(globalStrainSymm, rotTensorGlobalToLocal);
            double curDamage = cell_damage[q]->damage;
            localStress = problem.getLocalStressTensor(localStrain, curDamage, curTemperature);
            auto globalStress = Physics::Transformations::basis_transformation(localStress, transpose(rotTensorGlobalToLocal));
            for (unsigned int i = 0; i <dim; ++i)
            { 
              for (unsigned int j=0; j<dim; ++j)
              {
                computed_quantities[q][Tensor<2, dim>::component_to_unrolled_index(TableIndices<2>(i, j))] = globalStress[i][j];
              }
            }
          }
        }
      private:
        ElasticProblem<dim> &problem;
        const double curTemperature;
    };

  class DamagePostprocessor : public DataPostprocessorVector<dim>
  {
    public:
      DamagePostprocessor(ElasticProblem<dim> &problem):
        problem(problem),
        DataPostprocessorVector<dim>("damage", update_gradients){}

      virtual void
      evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &input_data,
        std::vector<Vector<double>> &computed_quantities) const override
      {
        const auto cell = input_data.template get_cell<dim>();
        const auto cell_damage = problem.damageInQuadraturePoints.get_data(cell);

        const unsigned int n_q_points = computed_quantities.size();
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            computed_quantities[q][0] = cell_damage[q]->damage;
          }
      }
    private:
        ElasticProblem<dim> &problem;
  };

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
    
    static constexpr double T0 = 2000;
    static constexpr double a_kineticParam = 1.0e-2;
    static constexpr double m_kineticParam = 1.0;
    static constexpr double stressThreshold = 200e6;
    static constexpr double epsilonDamageIntegration = 1e-3;
    static const int maxIterationDamageIntegration = 100;
    static constexpr double displacementTolerance = 1e-6;

    void setup_system();
    void assemble_system(double curTemperature, double incTemperature);
    void solve();
    void predict_damage_tempInc(double curTemperature, double tempInc);
    void refine_grid();
    void output_results(const unsigned int cycle, const double curTemperature);
    void parse_cellToMaterial_data(std::string fileName);
    void saveDamageAsOld();

    bool checkActivationCriteria(SymmetricTensor<2,dim> localStress);

    double solve_kinetic_equation(SymmetricTensor<2,dim> localStress, double old_damage, double tempInc);
    double kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc);
    double derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc);


    SymmetricTensor<2,dim,double> getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature);
    SymmetricTensor<4,dim,double> getElasticityTensor(double curOmega, double curTemperature);
    SymmetricTensor<2,dim,double> getcteTensor(double curTemperature);
    Tensor<2,dim,double> getRotTensorGlobalToLocal(int cellId);

    std::istringstream parseMSHandPrepareStream(std::string mshFileName);

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
    const FEValuesExtractors::Scalar x_displacement{0};
    const FEValuesExtractors::Scalar y_displacement{1};
    const FEValuesExtractors::Scalar z_displacement{2};

  };

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
  ElasticProblem<dim>::ElasticProblem()
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
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
  SymmetricTensor<4,dim,double>  ElasticProblem<dim>::getElasticityTensor(double curOmega, double curTemperature)
  {
    FullMatrix<double> kelvin_matrix(6, 6);
    double C11 = 1060*1e9;
    double C33 = 36.5*1e9;
    double C44 = 0.25*1e9;
    double C12 = 180*1e9;
    double C13 = 15*1e9;
    double C66 = (C11 - C12) / 2;

    C33 = C33 * (1 - curOmega);

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

    SymmetricTensor<4, dim, double> elasticTensor;
    Physics::Notation::Kelvin::to_tensor(kelvin_matrix, elasticTensor);

    double temperatureMultiplicator = 0.4 + 0.6 * (curTemperature - 300) / T0; 

    elasticTensor = elasticTensor * temperatureMultiplicator;
    return elasticTensor;
  }

  template <int dim>
  SymmetricTensor<2,dim,double> ElasticProblem<dim>::getcteTensor(double curTemperature)
  {
    SymmetricTensor<2, dim, double> cteTensor;
    cteTensor[0][0] = 0.1;
    cteTensor[1][1] = 0.1;
    cteTensor[2][2] = 0.1;
    return cteTensor;
  }

  template <int dim>
  SymmetricTensor<2,dim> ElasticProblem<dim>::getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature) //provide localStrain!!!
  {
    auto localElasticityTensor = getElasticityTensor(curOmega, curTemperature); 
    double dT = curTemperature - T0;
    return localElasticityTensor* (localStrain - getcteTensor(curTemperature) * dT); 
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
    return localStress[dim-1][dim-1] > stressThreshold;
  }

  template <int dim>
  double ElasticProblem<dim>::kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc)
  { 
    double actingStressFactor = (localStress[dim-1][dim-1] - stressThreshold)/stressThreshold;
    if(actingStressFactor < 0) throw std::runtime_error("acting stress is negative, this function should've not be called");
    return curOmega - oldOmega - a_kineticParam * std::pow(actingStressFactor, m_kineticParam) * tempInc;
  };

  template <int dim>
  double ElasticProblem<dim>::derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc)
  {
    double actingStressFactor = (localStress[dim-1][dim-1] - stressThreshold)/stressThreshold;
    if (actingStressFactor < 0) throw std::runtime_error("acting stress is negative, this function should've not be called");
    return 1 - a_kineticParam * m_kineticParam * tempInc * std::pow(actingStressFactor, m_kineticParam ) / (1-curOmega);
  }

  template <int dim>
  void ElasticProblem<dim>::setup_system() //never before mesh refinement or damageInQuadraturePoints must be reinitialized
  {

    const QGauss<dim> quadrature_formula( fe.degree + 1);

    const unsigned int n_q_points    = quadrature_formula.size();

    damageInQuadraturePoints.initialize(triangulation.begin_active(),
                                          triangulation.end(),
                                          n_q_points);

    for (auto &cell : dof_handler.active_cell_iterators())
    {
      damageInQuadraturePoints.initialize(cell, quadrature_formula.size());
    }

    for (auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (cell->face(f)->at_boundary())
        {
          const Point<dim> face_center = cell->face(f)->center();
          if (std::abs(face_center[0]) < 1e-12) 
          {
            cell->face(f)->set_boundary_id(1); // here 1 is for fixed boundary with zero displacement, and 2 is for boundary with prescribed non-zero displacement
          }
          if (std::abs(face_center[0] - 1.0) < 1e-12) 
          {
            cell->face(f)->set_boundary_id(2);
          }
        }
      }
    }

    dof_handler.distribute_dofs(fe);
    solDisplacement.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    //here creating displacement constraints for test case of 1d extension, in general case it should be created based on actual boundary conditions, for example read from msh file

    Functions::ZeroFunction<dim> zero_boundary_function(dim);
    Tensor<1, dim> displacementVector;
    displacementVector[0] = 1e-2;
    displacementVector[1] = 0.0;
    displacementVector[2] = 0.0;

    const ComponentMask x_mask = fe.component_mask(x_displacement);
    const ComponentMask y_mask = fe.component_mask(y_displacement);
    const ComponentMask z_mask = fe.component_mask(z_displacement);

    constraints.clear();

    VectorTools::interpolate_boundary_values(dof_handler,
                                            types::boundary_id(1),
                                            zero_boundary_function,
                                            constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                            types::boundary_id(2),
                                            Functions::ConstantFunction<dim>(displacementVector[0], dim),
                                            constraints,
                                            x_mask);
    
    VectorTools::interpolate_boundary_values(dof_handler,
                                            types::boundary_id(2),
                                            Functions::ConstantFunction<dim>(displacementVector[1], dim),
                                            constraints,
                                            y_mask);
    
    VectorTools::interpolate_boundary_values(dof_handler,
                                            types::boundary_id(2),
                                            Functions::ConstantFunction<dim>(displacementVector[2], dim),
                                            constraints,
                                            z_mask);

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
  void ElasticProblem<dim>::assemble_system(double curTemperature, double incTemperature)
  {
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

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
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
            localElasticTensor = getElasticityTensor(damageInQpoint, curTemperature);
            globalElasticTensor = Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
            cell_matrix(i, j) += (iShapeGradSymm * (globalElasticTensor * jShapeGradSymm))*JxW;              
          }
        }
        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
          iShapeGradSymm = symmetrize(fe_values[displacement].gradient(i, q));
          JxW = fe_values.JxW(q);
          damageInQpoint = cell_damage[q]->damage;
          localcteTensor = getcteTensor(curTemperature);
          globalcteTensor = Physics::Transformations::basis_transformation(localcteTensor, rotTensorLocalToGlobal);
          localElasticTensor = getElasticityTensor(damageInQpoint, curTemperature);
          globalElasticTensor = Physics::Transformations::basis_transformation(localElasticTensor, rotTensorLocalToGlobal);
          cell_rhs(i) += ((iShapeGradSymm * (globalElasticTensor * globalcteTensor)) * (curTemperature - T0)) * JxW;        
        }
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  }

  template <int dim>
  void ElasticProblem<dim>::solve()
  {
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

  template <int dim>
  void ElasticProblem<dim>::predict_damage_tempInc(double curTemperature, double tempInc)
  {
    SymmetricTensor<2, dim> globalStrain;
    SymmetricTensor<2, dim> localStrain;
    SymmetricTensor<2, dim> localStress;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_quadrature_points);
    const unsigned int n_q_points = quadrature_formula.size();
    Tensor<2,dim> rotTensorGlobalToLocal;
    for (auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      rotTensorGlobalToLocal = getRotTensorGlobalToLocal(cell->material_id());

      fe_values.reinit(cell);
      std::vector<Tensor<2, dim>> global_displacement_gradients(quadrature_formula.size());
      global_displacement_gradients.resize(quadrature_formula.size());
      fe_values[displacement].get_function_gradients(solDisplacement, global_displacement_gradients); //check if it is calculated right, maybe specific test for this function?
      for (unsigned int q = 0; q < n_q_points; q++)
      {

        globalStrain = dealii::symmetrize(global_displacement_gradients[q]); //is it right? double-check!
        localStrain = Physics::Transformations::basis_transformation(globalStrain, rotTensorGlobalToLocal);
        double old_damage = cell_damage[q]->old_damage;
        double curDamage = cell_damage[q]->damage;
        localStress = getLocalStressTensor(localStrain, curDamage, curTemperature);
        if (!checkActivationCriteria(localStress)) continue;
        cell_damage[q]->damage = solve_kinetic_equation(localStress, old_damage, tempInc);
      } 
    }
  }

  template <int dim>
  void ElasticProblem<dim>::saveDamageAsOld()
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      auto cell_damage = damageInQuadraturePoints.get_data(cell);
      for (unsigned int q = 0; q < n_q_points; q++)
      {
        cell_damage[q]->old_damage = cell_damage[q]->damage;
      }
    }
  }

  template <int dim>
  double ElasticProblem<dim>::solve_kinetic_equation(SymmetricTensor<2,dim> localStress, double old_damage, double tempInc)
  {
    double omegaPrevIter = old_damage;
    double dOmega = -1.0 * kineticResidual(localStress, omegaPrevIter, old_damage, tempInc) /  derivativeKineticResidual(localStress, omegaPrevIter, tempInc);
    double omegaNextIter = omegaPrevIter + dOmega;
    double resOmega = abs(omegaPrevIter - omegaNextIter);
    int nIter = 0;
    while (resOmega > epsilonDamageIntegration)
    {
      nIter++;
      if (nIter > maxIterationDamageIntegration) throw std::runtime_error("max number of iterations for kinetic damage equation reached!");
      omegaPrevIter = omegaNextIter;
      dOmega = -1.0 * kineticResidual(localStress, omegaPrevIter, old_damage, tempInc) /  derivativeKineticResidual(localStress, omegaPrevIter, tempInc);
      omegaNextIter = omegaPrevIter + dOmega;
      resOmega = abs(omegaPrevIter - omegaNextIter);
    }
    return omegaNextIter;
  }

  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle, double curTemperature)
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

    data_out.add_data_vector(solDisplacement,  solDisplacement_names);

    StressPostprocessor stress_post(*this, curTemperature);
    data_out.add_data_vector(solDisplacement, stress_post);

    DamagePostprocessor damage_post(*this);
    data_out.add_data_vector(solDisplacement, damage_post);

    // Vector<double> cell_damage(triangulation.n_active_cells()); 
    // for (const auto &cell : dof_handler.active_cell_iterators()) //temporary solution to output damage, in general case it should be done through DataPostprocessor
    // {
    //     if (!cell->is_locally_owned())
    //         continue;                     // essential in parallel computations

    //     fe_values.reinit(cell);

    //     // Retrieve the per‑quadrature‑point history
    //     const std::vector<std::shared_ptr<const double>> qp_damage =
    //         damageInQuadraturePoints.get_data(cell);

    //     double average = 0.0;
    //     for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
    //         average += *(qp_damage[q]);   // dereference the shared_ptr
    //     average /= quadrature_formula.size();

    //     // Index by active cell index is mandatory for DataOut
    //     cell_damage[cell->active_cell_index()] = average;
    // }
    // data_out.add_data_vector(cell_damage, "damage");

    data_out.build_patches();

    std::ofstream output("solDisplacement-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output);
  }

  template <int dim>
  void ElasticProblem<dim>::meshTest()
  { 
    std::string mshFileName = "testMSHreplaceTag.msh";
    std::string cellToMaterialFileName = "groupFile";
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName); 
    auto mshStream = parseMSHandPrepareStream(mshFileName);
    grid_in.read_msh(mshStream); //it will read only one phsycial tag, related to id of cell (so u need to use special map material[cellId] for material, rotation have direct relation to cell id)
    parse_cellToMaterial_data(cellToMaterialFileName);

    setup_system();
  }

  template <int dim>
  void ElasticProblem<dim>::elasticCalculationTest()
  {
    std::string mshFileName = "testMSHreplaceTag.msh";
    std::string cellToMaterialFileName = "groupFile";
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName); 
    auto mshStream = parseMSHandPrepareStream(mshFileName);
    grid_in.read_msh(mshStream); //it will read only one phsycial tag, related to id of cell (so u need to use special map material[cellId] for material, rotation have direct relation to cell id)
    parse_cellToMaterial_data(cellToMaterialFileName);

    double curT = 1700;
    double dT = 300;

    setup_system();
    assemble_system(curT, dT);
    solve();
  }

  template <int dim>
  void ElasticProblem<dim>::elasticAndDamageTest()
  { 
    std::string mshFileName = "smallCube.msh";
    std::string cellToMaterialFileName = "groupFile";
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName); 
    grid_in.read_msh(mshFileName); //it will read only one phsycial tag, related to id of cell (so u need to use special map material[cellId] for material, rotation have direct relation to cell id)
    parse_cellToMaterial_data(cellToMaterialFileName);

    //here manually setting one main orientation for all cells, in general case it should be read from msh file and pushed in
    //it is done for test purposes, to check if damage is predicted correctly for one orientation
    Tensor<1, 3> axis;
    axis[0] = 0.0;
    axis[1] = 1.0;
    axis[2] = 0.0;

    const double angle = numbers::PI / 2.0;

    auto rotation_matrix = Physics::Transformations::Rotations::rotation_matrix_3d(axis, angle);

    rotmatOrientationCell.push_back(rotation_matrix); //for this test we will use only one orientation, so we push only one identity matrix, but in general case it should be read from msh file and pushed in this vector in order of cell id

    double curT = 2000;
    double dT = 1;

    setup_system();
    assemble_system(curT, dT);
    solve();
    predict_damage_tempInc(curT, dT);
    output_results(1, curT);
  }

  template <int dim>
  void ElasticProblem<dim>::elasticDamageSelfConsistentTest()
  {
    std::string mshFileName = "smallCube.msh";
    std::string cellToMaterialFileName = "groupFile";
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName);
    grid_in.read_msh(mshFileName); //it will read only one phsycial tag, related to id of cell (so u need to use special map material[cellId] for material, rotation have direct relation to cell id)
    parse_cellToMaterial_data(cellToMaterialFileName);

    //here manually setting one main orientation for all cells, in general case it should be read from msh file and pushed in
    //it is done for test purposes, to check if damage is predicted correctly for one orientation
    Tensor<1, 3> axis;
    axis[0] = 0.0;
    axis[1] = 1.0;
    axis[2] = 0.0;

    const double angle = numbers::PI / 2.0;

    auto rotation_matrix = Physics::Transformations::Rotations::rotation_matrix_3d(axis, angle);

    rotmatOrientationCell.push_back(rotation_matrix); //for this test we will use only one orientation, so we push only one identity matrix, but in general case it should be read from msh file and pushed in this vector in order of cell id

    double curT = 2000;
    double dT = 1;

    setup_system();

    int iterationCounter = 0;

    Vector<double> oldSolution;
    Vector<double> solutionDifference;

    assemble_system(curT, dT);
    solve();
    predict_damage_tempInc(curT, dT);
    output_results(iterationCounter, curT);
    iterationCounter++;

    oldSolution = solDisplacement;
    double displacementResidual = 1.0;

    while (displacementResidual > displacementTolerance) //some stopping criteria for iteration, for example based on damage residual or stress residual, it is just an example, it should be defined based on physics of the problem and expected solution
    {
      assemble_system(curT, dT);
      solve();
      predict_damage_tempInc(curT, dT);
      output_results(iterationCounter, curT);
      iterationCounter++;
      solutionDifference = solDisplacement;
      solutionDifference -= oldSolution;
      double maxDisplacement = *std::max_element(solDisplacement.begin(), solDisplacement.end()); //should be made more robust
      double displacementMaxDiff = *std::max_element(solutionDifference.begin(), solutionDifference.end());
      displacementResidual = displacementMaxDiff / maxDisplacement;
      oldSolution = solDisplacement;
    }
    saveDamageAsOld();
  }

} // namespace StepCooling


int main()
{
  try
    {
      StepCooling::ElasticProblem<3> myProblem;
      myProblem.elasticDamageSelfConsistentTest();
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
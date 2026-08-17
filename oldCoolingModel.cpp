
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>

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

  private:

    struct DamageQData : public TransferableQuadraturePointData
    {
      double damage = 0.0;
      unsigned int number_of_values() const override
      {
        return 1;
      }

      void pack_values(std::vector<double> &values) const override
      {
        AssertDimension(values.size(), 1);
        values[0] = damage;
      }

      void unpack_values(const std::vector<double> &values) override
      {
        AssertDimension(values.size(), 1);
        damage = values[0];
      }
    };
    
    static constexpr double T0 = 2000;
    static constexpr double a_kineticParam = 1.0;
    static constexpr double m_kineticParam = 2.0;
    static constexpr double stressThreshold = 200e6;
    static constexpr double epsilonDamageIntegration = 1e-3;
    static const int maxIterationDamageIntegration = 100;

    void setup_system();
    void assemble_system(double curTemperature, double incTemperature);
    void solve();
    void predict_damage_tempInc(double curTemperature, double tempInc);
    void refine_grid();
    void output_results(const unsigned int cycle = 1) const;
    void parse_cellToMaterial_data(std::string fileName);

    bool checkActivationCriteria(SymmetricTensor<2,dim> localStress);

    double solve_kinetic_equation(SymmetricTensor<2,dim> localStress, double old_damage, double tempInc);
    double kineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double oldOmega, double tempInc);
    double derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc);

    SymmetricTensor<2,dim,double> getLocalStressTensor(SymmetricTensor<2,dim> localStrain, double curOmega, double curTemperature);
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
  SymmetricTensor<2,dim> ElasticProblem<dim>::getLocalStressTensor(SymmetricTensor<2,dim> localStrain, double curOmega, double curTemperature) //provide localStrain!!!
  {
    return getElasticityTensor(curOmega, curTemperature) * (localStrain - getcteTensor(curTemperature) * (curTemperature - T0)); 
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
    double actingStress = localStress[dim-1][dim-1] - stressThreshold;
    if(actingStress < 0) throw std::runtime_error("acting stress is negative, this function should've not be called");
    return curOmega - oldOmega - a_kineticParam * std::pow((actingStress /(1-curOmega)), m_kineticParam) * tempInc;
  };

  template <int dim>
  double ElasticProblem<dim>::derivativeKineticResidual(SymmetricTensor<2,dim> localStress, double curOmega, double tempInc)
  {
    double actingStress = localStress[dim-1][dim-1] - stressThreshold;
    if (actingStress < 0) throw std::runtime_error("acting stress is negative, this function should've not be called");
    return 1 - a_kineticParam * m_kineticParam * tempInc * std::pow(actingStress / (1-curOmega), m_kineticParam) / (1-curOmega);
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

    dof_handler.distribute_dofs(fe);
    solDisplacement.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    boundaryDisplacementFunction<dim> prescribed_displacement;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             types::boundary_id(0),
                                             prescribed_displacement,
                                             constraints);
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

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    boundaryDisplacementFunction<dim> prescribed_displacement;
    prescribed_displacement.setCurrentTemperature(curTemperature);
    prescribed_displacement.setInitialTemperarure(T0);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             types::boundary_id(0),
                                             prescribed_displacement,
                                             constraints);
    constraints.close();

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
        double old_damage = cell_damage[q]->damage;
        localStress = getLocalStressTensor(localStrain, old_damage, curTemperature);
        if (!checkActivationCriteria(localStress)) continue;
        cell_damage[q]->damage = solve_kinetic_equation(localStress, old_damage, tempInc);
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
    predict_damage_tempInc(curT, dT);
  }

} // namespace StepCooling


int main()
{
  try
    {
      StepCooling::ElasticProblem<3> myProblem;
      myProblem.elasticAndDamageTest();
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
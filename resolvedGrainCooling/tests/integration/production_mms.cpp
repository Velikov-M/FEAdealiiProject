#include "resolved_grain/cell_assembly.h"
#include "resolved_grain/material.h"

#include "symbolic_oracles.h"

#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;
using namespace StepCooling;

namespace
{
  class ParsedVectorFunction : public Function<3>
  {
  public:
    explicit ParsedVectorFunction(const std::array<std::string, 3> &expressions,
                                  const std::array<std::string, 9> *gradientExpressions = nullptr)
      : Function<3>(3)
      , parser(3)
    {
      parser.initialize("x,y,z", std::vector<std::string>(expressions.begin(), expressions.end()),
                        {{"pi", numbers::PI}});
      if (gradientExpressions != nullptr)
        {
          analyticGradients = true;
          for (unsigned int i = 0; i < gradientExpressions->size(); ++i)
            gradients[i].initialize("x,y,z", (*gradientExpressions)[i], {{"pi", numbers::PI}});
        }
    }

    Tensor<1, 3> tensor_value(const Point<3> &point) const
    {
      Vector<double> values(3);
      parser.vector_value(point, values);
      return Tensor<1, 3>({values[0], values[1], values[2]});
    }

    void vector_value(const Point<3> &point, Vector<double> &values) const override
    {
      parser.vector_value(point, values);
    }

    Tensor<1, 3> gradient(const Point<3> &point,
                          const unsigned int component = 0) const override
    {
      Tensor<1, 3> result;
      if (!analyticGradients)
        return parser.gradient(point, component);
      for (unsigned int d = 0; d < 3; ++d)
        result[d] = gradients[3 * component + d].value(point);
      return result;
    }

  private:
    FunctionParser<3> parser;
    std::array<FunctionParser<3>, 9> gradients;
    bool analyticGradients = false;
  };

  struct Material
  {
    Material()
    {
      prm.enter_subsection("MaterialParameters");
      prm.enter_subsection("LinearElasticityParameters");
      prm.declare_entry("C11_0", "760e9");
      prm.declare_entry("C12_0", "80e9");
      prm.declare_entry("C13_0", "13e9");
      prm.declare_entry("C33_0", "12e9");
      prm.declare_entry("C44_0", "0.48e9");
      prm.declare_entry("C11_functionOfTemperature", "1 + 1e-4*(T-2000)");
      prm.declare_entry("C12_functionOfTemperature", "1 - 2e-4*(T-2000)");
      prm.declare_entry("C13_functionOfTemperature", "1 + 3e-4*(T-2000)");
      prm.declare_entry("C33_functionOfTemperature", "1 - 4e-4*(T-2000)");
      prm.declare_entry("C44_functionOfTemperature", "1 + 5e-4*(T-2000)");
      prm.leave_subsection();
      prm.enter_subsection("ThermoElasticParameters");
      prm.declare_entry("alpha11_0", "1.2e-6");
      prm.declare_entry("alpha33_0", "32e-6");
      prm.declare_entry("alpha11_functionOfTemperature", "1 + 1e-3*(T-2000)");
      prm.declare_entry("alpha33_functionOfTemperature", "1 - 1e-3*(T-2000)");
      prm.leave_subsection();
      prm.leave_subsection();
      elasticity.setFromPrm(prm);
      cte.setFromPrm(prm);
    }

    ParameterHandler prm;
    ElasticityTensor<3> elasticity;
    CteTensor<3> cte;
  };

  struct ErrorRow
  {
    unsigned int cells;
    unsigned int dofs;
    double l2;
    double h1;
    double indicator;
  };

  ErrorRow solveOnMesh(Triangulation<3> &triangulation,
                       const ParsedVectorFunction &exact,
                       const ParsedVectorFunction &bodyForce,
                       const FunctionParser<3> &temperature,
                       const Material &material,
                       const bool adaptAfterSolve)
  {
    FESystem<3> fe(FE_Q<3>(1), 3);
    DoFHandler<3> dofs(triangulation);
    dofs.distribute_dofs(fe);
    const FEValuesExtractors::Vector displacement(0);

    AffineConstraints<double> constraints;
    DoFTools::make_hanging_node_constraints(dofs, constraints);
    VectorTools::interpolate_boundary_values(dofs, 0, exact, constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dofs.n_dofs());
    DoFTools::make_sparsity_pattern(dofs, dsp, constraints, false);
    SparsityPattern sparsity;
    sparsity.copy_from(dsp);
    SparseMatrix<double> matrix(sparsity);
    Vector<double> rhs(dofs.n_dofs()), solution(dofs.n_dofs());

    const QGauss<3> quadrature(2);
    FEValues<3> feValues(fe, quadrature,
                         update_values | update_gradients | update_quadrature_points |
                         update_JxW_values);
    const unsigned int dofsPerCell = fe.n_dofs_per_cell();
    const unsigned int points = quadrature.size();
    AssemblyScratch::Copy copy;
    copy.cell_matrix.reinit(dofsPerCell, dofsPerCell);
    copy.cell_rhs.reinit(dofsPerCell);
    copy.local_dof_indices.resize(dofsPerCell);
    std::vector<SymmetricTensor<4, 3>> elasticity(points);
    std::vector<SymmetricTensor<2, 3>> cte(points);
    std::vector<double> temperatureChange(points);
    std::vector<Tensor<1, 3>> body(points);
    std::vector<std::vector<SymmetricTensor<2, 3>>> gradients(
      dofsPerCell, std::vector<SymmetricTensor<2, 3>>(points));
    std::vector<SymmetricTensor<2, 3>> macroCoupling(dofsPerCell);

    for (const auto &cell : dofs.active_cell_iterators())
      {
        feValues.reinit(cell);
        for (const unsigned int q : feValues.quadrature_point_indices())
          {
            const Point<3> &point = feValues.quadrature_point(q);
            const double T = temperature.value(point);
            elasticity[q] = material.elasticity.getElasticityTensor(0.0, T);
            cte[q] = material.cte.getcteTensor(0.0, T);
            temperatureChange[q] = T - SymbolicOracles::mmsReferenceTemperature;
            body[q] = bodyForce.tensor_value(point);
          }
        assembleCell<3>(feValues, displacement, elasticity, cte, temperatureChange,
                        &body, false, gradients, macroCoupling, copy);
        cell->get_dof_indices(copy.local_dof_indices);
        constraints.distribute_local_to_global(copy.cell_matrix, copy.cell_rhs,
                                               copy.local_dof_indices, matrix, rhs);
      }

    SparseDirectUMFPACK direct;
    direct.initialize(matrix);
    direct.vmult(solution, rhs);
    constraints.distribute(solution);

    Vector<float> difference(triangulation.n_active_cells());
    const QGauss<3> errorQuadrature(4);
    VectorTools::integrate_difference(dofs, solution, exact, difference, errorQuadrature,
                                      VectorTools::L2_norm);
    const double l2 = VectorTools::compute_global_error(triangulation, difference,
                                                        VectorTools::L2_norm);
    VectorTools::integrate_difference(dofs, solution, exact, difference, errorQuadrature,
                                      VectorTools::H1_seminorm);
    const double h1 = VectorTools::compute_global_error(triangulation, difference,
                                                        VectorTools::H1_seminorm);

    Vector<float> indicator(triangulation.n_active_cells());
    KellyErrorEstimator<3>::estimate(dofs, QGauss<2>(2), {}, solution, indicator);
    const double indicatorNorm = indicator.l2_norm();
    const unsigned int solvedCells = triangulation.n_active_cells();
    if (adaptAfterSolve)
      {
        GridRefinement::refine_and_coarsen_fixed_number(triangulation, indicator, 0.3, 0.0);
        triangulation.prepare_coarsening_and_refinement();
        triangulation.execute_coarsening_and_refinement();
      }
    return {solvedCells, dofs.n_dofs(), l2, h1, indicatorNorm};
  }

  ErrorRow runLevel(const unsigned int refinements,
                    const ParsedVectorFunction &exact,
                    const ParsedVectorFunction &bodyForce,
                    const FunctionParser<3> &temperature,
                    const Material &material)
  {
    Triangulation<3> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(refinements);
    return solveOnMesh(triangulation, exact, bodyForce, temperature, material, false);
  }

  std::vector<ErrorRow> runAdaptive(const ParsedVectorFunction &exact,
                                    const ParsedVectorFunction &bodyForce,
                                    const FunctionParser<3> &temperature,
                                    const Material &material)
  {
    Triangulation<3> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(2);
    std::vector<ErrorRow> rows;
    for (unsigned int cycle = 0; cycle < 4; ++cycle)
      rows.push_back(solveOnMesh(triangulation, exact, bodyForce, temperature, material,
                                 cycle + 1 < 4));
    return rows;
  }

  void writeRows(std::ostream &output, const std::vector<ErrorRow> &rows)
  {
    for (unsigned int i = 0; i < rows.size(); ++i)
      {
        if (i != 0)
          output << ',';
        output << "{\"cells\":" << rows[i].cells << ",\"dofs\":" << rows[i].dofs
               << ",\"L2\":" << rows[i].l2 << ",\"H1\":" << rows[i].h1
               << ",\"indicator\":" << rows[i].indicator << '}';
      }
  }
}

int main(int argc, char **argv)
{
  if (argc != 2)
    {
      std::cerr << "usage: resolved_grain_mms OUTPUT.json" << std::endl;
      return 2;
    }
  try
    {
      const std::array<std::string, 3> exactExpressions{{SymbolicOracles::mmsDisplacement[0],
                                                          SymbolicOracles::mmsDisplacement[1],
                                                          SymbolicOracles::mmsDisplacement[2]}};
      const std::array<std::string, 3> bodyExpressions{{SymbolicOracles::mmsBodyForce[0],
                                                         SymbolicOracles::mmsBodyForce[1],
                                                         SymbolicOracles::mmsBodyForce[2]}};
      const std::array<std::string, 9> exactGradient{{
        SymbolicOracles::mmsDisplacementGradient[0], SymbolicOracles::mmsDisplacementGradient[1],
        SymbolicOracles::mmsDisplacementGradient[2], SymbolicOracles::mmsDisplacementGradient[3],
        SymbolicOracles::mmsDisplacementGradient[4], SymbolicOracles::mmsDisplacementGradient[5],
        SymbolicOracles::mmsDisplacementGradient[6], SymbolicOracles::mmsDisplacementGradient[7],
        SymbolicOracles::mmsDisplacementGradient[8],
      }};
      const ParsedVectorFunction exact(exactExpressions, &exactGradient);
      const ParsedVectorFunction body(bodyExpressions);
      FunctionParser<3> temperature(1);
      temperature.initialize("x,y,z", SymbolicOracles::mmsTemperature, {{"pi", numbers::PI}});
      const Material material;
      std::vector<ErrorRow> rows;
      for (const unsigned int refinements : {2, 3, 4})
        {
          rows.push_back(runLevel(refinements, exact, body, temperature, material));
          const auto &row = rows.back();
          std::cout << "global: cells=" << row.cells << " dofs=" << row.dofs
                    << " L2=" << row.l2 << " H1=" << row.h1
                    << " indicator=" << row.indicator << std::endl;
        }
      const std::vector<ErrorRow> adaptiveRows = runAdaptive(exact, body, temperature, material);
      for (const auto &row : adaptiveRows)
        std::cout << "adaptive: cells=" << row.cells << " dofs=" << row.dofs
                  << " L2=" << row.l2 << " H1=" << row.h1
                  << " indicator=" << row.indicator << std::endl;

      std::ofstream output(argv[1]);
      output.precision(17);
      output << "{\"rows\":[";
      writeRows(output, rows);
      output << "],\"adaptive_rows\":[";
      writeRows(output, adaptiveRows);
      output << "]}\n";
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << std::endl;
      return 1;
    }
  return 0;
}

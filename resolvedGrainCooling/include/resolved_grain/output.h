#ifndef RESOLVED_GRAIN_OUTPUT_H
#define RESOLVED_GRAIN_OUTPUT_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <array>
#include <vector>

namespace StepCooling
{
  using namespace dealii;

  /** Own cell-centered arrays referenced by DataOut during VTU construction.
   *
   * Scalar stresses are in Pa, strains and damage are dimensionless, and the damage-driving
   * ratio is normalized by the configured stress threshold. The complete object must outlive
   * DataOut::build_patches() and DataOut::write_vtu().
   */
  struct CellOutputFields
  {
    /** Allocate arrays for all scalar fields and requested full tensors.
     * @param[in] cellCount Number of active cells.
     * @param[in] writeFullStress Whether to allocate local/global stress component arrays.
     * @param[in] writeFullStrain Whether to allocate local/global strain component arrays.
     */
    CellOutputFields(unsigned int cellCount, bool writeFullStress, bool writeFullStrain);

    Vector<double> damage;             ///< Cell-average damage.
    Vector<double> sigma33;            ///< Local c-axis normal stress.
    Vector<double> eps33;              ///< Local total strain normal to the c-axis.
    Vector<double> basalShear;         ///< Local basal-plane shear magnitude.
    Vector<double> vonMises;           ///< Frame-invariant von Mises stress.
    Vector<double> pressure;           ///< Frame-invariant pressure, positive in compression.
    Vector<double> damageDrivingRatio; ///< Local sigma_33 divided by stress threshold.

    std::array<Vector<double>, 6> macroscopicStrain; ///< Uniform arrays in tensor-component order.

    std::array<Vector<double>, 5> localStress;  ///< Order 11, 22, 12, 13, 23.
    std::array<Vector<double>, 6> globalStress; ///< Order xx, yy, zz, xy, xz, yz.
    std::array<Vector<double>, 5> localStrain;  ///< Order 11, 22, 12, 13, 23.
    std::array<Vector<double>, 6> globalStrain; ///< Order xx, yy, zz, xy, xz, yz.

    bool writeFullStress; ///< Whether full stress arrays are populated and exported.
    bool writeFullStrain; ///< Whether full strain arrays are populated and exported.
  };

  /** Export each grain's c-axis as a vector for ParaView line glyphs.
   *
   * The orientation array is referenced, not copied, and must outlive DataOut evaluation.
   * The c-axis is a director: c and -c are equivalent, so arrow glyphs would imply false polarity.
   */
  template <int dim>
  class CAxisPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    /** Construct the postprocessor with persistent per-grain orientations.
     * @param[in] orientations Global-to-local matrices indexed by grain material_id - 1.
     */
    CAxisPostprocessor(const std::vector<Tensor<2, dim>> &orientations)
      : DataPostprocessorVector<dim>("c_axis", update_default)
      , orientations(orientations)
    {}

    /** Evaluate c-axis vectors on the current cell.
     * @param[in] inputs Solution values and active-cell context supplied by DataOut.
     * @param[out] computed_quantities Destination c-axis vectors.
     */
    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      const auto cell = inputs.template get_cell<dim>();
      // Row 2 of the global->local rotation is the c-axis written in global coordinates.
      const Tensor<2, dim> &R = orientations[cell->material_id() - 1];
      for (auto &q : computed_quantities)
        for (unsigned int d = 0; d < dim; ++d)
          q(d) = R[dim - 1][d];
    }

  private:
    const std::vector<Tensor<2, dim>> &orientations;
  };

  /** Reconstruct total displacement u = Ebar.x + u_periodic for visualization.
   *
   * The solver stores only the fluctuation; macroscopicStrain supplies the affine part and is
   * zero for non-periodic boundary conditions. The referenced tensor must outlive evaluation.
   */
  template <int dim>
  class TotalDisplacementPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    /** Construct the postprocessor with the current macroscopic strain.
     * @param[in] macroStrain Macroscopic strain tensor, dimensionless.
     */
    TotalDisplacementPostprocessor(const SymmetricTensor<2, dim> &macroStrain)
      : DataPostprocessorVector<dim>("displacement_total",
                                     update_values | update_quadrature_points)
      , macroStrain(macroStrain)
    {}

    /** Evaluate total displacement at all requested points.
     * @param[in] inputs Fluctuation values and evaluation points supplied by DataOut.
     * @param[out] computed_quantities Destination total displacement vectors in m.
     */
    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      for (unsigned int q = 0; q < computed_quantities.size(); ++q)
      {
        const Point<dim> &x = inputs.evaluation_points[q];
        for (unsigned int d = 0; d < dim; ++d)
        {
          double macroPart = 0.0;
          for (unsigned int e = 0; e < dim; ++e)
            macroPart += macroStrain[d][e] * x[e];
          computed_quantities[q](d) = inputs.solution_values[q](d) + macroPart;
        }
      }
    }

  private:
    const SymmetricTensor<2, dim> &macroStrain;
  };
}

#endif

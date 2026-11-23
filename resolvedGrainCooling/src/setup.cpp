#include "resolved_grain/elastic_problem.h"

namespace StepCooling
{
  using namespace dealii;

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
  std::istringstream ElasticProblem<dim>::parseMSHandPrepareStream(std::string mshFileName)
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
    outFile << out.str();
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
  SymmetricTensor<2,dim> ElasticProblem<dim>::getLocalStressTensor(const SymmetricTensor<2,dim> &localStrain, double curOmega, double curTemperature) const
  {
    double T0 = prm.get_double({"ModelParameters"}, "InitialTemperature");
    return elasticityTensor.getElasticityTensor(curOmega, curTemperature) * (localStrain - cteTensor.getcteTensor(curOmega, curTemperature) * (curTemperature - T0)); 
  }

  template <int dim>
  Tensor<2,dim,double> ElasticProblem<dim>::getRotTensorGlobalToLocal(int cellId) const
  {
    Assert(dim == 3, ExcNotImplemented());
    return rotmatOrientationCell[cellId - 1];
  }

  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
    elasticityTensor.setFromPrm(prm);
    cteTensor.setFromPrm(prm);
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

    // The sparsity pattern must be built from a constraint set that already contains every
    // constraint that will ever couple two DoFs. Hanging nodes and Dirichlet conditions only
    // ever eliminate rows, but PERIODICITY genuinely couples DoFs on opposite faces: leaving it
    // out here would give a pattern with no entry for those couplings, and
    // distribute_local_to_global() would then silently drop them. So build the real constraint
    // set once here, purely for the pattern.
    applyBoundaryConstraints();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    // L's six columns live on the same DoF numbering as the solution vector.
    macroCouplingColumns.assign(nMacroDofs, Vector<double>(dof_handler.n_dofs()));
    macroscopicStrain = 0;
  }

  template <int dim>
  void ElasticProblem<dim>::assignGrains()
  {
    const double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    const double grainSize = L / grainsPerSide;

    for (const auto &cell : triangulation.active_cell_iterators())
    {
      const Point<dim> c = cell->center();
      unsigned int index = 0, stride = 1;
      for (unsigned int d = 0; d < dim; ++d)
      {
        // clamp guards the boundary: a centroid exactly at x = L would otherwise index one past
        // the last grain. Centroids are cell centres so this cannot trigger on a conforming
        // mesh, but it costs nothing and turns a silent out-of-range material_id into a no-op.
        const int slot = std::clamp(static_cast<int>(c[d] / grainSize),
                                    0, static_cast<int>(grainsPerSide) - 1);
        index += static_cast<unsigned int>(slot) * stride;
        stride *= grainsPerSide;
      }
      cell->set_material_id(index + 1);   // 1-based: getRotTensorGlobalToLocal indexes [id - 1]
    }

    const unsigned int nGrains = grainsPerSide * grainsPerSide * grainsPerSide;
    std::cout << "Grains: " << grainsPerSide << "^3 = " << nGrains << ", "
              << triangulation.n_active_cells() << " cells => "
              << double(triangulation.n_active_cells()) / nGrains
              << " elements per grain" << std::endl;
  }

  template <int dim>
  void ElasticProblem<dim>::refreshGrainStiffnessCache(double curTemperature)
  {
    // Cheap guard: the cache depends only on orientation (fixed for the run) and temperature.
    if (grainCacheTemperature == curTemperature &&
        grainStiffnessZeroDamage.size() == rotmatOrientationCell.size())
      return;

    TimerOutput::Scope timer_section(timer, "Grain stiffness cache");
    const unsigned int nGrains = rotmatOrientationCell.size();
    grainStiffnessZeroDamage.resize(nGrains);
    grainDamageDirection.resize(nGrains);
    grainCteZeroDamage.resize(nGrains);
    grainCaxisOuter.resize(nGrains);

    const SymmetricTensor<4, dim> localZeroDamage =
      elasticityTensor.getElasticityTensor(0.0, curTemperature);
    const SymmetricTensor<2, dim> localCteZeroDamage =
      cteTensor.getcteTensor(0.0, curTemperature);

    for (unsigned int g = 0; g < nGrains; ++g)
    {
      const Tensor<2, dim> rotLocalToGlobal = transpose(rotmatOrientationCell[g]);
      grainStiffnessZeroDamage[g] =
        Physics::Transformations::basis_transformation(localZeroDamage, rotLocalToGlobal);

      // n = the grain's c-axis in global coordinates = row 2 of the global->local rotation.
      SymmetricTensor<2, dim> nn;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = i; j < dim; ++j)
          nn[i][j] = rotmatOrientationCell[g][dim-1][i] * rotmatOrientationCell[g][dim-1][j];
      grainDamageDirection[g] = outer_product(nn, nn);
      grainCaxisOuter[g] = nn;
      grainCteZeroDamage[g] =
        Physics::Transformations::basis_transformation(localCteZeroDamage, rotLocalToGlobal);
    }

    grainCacheTemperature = curTemperature;
    grainCacheC33 = elasticityTensor.getC33AtZeroDamage(curTemperature);
    grainCacheAlpha33 = cteTensor.getAlpha33AtZeroDamage(curTemperature);
  }

  // Label the six faces of the cube geometrically. The .msh carries no surface elements at all,
  // so every face arrives with the default id 0 and there is nothing to inherit from the file.
  // Called once after the mesh is read: child faces inherit their parent's boundary id, so
  // refinement preserves this classification and it must NOT be redone per cycle.
  template <int dim>
  void ElasticProblem<dim>::assignBoundaryIds()
  {
    const double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    // Loose relative to L: the tolerance only has to separate a face from the cell layer behind
    // it, and mesh coordinates come from a text file with finite precision.
    const double tol = 1e-8 * L;

    std::array<unsigned int, 6> counts{};
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
      {
        if (!face->at_boundary())
          continue;
        const Point<dim> c = face->center();
        if      (std::abs(c[0])     < tol) { face->set_boundary_id(BoundaryIds::xMinus); ++counts[0]; }
        else if (std::abs(c[0] - L) < tol) { face->set_boundary_id(BoundaryIds::xPlus);  ++counts[1]; }
        else if (std::abs(c[1])     < tol) { face->set_boundary_id(BoundaryIds::yMinus); ++counts[2]; }
        else if (std::abs(c[1] - L) < tol) { face->set_boundary_id(BoundaryIds::yPlus);  ++counts[3]; }
        else if (std::abs(c[2])     < tol) { face->set_boundary_id(BoundaryIds::zMinus); ++counts[4]; }
        else if (std::abs(c[2] - L) < tol) { face->set_boundary_id(BoundaryIds::zPlus);  ++counts[5]; }
        else
          throw std::runtime_error(
            "boundary face at (" + std::to_string(c[0]) + "," + std::to_string(c[1]) + "," +
            std::to_string(c[2]) + ") lies on none of the six planes of the cube [0,L]^3 with "
            "L = " + std::to_string(L) + " -- LengthOfTheBody does not match the mesh, or the "
            "mesh is not the axis-aligned cube this model assumes.");
      }

    // Periodic matching needs the pairs to be the same size; catching it here beats a confusing
    // failure inside collect_periodic_faces later.
    for (unsigned int pair = 0; pair < 3; ++pair)
      if (counts[2 * pair] != counts[2 * pair + 1])
        throw std::runtime_error("boundary face counts differ across face pair " +
                                  std::to_string(pair) + " (" +
                                  std::to_string(counts[2 * pair]) + " vs " +
                                  std::to_string(counts[2 * pair + 1]) +
                                  ") -- the mesh is not periodic in that direction.");

    std::cout << "Boundary faces per pair: x=" << counts[0] << ", y=" << counts[2]
              << ", z=" << counts[4] << std::endl;
  }

  template <int dim>
  void ElasticProblem<dim>::constrainRigidBodyModes()
  {
    const double L = prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    const double tol = 1e-8 * L;

    // Support point, and which displacement components are fixed there.
    struct Support { Point<dim> at; std::vector<unsigned int> components; };
    const std::array<Support, 3> supports{{
      { Point<dim>(0.0, 0.0, 0.0), {0, 1, 2} },  // all three translations
      { Point<dim>(L,   0.0, 0.0), {1, 2}    },  // rotations about e_z and e_y; u_x LEFT FREE
      { Point<dim>(0.0, L,   0.0), {2}       }   // rotation about e_x
    }};
    // u_x must stay free at the second support, and u_x and u_y at the third: fixing them would
    // stop the body contracting along those axes, which is a real applied load, not a support.

    std::vector<Point<dim>> supportLocations(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, supportLocations);

    std::array<std::array<bool, dim>, 3> constrained{};
    std::vector<types::global_dof_index> dofsOnCell(fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(dofsOnCell);
      for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k)
      {
        const unsigned int component = fe.system_to_component_index(k).first;
        const types::global_dof_index gi = dofsOnCell[k];
        for (unsigned int s = 0; s < supports.size(); ++s)
        {
          if (constrained[s][component] ||
              supportLocations[gi].distance(supports[s].at) > tol)
            continue;
          const auto &comps = supports[s].components;
          if (std::find(comps.begin(), comps.end(), component) == comps.end())
            continue;
          if (!constraints.is_constrained(gi))
            constraints.add_line(gi);   // homogeneous: the inhomogeneity defaults to 0
          constrained[s][component] = true;
        }
      }
    }

    unsigned int nConstraints = 0;
    for (const auto &perSupport : constrained)
      for (unsigned int d = 0; d < dim; ++d)
        if (perSupport[d]) ++nConstraints;
    if (nConstraints != 6)
      throw std::runtime_error(
        "minimal kinematic support produced " + std::to_string(nConstraints) +
        " constraints instead of 6 -- the three support corners were not all found as mesh "
        "vertices. Check LengthOfTheBody against the mesh.");
  }

  template <int dim>
  void ElasticProblem<dim>::constrainOneVertex(const Point<dim> &where)
  {
    const double tol = 1e-8 * prm.get_double({"ModelParameters"}, "LengthOfTheBody");
    std::vector<Point<dim>> supportLocations(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, supportLocations);

    std::array<bool, dim> pinned{};
    std::vector<types::global_dof_index> dofsOnCell(fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(dofsOnCell);
      for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k)
      {
        const unsigned int component = fe.system_to_component_index(k).first;
        const types::global_dof_index gi = dofsOnCell[k];
        if (pinned[component] || supportLocations[gi].distance(where) > tol)
          continue;
        // Pinning a DoF that periodicity already constrained would be an over-determined
        // constraint, which AffineConstraints resolves silently rather than reporting. The point
        // of pinning the origin is that it is a periodicity MASTER; verify that rather than
        // trust it.
        if (constraints.is_constrained(gi))
          throw std::runtime_error(
            "the vertex chosen for pinning is already constrained by periodicity -- pick a "
            "master vertex (the origin should be one, since periodicity slaves the '+' faces "
            "to the '-' faces).");
        constraints.add_line(gi);   // homogeneous: inhomogeneity defaults to 0
        pinned[component] = true;
      }
    }

    for (unsigned int d = 0; d < dim; ++d)
      if (!pinned[d])
        throw std::runtime_error(
          "could not pin component " + std::to_string(d) + " at the requested vertex -- it was "
          "not found as a mesh vertex. Check LengthOfTheBody against the mesh.");
  }

  // Rebuilds `constraints` from scratch for the selected BC family. Must be called before every
  // assembly, because AffineConstraints cannot be added to once closed.
  template <int dim>
  void ElasticProblem<dim>::applyBoundaryConstraints()
  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    switch (boundaryConditionType)
    {
      case BoundaryConditionType::zeroDisplacement:
      {
        // u = 0 on the whole boundary. Kills every rigid-body mode as a side effect, so no
        // extra support is needed. Applied per face id rather than to id 0, because
        // assignBoundaryIds() has already relabelled everything.
        const Functions::ZeroFunction<dim> zero(dim);
        for (const types::boundary_id id : {BoundaryIds::xMinus, BoundaryIds::xPlus,
                                            BoundaryIds::yMinus, BoundaryIds::yPlus,
                                            BoundaryIds::zMinus, BoundaryIds::zPlus})
          VectorTools::interpolate_boundary_values(dof_handler, id, zero, constraints);
        break;
      }

      case BoundaryConditionType::freeSurface:
      {
        // Traction-free needs NO boundary term at all: zero traction is the natural boundary
        // condition of the weak form, so simply not integrating anything over the boundary is
        // exactly right. Only the rigid-body modes need removing.
        constrainRigidBodyModes();
        break;
      }

      case BoundaryConditionType::periodic:
      {
        // Homogeneous constraints on the FLUCTUATION u~ only. Ebar does not appear here: it is
        // carried as six separate unknowns and enters assembly as an eigenstrain
        // (homogenization.md section 8), which is why no inhomogeneity is needed and why the
        // existing thermal-load code path is reused unchanged.
        std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
          matchedPairs;
        GridTools::collect_periodic_faces(dof_handler, BoundaryIds::xMinus, BoundaryIds::xPlus,
                                          /*direction=*/0, matchedPairs);
        GridTools::collect_periodic_faces(dof_handler, BoundaryIds::yMinus, BoundaryIds::yPlus,
                                          /*direction=*/1, matchedPairs);
        GridTools::collect_periodic_faces(dof_handler, BoundaryIds::zMinus, BoundaryIds::zPlus,
                                          /*direction=*/2, matchedPairs);
        DoFTools::make_periodicity_constraints<dim, dim>(matchedPairs, constraints);

        // Pin ONE vertex, three components. Periodicity leaves exactly the three rigid
        // TRANSLATIONS unconstrained -- rigid rotations are already excluded, because u~ = W.x
        // with W skew is not periodic -- so three constraints is exactly enough, and pinning any
        // more would apply a real load and destroy <sigma> = 0.
        //
        // The origin is safe to pin: make_periodicity_constraints slaves the "+" face to the
        // "-" face, and (0,0,0) lies on all three "-" faces, so it is a master everywhere. That
        // is checked rather than assumed -- pinning an already-constrained DoF would silently
        // produce an over-determined constraint.
        constrainOneVertex(Point<dim>(0.0, 0.0, 0.0));
        break;
      }    }

    constraints.close();
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

    // setup_system() resets damage after refinement; refinement cycles restart from zero damage.
    triangulation.execute_coarsening_and_refinement();
    assignGrains();   // pure function of position, so the grain field is unchanged by refining
    setup_system();
  }

  template <int dim>
  void ElasticProblem<dim>::prepareRefinementCycle(const unsigned int refinementCycle)
  {
    if (refinementCycle == 0)
    {
      setup_system();
      return;
    }

    const std::string refinementStrategy =
      prm.get({"MeshRefinementParameters"}, "RefinementStrategy");
    if (refinementStrategy == "uniform")
    {
      triangulation.refine_global(1);
      assignGrains();
      setup_system();
    }
    else
    {
      const double fractionToRefine =
        prm.get_double({"MeshRefinementParameters"}, "FractionToRefine");
      const double fractionToCoarsen =
        prm.get_double({"MeshRefinementParameters"}, "FractionToCoarsen");
      refine_grid(fractionToRefine, fractionToCoarsen);
    }
  }

  template <int dim>
  void ElasticProblem<dim>::initializeMicrostructure()
  {
    std::string mshFileName = prm.get({"InputFiles"}, "MeshFile");
    std::string cellToMaterialFileName = prm.get({"InputFiles"}, "CellToMaterialFile");
    GridIn <dim> grid_in;
    grid_in.attach_triangulation(this->triangulation);
    std::ifstream input_file(mshFileName);
    auto mshStream = parseMSHandPrepareStream(mshFileName);
    grid_in.read_msh(mshStream);
    parse_cellToMaterial_data(cellToMaterialFileName);

    // Grain identity is a 1-based material_id inherited by children on refinement. Orientations
    // are generated per grain below, replacing any metadata read from the mesh.
    const bool uniformOrientation = prm.get_bool({"BoundaryConditions"}, "UniformOrientation");
    const bool useOrientationFile = prm.get_bool({"BoundaryConditions"}, "UseOrientationFile");
    if (uniformOrientation && useOrientationFile)
      throw std::runtime_error("UniformOrientation and UseOrientationFile cannot both be true");
    const unsigned int orientationSeed =
      prm.get_integer({"BoundaryConditions"}, "OrientationSeed");
    // Boundary IDs must be assigned on the coarse mesh before initial refinement.
    const unsigned int initialRefinements =
      prm.get_integer({"MeshRefinementParameters"}, "InitialGlobalRefinements");
    assignBoundaryIds();
    if (initialRefinements > 0)
      triangulation.refine_global(initialRefinements);

    grainsPerSide = prm.get_integer({"BoundaryConditions"}, "GrainsPerSide");
    const unsigned int nGrains = grainsPerSide * grainsPerSide * grainsPerSide;
    if (uniformOrientation)
      std::cout << "*** UniformOrientation smoke test: every grain identically oriented. "
                   "Expect Cbar = C, u~ = 0, sigma = 0 pointwise. ***" << std::endl;

    // Generate one orientation per grain; positional grain assignment preserves them on refinement.
    std::mt19937_64 rng(orientationSeed);
    rotmatOrientationCell.clear();
    rotmatOrientationCell.reserve(nGrains);
    std::ifstream orientationInput;
    if (useOrientationFile)
      {
        const std::string fileName = prm.get({"InputFiles"}, "OrientationFile");
        orientationInput.open(fileName);
        if (!orientationInput)
          throw std::runtime_error("cannot open OrientationFile: " + fileName);
      }
    for (unsigned int g = 0; g < nGrains; ++g)
      {
        Tensor<2, dim> orientation;
        if (uniformOrientation)
          orientation = Tensor<2, dim>(unit_symmetric_tensor<dim>());
        else if (useOrientationFile)
          {
            for (unsigned int i = 0; i < dim; ++i)
              for (unsigned int j = 0; j < dim; ++j)
                if (!(orientationInput >> orientation[i][j]))
                  throw std::runtime_error(
                    "OrientationFile must contain 9 numeric entries per grain");
            Tensor<2, dim> identity;
            for (unsigned int d = 0; d < dim; ++d)
              identity[d][d] = 1.0;
            if ((orientation * transpose(orientation) - identity).norm() > 1e-10 ||
                std::abs(determinant(orientation) - 1.0) > 1e-10)
              throw std::runtime_error("OrientationFile contains a non-orthonormal rotation");
          }
        else
          orientation = randomOrientationUniformOnSphere<dim>(rng);
        rotmatOrientationCell.push_back(orientation);
      }

    assignGrains();

    // Reject grain counts that cannot be represented by this build's material_id type.
    if (nGrains >= numbers::invalid_material_id)
      throw std::runtime_error(
        "GrainsPerSide^3 = " + std::to_string(nGrains) + " grains exceeds what "
        "types::material_id can represent on this deal.II build.");

    // Report sampling moments as a diagnostic for the orientation distribution.
    {
      Tensor<1, dim> meanAxis;
      double meanCzSquared = 0.0;
      for (const auto &R : rotmatOrientationCell)
      {
        for (unsigned int d = 0; d < dim; ++d) meanAxis[d] += R[dim-1][d];
        meanCzSquared += R[dim-1][dim-1] * R[dim-1][dim-1];
      }
      meanAxis /= rotmatOrientationCell.size();
      meanCzSquared /= rotmatOrientationCell.size();
      std::cout << "Orientations: seed=" << orientationSeed
                << ", |mean c-axis|=" << meanAxis.norm();
      if (!uniformOrientation && !useOrientationFile)
        std::cout << " (random expectation ~0)";
      std::cout << ", mean c_z^2=" << meanCzSquared;
      if (!uniformOrientation && !useOrientationFile)
        std::cout << " (random expectation ~0.3333)";
      std::cout << std::endl;
    }
  }

  template class ElasticProblem<3>;
}

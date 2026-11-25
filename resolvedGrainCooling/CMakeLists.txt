cmake_minimum_required(VERSION 3.13.4)

set(TARGET "main")
set(TARGET_SRC main.cpp)

find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR
    "Could not locate deal.II 9.8.0 or newer. "
    "Pass -DDEAL_II_DIR=/path/to/deal.II or set DEAL_II_DIR."
  )
endif()

deal_ii_initialize_cached_variables()
project(${TARGET})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(resolved_grain STATIC
  src/parameters.cpp
  src/material.cpp
  src/damage_law.cpp
  src/quadrature_transfer.cpp
  src/cell_assembly.cpp
  src/setup.cpp
  src/assembly.cpp
  src/solvers.cpp
  src/damage.cpp
  src/periodic.cpp
  src/output.cpp
  src/run.cpp
)
deal_ii_setup_target(resolved_grain)

deal_ii_invoke_autopilot()
target_link_libraries(main resolved_grain)

find_package(Doxygen QUIET)
if(Doxygen_FOUND)
  configure_file(Doxyfile Doxyfile @ONLY)
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generate resolved-grain API documentation"
    VERBATIM
  )
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(GTest 1.11 REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
  set(ORACLE_HEADER ${GENERATED_DIR}/symbolic_oracles.h)
  add_custom_command(
    OUTPUT ${ORACLE_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/symbolic/generate_oracles.py
            --output ${ORACLE_HEADER}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/symbolic/generate_oracles.py
    COMMENT "Generate independent SymPy verification oracles"
    VERBATIM
  )

  add_custom_target(symbolic_oracles DEPENDS ${ORACLE_HEADER})

  add_executable(resolved_grain_tests
    tests/unit/test_main.cpp
    tests/unit/test_material.cpp
    tests/unit/test_damage_law.cpp
    tests/unit/test_orientations.cpp
    tests/unit/test_quadrature_transfer.cpp
    tests/unit/test_convergence.cpp
  )
  add_dependencies(resolved_grain_tests symbolic_oracles)
  target_include_directories(resolved_grain_tests PRIVATE ${GENERATED_DIR})
  target_link_libraries(resolved_grain_tests resolved_grain GTest::gtest)
  deal_ii_setup_target(resolved_grain_tests)

  include(GoogleTest)
  gtest_discover_tests(resolved_grain_tests
    DISCOVERY_TIMEOUT 60
    PROPERTIES LABELS unit
  )

  add_executable(resolved_grain_mms
    tests/integration/production_mms.cpp
  )
  add_dependencies(resolved_grain_mms symbolic_oracles)
  target_include_directories(resolved_grain_mms PRIVATE ${GENERATED_DIR})
  target_link_libraries(resolved_grain_mms resolved_grain)
  deal_ii_setup_target(resolved_grain_mms)
  add_test(NAME production.mms
    COMMAND resolved_grain_mms ${CMAKE_CURRENT_BINARY_DIR}/production_mms.json
  )
  set_tests_properties(production.mms PROPERTIES LABELS integration TIMEOUT 120)
endif()

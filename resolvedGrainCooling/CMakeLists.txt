cmake_minimum_required(VERSION 3.13.4)

set(TARGET "main")
set(TARGET_SRC
  main.cpp
  src/parameters.cpp
  src/material.cpp
  src/setup.cpp
  src/assembly.cpp
  src/solvers.cpp
  src/damage.cpp
  src/periodic.cpp
  src/output.cpp
  src/run.cpp
)

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
deal_ii_invoke_autopilot()

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

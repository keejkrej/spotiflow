cmake_minimum_required(VERSION 3.15)
project(spotiflow_extensions CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python interpreter, development headers, and NumPy
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# OpenMP is optional but used for parallelism when available
find_package(OpenMP)

# nanoflann is a header-only KD-tree library bundled in the repo
set(NANOFLANN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/spotiflow/lib/external/nanoflann")

# Helper: build one Python C-extension module and install it under spotiflow/lib/
function(add_spotiflow_ext NAME SOURCE)
    Python_add_library(${NAME} MODULE "${SOURCE}" WITH_SOABI)
    target_include_directories(${NAME} PRIVATE
        "${Python_NumPy_INCLUDE_DIRS}"
        "${NANOFLANN_INCLUDE_DIR}"
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    # Install into spotiflow/lib/ so the module is importable as
    # spotiflow.lib.<NAME> via Python 3 namespace packages
    install(TARGETS ${NAME} DESTINATION spotiflow/lib)
endfunction()

add_spotiflow_ext(spotflow2d  spotiflow/lib/spotflow2d.cpp)
add_spotiflow_ext(spotflow3d  spotiflow/lib/spotflow3d.cpp)
add_spotiflow_ext(point_nms   spotiflow/lib/point_nms.cpp)
add_spotiflow_ext(point_nms3d spotiflow/lib/point_nms3d.cpp)
add_spotiflow_ext(filters     spotiflow/lib/filters.cpp)
add_spotiflow_ext(filters3d   spotiflow/lib/filters3d.cpp)

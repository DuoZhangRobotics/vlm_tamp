cmake_minimum_required(VERSION 3.16)
project(vlm_tamp_native LANGUAGES CXX)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(VAMP_ROOT "${PROJECT_ROOT}/third_party/mc-vamp" CACHE PATH "Pinned VAMP checkout")
set(VAMP_DEPS "${VAMP_ROOT}/build/cp310-cp310-linux_x86_64/_deps" CACHE PATH "Existing dependencies")
set(EIGEN_INCLUDE "" CACHE PATH "Directory containing Eigen/Core")
add_library(mcvamp_contact_numerical SHARED mcvamp_contact_numerical.cc)
target_compile_features(mcvamp_contact_numerical PRIVATE cxx_std_17)
set_target_properties(mcvamp_contact_numerical PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(mcvamp_contact_numerical PRIVATE
    -march=native -mavx2 -Wno-ignored-attributes -Wno-write-strings)
target_include_directories(mcvamp_contact_numerical PRIVATE
    "${VAMP_ROOT}/src/impl" "${EIGEN_INCLUDE}" "${VAMP_DEPS}/nigh-src/src"
    "${VAMP_DEPS}/pdqsort-src" "${VAMP_DEPS}/simdxorshift-src/include")
enable_testing()
add_subdirectory("${PROJECT_ROOT}/tests/native" "tests")

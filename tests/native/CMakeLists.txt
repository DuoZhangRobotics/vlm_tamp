cmake_minimum_required(VERSION 3.16)
project(vlm_tamp_native_tests LANGUAGES CXX)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(VAMP_ROOT "${PROJECT_ROOT}/third_party/mc-vamp" CACHE PATH "Pinned VAMP checkout")
set(VAMP_DEPS "${VAMP_ROOT}/build/cp310-cp310-linux_x86_64/_deps" CACHE PATH "Existing build dependencies")

add_executable(test_capt_collision test_capt_collision.cc)
target_compile_features(test_capt_collision PRIVATE cxx_std_17)
target_compile_options(test_capt_collision PRIVATE -O3 -DNDEBUG -march=native -mavx2)
target_include_directories(test_capt_collision PRIVATE
    "${VAMP_ROOT}/src/impl" "${VAMP_DEPS}/pdqsort-src")
enable_testing()
add_test(NAME capt_collision_contract COMMAND test_capt_collision)

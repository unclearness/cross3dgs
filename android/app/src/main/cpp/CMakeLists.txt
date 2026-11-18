cmake_minimum_required(VERSION 3.28)
project(cross3dgs_android CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(CROSS_3DGS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CROSS_3DGS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

get_filename_component(CROSS_3DGS_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
add_subdirectory("${CROSS_3DGS_ROOT}" cross3dgs_root)

add_library(cross3dgs_android SHARED
    jni_bridge.cpp
    surface_renderer.cpp
    device_caps.cpp
    "${CROSS_3DGS_ROOT}/tests/gpu/atomic_free_probe.cpp"
    "${CROSS_3DGS_ROOT}/tests/gpu/test_cpu_oracle.cpp")
target_compile_definitions(cross3dgs_android PRIVATE
    CROSS_3DGS_ATOMIC_FREE_PROBE_LIBRARY=1
    CROSS_3DGS_CPU_ORACLE_LIBRARY=1)
target_link_libraries(cross3dgs_android PRIVATE
    cross3dgs
    spdlog::spdlog
    android
    log
    nativewindow)
target_compile_options(cross3dgs_android PRIVATE
    -Wall -Wextra -Wpedantic)

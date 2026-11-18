cmake_minimum_required(VERSION 3.28)
project(cross_3dgs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

option(CROSS_3DGS_BUILD_TESTS "Build cross_3dgs tests" ON)
option(CROSS_3DGS_BUILD_TOOLS "Build cross_3dgs desktop tools" ON)

# The train_cli live viewer (--viewer) is a debugging aid: it opens its own
# GLFW window with its own Vulkan instance and function table alongside the
# training context. Official benchmarks always run headless, so the viewer can
# be dropped without touching the measured path. Default it off on Apple,
# where the GLFW window path has not been verified against MoltenVK; --viewer
# then reports "viewer unavailable" exactly as it already does when the window
# cannot be created at run time.
if(APPLE)
    set(_cross_3dgs_viewer_default OFF)
else()
    set(_cross_3dgs_viewer_default ON)
endif()
option(CROSS_3DGS_BUILD_VIEWER "Build the train_cli live viewer (--viewer)"
    ${_cross_3dgs_viewer_default})

add_subdirectory(external EXCLUDE_FROM_ALL)

include(FetchContent)
# GLM 0.9.9.8 predates CMake 4's policy floor. Keep the dependency pinned to
# VkSplat's exact version while opting it into the modern policy baseline.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glm)

add_subdirectory(libs/cross3dgs_gpu)
add_subdirectory(libs/cross3dgs)
if(CROSS_3DGS_BUILD_TOOLS)
    add_subdirectory(tools/train_cli)
    add_subdirectory(tools/schedule_spec_cli)
endif()

if(CROSS_3DGS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

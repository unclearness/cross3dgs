# CPU COLMAP sparse SfM shared by the Android and iOS capture-to-view apps.
#
# One SfM implementation (src/cross3dgs_sfm.cc, C API in
# include/cross3dgs_sfm.h) built against the COLMAP fork in external/colmap.
# This is a standalone project configured with the vcpkg toolchain and the
# target platform's toolchain; it is not part of the desktop build.
#   iOS:     scripts/build_ios_sfm.sh merges cross3dgs_sfm and every static
#            dependency into libcross3dgs_sfm.a, which ios/CMakeLists.txt
#            links through CROSS_3DGS_IOS_SFM_LIBRARY.
#   Android: scripts/build_android_sfm.ps1 builds libcross3dgs_sfm_jni.so
#            (cross3dgs_sfm plus the JNI wrapper in jni/), which the :e2e APK
#            packages through -PsfmJniDir.
cmake_minimum_required(VERSION 3.28)
project(cross3dgs_sfm LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME MATCHES "^(iOS|Android)$")
    message(FATAL_ERROR "Configure with CMAKE_SYSTEM_NAME iOS or Android "
        "(see scripts/build_ios_sfm.sh and scripts/build_android_sfm.ps1)")
endif()

set(CROSS3DGS_SFM_COLMAP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../external/colmap"
    CACHE PATH "COLMAP fork source (the external/colmap submodule)")
if(NOT EXISTS "${CROSS3DGS_SFM_COLMAP_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "COLMAP fork not found at ${CROSS3DGS_SFM_COLMAP_SOURCE}; "
        "run git submodule update --init external/colmap")
endif()

# CACHE FORCE: COLMAP's own CMakeLists runs under policy CMP0077=OLD, where
# option() would overwrite plain variables set here.
set(CUDA_ENABLED OFF CACHE BOOL "" FORCE)
set(GUI_ENABLED OFF CACHE BOOL "" FORCE)
set(MVS_ENABLED OFF CACHE BOOL "" FORCE)
set(OPENGL_ENABLED OFF CACHE BOOL "" FORCE)
set(ONNX_ENABLED OFF CACHE BOOL "" FORCE)
set(CGAL_ENABLED OFF CACHE BOOL "" FORCE)
set(DOWNLOAD_ENABLED OFF CACHE BOOL "" FORCE)
set(CHOLMOD_ENABLED OFF CACHE BOOL "" FORCE)
set(TESTS_ENABLED OFF CACHE BOOL "" FORCE)
set(IPO_ENABLED OFF CACHE BOOL "" FORCE)
set(SIMD_ENABLED OFF CACHE BOOL "" FORCE)
set(UNINSTALL_ENABLED OFF CACHE BOOL "" FORCE)
set(CCACHE_ENABLED OFF CACHE BOOL "" FORCE)

add_subdirectory("${CROSS3DGS_SFM_COLMAP_SOURCE}" colmap)

# COLMAP's own SQLite target is scoped to its directory; the live preview
# reads the database through a separate read-only connection.
find_package(SQLite3 REQUIRED)

add_library(cross3dgs_sfm STATIC src/cross3dgs_sfm.cc src/sfm_preview.cc)
target_link_libraries(cross3dgs_sfm PRIVATE
    colmap_controllers
    colmap_scene
    colmap_sensor
    colmap_util
    SQLite::SQLite3)
target_include_directories(cross3dgs_sfm
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CROSS3DGS_SFM_COLMAP_SOURCE}/src")

if(ANDROID)
    find_package(OpenBLAS CONFIG REQUIRED COMPONENTS serial)
    target_link_libraries(cross3dgs_sfm PRIVATE OpenBLAS::OpenBLAS)
    add_library(cross3dgs_sfm_jni SHARED jni/cross3dgs_sfm_jni.cc)
    target_link_libraries(cross3dgs_sfm_jni PRIVATE cross3dgs_sfm log)
    set_target_properties(cross3dgs_sfm_jni PROPERTIES
        LINK_FLAGS "-static-openmp -Wl,-z,max-page-size=16384")

    option(CROSS3DGS_SFM_BUILD_BLAS_SMOKE
        "Build an on-device concurrent BLAS correctness smoke (no timings)" OFF)
    if(CROSS3DGS_SFM_BUILD_BLAS_SMOKE)
        find_package(Threads REQUIRED)
        add_executable(cross3dgs_sfm_blas_smoke tests/blas_smoke.cc)
        target_compile_features(cross3dgs_sfm_blas_smoke PRIVATE cxx_std_17)
        target_link_libraries(cross3dgs_sfm_blas_smoke PRIVATE
            OpenBLAS::OpenBLAS Threads::Threads)
        target_link_options(cross3dgs_sfm_blas_smoke PRIVATE
            "-Wl,-z,max-page-size=16384")
    endif()
endif()

#!/usr/bin/env bash
# Builds libcross3dgs_sfm.a: the CPU COLMAP sparse SfM shared with Android
# (libs/cross3dgs_sfm, COLMAP fork in external/colmap) and every static
# dependency, merged into one arm64 iOS archive for the capture-to-view app
# (ios/CMakeLists.txt, -DCROSS_3DGS_IOS_SFM_LIBRARY=<out>/libcross3dgs_sfm.a).
#
# iOS counterpart of scripts/build_android_sfm.ps1:
#   1. libomp for iOS from the LLVM 19.1.7 OpenMP runtime release sources
#      (Apple clang ships no OpenMP runtime; COLMAP and faiss require one),
#   2. vcpkg dependencies for arm64 iOS (libs/cross3dgs_sfm/vcpkg.json,
#      overlay triplet and the ports/ios gklib port; BLAS/LAPACK come from
#      Accelerate),
#   3. COLMAP + the C bridge, then one merged archive.
#
# Usage:
#   scripts/build_ios_sfm.sh --vcpkg-root <vcpkg root> [--out build-ios-sfm]
# Host requirements: Xcode, CMake >= 3.28, Ninja, pkg-config, curl.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
vcpkg_root="${VCPKG_ROOT:-}"
out="${repo_root}/build-ios-sfm"
llvm_version="19.1.7"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vcpkg-root) vcpkg_root="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -f "${repo_root}/external/colmap/CMakeLists.txt" ]] || {
    echo "external/colmap is missing; run git submodule update --init external/colmap" >&2; exit 2; }
[[ -x "${vcpkg_root}/vcpkg" ]] || {
    echo "--vcpkg-root (or VCPKG_ROOT) must contain a bootstrapped vcpkg" >&2; exit 2; }
mkdir -p "${out}"
out="$(cd "${out}" && pwd)"
sfm_source="${repo_root}/libs/cross3dgs_sfm"

# 1. libomp (static, arm64 iOS).
omp_prefix="${out}/omp"
if [[ ! -f "${omp_prefix}/lib/libomp.a" ]]; then
    release="https://github.com/llvm/llvm-project/releases/download/llvmorg-${llvm_version}"
    mkdir -p "${out}/omp-src"
    curl -sSfL -o "${out}/omp-src/openmp.tar.xz" "${release}/openmp-${llvm_version}.src.tar.xz"
    curl -sSfL -o "${out}/omp-src/cmake.tar.xz" "${release}/cmake-${llvm_version}.src.tar.xz"
    tar -xf "${out}/omp-src/openmp.tar.xz" -C "${out}/omp-src"
    tar -xf "${out}/omp-src/cmake.tar.xz" -C "${out}/omp-src"
    rm -rf "${out}/omp-src/cmake"
    mv "${out}/omp-src/cmake-${llvm_version}.src" "${out}/omp-src/cmake"
    cmake -S "${out}/omp-src/openmp-${llvm_version}.src" -B "${out}/omp-build" -G Ninja \
        -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_BUILD_TYPE=Release \
        -DLIBOMP_ENABLE_SHARED=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
        -DLIBOMP_OMPT_SUPPORT=OFF -DOPENMP_ENABLE_OMPT_TOOLS=OFF \
        -DLIBOMP_USE_HWLOC=OFF -DCMAKE_INSTALL_PREFIX="${omp_prefix}"
    cmake --build "${out}/omp-build"
    cmake --install "${out}/omp-build"
fi

# 2+3. vcpkg dependencies are installed by the manifest-mode configure.
omp_flags="-Xclang -fopenmp -I${omp_prefix}/include"
cmake -S "${sfm_source}" -B "${out}/build" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${vcpkg_root}/scripts/buildsystems/vcpkg.cmake" \
    -DVCPKG_TARGET_TRIPLET=arm64-ios-cross3dgs \
    -DVCPKG_OVERLAY_TRIPLETS="${sfm_source}/triplets" \
    -DVCPKG_OVERLAY_PORTS="${sfm_source}/ports/ios" \
    -DVCPKG_INSTALLED_DIR="${out}/vcpkg_installed" \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBLA_VENDOR=Apple \
    -DOpenMP_C_FLAGS="${omp_flags}" -DOpenMP_CXX_FLAGS="${omp_flags}" \
    -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp \
    -DOpenMP_omp_LIBRARY="${omp_prefix}/lib/libomp.a"
cmake --build "${out}/build" --target cross3dgs_sfm

# Merge: every static archive the build produced (COLMAP modules, fetched
# PoseLib/faiss, the bridge), every vcpkg dependency, and libomp. Archive
# members are pulled into the app only when referenced.
archives=()
while IFS= read -r archive; do archives+=("${archive}"); done < <(
    { find "${out}/build" -name '*.a' -type f
      find "${out}/vcpkg_installed/arm64-ios-cross3dgs/lib" -maxdepth 1 -name '*.a' -type f
      echo "${omp_prefix}/lib/libomp.a"; } | sort -u)
rm -f "${out}/libcross3dgs_sfm.a"
xcrun libtool -static -no_warning_for_no_symbols -o "${out}/libcross3dgs_sfm.a" "${archives[@]}" 2>/dev/null
cp "${sfm_source}/include/cross3dgs_sfm.h" "${out}/cross3dgs_sfm.h"
echo "merged ${#archives[@]} archives -> ${out}/libcross3dgs_sfm.a ($(du -h "${out}/libcross3dgs_sfm.a" | cut -f1))"

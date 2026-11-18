# Static arm64 iOS triplet for the capture-to-view SfM dependencies. The
# deployment target matches ios/CMakeLists.txt; release-only keeps the build
# time and the merged static library small.
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME iOS)
set(VCPKG_OSX_DEPLOYMENT_TARGET 15.0)
set(VCPKG_BUILD_TYPE release)

# OpenColorIO (an OpenImageIO dependency) queries display ICC profiles via
# IOKit/graphics on __APPLE__, which the iOS SDK lacks. Its headless branch
# compiles the empty monitor list instead; nothing else in the library
# reads the define.
if(PORT STREQUAL "opencolorio")
    set(VCPKG_CXX_FLAGS "-DOCIO_HEADLESS_ENABLED")
    set(VCPKG_C_FLAGS "")
endif()

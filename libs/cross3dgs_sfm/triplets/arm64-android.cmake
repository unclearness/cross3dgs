# Overlay copy of vcpkg's arm64-android triplet with project-wide settings
# pinned in one place: Android API level 28, static libraries, and 16 KB
# page-size alignment for Android 15+ devices.
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Android)
set(VCPKG_CMAKE_SYSTEM_VERSION 28)
set(VCPKG_MAKE_BUILD_TRIPLET "--host=aarch64-linux-android")
set(VCPKG_CMAKE_CONFIGURE_OPTIONS
    -DANDROID_ABI=arm64-v8a
    -DANDROID_PLATFORM=android-28
    -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON)
set(VCPKG_LINKER_FLAGS "-Wl,-z,max-page-size=16384")

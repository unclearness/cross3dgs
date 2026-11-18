# cross_3dgs iOS bench app (Phase I0 — engineering only).
#
# Configure with the Xcode generator and the iOS system name; everything else
# (glslang/SPIRV source build, MoltenVK static link, no Vulkan loader) follows
# from that:
#
#   cmake -S ios -B build-ios -G Xcode -DCMAKE_SYSTEM_NAME=iOS \
#         -DCROSS_3DGS_IOS_TEAM=<team id>
#   cmake --build build-ios --config Release -- -allowProvisioningUpdates
#
# See ios/README.md for signing, install, launch, and artifact retrieval.
cmake_minimum_required(VERSION 3.28)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(FATAL_ERROR "Configure with -DCMAKE_SYSTEM_NAME=iOS (got '${CMAKE_SYSTEM_NAME}')")
endif()

# os_proc_available_memory needs iOS 13; empty-dict UILaunchScreen needs 14.
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum iOS version")

project(cross3dgs_ios CXX C OBJC OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Same forced-off set as the Android app CMakeLists.
set(CROSS_3DGS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CROSS_3DGS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(CROSS_3DGS_BUILD_VIEWER OFF CACHE BOOL "" FORCE)

get_filename_component(CROSS_3DGS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_subdirectory("${CROSS_3DGS_ROOT}" cross3dgs_root)

# MoltenVK: iOS has no Vulkan loader, so the static ICD from the macOS Vulkan
# SDK is linked directly (ios-arm64 slice of the shipped xcframework).
set(CROSS_3DGS_MOLTENVK_XCFRAMEWORK "" CACHE PATH
    "Path to MoltenVK.xcframework (default: newest $HOME/VulkanSDK/*/macOS/lib)")
if(NOT CROSS_3DGS_MOLTENVK_XCFRAMEWORK)
    file(GLOB _mvk_candidates "$ENV{HOME}/VulkanSDK/*/macOS/lib/MoltenVK.xcframework")
    if(_mvk_candidates)
        list(SORT _mvk_candidates)
        list(GET _mvk_candidates -1 CROSS_3DGS_MOLTENVK_XCFRAMEWORK)
    endif()
endif()
set(_mvk_lib "${CROSS_3DGS_MOLTENVK_XCFRAMEWORK}/ios-arm64/libMoltenVK.a")
if(NOT EXISTS "${_mvk_lib}")
    message(FATAL_ERROR "MoltenVK ios-arm64 static library not found: ${_mvk_lib}\n"
        "Install the LunarG Vulkan SDK for macOS or set CROSS_3DGS_MOLTENVK_XCFRAMEWORK.")
endif()
message(STATUS "MoltenVK (iOS): ${_mvk_lib}")

add_executable(cross3dgs_ios MACOSX_BUNDLE
    app/main.mm
    app/AppDelegate.mm
    # Platform-neutral Vulkan capability probe shared verbatim with the
    # Android app (the namespace name is historical).
    "${CROSS_3DGS_ROOT}/android/app/src/main/cpp/device_caps.cpp"
    # The PC CLI entry point, renamed via CROSS_3DGS_TRAIN_MAIN_AS_LIBRARY so
    # the app can call it with a synthesized argv. This keeps the argument
    # surface and bench.json output identical to the desktop path.
    "${CROSS_3DGS_ROOT}/tools/train_cli/main.cpp"
    # The selected-schedule CPU double oracle, compiled as a library the same
    # way the Android app embeds it, so the oracle runs on the measured
    # device (launch with --ios-cpu-oracle <output dir>).
    "${CROSS_3DGS_ROOT}/tests/gpu/test_cpu_oracle.cpp")

target_compile_definitions(cross3dgs_ios PRIVATE
    CROSS_3DGS_VIEWER=0
    CROSS_3DGS_TRAIN_MAIN_AS_LIBRARY=1
    CROSS_3DGS_CPU_ORACLE_LIBRARY=1)

target_link_libraries(cross3dgs_ios PRIVATE
    cross3dgs
    spdlog::spdlog
    "${_mvk_lib}"
    "-framework Metal"
    "-framework QuartzCore"
    "-framework IOSurface"
    "-framework CoreGraphics"
    "-framework CoreFoundation"
    "-framework Foundation"
    "-framework UIKit")

# Bundled dataset: the app reads the scene directly from the bundle (no
# first-launch extraction, unlike Android's AssetInstaller).
option(CROSS_3DGS_IOS_BUNDLE_DATASET
    "Bundle data/360_v2 scenes (images_4 + sparse/0) into the app" ON)
# Which Mip-NeRF 360 scenes to bundle (semicolon-separated), for example
# "bicycle;garden".
set(CROSS_3DGS_IOS_BUNDLE_SCENES "bicycle" CACHE STRING
    "Semicolon-separated data/360_v2 scenes to bundle into the app")
if(CROSS_3DGS_IOS_BUNDLE_DATASET)
    foreach(_ios_scene IN LISTS CROSS_3DGS_IOS_BUNDLE_SCENES)
        file(GLOB _ios_images
            "${CROSS_3DGS_ROOT}/data/360_v2/${_ios_scene}/images_4/*")
        file(GLOB _ios_sparse
            "${CROSS_3DGS_ROOT}/data/360_v2/${_ios_scene}/sparse/0/*")
        if(NOT _ios_images OR NOT _ios_sparse)
            message(FATAL_ERROR "data/360_v2/${_ios_scene} is incomplete; provision the dataset "
                "or configure with -DCROSS_3DGS_IOS_BUNDLE_DATASET=OFF")
        endif()
        set_source_files_properties(${_ios_images} PROPERTIES
            MACOSX_PACKAGE_LOCATION "data/360_v2/${_ios_scene}/images_4")
        set_source_files_properties(${_ios_sparse} PROPERTIES
            MACOSX_PACKAGE_LOCATION "data/360_v2/${_ios_scene}/sparse/0")
        target_sources(cross3dgs_ios PRIVATE ${_ios_images} ${_ios_sparse})
    endforeach()
endif()

set(CROSS_3DGS_IOS_TEAM "" CACHE STRING "Apple Development team id for code signing")
option(CROSS_3DGS_IOS_INCREASED_MEMORY_LIMIT
    "Sign with com.apple.developer.kernel.increased-memory-limit" ON)

set_target_properties(cross3dgs_ios PROPERTIES
    OUTPUT_NAME "cross3dgs"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/app/Info.plist.in"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "org.unclearness.cross3dgs"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC "YES"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
    XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${CROSS_3DGS_IOS_TEAM}")
if(CROSS_3DGS_IOS_INCREASED_MEMORY_LIMIT)
    set_target_properties(cross3dgs_ios PROPERTIES
        XCODE_ATTRIBUTE_CODE_SIGN_ENTITLEMENTS
        "${CMAKE_CURRENT_SOURCE_DIR}/app/cross3dgs.entitlements")
endif()

# ---------------------------------------------------------------------------
# Capture-to-view app (iOS counterpart of the Android :e2e module): ARKit
# capture -> frame extraction -> CPU COLMAP SfM -> Cross3DGS optimization ->
# free-view viewer. A separate app with its own bundle id, like the separate
# Android application; the bench app above gains no camera/SfM code.
# Needs the merged SfM archive from scripts/build_ios_sfm.sh.
set(CROSS_3DGS_IOS_SFM_LIBRARY "" CACHE FILEPATH
    "libcross3dgs_sfm.a from scripts/build_ios_sfm.sh; enables the cross3dgs_e2e_ios target")
if(CROSS_3DGS_IOS_SFM_LIBRARY)
    if(NOT EXISTS "${CROSS_3DGS_IOS_SFM_LIBRARY}")
        message(FATAL_ERROR "CROSS_3DGS_IOS_SFM_LIBRARY not found: ${CROSS_3DGS_IOS_SFM_LIBRARY}")
    endif()
    set(_e2e_dir "${CMAKE_CURRENT_SOURCE_DIR}/e2e/app")
    add_executable(cross3dgs_e2e_ios MACOSX_BUNDLE
        "${_e2e_dir}/main.mm"
        "${_e2e_dir}/E2eViewController.mm"
        "${_e2e_dir}/E2eCaptureViewController.mm"
        "${_e2e_dir}/E2eFrameExtractor.mm"
        "${_e2e_dir}/E2eProject.mm"
        "${_e2e_dir}/E2eTelemetry.mm"
        "${_e2e_dir}/E2eEngine.cpp"
        "${_e2e_dir}/E2ePipeline.cpp"
        "${CROSS_3DGS_ROOT}/android/app/src/main/cpp/device_caps.cpp")
    target_include_directories(cross3dgs_e2e_ios PRIVATE
        "${_e2e_dir}"
        "${CROSS_3DGS_ROOT}/libs/cross3dgs_sfm/include")
    target_link_libraries(cross3dgs_e2e_ios PRIVATE
        cross3dgs
        spdlog::spdlog
        "${_mvk_lib}"
        # -force_load is not needed: the bridge's two C entry points pull in
        # everything they reference from the merged archive.
        "${CROSS_3DGS_IOS_SFM_LIBRARY}"
        z
        compression
        "-framework Accelerate"
        "-framework ARKit"
        "-framework AVFoundation"
        "-framework CoreMedia"
        "-framework CoreMotion"
        "-framework CoreVideo"
        "-framework ImageIO"
        "-framework SceneKit"
        "-framework UniformTypeIdentifiers"
        "-framework Metal"
        "-framework QuartzCore"
        "-framework IOSurface"
        "-framework CoreGraphics"
        "-framework CoreFoundation"
        "-framework Foundation"
        "-framework UIKit")
    set_target_properties(cross3dgs_e2e_ios PROPERTIES
        OUTPUT_NAME "cross3dgs_e2e"
        MACOSX_BUNDLE_INFO_PLIST "${_e2e_dir}/Info.plist.in"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "org.unclearness.cross3dgs.e2e"
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC "YES"
        XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
        XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${CROSS_3DGS_IOS_TEAM}")
    if(CROSS_3DGS_IOS_INCREASED_MEMORY_LIMIT)
        set_target_properties(cross3dgs_e2e_ios PROPERTIES
            XCODE_ATTRIBUTE_CODE_SIGN_ENTITLEMENTS
            "${CMAKE_CURRENT_SOURCE_DIR}/app/cross3dgs.entitlements")
    endif()
endif()

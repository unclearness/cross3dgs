# Third-party dependencies of the Vulkan compute runtime (libs/cross3dgs_gpu).
# Added with EXCLUDE_FROM_ALL from the top-level CMakeLists, so only what the
# cross_3dgs targets actually link is built.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
if(NOT ANDROID AND NOT IOS)
    # The desktop Vulkan SDK provides headers/libs (glslang etc.). On Android
    # and iOS everything comes from source-built Khronos repos instead (the
    # SDK archives are host-arch only); find_package(Vulkan) would create a
    # Vulkan::Headers target that collides with the fetched Vulkan-Headers
    # project, so it is skipped there.
    include(FindVulkanSdk)
    find_package(Vulkan REQUIRED)
endif()

include(FetchContent)

# glslang (runtime GLSL -> SPIR-V compilation for vkw).
# Desktop: from the Vulkan SDK — its glslang-config.cmake is broken
# (references a nonexistent <SDK>/Lib/lib/... path), so link the static libs
# directly. Android/iOS: the SDK ships host-arch archives only, so build the
# whole SPIR-V stack from source, pinned to the SDK's lockstep tag (the three
# repos tag every SDK release in mutually compatible states).
add_library(cross3dgs_glslang INTERFACE)
set(_glslang_libs
    glslang SPIRV MachineIndependent GenericCodeGen OSDependent
    glslang-default-resource-limits SPIRV-Tools-opt SPIRV-Tools)
if(ANDROID OR IOS)
    set(CROSS_3DGS_GLSLANG_TAG "vulkan-sdk-1.4.350.0"
        CACHE STRING "Lockstep vulkan-sdk tag for glslang/SPIRV-Tools/SPIRV-Headers")
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
    set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
    set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
    set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
    # ENABLE_OPT=1 matches the desktop cross3dgs_vkw configuration.
    set(ENABLE_OPT ON CACHE BOOL "" FORCE)
    set(ALLOW_EXTERNAL_SPIRV_TOOLS ON CACHE BOOL "" FORCE)
    FetchContent_Declare(spirv-headers
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
        GIT_TAG ${CROSS_3DGS_GLSLANG_TAG}
        GIT_SHALLOW TRUE)
    # iOS uses the Xcode generator, which rejects this pinned SPIRV-Tools
    # checkout as-is (see cmake/PatchSpirvToolsXcode.cmake). The patch is
    # applied at population, and only for iOS so the Android build inputs
    # stay byte-identical.
    if(IOS)
        set(_spirv_tools_patch PATCH_COMMAND ${CMAKE_COMMAND} -P
            "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/PatchSpirvToolsXcode.cmake")
    else()
        set(_spirv_tools_patch "")
    endif()
    FetchContent_Declare(spirv-tools
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
        GIT_TAG ${CROSS_3DGS_GLSLANG_TAG}
        GIT_SHALLOW TRUE
        ${_spirv_tools_patch})
    FetchContent_Declare(glslang
        GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
        GIT_TAG ${CROSS_3DGS_GLSLANG_TAG}
        GIT_SHALLOW TRUE)
    # The NDK sysroot ships only the C headers (vulkan/vulkan.h); vkw needs
    # Vulkan-Hpp (vulkan/vulkan.hpp). Fetch the lockstep Vulkan-Headers and
    # let its include dir shadow the sysroot's older vulkan.h so the .h/.hpp
    # pair stays consistent (and matches the desktop SDK version).
    FetchContent_Declare(vulkan-headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG ${CROSS_3DGS_GLSLANG_TAG}
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(vulkan-headers)
    FetchContent_MakeAvailable(spirv-headers)
    # SPIRV-Tools looks for its headers via SPIRV-Headers_SOURCE_DIR;
    # FetchContent only sets the lowercased spirv-headers_SOURCE_DIR.
    set(SPIRV-Headers_SOURCE_DIR "${spirv-headers_SOURCE_DIR}" CACHE PATH "" FORCE)
    FetchContent_MakeAvailable(spirv-tools glslang)
    # Source-built targets propagate their own include dirs; no manual paths.
    target_link_libraries(cross3dgs_glslang INTERFACE
        glslang SPIRV glslang-default-resource-limits SPIRV-Tools-opt SPIRV-Tools)
elseif(WIN32)
    # Windows SDK nests glslang's whole tree under Include/glslang, so that one
    # include dir resolves both <SPIRV/...> and <glslang/Public/...>. Libs are
    # <name>.lib / <name>d.lib with per-config debug ("d" suffix) selection.
    target_include_directories(cross3dgs_glslang SYSTEM INTERFACE
        "${CROSS_3DGS_VULKAN_SDK_INCLUDE}/glslang")
    foreach(_lib IN LISTS _glslang_libs)
        target_link_libraries(cross3dgs_glslang INTERFACE
            "$<IF:$<CONFIG:Debug>,${CROSS_3DGS_VULKAN_SDK_LIB}/${_lib}d.lib,${CROSS_3DGS_VULKAN_SDK_LIB}/${_lib}.lib>")
    endforeach()
elseif(APPLE)
    # The macOS SDK lays its headers out like the Linux one, but ld64 rejects
    # -Wl,--start-group. It does not need the group: ld64 resolves circular
    # static archives by rescanning the whole command line, so listing the
    # archives plainly is enough.
    target_include_directories(cross3dgs_glslang SYSTEM INTERFACE
        "${CROSS_3DGS_VULKAN_SDK_INCLUDE}"
        "${CROSS_3DGS_VULKAN_SDK_INCLUDE}/glslang")
    foreach(_lib IN LISTS _glslang_libs)
        target_link_libraries(cross3dgs_glslang INTERFACE
            "${CROSS_3DGS_VULKAN_SDK_LIB}/lib${_lib}.a")
    endforeach()
else()
    # Linux SDK flattens glslang's headers directly under include/glslang, so
    # <SPIRV/...> resolves under include/glslang while <glslang/Public/...>
    # resolves under include; add both. The archives are static lib<name>.a
    # with circular interdependencies (glslang <-> MachineIndependent <-> SPIRV),
    # so wrap them in a link group for GNU ld to resolve regardless of order.
    target_include_directories(cross3dgs_glslang SYSTEM INTERFACE
        "${CROSS_3DGS_VULKAN_SDK_INCLUDE}"
        "${CROSS_3DGS_VULKAN_SDK_INCLUDE}/glslang")
    set(_glslang_paths "")
    foreach(_lib IN LISTS _glslang_libs)
        list(APPEND _glslang_paths "${CROSS_3DGS_VULKAN_SDK_LIB}/lib${_lib}.a")
    endforeach()
    target_link_libraries(cross3dgs_glslang INTERFACE
        -Wl,--start-group ${_glslang_paths} -Wl,--end-group)
endif()

# GLFW: vkw.h includes <GLFW/glfw3.h> on desktop even for headless use.
# On Android vkw.h switches to <android/native_window.h>; on iOS it is
# windowless. GLFW is unused on both.
if(NOT ANDROID AND NOT IOS)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
if(UNIX AND NOT APPLE)
    # Build the X11 backend only. Wayland would additionally require
    # wayland-scanner + libwayland-dev + libxkbcommon-dev; X11-only GLFW still
    # runs under Wayland compositors via XWayland, so this keeps the build
    # dependency-light. Override with -DGLFW_BUILD_WAYLAND=ON if desired.
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glfw)
endif()

# stb image loaders (header-only usage by the dataset loader).
# stb has no release tags; pin an explicit commit for reproducibility (never
# track the moving `master`). This matches VulkanWrapper's STB_TAG so both stb
# copies in the build are identical. GIT_SHALLOW is OFF because a shallow clone
# cannot reliably reference an arbitrary commit hash.
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG c9064e317699d2e495f36ba4f9ac037e88ee371a
    GIT_SHALLOW FALSE
    SOURCE_SUBDIR nonexistent_disable_add_subdirectory)
FetchContent_MakeAvailable(stb)
add_library(cross3dgs_stb INTERFACE)
target_include_directories(cross3dgs_stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")

# Logging / performance measurement
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(spdlog)

# ------------------------------------------------------------------------------
# cross3dgs_vkw: takiyu/VulkanWrapper compiled by our own target (its bundled
# CMake/third_party predate CMake 4.x and current glslang, so we do not
# add_subdirectory it; we link against the Vulkan SDK instead).
# ------------------------------------------------------------------------------
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VulkanWrapper/src/vkw.cpp")
    message(FATAL_ERROR
        "external/VulkanWrapper is missing. Run:\n"
        "  git submodule update --init --recursive")
endif()
add_library(cross3dgs_vkw STATIC
    VulkanWrapper/src/vkw.cpp)
target_include_directories(cross3dgs_vkw SYSTEM PUBLIC
    VulkanWrapper/include)
# Dynamic dispatch (upstream vkw's mode): debug-utils/report extension
# functions are unresolvable with the static dispatcher + vulkan-1.lib.
# iOS has no Vulkan loader to dlopen: MoltenVK is linked statically by the
# app, so the dispatcher is initialized from the linked vkGetInstanceProcAddr
# instead (vkw.cpp), and the dlopen tool is disabled.
if(IOS)
    target_compile_definitions(cross3dgs_vkw PUBLIC
        VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=0
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        ENABLE_OPT=1)
else()
    target_compile_definitions(cross3dgs_vkw PUBLIC
        VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        ENABLE_OPT=1)
endif()
if(WIN32)
    target_compile_definitions(cross3dgs_vkw PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_link_libraries(cross3dgs_vkw PUBLIC cross3dgs_glslang)
if(ANDROID)
    # Vulkan::Headers (the fetched 1.4.x set, .h and .hpp consistent) must
    # shadow the sysroot's older vulkan.h; the NDK's libvulkan.so is linked
    # directly. vkw.h uses ANativeWindow + __android_log under __ANDROID__.
    target_link_libraries(cross3dgs_vkw PUBLIC Vulkan::Headers vulkan android log)
elseif(IOS)
    # Headers only: there is no loader library. The final app links
    # MoltenVK.xcframework (static) plus its Metal/QuartzCore/IOSurface
    # framework dependencies, which resolves vkGetInstanceProcAddr.
    target_link_libraries(cross3dgs_vkw PUBLIC Vulkan::Headers)
else()
    target_link_libraries(cross3dgs_vkw PUBLIC Vulkan::Vulkan glfw)
endif()
if(NOT WIN32)
    # The glslang/SPIRV-Tools static archives pull in pthread and dlopen.
    find_package(Threads REQUIRED)
    target_link_libraries(cross3dgs_vkw PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
if(MSVC)
    target_compile_options(cross3dgs_vkw PRIVATE /bigobj)
endif()

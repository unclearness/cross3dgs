# Locate the Vulkan SDK even when the VULKAN_SDK environment variable is unset
# (common on machines where the SDK installer ran without a shell restart, or
# where the Linux tarball's setup-env.sh was not sourced).
#
# Priority:
#   1. CROSS_3DGS_VULKAN_SDK_DIR cache variable (manual override)
#   2. VULKAN_SDK environment variable
#   3. Newest version directory under the platform's usual install root
#      (Windows: C:/VulkanSDK/<ver>; Linux: ~/vulkan/<ver>/x86_64, the layout
#       the LunarG tarball extracts to and that setup-env.sh points at).
#
# Result: ENV{VULKAN_SDK} is set so that CMake's FindVulkan module works, and
# CROSS_3DGS_VULKAN_SDK_ROOT / _INCLUDE / _LIB expose the SDK's include and
# library directories with the correct casing for the host OS (Windows
# Include/Lib vs Linux include/lib), so downstream targets can link glslang
# directly.

set(CROSS_3DGS_VULKAN_SDK_DIR "" CACHE PATH "Manual Vulkan SDK root override")

if(CROSS_3DGS_VULKAN_SDK_DIR)
    set(_cross_3dgs_sdk_root "${CROSS_3DGS_VULKAN_SDK_DIR}")
elseif(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
    set(_cross_3dgs_sdk_root "$ENV{VULKAN_SDK}")
elseif(WIN32)
    file(GLOB _cross_3dgs_sdk_candidates "C:/VulkanSDK/*")
    set(_cross_3dgs_sdk_root "")
    foreach(_cand IN LISTS _cross_3dgs_sdk_candidates)
        if(IS_DIRECTORY "${_cand}" AND EXISTS "${_cand}/Include/vulkan/vulkan.h")
            if("${_cand}" STRGREATER "${_cross_3dgs_sdk_root}")
                set(_cross_3dgs_sdk_root "${_cand}")
            endif()
        endif()
    endforeach()
else()
    # Linux: the LunarG tarball extracts to <dir>/<version>/x86_64 and its
    # setup-env.sh sets VULKAN_SDK to that x86_64 dir.
    # macOS: the LunarG installer writes <dir>/<version>/macOS instead, and the
    # arch component is "macOS" on both Apple Silicon and Intel, so none of the
    # x86_64 globs ever match there.
    # If setup-env.sh was never sourced, discover the newest SDK under the
    # conventional install roots for the host.
    set(_cross_3dgs_sdk_root "")
    file(GLOB _cross_3dgs_sdk_candidates
        "$ENV{HOME}/vulkan/*/x86_64"
        "$ENV{HOME}/VulkanSDK/*/x86_64"
        "/opt/vulkan/*/x86_64"
        "$ENV{HOME}/vulkan/*/macOS"
        "$ENV{HOME}/VulkanSDK/*/macOS"
        "/opt/vulkan/*/macOS")
    foreach(_cand IN LISTS _cross_3dgs_sdk_candidates)
        if(IS_DIRECTORY "${_cand}" AND EXISTS "${_cand}/include/vulkan/vulkan.h")
            if("${_cand}" STRGREATER "${_cross_3dgs_sdk_root}")
                set(_cross_3dgs_sdk_root "${_cand}")
            endif()
        endif()
    endforeach()
endif()

# Resolve the include/lib subdirectory casing (Windows Include/Lib, Linux
# include/lib) and validate that we actually found a usable SDK.
set(CROSS_3DGS_VULKAN_SDK_INCLUDE "")
set(CROSS_3DGS_VULKAN_SDK_LIB "")
if(_cross_3dgs_sdk_root)
    if(EXISTS "${_cross_3dgs_sdk_root}/Include/vulkan/vulkan.h")
        set(CROSS_3DGS_VULKAN_SDK_INCLUDE "${_cross_3dgs_sdk_root}/Include")
    elseif(EXISTS "${_cross_3dgs_sdk_root}/include/vulkan/vulkan.h")
        set(CROSS_3DGS_VULKAN_SDK_INCLUDE "${_cross_3dgs_sdk_root}/include")
    endif()
    if(IS_DIRECTORY "${_cross_3dgs_sdk_root}/Lib")
        set(CROSS_3DGS_VULKAN_SDK_LIB "${_cross_3dgs_sdk_root}/Lib")
    elseif(IS_DIRECTORY "${_cross_3dgs_sdk_root}/lib")
        set(CROSS_3DGS_VULKAN_SDK_LIB "${_cross_3dgs_sdk_root}/lib")
    endif()
endif()

if(NOT _cross_3dgs_sdk_root OR NOT CROSS_3DGS_VULKAN_SDK_INCLUDE
   OR NOT CROSS_3DGS_VULKAN_SDK_LIB)
    message(FATAL_ERROR
        "Vulkan SDK not found. Install it or set -DCROSS_3DGS_VULKAN_SDK_DIR=<path>.\n"
        "  Windows: the LunarG installer under C:/VulkanSDK/<ver>.\n"
        "  Linux:   the LunarG tarball; point at its x86_64 dir (e.g.\n"
        "           ~/vulkan/<ver>/x86_64) or source its setup-env.sh first.\n"
        "  macOS:   the LunarG installer; point at its macOS dir (e.g.\n"
        "           ~/VulkanSDK/<ver>/macOS) or source its setup-env.sh first.")
endif()

set(ENV{VULKAN_SDK} "${_cross_3dgs_sdk_root}")
list(APPEND CMAKE_PREFIX_PATH "${CROSS_3DGS_VULKAN_SDK_LIB}/cmake")
set(CROSS_3DGS_VULKAN_SDK_ROOT "${_cross_3dgs_sdk_root}")
message(STATUS "Vulkan SDK: ${CROSS_3DGS_VULKAN_SDK_ROOT}")

# Xcode-generator compatibility patch for the pinned SPIRV-Tools checkout
# (vulkan-sdk-1.4.350.0), applied as a FetchContent PATCH_COMMAND for iOS
# builds only (Android/desktop use Ninja/MSBuild and do not need it).
#
# Xcode's "new build system" refuses a custom command whose output is attached
# to sibling targets (SPIRV-Tools-static / SPIRV-Tools-shared) with no common
# producing dependency. The grammar tables in source/CMakeLists.txt already
# use the sanctioned pattern — a producing add_custom_target that both library
# targets depend on via add_dependencies — but build-version.inc is attached
# through a file-level OBJECT_DEPENDS on software_version.cpp instead, which
# pulls the custom command into every target compiling that file. Rewrite it
# to the tables pattern: a producing target, target-level dependencies, and no
# OBJECT_DEPENDS. (The OBJECT_DEPENDS existed to force regeneration when the
# .inc changes; the target-level dependency preserves the clean-build ordering
# that matters here.) Idempotent: a re-run leaves a patched tree unchanged.
#
# Runs with the SPIRV-Tools source dir as the working directory.
set(_file "source/CMakeLists.txt")
file(READ "${_file}" _content)
if(NOT _content MATCHES "spirv-tools-build-version")
    # 1. Producing target for build-version.inc.
    set(_anchor
        "COMMENT \"Update build-version.inc in the SPIRV-Tools build directory (if necessary).\")")
    string(REPLACE
        "${_anchor}"
        "${_anchor}\nadd_custom_target(spirv-tools-build-version\n  DEPENDS \${SPIRV_TOOLS_BUILD_VERSION_INC})"
        _patched "${_content}")
    if(_patched STREQUAL _content)
        message(FATAL_ERROR "PatchSpirvToolsXcode: custom-command anchor not found in ${_file}")
    endif()
    # 2. Drop the file-level attachment that names both libraries as
    #    consumers of the custom command.
    set(_object_depends "set_source_files_properties(
  \${CMAKE_CURRENT_SOURCE_DIR}/software_version.cpp
  PROPERTIES OBJECT_DEPENDS \"\${SPIRV_TOOLS_BUILD_VERSION_INC}\")")
    string(REPLACE "${_object_depends}" "" _patched2 "${_patched}")
    if(_patched2 STREQUAL _patched)
        message(FATAL_ERROR "PatchSpirvToolsXcode: OBJECT_DEPENDS anchor not found in ${_file}")
    endif()
    # 3. Target-level dependency alongside the existing tables targets.
    set(_add_deps "add_dependencies(\${target} core_tables extinst_tables)")
    string(REPLACE "${_add_deps}"
        "add_dependencies(\${target} core_tables extinst_tables spirv-tools-build-version)"
        _patched3 "${_patched2}")
    if(_patched3 STREQUAL _patched2)
        message(FATAL_ERROR "PatchSpirvToolsXcode: add_dependencies anchor not found in ${_file}")
    endif()
    file(WRITE "${_file}" "${_patched3}")
    message(STATUS "PatchSpirvToolsXcode: build-version.inc rewired to the tables pattern")
endif()

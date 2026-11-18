# Vulkan compute runtime: pre-recorded multi-pass compute programs with
# explicit buffers and barriers, runtime GLSL -> SPIR-V compilation, and the
# device capability query. Nothing here is specific to Gaussian splatting.
add_library(cross3dgs_gpu STATIC
    src/gpu_primitives.cpp
    src/gpu_program.cpp
    src/vk_context.cpp)
target_include_directories(cross3dgs_gpu PUBLIC include PRIVATE src)
target_link_libraries(cross3dgs_gpu PUBLIC cross3dgs_vkw)
if(MSVC)
    target_compile_options(cross3dgs_gpu PRIVATE /W4 /permissive- /bigobj /utf-8)
endif()

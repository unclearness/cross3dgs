add_library(cross3dgs STATIC
    src/dataset.cpp
    src/gaussian_program.cpp
    src/orbit_view.cpp
    src/gaussian/backward_passes.cpp
    src/gaussian/density_passes.cpp
    src/gaussian/forward_passes.cpp
    src/gaussian/program_setup.cpp
    src/gaussian/optimizer_passes.cpp
    src/gaussian/shaders_common.cpp
    src/gaussian/shaders_forward.cpp
    src/gaussian/shaders_gather_backward.cpp
    src/gaussian/shaders_igs.cpp
    src/gaussian/shaders_loss.cpp
    src/gaussian/shaders_mcmc.cpp
    src/gaussian/shaders_optimizer.cpp
    src/gaussian/shaders_projection_backward.cpp
    src/gaussian/shaders_raster_backward.cpp
    src/radix_sort.cpp
    src/schedule_ledger.cpp
    src/gaussian_state.cpp
    src/training_preview.cpp)
target_include_directories(cross3dgs PUBLIC include)
# The reordered SH-rest layout is on for subgroups of 32 lanes or more, and,
# with this switch (default ON), for narrower subgroups too. OFF keeps Mali on
# the linear layout.
option(CROSS3DGS_SH_REORDER_NARROW_SUBGROUPS
       "Use the reordered SH-rest layout on subgroups narrower than 32" ON)
target_compile_definitions(cross3dgs PRIVATE
    CROSS3DGS_SH_REORDER_NARROW_SUBGROUPS=$<BOOL:${CROSS3DGS_SH_REORDER_NARROW_SUBGROUPS}>)
target_link_libraries(cross3dgs PUBLIC cross3dgs_gpu glm::glm
                      PRIVATE cross3dgs_stb spdlog::spdlog)
if(MSVC)
    target_compile_options(cross3dgs PRIVATE /W4 /permissive- /utf-8)
endif()

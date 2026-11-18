add_executable(cross_3dgs_train main.cpp)
target_link_libraries(cross_3dgs_train PRIVATE
    cross3dgs spdlog::spdlog)
if(CROSS_3DGS_BUILD_VIEWER)
    target_sources(cross_3dgs_train PRIVATE vk_viewer.cpp)
    target_link_libraries(cross_3dgs_train PRIVATE glfw)
    target_compile_definitions(cross_3dgs_train PRIVATE CROSS_3DGS_VIEWER=1)
else()
    target_compile_definitions(cross_3dgs_train PRIVATE CROSS_3DGS_VIEWER=0)
endif()
if(MSVC)
    target_compile_options(cross_3dgs_train PRIVATE /W4 /permissive- /utf-8)
endif()


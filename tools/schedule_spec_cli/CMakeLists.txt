add_executable(cross_3dgs_schedule_spec main.cpp)
target_link_libraries(cross_3dgs_schedule_spec PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_schedule_spec PRIVATE
        /W4 /permissive- /utf-8)
endif()

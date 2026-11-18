add_executable(cross_3dgs_tests cpu/test_dataset.cpp)
target_link_libraries(cross_3dgs_tests PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_tests PRIVATE /W4 /permissive- /utf-8)
endif()
set(CROSS3DGS_BICYCLE_TEST_ROOT "${CMAKE_SOURCE_DIR}/data/360_v2/bicycle")
if(EXISTS "${CROSS3DGS_BICYCLE_TEST_ROOT}")
    add_test(NAME Cross3dgs.DatasetBicycle
        COMMAND cross_3dgs_tests "${CROSS3DGS_BICYCLE_TEST_ROOT}")
else()
    message(STATUS
        "Mip-NeRF 360 bicycle is absent; skipping dataset-dependent CTest")
endif()

add_executable(cross_3dgs_forward_test gpu/test_gaussian_forward.cpp)
target_link_libraries(cross_3dgs_forward_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_forward_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.GaussianForward COMMAND cross_3dgs_forward_test)

# Validation/profiling probe. Kept out of CTest because it performs
# repeated timestamped GPU runs and writes structured artifacts on request.
add_executable(cross_3dgs_atomic_free_probe gpu/atomic_free_probe.cpp)
target_link_libraries(cross_3dgs_atomic_free_probe PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_atomic_free_probe PRIVATE /W4 /permissive- /utf-8)
endif()

add_executable(cross_3dgs_cpu_oracle_test gpu/test_cpu_oracle.cpp)
target_link_libraries(cross_3dgs_cpu_oracle_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_cpu_oracle_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.CpuDoubleOracle COMMAND cross_3dgs_cpu_oracle_test)

add_executable(cross_3dgs_schedule_ledger_test
    cpu/test_schedule_ledger.cpp)
target_link_libraries(cross_3dgs_schedule_ledger_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_schedule_ledger_test
        PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.ScheduleLedger
    COMMAND cross_3dgs_schedule_ledger_test)

add_executable(cross_3dgs_orbit_view_test cpu/test_orbit_view.cpp)
target_link_libraries(cross_3dgs_orbit_view_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_orbit_view_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.OrbitView COMMAND cross_3dgs_orbit_view_test)

add_executable(cross_3dgs_ply_export_test cpu/test_ply_export.cpp)
target_link_libraries(cross_3dgs_ply_export_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_ply_export_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.PlyExport COMMAND cross_3dgs_ply_export_test)

add_executable(cross_3dgs_training_preview_test cpu/test_training_preview.cpp)
target_link_libraries(cross_3dgs_training_preview_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_training_preview_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.TrainingPreview COMMAND cross_3dgs_training_preview_test)

add_executable(cross_3dgs_radix_test gpu/test_radix_sort.cpp)
target_link_libraries(cross_3dgs_radix_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_radix_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.RadixSort COMMAND cross_3dgs_radix_test)

add_executable(cross_3dgs_buffer_zero_init_test gpu/test_buffer_zero_init.cpp)
target_link_libraries(cross_3dgs_buffer_zero_init_test PRIVATE cross3dgs)
if(MSVC)
    target_compile_options(cross_3dgs_buffer_zero_init_test PRIVATE /W4 /permissive- /utf-8)
endif()
add_test(NAME Cross3dgs.BufferZeroInit COMMAND cross_3dgs_buffer_zero_init_test)

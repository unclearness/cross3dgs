# Android SfM OpenBLAS overlay

This is the OpenBLAS 0.3.29 port copied from the manifest's vcpkg baseline
`66c0373dc7fca549e5803087b9487edfe3aca0a1` (`ports/openblas`). Its source archive
hash and upstream patches are unchanged. The local port revision is 1.
See the [shared library](../../../README.md).

Only arm64 Android adds `TARGET=CORTEXA57`, `NUM_THREADS=8` (the original
Android buffer-capacity setting), and `USE_LOCKING=ON`.
The host build retains the upstream options. The Android build stays
single-threaded: COLMAP parallelizes image pairs and FAISS uses one thread per
index, so an additional BLAS worker pool would compete with those workers.
Requesting the port's `threads` feature on Android fails explicitly.

The [OpenBLAS Android build guidance](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.29/docs/install.md#android)
recommends CORTEXA57's optimized routines as compatible with other ARMv8 CPUs.
This changes the kernel choice without requiring a newer instruction set or
host-native compiler flags.

The [OpenBLAS build options](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.29/docs/build_system.md)
require locking to safely call a single-threaded library from multiple threads.
FAISS's centroid search/training calls SGEMM from COLMAP's matcher workers.
`USE_LOCKING` protects the BLAS packing-buffer allocator; it does not create
BLAS worker threads. A single-threaded build without locking
(`USE_THREAD=OFF` and `USE_LOCKING=OFF`) is not safe for these concurrent
calls.

Use the [concurrent correctness smoke](../../../tests/blas_smoke.cc) after
rebuilding. The build script's `-BlasSmoke` flag creates the executable.
The smoke exercises simultaneous SGEMM and DGEMM, both normal and transposed
inputs, odd matrix dimensions, and beta accumulation against an integer oracle.
It records no timings and does not replace a frozen performance experiment.

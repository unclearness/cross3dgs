# Gaussian training implementation

The program handles forward rendering, backward gradients, optimizer updates, and
density control. The public name `GaussianForwardProgram` and its header retain
their historical spelling for source compatibility.

The public interface is [gaussian_forward.h](../../include/cross3dgs/gaussian_forward.h).
[gaussian_program.cpp](../gaussian_program.cpp) implements state/target upload,
execution, readback, and diagnostics. Everything in this directory is private
to the `cross3dgs` library and compiled through its [CMake target](../../CMakeLists.txt).

Start with the constructor in [program_setup.cpp](program_setup.cpp). It selects
the device schedule, allocates buffers, registers passes in execution order,
then builds the program and uploads its initial state. No pass registration
occurs in the per-iteration execution path.

| File | Responsibility |
| --- | --- |
| [program_impl.h](program_impl.h) | Private PImpl, GPU buffers, and construction-only decisions |
| [program_setup.cpp](program_setup.cpp) | Validation, schedule selection, allocation, initial uploads |
| [forward_passes.cpp](forward_passes.cpp) | Camera selection, projection, binning/sorting, forward raster |
| [backward_passes.cpp](backward_passes.cpp) | Gradient clears, loss, raster backward, projection VJP/fused Adam |
| [optimizer_passes.cpp](optimizer_passes.cpp) | Separate Adam passes |
| [density_passes.cpp](density_passes.cpp) | MCMC and IGS densification/pruning |
| [shaders.h](shaders.h) | Private shader-builder declarations |
| [shaders_common.cpp](shaders_common.cpp) | Token substitution and target-storage decoding |
| [shaders_forward.cpp](shaders_forward.cpp) | Camera, projection, binning, raster GLSL |
| [shaders_loss.cpp](shaders_loss.cpp) | Reference, separable, and Adreno SSIM GLSL |
| [shaders_raster_backward.cpp](shaders_raster_backward.cpp) | Pixel, per-splat, and tiled scatter GLSL |
| [shaders_gather_backward.cpp](shaders_gather_backward.cpp) | Sobol and exact atomic-free backward GLSL |
| [shaders_projection_backward.cpp](shaders_projection_backward.cpp) | Projection/covariance/SH VJP and fused Adam GLSL |
| [shaders_optimizer.cpp](shaders_optimizer.cpp) | Separate Adam and optimizer-step GLSL |
| [shaders_mcmc.cpp](shaders_mcmc.cpp) | MCMC GLSL |
| [shaders_igs.cpp](shaders_igs.cpp) | IGS and Mip3D-filter GLSL |

Keep shader text/substitution order, buffer binding order, resource declarations,
and pass order intact when moving code. In particular, Qualcomm clears gradients
before the forward chain; other devices clear them immediately before backward.
The optimizer-step increment follows density control. `GaussianBuildOptions`
exists only during construction; execution uses the existing PImpl state.

The hashed `selector_source` strings in [schedule_ledger.cpp](../schedule_ledger.cpp)
retain their historical `gaussian_forward.cpp` locations to keep serialized
catalog identities unchanged. Use this map to locate the current implementation.

See the [repository README](../../../../README.md).

# Third-party notices

These components keep their own licenses whether Cross3DGS is used under the
AGPL or under a commercial license. The full texts of the licenses that
apply to code in this repository are in [LICENSES/](LICENSES/): the Apache
License 2.0 in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) (VkSplat,
Brush) and the MIT notices named below.

The COLMAP world-normalization and sparse-point KNN initialization in
`libs/cross3dgs/src/dataset.cpp` and `src/gaussian_state.cpp` are adapted
from VkSplat, revision
`e26c254938c81ff85998cd357a9e005e255d9b03`, licensed under the Apache License
2.0. VkSplat is Copyright its contributors.

The stable uint32 radix-sort pass layout and GLSL implementation in
`libs/cross3dgs/src/radix_sort.cpp` are adapted from the radix-sort shaders in
VkSplat at the same revision (`vksplat/shader/radix_sort/`). Those shaders
carry their own MIT License, Copyright (c) 2024 jaesung-cs, reproduced in
[LICENSES/MIT-radix-sort.txt](LICENSES/MIT-radix-sort.txt). Any changes VkSplat
made to them, like the rest of VkSplat, are under the Apache License 2.0.

The tiled 8x8 raster backward (`tiled_8x8`) in
`libs/cross3dgs/src/gaussian/shaders_raster_backward.cpp` is derived from
VkSplat's backward at the same revision, under the Apache License 2.0.

The patches under `patches/` modify Brush (Apache License 2.0) and VkSplat
(Apache License 2.0) for the baseline runs; the code they touch keeps those
licenses.

The overlay ports under `libs/cross3dgs_sfm/ports/` are modified from vcpkg
ports, which are licensed under the MIT License (Copyright (c) Microsoft
Corporation), reproduced in [LICENSES/MIT-vcpkg.txt](LICENSES/MIT-vcpkg.txt). The libraries they build (CLAPACK, OpenBLAS, GKlib) keep their
own licenses; OpenBLAS's is in `ports/android/openblas/LICENSE.txt`.

COLMAP is included as a git submodule (`external/colmap`, a fork) and is
licensed under the new BSD license.

VulkanWrapper is included as a git submodule (`external/VulkanWrapper`, a fork
of takiyu/VulkanWrapper) and is licensed under the MIT License. Its own
dependency notices and licenses remain in that submodule.

The build fetches the following dependencies at configure time; none of them
is vendored here and each remains under its own license: GLM, GLFW, spdlog,
and stb on every platform, plus glslang, SPIRV-Tools, SPIRV-Headers, and
Vulkan-Headers on Android and iOS (desktop builds use the copies shipped with
the Vulkan SDK).

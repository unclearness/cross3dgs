#ifndef CROSS3DGS_GPU_GPU_PRIMITIVES_H
#define CROSS3DGS_GPU_GPU_PRIMITIVES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cross3dgs/gpu/gpu_program.h"

namespace cross3dgs::gpu {

struct ScanPlan {
    std::vector<BufferVariable> scratch;
};

// Appends a hierarchical, work-efficient U32 exclusive scan to `program`.
// Input and output may alias. `count == 0` appends no work.
// If `indirect` is present, every generated pass consumes the next packed
// VkDispatchIndirectCommand and advances `indirect_command_index`.
ScanPlan AddExclusiveScanU32(CustomGpuProgram& program,
                             const BufferVariable& input,
                             const BufferVariable& output, size_t count,
                             const std::string& label = "exclusive_scan",
                             const BufferVariable& indirect = {},
                             uint32_t* indirect_command_index = nullptr);

}  // namespace cross3dgs::gpu

#endif  // CROSS3DGS_GPU_GPU_PRIMITIVES_H

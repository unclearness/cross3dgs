#ifndef CROSS3DGS_RADIX_SORT_H
#define CROSS3DGS_RADIX_SORT_H

#include <cstddef>
#include <string>
#include <vector>

#include "cross3dgs/gpu/gpu_program.h"

namespace cross3dgs {

// Adds a stable uint32 key/value LSD radix sort. The meta buffer contains
// four uvec4 records whose x lane is the GPU-produced active element count.
// key_bits bounds the significant low bits of every key; the pass count is
// ceil(key_bits/8) rounded up to an even number so the sorted result always
// lands back in keys_a/values_a (32 -> the classic four passes).
std::vector<gpu::BufferVariable> AddRadixSortU32(
        gpu::CustomGpuProgram& program,
        const gpu::BufferVariable& keys_a,
        const gpu::BufferVariable& values_a,
        const gpu::BufferVariable& keys_b,
        const gpu::BufferVariable& values_b,
        const gpu::BufferVariable& meta, size_t capacity,
        const std::string& label, bool force_portable = false,
        bool unsafe_forbid_portable = false, uint32_t key_bits = 32);

}  // namespace cross3dgs

#endif

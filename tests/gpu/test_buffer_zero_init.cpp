#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cross3dgs/gpu/gpu_program.h"

namespace {

cross3dgs::gpu::TensorDesc U32(size_t count, bool host_visible) {
    cross3dgs::gpu::TensorDesc desc;
    desc.scalar_type = cross3dgs::gpu::ScalarType::U32;
    desc.lanes = 1;
    desc.logical_shape = {1, {static_cast<uint32_t>(count), 1, 1, 1}};
    desc.capacity_bytes = count * sizeof(uint32_t);
    desc.host_visible = host_visible;
    return desc;
}

}  // namespace

// Optimizer passes read Adam moments before writing them, so a fresh buffer
// must read back as zero even when the allocator hands out memory that a
// previous program filled with a nonzero pattern.
int main() {
    constexpr size_t kCount = 16u << 20;  // 64 MiB per buffer
    constexpr int kRounds = 4;
    for (int round = 0; round < kRounds; round++) {
        for (const bool host_visible : {false, true}) {
            {
                cross3dgs::gpu::CustomGpuProgram poison;
                std::vector<cross3dgs::gpu::BufferVariable> buffers;
                const std::vector<uint32_t> pattern(kCount, 0xa5a5a5a5u);
                for (int i = 0; i < 4; i++) {
                    buffers.push_back(
                            poison.createBuffer(U32(kCount, host_visible)));
                    poison.upload(buffers.back(), pattern.data(),
                                  pattern.size() * sizeof(uint32_t));
                }
            }
            cross3dgs::gpu::CustomGpuProgram fresh;
            for (int i = 0; i < 4; i++) {
                const auto buffer =
                        fresh.createBuffer(U32(kCount, host_visible));
                const auto bytes = fresh.download(buffer);
                const bool zero = std::all_of(
                        bytes.begin(), bytes.end(),
                        [](std::byte value) { return value == std::byte{0}; });
                if (!zero) {
                    std::cerr << "fresh buffer is not zero (round " << round
                              << ", host_visible " << host_visible
                              << ", buffer " << i << ")\n";
                    return 1;
                }
            }
        }
    }
    std::cout << "fresh buffers read back as zero\n";
    return 0;
}

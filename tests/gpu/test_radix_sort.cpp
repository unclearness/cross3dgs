#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <glm/glm.hpp>

#include "cross3dgs/gpu/gpu_program.h"
#include "cross3dgs/radix_sort.h"

namespace {

cross3dgs::gpu::TensorDesc U32(size_t count, uint32_t lanes = 1) {
    cross3dgs::gpu::TensorDesc desc;
    desc.scalar_type = cross3dgs::gpu::ScalarType::U32;
    desc.lanes = lanes;
    desc.logical_shape = {
            1, {static_cast<uint32_t>(count), 1, 1, 1}};
    desc.capacity_bytes = count * lanes * sizeof(uint32_t);
    return desc;
}

}  // namespace

int main() {
    constexpr size_t kCapacity = 2048;
    constexpr size_t kCount = 1371;
    cross3dgs::gpu::CustomGpuProgram program;
    const auto keys_a = program.createBuffer(U32(kCapacity));
    const auto values_a = program.createBuffer(U32(kCapacity));
    const auto keys_b = program.createBuffer(U32(kCapacity));
    const auto values_b = program.createBuffer(U32(kCapacity));
    const auto meta = program.createBuffer(U32(4, 4));
    cross3dgs::AddRadixSortU32(
            program, keys_a, values_a, keys_b, values_b, meta, kCapacity,
            "test_radix");
    program.build();

    std::mt19937 random(42);
    std::vector<uint32_t> keys(kCapacity, 0xffffffffu);
    std::vector<uint32_t> values(kCapacity, 0u);
    for (size_t i = 0; i < kCount; i++) {
        keys[i] = random();
        values[i] = static_cast<uint32_t>(i);
    }
    std::vector<glm::uvec4> metadata(
            4, glm::uvec4(static_cast<uint32_t>(kCount), 0, 0, 0));
    program.upload(keys_a, keys.data(), keys.size() * sizeof(uint32_t));
    program.upload(values_a, values.data(),
                   values.size() * sizeof(uint32_t));
    program.upload(meta, metadata.data(),
                   metadata.size() * sizeof(glm::uvec4));
    program.execute();

    const auto key_bytes =
            program.download(keys_a, kCount * sizeof(uint32_t));
    const auto value_bytes =
            program.download(values_a, kCount * sizeof(uint32_t));
    const auto* sorted_keys =
            reinterpret_cast<const uint32_t*>(key_bytes.data());
    const auto* sorted_values =
            reinterpret_cast<const uint32_t*>(value_bytes.data());
    for (size_t i = 1; i < kCount; i++) {
        if (sorted_keys[i - 1] > sorted_keys[i]) {
            std::cerr << "radix keys are not sorted at " << i << "\n";
            return 1;
        }
    }
    for (size_t i = 0; i < kCount; i++) {
        if (sorted_values[i] >= kCount ||
            keys[sorted_values[i]] != sorted_keys[i]) {
            std::cerr << "radix key/value association mismatch\n";
            return 1;
        }
    }
    std::cout << "GPU uint32 radix sort validated\n";

    // Stability under heavy key duplication with a reduced key width (the
    // precondition the full-depth-sort tile ordering relies on): equal keys
    // must preserve the emission order of their values, and 12-bit keys must
    // sort correctly with the two-pass configuration.
    cross3dgs::gpu::CustomGpuProgram stable_program;
    const auto stable_keys_a = stable_program.createBuffer(U32(kCapacity));
    const auto stable_values_a = stable_program.createBuffer(U32(kCapacity));
    const auto stable_keys_b = stable_program.createBuffer(U32(kCapacity));
    const auto stable_values_b = stable_program.createBuffer(U32(kCapacity));
    const auto stable_meta = stable_program.createBuffer(U32(4, 4));
    cross3dgs::AddRadixSortU32(
            stable_program, stable_keys_a, stable_values_a, stable_keys_b,
            stable_values_b, stable_meta, kCapacity, "test_radix_stable",
            false, false, /*key_bits=*/12);
    stable_program.build();
    std::vector<uint32_t> duplicate_keys(kCapacity, 0xfffu);
    for (size_t i = 0; i < kCount; i++) {
        duplicate_keys[i] = random() % 7u;
    }
    stable_program.upload(stable_keys_a, duplicate_keys.data(),
                          duplicate_keys.size() * sizeof(uint32_t));
    stable_program.upload(stable_values_a, values.data(),
                          values.size() * sizeof(uint32_t));
    stable_program.upload(stable_meta, metadata.data(),
                          metadata.size() * sizeof(glm::uvec4));
    stable_program.execute();
    const auto stable_key_bytes =
            stable_program.download(stable_keys_a, kCount * sizeof(uint32_t));
    const auto stable_value_bytes =
            stable_program.download(stable_values_a,
                                    kCount * sizeof(uint32_t));
    const auto* stable_sorted_keys =
            reinterpret_cast<const uint32_t*>(stable_key_bytes.data());
    const auto* stable_sorted_values =
            reinterpret_cast<const uint32_t*>(stable_value_bytes.data());
    for (size_t i = 1; i < kCount; i++) {
        if (stable_sorted_keys[i - 1] > stable_sorted_keys[i]) {
            std::cerr << "duplicate-key radix keys are not sorted at " << i
                      << "\n";
            return 1;
        }
        if (stable_sorted_keys[i - 1] == stable_sorted_keys[i] &&
            stable_sorted_values[i - 1] >= stable_sorted_values[i]) {
            std::cerr << "radix sort is not stable at " << i << "\n";
            return 1;
        }
    }
    for (size_t i = 0; i < kCount; i++) {
        if (stable_sorted_values[i] >= kCount ||
            duplicate_keys[stable_sorted_values[i]] !=
                    stable_sorted_keys[i]) {
            std::cerr << "duplicate-key radix association mismatch\n";
            return 1;
        }
    }
    std::cout << "GPU radix sort stability validated\n";
    return 0;
}

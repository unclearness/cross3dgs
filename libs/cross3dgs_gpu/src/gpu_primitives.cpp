#include "cross3dgs/gpu/gpu_primitives.h"

#include <string>

#include "check.h"

namespace cross3dgs::gpu {

namespace {

constexpr size_t kThreads = 256;
constexpr size_t kValuesPerGroup = 2 * kThreads;

TensorDesc U32Desc(size_t count) {
    TensorDesc desc;
    desc.scalar_type = ScalarType::U32;
    desc.logical_shape =
            {1, {static_cast<uint32_t>(count == 0 ? 1 : count), 1, 1, 1}};
    desc.capacity_bytes = (count == 0 ? 1 : count) * sizeof(uint32_t);
    return desc;
}

std::string ScanShader() {
    return R"(#version 450
layout(local_size_x=256) in;
layout(std430, binding=0) readonly buffer Input { uint value[]; } input_data;
layout(std430, binding=1) writeonly buffer Output { uint value[]; } output_data;
layout(std430, binding=2) writeonly buffer Sums { uint value[]; } block_sums;
shared uint values[512];
void main() {
    uint tid = gl_LocalInvocationID.x;
    uint base = gl_WorkGroupID.x * 512u;
    uint ai = tid;
    uint bi = tid + 256u;
    values[ai] = base + ai < COUNT ? input_data.value[base + ai] : 0u;
    values[bi] = base + bi < COUNT ? input_data.value[base + bi] : 0u;

    uint offset = 1u;
    for (uint n_active = 256u; n_active > 0u; n_active >>= 1u) {
        barrier();
        if (tid < n_active) {
            uint left = offset * (2u * tid + 1u) - 1u;
            uint right = offset * (2u * tid + 2u) - 1u;
            values[right] += values[left];
        }
        offset <<= 1u;
    }
    if (tid == 0u) {
        block_sums.value[gl_WorkGroupID.x] = values[511];
        values[511] = 0u;
    }
    for (uint n_active = 1u; n_active < 512u; n_active <<= 1u) {
        offset >>= 1u;
        barrier();
        if (tid < n_active) {
            uint left = offset * (2u * tid + 1u) - 1u;
            uint right = offset * (2u * tid + 2u) - 1u;
            uint temporary = values[left];
            values[left] = values[right];
            values[right] += temporary;
        }
    }
    barrier();
    if (base + ai < COUNT) output_data.value[base + ai] = values[ai];
    if (base + bi < COUNT) output_data.value[base + bi] = values[bi];
})";
}

std::string AddOffsetShader() {
    return R"(#version 450
layout(local_size_x=256) in;
layout(std430, binding=0) buffer Values { uint value[]; } values;
layout(std430, binding=1) readonly buffer Offsets { uint value[]; } offsets;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < COUNT) values.value[i] += offsets.value[i / 512u];
})";
}

std::string WithCount(std::string shader, size_t count) {
    const std::string token = "COUNT";
    size_t position = 0;
    while ((position = shader.find(token, position)) != std::string::npos) {
        const std::string replacement = std::to_string(count) + "u";
        shader.replace(position, token.size(), replacement);
        position += replacement.size();
    }
    return shader;
}

BufferVariable AddLevel(CustomGpuProgram& program,
                        const BufferVariable& input,
                        const BufferVariable& output, size_t count,
                        const std::string& label, ScanPlan& plan,
                        const BufferVariable& indirect,
                        uint32_t* indirect_command_index) {
    const auto dispatch = [&](uint32_t groups) {
        if (!indirect) return DispatchDesc::Direct(groups);
        CROSS3DGS_CHECK(indirect_command_index != nullptr,
                        "indirect scan requires a command index");
        const uint32_t offset = 3 * sizeof(uint32_t) *
                (*indirect_command_index)++;
        return DispatchDesc::Indirect(indirect, offset);
    };
    const size_t group_count = (count + kValuesPerGroup - 1) / kValuesPerGroup;
    const auto sums = program.createBuffer(U32Desc(group_count), group_count);
    plan.scratch.push_back(sums);
    program.addPass({
            label + "_blocks_" + std::to_string(count),
            WithCount(ScanShader(), count),
            {input, output, sums},
            {{input, PipelineStage::Compute, AccessMode::Read, {}},
             {output, PipelineStage::Compute, AccessMode::Write, {}},
             {sums, PipelineStage::Compute, AccessMode::Write, {}}},
            dispatch(static_cast<uint32_t>(group_count))});

    if (group_count > 1) {
        const auto scanned_sums =
                program.createBuffer(U32Desc(group_count), group_count);
        plan.scratch.push_back(scanned_sums);
        AddLevel(program, sums, scanned_sums, group_count, label + "_upper",
                 plan, indirect, indirect_command_index);
        const uint32_t add_groups =
                static_cast<uint32_t>((count + kThreads - 1) / kThreads);
        program.addPass({
                label + "_add_offsets_" + std::to_string(count),
                WithCount(AddOffsetShader(), count),
                {output, scanned_sums},
                {{output, PipelineStage::Compute, AccessMode::ReadWrite, {}},
                 {scanned_sums, PipelineStage::Compute, AccessMode::Read, {}}},
                dispatch(add_groups)});
    }
    return sums;
}

}  // namespace

ScanPlan AddExclusiveScanU32(CustomGpuProgram& program,
                             const BufferVariable& input,
                             const BufferVariable& output, size_t count,
                             const std::string& label,
                             const BufferVariable& indirect,
                             uint32_t* indirect_command_index) {
    CROSS3DGS_CHECK(input && output, "scan input/output must not be null");
    CROSS3DGS_CHECK(input.desc().scalar_type == ScalarType::U32 &&
                            output.desc().scalar_type == ScalarType::U32 &&
                            input.desc().lanes == 1 && output.desc().lanes == 1,
                    "exclusive scan requires scalar U32 buffers");
    CROSS3DGS_CHECK(count * sizeof(uint32_t) <= input.capacityBytes() &&
                            count * sizeof(uint32_t) <= output.capacityBytes(),
                    "exclusive scan count exceeds input/output capacity");
    ScanPlan plan;
    if (count != 0) {
        AddLevel(program, input, output, count, label, plan, indirect,
                 indirect_command_index);
    }
    return plan;
}

}  // namespace cross3dgs::gpu

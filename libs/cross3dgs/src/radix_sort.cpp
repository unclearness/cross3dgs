// The radix layout and GLSL are adapted from VkSplat's radix-sort shaders,
// which are under the MIT License, Copyright (c) 2024 jaesung-cs.
// See THIRD_PARTY_NOTICES.md and LICENSES/MIT-radix-sort.txt.
#include "cross3dgs/radix_sort.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cross3dgs {
namespace {

constexpr uint32_t kWorkgroupSize = 256;
constexpr uint32_t kPartitionDivision = 8;
constexpr uint32_t kPartitionSize =
        kWorkgroupSize * kPartitionDivision;
constexpr uint32_t kRadix = 256;
constexpr uint32_t kPasses = 4;

gpu::TensorDesc U32Desc(size_t count, uint32_t lanes = 1) {
    gpu::TensorDesc desc;
    desc.scalar_type = gpu::ScalarType::U32;
    desc.lanes = lanes;
    desc.logical_shape = {
            1, {static_cast<uint32_t>(count), 1, 1, 1}};
    desc.capacity_bytes = count * lanes * sizeof(uint32_t);
    return desc;
}

std::string Replace(std::string source, const std::string& token,
                    const std::string& replacement) {
    size_t position = 0;
    while ((position = source.find(token, position)) != std::string::npos) {
        source.replace(position, token.size(), replacement);
        position += replacement.size();
    }
    return source;
}

std::string ClearHistogramShader() {
    return R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) writeonly buffer Histogram { uint value[]; } histogram;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if(i<1024u) histogram.value[i]=0u;
})";
}

std::string UpsweepShader(uint32_t pass) {
    auto source = R"(#version 450
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Keys { uint value[]; } keys;
layout(std430,binding=1) buffer GlobalHistogram { uint value[]; } global_histogram;
layout(std430,binding=2) writeonly buffer PartitionHistogram { uint value[]; } partition_histogram;
layout(std430,binding=3) readonly buffer Meta { uvec4 value[]; } meta;
shared uint local_histogram[256];
void main() {
    uint index=gl_LocalInvocationID.x;
    uint element_count=meta.value[PASS].x;
    uint partition_index=gl_WorkGroupID.x;
    uint start=partition_index*2048u;
    if(start>=element_count) return;
    local_histogram[index]=0u;
    barrier();
    for(uint i=0u;i<8u;i++) {
        uint key_index=start+256u*i+index;
        uint key=key_index<element_count ? keys.value[key_index] : 0xffffffffu;
        uint digit=(key>>(8u*PASS))&255u;
        atomicAdd(local_histogram[digit],1u);
    }
    barrier();
    partition_histogram.value[partition_index*256u+index]=local_histogram[index];
    atomicAdd(global_histogram.value[PASS*256u+index],
              local_histogram[index]);
})";
    return Replace(std::move(source), "PASS", std::to_string(pass) + "u");
}

std::string SpineShader(uint32_t pass) {
    auto source = R"(#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x=256) in;
layout(std430,binding=0) buffer GlobalHistogram { uint value[]; } global_histogram;
layout(std430,binding=1) buffer PartitionHistogram { uint value[]; } partition_histogram;
layout(std430,binding=2) readonly buffer Meta { uvec4 value[]; } meta;
shared uint reduction;
shared uint intermediate[32];
void main() {
    uint index=gl_LocalInvocationID.x;
    uint subgroup_index=gl_SubgroupID;
    uint digit=gl_WorkGroupID.x;
    uint element_count=meta.value[PASS].x;
    uint partition_count=(element_count+2047u)/2048u;
    if(index==0u) reduction=0u;
    barrier();
    for(uint base=0u;base<partition_count;base+=256u) {
        uint partition_index=base+index;
        uint input_value=partition_index<partition_count ?
            partition_histogram.value[partition_index*256u+digit] : 0u;
        uint exclusive=subgroupExclusiveAdd(input_value)+reduction;
        uint sum=subgroupAdd(input_value);
        if(subgroupElect()) intermediate[subgroup_index]=sum;
        barrier();
        if(index<gl_NumSubgroups) {
            uint upper_exclusive=subgroupExclusiveAdd(intermediate[index]);
            uint upper_sum=subgroupAdd(intermediate[index]);
            intermediate[index]=upper_exclusive;
            if(index==0u) reduction+=upper_sum;
        }
        barrier();
        if(partition_index<partition_count)
            partition_histogram.value[partition_index*256u+digit]=
                exclusive+intermediate[subgroup_index];
        barrier();
    }
    if(digit==0u) {
        uint value=global_histogram.value[PASS*256u+index];
        uint exclusive=subgroupExclusiveAdd(value);
        uint sum=subgroupAdd(value);
        if(subgroupElect()) intermediate[subgroup_index]=sum;
        barrier();
        if(index<gl_NumSubgroups)
            intermediate[index]=subgroupExclusiveAdd(intermediate[index]);
        barrier();
        global_histogram.value[PASS*256u+index]=
            exclusive+intermediate[subgroup_index];
    }
})";
    return Replace(std::move(source), "PASS", std::to_string(pass) + "u");
}

std::string PortableSpineShader(uint32_t pass) {
    auto source = R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) buffer GlobalHistogram { uint value[]; } global_histogram;
layout(std430,binding=1) buffer PartitionHistogram { uint value[]; } partition_histogram;
layout(std430,binding=2) readonly buffer Meta { uvec4 value[]; } meta;
void main() {
    uint digit=gl_WorkGroupID.x;
    uint element_count=meta.value[PASS].x;
    uint partition_count=(element_count+2047u)/2048u;
    uint running=0u;
    for(uint partition_index=0u;partition_index<partition_count;
        partition_index++) {
        uint index=partition_index*256u+digit;
        uint count=partition_histogram.value[index];
        partition_histogram.value[index]=running;
        running+=count;
    }
    if(digit==0u) {
        running=0u;
        for(uint d=0u;d<256u;d++) {
            uint index=PASS*256u+d;
            uint count=global_histogram.value[index];
            global_histogram.value[index]=running;
            running+=count;
        }
    }
})";
    return Replace(std::move(source), "PASS", std::to_string(pass) + "u");
}

std::string DownsweepShader(uint32_t pass) {
    std::string source = R"(#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer GlobalHistogram { uint value[]; } global_histogram;
layout(std430,binding=1) readonly buffer PartitionHistogram { uint value[]; } partition_histogram;
layout(std430,binding=2) readonly buffer KeysIn { uint value[]; } keys_in;
layout(std430,binding=3) readonly buffer ValuesIn { uint value[]; } values_in;
layout(std430,binding=4) writeonly buffer KeysOut { uint value[]; } keys_out;
layout(std430,binding=5) writeonly buffer ValuesOut { uint value[]; } values_out;
layout(std430,binding=6) readonly buffer Meta { uvec4 value[]; } meta;
shared uint histogram[HISTOGRAM_CAPACITY];
shared uint sums[512];
void main() {
    uint lane=gl_SubgroupInvocationID;
    uint subgroup=gl_SubgroupID;
    uint index=gl_LocalInvocationID.x;
    uint partition_index=gl_WorkGroupID.x;
    uint start=partition_index*2048u;
    uint element_count=meta.value[PASS].x;
    if(start>=element_count) return;
    if(index<256u) {
        for(uint i=0u;i<gl_NumSubgroups;i++)
            histogram[gl_NumSubgroups*index+i]=0u;
    }
    barrier();

    uint local_keys[8];
    uint local_values[8];
    uint local_digits[8];
    uint local_offsets[8];
    uint representative_counts[8];
    for(uint i=0u;i<8u;i++) {
        uint key_index=start+(8u*gl_SubgroupSize)*subgroup+
                       i*gl_SubgroupSize+lane;
        uint key=key_index<element_count ? keys_in.value[key_index]
                                        : 0xffffffffu;
        local_keys[i]=key;
        local_values[i]=key_index<element_count ? values_in.value[key_index]
                                               : 0u;
        uint digit=(key>>(8u*PASS))&255u;
        local_digits[i]=digit;
        uvec4 matching=subgroupBallot(true);
        for(uint bit=0u;bit<8u;bit++) {
            uint bit_value=(digit>>bit)&1u;
            uvec4 ones=subgroupBallot(bit_value==1u);
            matching&=uvec4(bit_value-1u)^ones;
        }
        uint subgroup_offset=subgroupBallotExclusiveBitCount(matching);
        uint digit_count=subgroupBallotBitCount(matching);
        if(subgroup_offset==0u) {
            atomicAdd(histogram[gl_NumSubgroups*digit+subgroup],
                      digit_count);
            representative_counts[i]=digit_count;
        } else {
            representative_counts[i]=0u;
        }
        local_offsets[i]=subgroup_offset;
    }
    barrier();

    for(uint i=index;i<256u*gl_NumSubgroups;i+=256u) {
        uint value=histogram[i];
        uint sum=subgroupAdd(value);
        histogram[i]=subgroupExclusiveAdd(value);
        if(lane==0u) sums[i/gl_SubgroupSize]=sum;
    }
    barrier();
    uint upper_count=256u*gl_NumSubgroups/gl_SubgroupSize;
    if(index<upper_count) {
        uint value=sums[index];
        uint sum=subgroupAdd(value);
        sums[index]=subgroupExclusiveAdd(value);
        if(lane==0u) sums[upper_count+index/gl_SubgroupSize]=sum;
    }
    barrier();
    uint top_count=(upper_count+gl_SubgroupSize-1u)/gl_SubgroupSize;
    if(index<top_count)
        sums[upper_count+index]=
            subgroupExclusiveAdd(sums[upper_count+index]);
    barrier();
    if(index<upper_count)
        sums[index]+=sums[upper_count+index/gl_SubgroupSize];
    barrier();
    for(uint i=index;i<256u*gl_NumSubgroups;i+=256u)
        histogram[i]+=sums[i/gl_SubgroupSize];
    barrier();

    for(uint i=0u;i<8u;i++) {
        uint digit=local_digits[i];
        local_offsets[i]+=histogram[gl_NumSubgroups*digit+subgroup];
        barrier();
        if(representative_counts[i]>0u)
            atomicAdd(histogram[gl_NumSubgroups*digit+subgroup],
                      representative_counts[i]);
        barrier();
    }
    if(index<256u) {
        uint prior=index==0u ? 0u :
            histogram[gl_NumSubgroups*index-1u];
        sums[index]=global_histogram.value[PASS*256u+index]+
                    partition_histogram.value[partition_index*256u+index]-prior;
    }
    barrier();

    for(uint i=0u;i<8u;i++)
        histogram[local_offsets[i]]=local_keys[i];
    barrier();
    uint destination_offsets[8];
    for(uint i=index;i<2048u;i+=256u) {
        uint key=histogram[i];
        uint digit=(key>>(8u*PASS))&255u;
        uint destination=sums[digit]+i;
        if(destination<element_count) keys_out.value[destination]=key;
        destination_offsets[i/256u]=destination;
    }
    barrier();
    for(uint i=0u;i<8u;i++)
        histogram[local_offsets[i]]=local_values[i];
    barrier();
    for(uint i=index;i<2048u;i+=256u) {
        uint destination=destination_offsets[i/256u];
        if(destination<CAPACITY)
            values_out.value[destination]=histogram[i];
    }
})";
    source = Replace(std::move(source), "PASS",
                     std::to_string(pass) + "u");
    return source;
}

std::string PortableDownsweepShader(uint32_t pass) {
    std::string source = R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer GlobalHistogram { uint value[]; } global_histogram;
layout(std430,binding=1) readonly buffer PartitionHistogram { uint value[]; } partition_histogram;
layout(std430,binding=2) readonly buffer KeysIn { uint value[]; } keys_in;
layout(std430,binding=3) readonly buffer ValuesIn { uint value[]; } values_in;
layout(std430,binding=4) writeonly buffer KeysOut { uint value[]; } keys_out;
layout(std430,binding=5) writeonly buffer ValuesOut { uint value[]; } values_out;
layout(std430,binding=6) readonly buffer Meta { uvec4 value[]; } meta;
shared uint digit_counts[256];
void main() {
    uint partition_index=gl_WorkGroupID.x;
    uint start=partition_index*2048u;
    uint element_count=meta.value[PASS].x;
    if(start>=element_count) return;
    for(uint digit=0u;digit<256u;digit++) digit_counts[digit]=0u;
    uint end=min(start+2048u,element_count);
    for(uint index=start;index<end;index++) {
        uint key=keys_in.value[index];
        uint digit=(key>>(8u*PASS))&255u;
        uint destination=global_histogram.value[PASS*256u+digit]+
                partition_histogram.value[partition_index*256u+digit]+
                digit_counts[digit]++;
        keys_out.value[destination]=key;
        values_out.value[destination]=values_in.value[index];
    }
})";
    return Replace(std::move(source), "PASS", std::to_string(pass) + "u");
}

}  // namespace

std::vector<gpu::BufferVariable> AddRadixSortU32(
        gpu::CustomGpuProgram& program,
        const gpu::BufferVariable& keys_a,
        const gpu::BufferVariable& values_a,
        const gpu::BufferVariable& keys_b,
        const gpu::BufferVariable& values_b,
        const gpu::BufferVariable& meta, size_t capacity,
        const std::string& label, bool force_portable,
        bool unsafe_forbid_portable, uint32_t key_bits) {
    if (capacity == 0 || capacity % kPartitionSize != 0) {
        throw std::invalid_argument(
                "radix capacity must be a nonzero multiple of 2048");
    }
    if (key_bits == 0 || key_bits > 32) {
        throw std::invalid_argument("radix key_bits must be in [1, 32]");
    }
    const uint32_t passes = std::min(kPasses, ((key_bits + 7) / 8 + 1) & ~1u);
    const uint32_t partition_count =
            static_cast<uint32_t>(capacity / kPartitionSize);
    const uint32_t subgroup_size = program.deviceCaps().subgroup_size;
    if (subgroup_size == 0 || subgroup_size > kWorkgroupSize) {
        throw std::runtime_error("invalid subgroup size for radix sort");
    }
    const uint32_t subgroup_count =
            (kWorkgroupSize + subgroup_size - 1) / subgroup_size;
    // unsafe_forbid_portable exists only for the bounded failure probe:
    // it re-enables the subgroup-dependent spine/downsweep on devices whose
    // vendor evidence normally rejects it. Never set it for training runs.
    const bool portable_radix =
            (force_portable ||
             program.deviceCaps().vendor_id == 0x5143u) &&
            !unsafe_forbid_portable;
    const uint32_t histogram_capacity =
            std::max(kPartitionSize, kRadix * subgroup_count);
    const size_t shared_bytes =
            (static_cast<size_t>(histogram_capacity) + 512u) *
            sizeof(uint32_t);
    if (shared_bytes > program.deviceCaps().max_shared_memory) {
        throw std::runtime_error(
                "radix sort subgroup layout exceeds shared memory");
    }
    auto global_histogram =
            program.createBuffer(U32Desc(kPasses * kRadix));
    auto partition_histogram = program.createBuffer(
            U32Desc(static_cast<size_t>(partition_count) * kRadix));
    program.addPass({
            label + "_clear_histogram",
            ClearHistogramShader(),
            {global_histogram},
            {{global_histogram, gpu::PipelineStage::Compute,
              gpu::AccessMode::Write, {}}},
            gpu::DispatchDesc::Direct(4)});
    for (uint32_t pass = 0; pass < passes; pass++) {
        const bool even = (pass % 2) == 0;
        const auto& input_keys = even ? keys_a : keys_b;
        const auto& input_values = even ? values_a : values_b;
        const auto& output_keys = even ? keys_b : keys_a;
        const auto& output_values = even ? values_b : values_a;
        program.addPass({
                label + "_upsweep_" + std::to_string(pass),
                UpsweepShader(pass),
                {input_keys, global_histogram, partition_histogram, meta},
                {{input_keys, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}},
                 {global_histogram, gpu::PipelineStage::Compute,
                  gpu::AccessMode::ReadWrite, {}},
                 {partition_histogram, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Write, {}},
                 {meta, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}}},
                gpu::DispatchDesc::Direct(partition_count)});
        program.addPass({
                label + "_spine_" + std::to_string(pass),
                portable_radix ? PortableSpineShader(pass)
                               : SpineShader(pass),
                {global_histogram, partition_histogram, meta},
                {{global_histogram, gpu::PipelineStage::Compute,
                  gpu::AccessMode::ReadWrite, {}},
                 {partition_histogram, gpu::PipelineStage::Compute,
                  gpu::AccessMode::ReadWrite, {}},
                 {meta, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}}},
                gpu::DispatchDesc::Direct(kRadix)});
        program.addPass({
                label + "_downsweep_" + std::to_string(pass),
                portable_radix
                        ? PortableDownsweepShader(pass)
                        : Replace(
                                  Replace(DownsweepShader(pass),
                                          "HISTOGRAM_CAPACITY",
                                          std::to_string(histogram_capacity)),
                                  "CAPACITY",
                                  std::to_string(capacity) + "u"),
                {global_histogram, partition_histogram, input_keys,
                 input_values, output_keys, output_values, meta},
                {{global_histogram, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}},
                 {partition_histogram, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}},
                 {input_keys, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}},
                 {input_values, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}},
                 {output_keys, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Write, {}},
                 {output_values, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Write, {}},
                 {meta, gpu::PipelineStage::Compute,
                  gpu::AccessMode::Read, {}}},
                gpu::DispatchDesc::Direct(partition_count)});
    }
    return {global_histogram, partition_histogram};
}

}  // namespace cross3dgs

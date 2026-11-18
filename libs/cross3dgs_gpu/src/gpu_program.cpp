#include "cross3dgs/gpu/gpu_program.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <utility>

#include <vkw/vkw.h>

#include "check.h"
#include "vk_context.h"

namespace cross3dgs::gpu {

namespace {

std::atomic<uint64_t> g_next_buffer_id{1};

bool HasRead(AccessMode access) {
    return access == AccessMode::Read || access == AccessMode::ReadWrite;
}

bool HasWrite(AccessMode access) {
    return access == AccessMode::Write || access == AccessMode::ReadWrite;
}

vk::PipelineStageFlags VkStage(PipelineStage stage) {
    switch (stage) {
        case PipelineStage::Indirect:
            return vk::PipelineStageFlagBits::eDrawIndirect;
        case PipelineStage::Transfer:
            return vk::PipelineStageFlagBits::eTransfer;
        default:
            return vk::PipelineStageFlagBits::eComputeShader;
    }
}

vk::AccessFlags VkAccess(const ResourceUse& use) {
    vk::AccessFlags access;
    if (use.stage == PipelineStage::Indirect) {
        if (HasRead(use.access)) {
            access |= vk::AccessFlagBits::eIndirectCommandRead;
        }
        return access;
    }
    if (use.stage == PipelineStage::Transfer) {
        if (HasRead(use.access)) access |= vk::AccessFlagBits::eTransferRead;
        if (HasWrite(use.access)) access |= vk::AccessFlagBits::eTransferWrite;
        return access;
    }
    if (HasRead(use.access)) {
        access |= vk::AccessFlagBits::eShaderRead;
    }
    if (HasWrite(use.access)) {
        access |= vk::AccessFlagBits::eShaderWrite;
    }
    return access;
}

bool HasExtension(const vk::PhysicalDevice& physical_device,
                  const char* name) {
    const auto extensions = physical_device.enumerateDeviceExtensionProperties();
    return std::any_of(extensions.begin(), extensions.end(),
                       [&](const vk::ExtensionProperties& extension) {
                           return std::strcmp(extension.extensionName, name) == 0;
                       });
}

DeviceCaps QueryDeviceCaps(const VkContext& context) {
    DeviceCaps caps;
    const auto properties = context.physical_device.getProperties();
    const auto features = context.physical_device.getFeatures();
    caps.vendor_id = properties.vendorID;
    caps.device_id = properties.deviceID;
    caps.max_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
    caps.max_shared_memory = properties.limits.maxComputeSharedMemorySize;
    caps.max_storage_buffer_bindings = std::min(
            properties.limits.maxPerStageDescriptorStorageBuffers,
            properties.limits.maxDescriptorSetStorageBuffers);
    caps.max_storage_buffer_range = properties.limits.maxStorageBufferRange;
    caps.shader_int64 = features.shaderInt64 == VK_TRUE;

    vk::PhysicalDeviceSubgroupProperties subgroup;
    vk::PhysicalDeviceProperties2 properties2;
    properties2.pNext = &subgroup;
    context.physical_device.getProperties2(&properties2);
    caps.subgroup_size = subgroup.subgroupSize;
    caps.min_subgroup_size = subgroup.subgroupSize;
    caps.max_subgroup_size = subgroup.subgroupSize;
    caps.subgroup_supported_operations =
            static_cast<uint32_t>(subgroup.supportedOperations);
    caps.subgroup_supported_stages =
            static_cast<uint32_t>(subgroup.supportedStages);

    caps.f16_arithmetic = HasExtension(
            context.physical_device, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    caps.f16_storage = HasExtension(
            context.physical_device, VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float;
    vk::PhysicalDeviceFeatures2 features2;
    features2.pNext = &atomic_float;
    context.physical_device.getFeatures2(&features2);
    caps.float32_atomic_add =
            HasExtension(context.physical_device,
                         VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) &&
            atomic_float.shaderBufferFloat32AtomicAdd == VK_TRUE;
    caps.subgroup_size_control = HasExtension(
            context.physical_device, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    return caps;
}

size_t ScalarBytes(ScalarType type) {
    return type == ScalarType::F16 ? 2 : 4;
}

}  // namespace

size_t TensorShape::elementCount() const {
    CROSS3DGS_CHECK(rank >= 1 && rank <= dims.size(),
                    "TensorShape rank must be in [1, 4]");
    size_t count = 1;
    for (uint32_t i = 0; i < rank; i++) {
        CROSS3DGS_CHECK(dims[i] != 0, "TensorShape dimensions must be non-zero");
        count *= dims[i];
    }
    return count;
}

size_t TensorDesc::elementBytes() const {
    CROSS3DGS_CHECK(lanes >= 1 && lanes <= 4,
                    "TensorDesc lanes must be in [1, 4]");
    return ScalarBytes(scalar_type) * lanes;
}

DispatchDesc DispatchDesc::Direct(uint32_t x, uint32_t y, uint32_t z) {
    DispatchDesc result;
    result.mode = Mode::Direct;
    result.direct = {x, y, z};
    return result;
}

DispatchDesc DispatchDesc::Indirect(const BufferVariable& buffer,
                                    uint32_t offset) {
    DispatchDesc result;
    result.mode = Mode::Indirect;
    result.indirect = buffer;
    result.indirect_offset = offset;
    return result;
}

struct CustomGpuProgram::Impl {
    struct BufferState {
        BufferVariable variable;
        vkw::BufferPackPtr device;
        vkw::BufferPackPtr upload_staging;
        vkw::BufferPackPtr download_staging;
    };

    struct BuiltPass {
        ComputePassDesc desc;
        vkw::DescSetPackPtr descriptor_set;
        vkw::PipelinePackPtr pipeline;
    };

    struct LastUse {
        ResourceUse use;
    };

    explicit Impl(VkContextPtr shared_context)
        : ctx(shared_context ? std::move(shared_context) : CreateVkContext(false)),
          caps(QueryDeviceCaps(*ctx)) {}

    BufferState& state(const BufferVariable& variable) {
        CROSS3DGS_CHECK(variable, "null BufferVariable");
        auto found = buffers.find(variable.id());
        CROSS3DGS_CHECK(found != buffers.end(),
                        "BufferVariable does not belong to this program");
        return found->second;
    }

    const BufferState& state(const BufferVariable& variable) const {
        return const_cast<Impl*>(this)->state(variable);
    }

    void validateRange(const ResourceUse& use) const {
        const auto& buffer = state(use.resource).variable;
        CROSS3DGS_CHECK(use.range.offset <= buffer.capacityBytes(),
                        "resource range offset exceeds buffer capacity");
        const size_t size = use.range.size == 0
                                    ? buffer.capacityBytes() - use.range.offset
                                    : use.range.size;
        CROSS3DGS_CHECK(size <= buffer.capacityBytes() - use.range.offset,
                        "resource range exceeds buffer capacity");
    }

    VkContextPtr ctx;
    DeviceCaps caps;
    std::map<uint64_t, BufferState> buffers;
    std::vector<ComputePassDesc> pass_descs;
    std::vector<BuiltPass> passes;
    vkw::CommandBuffersPackPtr command_buffers;
    vkw::CommandBuffersPackPtr adreno_fenced_command_buffers;
    vkw::FencePtr fence;
    vkw::FencePtr adreno_fenced_fence;
    vk::UniqueQueryPool query_pool;
    std::vector<PassTiming> timings;
    std::vector<HostPassTiming> fenced_host_timings;
    bool profiling_enabled = true;
    bool conservative_barriers = false;
    bool built = false;
};

CustomGpuProgram::CustomGpuProgram(VkContextPtr context)
    : m_impl(std::make_unique<Impl>(std::move(context))) {}

CustomGpuProgram::~CustomGpuProgram() = default;
CustomGpuProgram::CustomGpuProgram(CustomGpuProgram&&) noexcept = default;
CustomGpuProgram& CustomGpuProgram::operator=(CustomGpuProgram&&) noexcept = default;

BufferVariable CustomGpuProgram::createBuffer(const TensorDesc& input_desc,
                                              size_t logical_count) {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(!impl.built, "cannot create buffers after build");
    TensorDesc desc = input_desc;
    const size_t shape_count = desc.logical_shape.elementCount();
    if (logical_count == 0) {
        logical_count = shape_count;
    }
    const size_t required = logical_count * desc.elementBytes();
    if (desc.capacity_bytes == 0) {
        desc.capacity_bytes = required;
    }
    CROSS3DGS_CHECK(desc.alignment != 0 &&
                            (desc.alignment & (desc.alignment - 1)) == 0,
                    "TensorDesc alignment must be a power of two");
    CROSS3DGS_CHECK(desc.capacity_bytes >= required,
                    "TensorDesc capacity is smaller than logical data");
    CROSS3DGS_CHECK(desc.capacity_bytes % desc.alignment == 0,
                    "TensorDesc capacity does not satisfy alignment");

    const uint64_t id = g_next_buffer_id.fetch_add(1);
    BufferVariable variable(id, desc, logical_count);
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eTransferSrc |
                                 vk::BufferUsageFlagBits::eTransferDst;
    if (desc.storage == StorageClass::UniformBuffer) {
        usage |= vk::BufferUsageFlagBits::eUniformBuffer;
    } else {
        usage |= vk::BufferUsageFlagBits::eStorageBuffer;
    }
    if (desc.storage == StorageClass::IndirectBuffer) {
        usage |= vk::BufferUsageFlagBits::eIndirectBuffer |
                 vk::BufferUsageFlagBits::eStorageBuffer;
    }
    auto device = vkw::CreateBufferPack(
            impl.ctx->physical_device, impl.ctx->device, desc.capacity_bytes,
            usage, desc.host_visible
                           ? vkw::HOST_VISIB_COHER_PROPS
                           : vk::MemoryPropertyFlagBits::eDeviceLocal);
    // Vulkan does not define the contents of new device memory. Passes such
    // as Adam read their state before ever writing it, so every buffer starts
    // zeroed instead of relying on the driver handing out cleared VRAM.
    if (desc.host_visible) {
        const auto& memory = device->dev_mem_pack;
        void* mapped = impl.ctx->device->mapMemory(memory->dev_mem.get(), 0,
                                                   memory->dev_mem_size);
        std::memset(mapped, 0, memory->dev_mem_size);
        impl.ctx->device->unmapMemory(memory->dev_mem.get());
    } else {
        RunOneShot(*impl.ctx, [&](const vk::UniqueCommandBuffer& command) {
            command->fillBuffer(*device->buf, 0, VK_WHOLE_SIZE, 0u);
            const vk::BufferMemoryBarrier barrier(
                    vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferRead |
                            vk::AccessFlagBits::eTransferWrite |
                            vk::AccessFlagBits::eIndirectCommandRead,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    *device->buf, 0, VK_WHOLE_SIZE);
            command->pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eComputeShader |
                            vk::PipelineStageFlagBits::eTransfer |
                            vk::PipelineStageFlagBits::eDrawIndirect,
                    {}, nullptr, barrier, nullptr);
        });
    }
    impl.buffers.emplace(id, Impl::BufferState{variable, std::move(device),
                                               nullptr, nullptr});
    return variable;
}

void CustomGpuProgram::setLogicalCount(BufferVariable& buffer,
                                       size_t logical_count) {
    auto& state = m_impl->state(buffer);
    CROSS3DGS_CHECK(logical_count * buffer.desc().elementBytes() <=
                            buffer.capacityBytes(),
                    "logical count exceeds fixed buffer capacity");
    buffer.m_logical_count = logical_count;
    state.variable.m_logical_count = logical_count;
}

void CustomGpuProgram::upload(const BufferVariable& buffer, const void* data,
                              size_t bytes, size_t offset) {
    Impl& impl = *m_impl;
    auto& state = impl.state(buffer);
    CROSS3DGS_CHECK(data != nullptr || bytes == 0, "upload data is null");
    CROSS3DGS_CHECK(offset <= buffer.capacityBytes() &&
                            bytes <= buffer.capacityBytes() - offset,
                    "upload exceeds buffer capacity");
    if (bytes == 0) {
        return;
    }
    if (buffer.desc().host_visible) {
        const auto& memory = state.device->dev_mem_pack;
        auto* mapped = static_cast<std::byte*>(impl.ctx->device->mapMemory(
                memory->dev_mem.get(), 0, memory->dev_mem_size));
        std::memcpy(mapped + offset, data, bytes);
        impl.ctx->device->unmapMemory(memory->dev_mem.get());
        return;
    }
    if (!state.upload_staging || state.upload_staging->size < bytes) {
        state.upload_staging = vkw::CreateBufferPack(
                impl.ctx->physical_device, impl.ctx->device, bytes,
                vk::BufferUsageFlagBits::eTransferSrc,
                vkw::HOST_VISIB_COHER_PROPS);
    }
    vkw::SendToDevice(impl.ctx->device, state.upload_staging, data, bytes);
    RunOneShot(*impl.ctx, [&](const vk::UniqueCommandBuffer& command) {
        if (impl.caps.vendor_id == 0x5143u) {
            const vk::BufferMemoryBarrier before(
                    vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eIndirectCommandRead,
                    vk::AccessFlagBits::eTransferWrite,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                    *state.device->buf, offset, bytes);
            command->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader |
                            vk::PipelineStageFlagBits::eDrawIndirect,
                    vk::PipelineStageFlagBits::eTransfer, {}, nullptr, before,
                    nullptr);
        }
        const vk::BufferCopy copy(0, offset, bytes);
        command->copyBuffer(*state.upload_staging->buf, *state.device->buf, copy);
        const vk::BufferMemoryBarrier barrier(
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead |
                        vk::AccessFlagBits::eShaderWrite |
                        vk::AccessFlagBits::eIndirectCommandRead,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                *state.device->buf, offset, bytes);
        command->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader |
                        vk::PipelineStageFlagBits::eDrawIndirect,
                {}, nullptr, barrier, nullptr);
    });
}

std::vector<std::byte> CustomGpuProgram::download(
        const BufferVariable& buffer, size_t bytes, size_t offset) {
    Impl& impl = *m_impl;
    auto& state = impl.state(buffer);
    if (bytes == 0) {
        bytes = buffer.logicalCount() * buffer.desc().elementBytes();
    }
    CROSS3DGS_CHECK(offset <= buffer.capacityBytes() &&
                            bytes <= buffer.capacityBytes() - offset,
                    "download exceeds buffer capacity");
    if (buffer.desc().host_visible) {
        const auto& memory = state.device->dev_mem_pack;
        const auto* mapped = static_cast<const std::byte*>(
                impl.ctx->device->mapMemory(memory->dev_mem.get(), 0,
                                            memory->dev_mem_size));
        std::vector<std::byte> result(bytes);
        std::memcpy(result.data(), mapped + offset, bytes);
        impl.ctx->device->unmapMemory(memory->dev_mem.get());
        return result;
    }
    if (!state.download_staging || state.download_staging->size < bytes) {
        state.download_staging = vkw::CreateBufferPack(
                impl.ctx->physical_device, impl.ctx->device, bytes,
                vk::BufferUsageFlagBits::eTransferDst,
                vkw::HOST_VISIB_COHER_PROPS);
    }
    RunOneShot(*impl.ctx, [&](const vk::UniqueCommandBuffer& command) {
        const vk::BufferMemoryBarrier before(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eTransferRead,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                *state.device->buf, offset, bytes);
        command->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                 vk::PipelineStageFlagBits::eTransfer, {},
                                 nullptr, before, nullptr);
        const vk::BufferCopy copy(offset, 0, bytes);
        command->copyBuffer(*state.device->buf, *state.download_staging->buf,
                            copy);
    });
    std::vector<std::byte> result(bytes);
    vkw::RecvFromDevice(impl.ctx->device, state.download_staging, result.data(),
                        bytes);
    return result;
}

void CustomGpuProgram::addPass(const ComputePassDesc& pass) {
    CROSS3DGS_CHECK(!m_impl->built, "cannot add passes after build");
    CROSS3DGS_CHECK(pass.fill_zero || !pass.shader_source.empty(),
                    "compute shader source is empty");
    CROSS3DGS_CHECK(!pass.fill_zero || !pass.buffers.empty(),
                    "fill_zero pass has no buffers to clear");
    CROSS3DGS_CHECK(!pass.label.empty(), "compute pass label is empty");
    m_impl->pass_descs.push_back(pass);
}

void CustomGpuProgram::setProfilingEnabled(bool enabled) {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(!impl.built, "profiling mode must be selected before build");
    impl.profiling_enabled = enabled;
}

void CustomGpuProgram::setConservativeBarriers(bool enabled) {
    CROSS3DGS_CHECK(!m_impl->built,
                    "barrier mode must be selected before build");
    m_impl->conservative_barriers = enabled;
}

void CustomGpuProgram::build() {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(!impl.pass_descs.empty(), "CustomGpuProgram has no passes");
    impl.passes.clear();
    impl.timings.clear();

    for (const auto& desc : impl.pass_descs) {
        std::set<uint64_t> declared;
        for (const auto& use : desc.resources) {
            impl.validateRange(use);
            declared.insert(use.resource.id());
        }
        for (const auto& buffer : desc.buffers) {
            impl.state(buffer);
            CROSS3DGS_CHECK(declared.count(buffer.id()) != 0,
                            "descriptor buffer is missing a ResourceUse");
        }
        if (desc.fill_zero) {
            // No pipeline or descriptor set: recorded as vkCmdFillBuffer.
            impl.passes.push_back({desc, nullptr, nullptr});
            impl.timings.push_back({desc.label, 0.0});
            continue;
        }
        if (desc.dispatch.mode == DispatchDesc::Mode::Direct) {
            CROSS3DGS_CHECK(desc.dispatch.direct.x != 0 &&
                                    desc.dispatch.direct.y != 0 &&
                                    desc.dispatch.direct.z != 0,
                            "direct dispatch dimensions must be non-zero");
        } else {
            const auto& indirect = impl.state(desc.dispatch.indirect).variable;
            CROSS3DGS_CHECK(
                    indirect.desc().storage == StorageClass::IndirectBuffer,
                    "indirect dispatch requires an IndirectBuffer");
            CROSS3DGS_CHECK(desc.dispatch.indirect_offset % 4 == 0,
                            "indirect dispatch offset must be 4-byte aligned");
            CROSS3DGS_CHECK(
                    desc.dispatch.indirect_offset + 3 * sizeof(uint32_t) <=
                            indirect.capacityBytes(),
                    "indirect dispatch arguments exceed buffer capacity");
        }

        std::vector<vkw::DescSetInfo> binding_infos;
        for (const auto& buffer : desc.buffers) {
            const vk::DescriptorType type =
                    buffer.desc().storage == StorageClass::UniformBuffer
                            ? vk::DescriptorType::eUniformBuffer
                            : vk::DescriptorType::eStorageBuffer;
            binding_infos.emplace_back(type, 1,
                                       vk::ShaderStageFlagBits::eCompute);
        }
        auto descriptor_set = vkw::CreateDescriptorSetPack(
                impl.ctx->device, binding_infos);
        auto writes = vkw::CreateWriteDescSetPack();
        for (uint32_t binding = 0; binding < desc.buffers.size(); binding++) {
            vkw::AddWriteDescSet(
                    writes, descriptor_set, binding,
                    {impl.state(desc.buffers[binding]).device});
        }
        vkw::UpdateDescriptorSets(impl.ctx->device, writes);
        vkw::PipelinePackPtr pipeline;
        try {
            const auto& compiler = desc.optimization ==
                                                   ShaderOptimization::Size
                                           ? impl.ctx->glsl_size_compiler
                                           : impl.ctx->glsl_compiler;
            auto shader = compiler->compileFromString(
                    impl.ctx->device, desc.shader_source,
                    vk::ShaderStageFlagBits::eCompute);
            pipeline = vkw::CreateComputePipeline(
                    impl.ctx->device, shader, {descriptor_set},
                    /*pipeline_cache=*/nullptr,
                    impl.ctx->forced_subgroup_size);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                    "compute pass '" + desc.label + "': " + error.what());
        }
        impl.passes.push_back(
                {desc, std::move(descriptor_set), std::move(pipeline)});
        impl.timings.push_back({desc.label, 0.0});
    }

    impl.command_buffers = vkw::CreateCommandBuffersPack(
            impl.ctx->device, impl.ctx->queue_family_idx, 2);
    impl.fence = vkw::CreateFence(impl.ctx->device);
    const bool timestamps = impl.profiling_enabled &&
                            impl.ctx->limits.timestampComputeAndGraphics;
    if (timestamps) {
        impl.query_pool = impl.ctx->device->createQueryPoolUnique(
                {{}, vk::QueryType::eTimestamp,
                 static_cast<uint32_t>(2 * impl.passes.size())});
    } else {
        impl.query_pool.reset();
    }

    for(uint32_t command_index=0;command_index<2;command_index++) {
    const auto& command = impl.command_buffers->cmd_bufs[command_index];
    const bool record_timestamps=command_index==0 && bool(impl.query_pool);
    vkw::BeginCommand(command, false);
    if (record_timestamps) {
        command->resetQueryPool(*impl.query_pool, 0,
                                static_cast<uint32_t>(2 * impl.passes.size()));
    }

    std::map<uint64_t, Impl::LastUse> last_uses;
    const uint32_t repetitions=command_index==0 ? 1u : 8u;
    for(uint32_t repetition=0;repetition<repetitions;repetition++) {
    for (size_t pass_index = 0; pass_index < impl.passes.size(); pass_index++) {
        const auto& pass = impl.passes[pass_index];
        std::vector<ResourceUse> uses = pass.desc.resources;
        if (pass.desc.dispatch.mode == DispatchDesc::Mode::Indirect) {
            uses.push_back({pass.desc.dispatch.indirect, PipelineStage::Indirect,
                            AccessMode::Read,
                            {pass.desc.dispatch.indirect_offset,
                             3 * sizeof(uint32_t)}});
        }

        vk::PipelineStageFlags source_stages;
        vk::PipelineStageFlags destination_stages;
        std::vector<vk::BufferMemoryBarrier> barriers;
        for (const auto& use : uses) {
            auto previous = last_uses.find(use.resource.id());
            if (previous != last_uses.end() &&
                (HasWrite(previous->second.use.access) || HasWrite(use.access))) {
                const auto& state = impl.state(use.resource);
                const size_t size = use.range.size == 0
                                            ? use.resource.capacityBytes() -
                                                      use.range.offset
                                            : use.range.size;
                barriers.emplace_back(
                        VkAccess(previous->second.use), VkAccess(use),
                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                        *state.device->buf, use.range.offset, size);
                source_stages |= VkStage(previous->second.use.stage);
                destination_stages |= VkStage(use.stage);
            }
            last_uses[use.resource.id()] = {use};
        }
        if (impl.conservative_barriers) {
            const vk::PipelineStageFlags all_stages =
                    vk::PipelineStageFlagBits::eComputeShader |
                    vk::PipelineStageFlagBits::eTransfer |
                    vk::PipelineStageFlagBits::eDrawIndirect;
            const vk::MemoryBarrier full(
                    vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferWrite,
                    vk::AccessFlagBits::eShaderRead |
                            vk::AccessFlagBits::eShaderWrite |
                            vk::AccessFlagBits::eTransferRead |
                            vk::AccessFlagBits::eTransferWrite |
                            vk::AccessFlagBits::eIndirectCommandRead);
            command->pipelineBarrier(all_stages, all_stages, {}, full, nullptr,
                                     nullptr);
        } else if (!barriers.empty()) {
            command->pipelineBarrier(source_stages, destination_stages, {},
                                     nullptr, barriers, nullptr);
        }
        if (record_timestamps) {
            command->writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,
                                    *impl.query_pool,
                                    static_cast<uint32_t>(2 * pass_index));
        }
        if (pass.desc.fill_zero) {
            for (const auto& buffer : pass.desc.buffers) {
                command->fillBuffer(*impl.state(buffer).device->buf, 0,
                                    VK_WHOLE_SIZE, 0u);
            }
        } else {
        vkw::CmdBindPipeline(command, pass.pipeline,
                             vk::PipelineBindPoint::eCompute);
        vkw::CmdBindDescSets(command, pass.pipeline, {pass.descriptor_set}, {},
                             vk::PipelineBindPoint::eCompute);
        if (pass.desc.dispatch.mode == DispatchDesc::Mode::Direct) {
            const auto size = pass.desc.dispatch.direct;
            vkw::CmdDispatch(command, size.x, size.y, size.z);
        } else {
            const auto& indirect = impl.state(pass.desc.dispatch.indirect);
            command->dispatchIndirect(*indirect.device->buf,
                                      pass.desc.dispatch.indirect_offset);
        }
        }
        if (record_timestamps) {
            command->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                                    *impl.query_pool,
                                    static_cast<uint32_t>(2 * pass_index + 1));
        }
    }
    }
    vkw::EndCommand(command);
    }
    impl.built = true;
}

void CustomGpuProgram::execute() {
    executeBatch(1);
}

void CustomGpuProgram::executeWithoutTimingReadback() {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(impl.built, "execute called before build");
    vkw::ResetFence(impl.ctx->device, impl.fence);
    vkw::QueueSubmit(impl.ctx->queue, impl.command_buffers->cmd_bufs[0],
                     impl.fence);
    vkw::WaitForFence(impl.ctx->device, impl.fence);
}

void CustomGpuProgram::executeBatch(uint32_t repeat_count) {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(impl.built, "execute called before build");
    CROSS3DGS_CHECK(repeat_count==1 || repeat_count==8,
                    "executeBatch supports one or eight repetitions");
    if (repeat_count == 8 && impl.ctx->split_batch_submits) {
        // Same iteration semantics as the 8x command buffer, but each
        // repetition is its own watchdog-sized submission (see
        // VkContext::split_batch_submits).
        for (uint32_t repetition = 0; repetition < repeat_count;
             repetition++) {
            vkw::ResetFence(impl.ctx->device, impl.fence);
            vkw::QueueSubmit(impl.ctx->queue,
                             impl.command_buffers->cmd_bufs[0], impl.fence);
            vkw::WaitForFence(impl.ctx->device, impl.fence);
        }
        return;
    }
    vkw::ResetFence(impl.ctx->device, impl.fence);
    const uint32_t command_index=repeat_count==8 ? 1u : 0u;
    vkw::QueueSubmit(impl.ctx->queue, impl.command_buffers->cmd_bufs[command_index],
                     impl.fence);
    vkw::WaitForFence(impl.ctx->device, impl.fence);
    if (!impl.query_pool || repeat_count!=1) {
        return;
    }
    std::vector<uint64_t> ticks(2 * impl.passes.size());
    const auto result = impl.ctx->device->getQueryPoolResults(
            *impl.query_pool, 0, static_cast<uint32_t>(ticks.size()),
            ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (result != vk::Result::eSuccess) {
        return;
    }
    for (size_t index = 0; index < impl.passes.size(); index++) {
        impl.timings[index].gpu_microseconds =
                double(ticks[2 * index + 1] - ticks[2 * index]) *
                impl.ctx->limits.timestampPeriod * 1e-3;
    }
}

void CustomGpuProgram::executeFenced() {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(impl.built, "executeFenced called before build");
    impl.fenced_host_timings.clear();
    impl.fenced_host_timings.reserve(impl.passes.size());
    for (const auto& pass : impl.passes) {
        impl.fenced_host_timings.push_back({pass.desc.label, 0.0});
    }
    const vk::MemoryBarrier barrier(
            vk::AccessFlagBits::eShaderWrite |
                    vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                    vk::AccessFlagBits::eTransferRead |
                    vk::AccessFlagBits::eTransferWrite |
                    vk::AccessFlagBits::eIndirectCommandRead);
    const vk::PipelineStageFlags stages =
            vk::PipelineStageFlagBits::eComputeShader |
            vk::PipelineStageFlagBits::eTransfer |
            vk::PipelineStageFlagBits::eDrawIndirect;
    auto fenced_commands = vkw::CreateCommandBuffersPack(
            impl.ctx->device, impl.ctx->queue_family_idx,
            static_cast<uint32_t>(impl.passes.size()));
    for (size_t index = 0; index < impl.passes.size(); index++) {
            const auto& pass = impl.passes[index];
            const auto& command =
                    fenced_commands->cmd_bufs[index];
            vkw::BeginCommand(command, false);
            command->pipelineBarrier(stages, stages, {}, barrier, nullptr,
                                     nullptr);
            if (pass.desc.fill_zero) {
                for (const auto& buffer : pass.desc.buffers) {
                    command->fillBuffer(*impl.state(buffer).device->buf, 0,
                                        VK_WHOLE_SIZE, 0u);
                }
            } else {
                vkw::CmdBindPipeline(command, pass.pipeline,
                                     vk::PipelineBindPoint::eCompute);
                vkw::CmdBindDescSets(command, pass.pipeline,
                                     {pass.descriptor_set}, {},
                                     vk::PipelineBindPoint::eCompute);
                if (pass.desc.dispatch.mode == DispatchDesc::Mode::Direct) {
                    const auto size = pass.desc.dispatch.direct;
                    vkw::CmdDispatch(command, size.x, size.y, size.z);
                } else {
                    const auto& indirect =
                            impl.state(pass.desc.dispatch.indirect);
                    command->dispatchIndirect(
                            *indirect.device->buf,
                            pass.desc.dispatch.indirect_offset);
                }
            }
            vkw::EndCommand(command);
    }
    for (size_t index = 0;
         index < fenced_commands->cmd_bufs.size(); index++) {
        const auto& command =
                fenced_commands->cmd_bufs[index];
        const auto fence = vkw::CreateFence(impl.ctx->device);
        const auto begin = std::chrono::steady_clock::now();
        vkw::QueueSubmit(impl.ctx->queue, command, fence);
        const vk::Result result =
                vkw::WaitForFence(impl.ctx->device, fence);
        const auto end = std::chrono::steady_clock::now();
        impl.fenced_host_timings[index].host_microseconds =
                std::chrono::duration<double, std::micro>(end - begin).count();
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(
                    "fenced pass failed: " + impl.passes[index].desc.label +
                    " (" + vk::to_string(result) + ")");
        }
    }
}

void CustomGpuProgram::executeAdrenoFencedBatch() {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(impl.built,
                    "executeAdrenoFencedBatch called before build");
    CROSS3DGS_CHECK(impl.caps.vendor_id == 0x5143,
                    "executeAdrenoFencedBatch requires a Qualcomm device");
    if (!impl.adreno_fenced_command_buffers) {
        const vk::MemoryBarrier barrier(
                vk::AccessFlagBits::eShaderWrite |
                        vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead |
                        vk::AccessFlagBits::eShaderWrite |
                        vk::AccessFlagBits::eTransferRead |
                        vk::AccessFlagBits::eTransferWrite |
                        vk::AccessFlagBits::eIndirectCommandRead);
        const vk::PipelineStageFlags stages =
                vk::PipelineStageFlagBits::eComputeShader |
                vk::PipelineStageFlagBits::eTransfer |
                vk::PipelineStageFlagBits::eDrawIndirect;
        impl.adreno_fenced_command_buffers = vkw::CreateCommandBuffersPack(
                impl.ctx->device, impl.ctx->queue_family_idx,
                static_cast<uint32_t>(impl.passes.size()));
        for (size_t index = 0; index < impl.passes.size(); index++) {
            const auto& pass = impl.passes[index];
            const auto& command =
                    impl.adreno_fenced_command_buffers->cmd_bufs[index];
            vkw::BeginCommand(command, false);
            command->pipelineBarrier(stages, stages, {}, barrier, nullptr,
                                     nullptr);
            if (pass.desc.fill_zero) {
                for (const auto& buffer : pass.desc.buffers) {
                    command->fillBuffer(*impl.state(buffer).device->buf, 0,
                                        VK_WHOLE_SIZE, 0u);
                }
            } else {
                vkw::CmdBindPipeline(command, pass.pipeline,
                                     vk::PipelineBindPoint::eCompute);
                vkw::CmdBindDescSets(command, pass.pipeline,
                                     {pass.descriptor_set}, {},
                                     vk::PipelineBindPoint::eCompute);
                if (pass.desc.dispatch.mode == DispatchDesc::Mode::Direct) {
                    const auto size = pass.desc.dispatch.direct;
                    vkw::CmdDispatch(command, size.x, size.y, size.z);
                } else {
                    const auto& indirect =
                            impl.state(pass.desc.dispatch.indirect);
                    command->dispatchIndirect(
                            *indirect.device->buf,
                            pass.desc.dispatch.indirect_offset);
                }
            }
            vkw::EndCommand(command);
        }
        impl.adreno_fenced_fence = vkw::CreateFence(impl.ctx->device);
    }

    std::vector<vk::CommandBuffer> group;
    group.reserve(impl.adreno_fenced_command_buffers->cmd_bufs.size());
    for (const auto& command :
         impl.adreno_fenced_command_buffers->cmd_bufs) {
        group.push_back(command.get());
    }
    vkw::ResetFence(impl.ctx->device, impl.adreno_fenced_fence);
    const vk::SubmitInfo submit_info({}, {}, group, {});
    const auto begin = std::chrono::steady_clock::now();
    impl.ctx->queue.submit(submit_info, impl.adreno_fenced_fence->get());
    const vk::Result result =
            vkw::WaitForFence(impl.ctx->device, impl.adreno_fenced_fence);
    const auto end = std::chrono::steady_clock::now();
    impl.fenced_host_timings = {{
            "adreno_fenced_batch",
            std::chrono::duration<double, std::micro>(end - begin).count()}};
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error(
                "Adreno fenced batch failed (" +
                vk::to_string(result) + ")");
    }
}

std::string CustomGpuProgram::firstHangingPass(uint64_t timeout_ns) {
    Impl& impl = *m_impl;
    CROSS3DGS_CHECK(impl.built, "firstHangingPass called before build");
    const auto& command = impl.command_buffers->cmd_bufs[0];
    const vk::MemoryBarrier barrier(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    const vk::PipelineStageFlags stages =
            vk::PipelineStageFlagBits::eComputeShader |
            vk::PipelineStageFlagBits::eDrawIndirect;
    for (const auto& pass : impl.passes) {
        const auto pass_fence = vkw::CreateFence(impl.ctx->device);
        vkw::ResetCommand(command);
        vkw::BeginCommand(command, false);
        command->pipelineBarrier(stages, stages, {}, barrier, nullptr, nullptr);
        if (pass.desc.fill_zero) {
            for (const auto& buffer : pass.desc.buffers) {
                command->fillBuffer(*impl.state(buffer).device->buf, 0,
                                    VK_WHOLE_SIZE, 0u);
            }
        } else {
            vkw::CmdBindPipeline(command, pass.pipeline,
                                 vk::PipelineBindPoint::eCompute);
            vkw::CmdBindDescSets(command, pass.pipeline, {pass.descriptor_set},
                                 {}, vk::PipelineBindPoint::eCompute);
            if (pass.desc.dispatch.mode == DispatchDesc::Mode::Direct) {
                const auto size = pass.desc.dispatch.direct;
                vkw::CmdDispatch(command, size.x, size.y, size.z);
            } else {
                const auto& indirect = impl.state(pass.desc.dispatch.indirect);
                command->dispatchIndirect(*indirect.device->buf,
                                          pass.desc.dispatch.indirect_offset);
            }
        }
        vkw::EndCommand(command);
        vkw::QueueSubmit(impl.ctx->queue, command, pass_fence);
        const vk::Result result =
                vkw::WaitForFence(impl.ctx->device, pass_fence, timeout_ns);
        if (result != vk::Result::eSuccess) {
            std::string dims;
            if (pass.desc.dispatch.mode == DispatchDesc::Mode::Direct) {
                const auto s = pass.desc.dispatch.direct;
                dims = " groups=" + std::to_string(s.x) + "x" +
                       std::to_string(s.y) + "x" + std::to_string(s.z);
            } else {
                dims = " indirect";
            }
            return "HANG@" + pass.desc.label + dims + " (ok=" +
                   std::to_string(&pass - impl.passes.data()) + ")";
        }
    }
    return {};
}

bool CustomGpuProgram::isBuilt() const {
    return m_impl->built;
}

const DeviceCaps& CustomGpuProgram::deviceCaps() const {
    return m_impl->caps;
}

const std::vector<PassTiming>& CustomGpuProgram::lastTimings() const {
    return m_impl->timings;
}

const std::vector<HostPassTiming>&
CustomGpuProgram::lastFencedHostTimings() const {
    return m_impl->fenced_host_timings;
}

size_t CustomGpuProgram::allocatedBytes() const {
    size_t bytes = 0;
    for (const auto& [id, state] : m_impl->buffers) {
        (void)id;
        bytes += state.variable.desc().capacity_bytes;
    }
    return bytes;
}

VkContextPtr CustomGpuProgram::context() const {
    return m_impl->ctx;
}

}  // namespace cross3dgs::gpu

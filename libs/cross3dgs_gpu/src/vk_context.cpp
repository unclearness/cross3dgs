#include "vk_context.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "cross3dgs/gpu/gpu_program.h"

namespace cross3dgs::gpu {

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

std::string DescribeDevices(const std::vector<vk::PhysicalDevice>& devices) {
    std::string description;
    for (size_t i = 0; i < devices.size(); i++) {
        description += "\n  [" + std::to_string(i) + "] " +
                       devices[i].getProperties().deviceName.data();
    }
    return description;
}

// CROSS3DGS_VK_DEVICE selects the physical device: either a 0-based index
// into the enumeration order, or a case-insensitive substring of the device
// name ("nvidia").  Unset keeps the previous behavior exactly -- the first
// enumerated device -- so no platform changes without opting in.
//
// The default is only a heuristic: enumeration order is not guaranteed, and
// Linux hosts routinely expose Mesa's lavapipe (a CPU software rasterizer)
// alongside the real GPU, so "the first one" is not always the intended
// device.  A wrong pick surfaces as an unrelated-looking failure (lavapipe
// dies with "radix sort subgroup layout exceeds shared memory"), hence the
// override.
vk::PhysicalDevice SelectPhysicalDevice(const vkw::InstancePackPtr& instance) {
    const char* requested = std::getenv("CROSS3DGS_VK_DEVICE");
    if (requested == nullptr || requested[0] == '\0') {
        return vkw::GetFirstPhysicalDevice(instance);
    }
    const auto devices = vkw::GetPhysicalDevices(instance);
    if (devices.empty()) {
        throw std::runtime_error("No physical devices");
    }
    const std::string request(requested);
    size_t selected = devices.size();
    if (std::all_of(request.begin(), request.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        })) {
        size_t index = devices.size();
        try {
            index = static_cast<size_t>(std::stoul(request));
        } catch (const std::exception&) {
            index = devices.size();  // overflow: report as out of range
        }
        if (index >= devices.size()) {
            throw std::runtime_error(
                    "CROSS3DGS_VK_DEVICE=" + request + " is out of range; " +
                    std::to_string(devices.size()) +
                    " device(s) available:" + DescribeDevices(devices));
        }
        selected = index;
    } else {
        const std::string needle = ToLower(request);
        for (size_t i = 0; i < devices.size(); i++) {
            const std::string name =
                    ToLower(devices[i].getProperties().deviceName.data());
            if (name.find(needle) != std::string::npos) {
                selected = i;
                break;
            }
        }
        if (selected == devices.size()) {
            throw std::runtime_error("CROSS3DGS_VK_DEVICE=\"" + request +
                                     "\" matches no device name:" +
                                     DescribeDevices(devices));
        }
    }
    vkw::PrintInfo("CROSS3DGS_VK_DEVICE=" + request + " -> using [" +
                   std::to_string(selected) + "] " +
                   devices[selected].getProperties().deviceName.data());
    return devices[selected];
}

}  // namespace

void ThrowCheckFailure(const std::string& msg, const char* file, int line) {
    std::ostringstream ss;
    ss << msg << " (" << file << ":" << line << ")";
    throw std::runtime_error(ss.str());
}

VkContextPtr CreateContext(bool debug) {
    const char* dbg = std::getenv("CROSS3DGS_VK_DEBUG");
    return CreateVkContext(debug || (dbg != nullptr && dbg[0] == '1'));
}

VkContextPtr CreateVkContext(bool debug_enable) {
    auto ctx = std::make_shared<VkContext>();
    ctx->instance = vkw::CreateInstance("cross3dgs", 1, "cross3dgs", 1,
                                        debug_enable,
                                        /*surface_enable=*/false);
    ctx->physical_device = SelectPhysicalDevice(ctx->instance);
    // Do NOT require the explicit TRANSFER bit (vkw's default): per the
    // spec, graphics or compute capability implies transfer support, and
    // Adreno omits the bit on its graphics+compute family ("No sufficient
    // queue" on Snapdragon otherwise).
    ctx->queue_family_idx = vkw::GetQueueFamilyIdx(
            ctx->physical_device,
            vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);
    // Enable the physical device's full core feature set.  Extension
    // features must also be explicitly chained into device creation:
    // merely advertising VK_EXT_shader_atomic_float is not sufficient.  The
    // Gaussian backward kernels use buffer float32 atomicAdd and Mali rejects
    // their compute pipelines when that feature was not enabled.
    auto features = vkw::GetPhysicalFeatures2(ctx->physical_device);
    std::vector<const char*> device_extensions;
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features;
    const auto extensions =
            ctx->physical_device.enumerateDeviceExtensionProperties();
    const auto has_extension = [&](const char* name) {
        return std::any_of(extensions.begin(), extensions.end(),
                           [&](const vk::ExtensionProperties& extension) {
                               return std::strcmp(extension.extensionName,
                                                  name) == 0;
                           });
    };
    // A portability implementation (MoltenVK) advertises
    // VK_KHR_portability_subset, and the spec then *requires* the device to
    // enable it (VUID-VkDeviceCreateInfo-pProperties-04451); vkCreateDevice is
    // invalid otherwise.  Spelled literally because the generated constant
    // lives in vulkan_beta.h, which only appears under
    // VK_ENABLE_BETA_EXTENSIONS -- defining that would drag in the whole
    // provisional extension set.  Conformant drivers never advertise it, so
    // Windows, Linux, and Android device creation is untouched.
    constexpr const char* kPortabilitySubsetExtension =
            "VK_KHR_portability_subset";
    if (has_extension(kPortabilitySubsetExtension)) {
        device_extensions.push_back(kPortabilitySubsetExtension);
    }
    const bool has_atomic_float =
            has_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    if (has_atomic_float) {
        features->pNext = &atomic_float_features;
        ctx->physical_device.getFeatures2(features.get());
        if (atomic_float_features.shaderBufferFloat32AtomicAdd) {
            device_extensions.push_back(
                    VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        } else {
            features->pNext = nullptr;
        }
    }
    // Intel Xe compilers may emit compute shaders at SIMD8/16 while device
    // properties report subgroupSize 32.  The training kernels specialize on
    // the reported value (workgroup==subgroup reductions, host-sized radix
    // scratch), so pin every compute pipeline to the reported size via
    // VK_EXT_subgroup_size_control.  Gated on the Intel vendor id: NVIDIA,
    // Mali, and Adreno keep the exact device setup and pipeline path they
    // had before.
    constexpr uint32_t kIntelVendorId = 0x8086;
    vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_control_features;
    if (ctx->physical_device.getProperties().vendorID == kIntelVendorId &&
        has_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
        vk::PhysicalDeviceFeatures2 query_features;
        query_features.pNext = &subgroup_control_features;
        ctx->physical_device.getFeatures2(&query_features);

        vk::PhysicalDeviceSubgroupProperties subgroup_properties;
        vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT control_properties;
        subgroup_properties.pNext = &control_properties;
        vk::PhysicalDeviceProperties2 query_properties;
        query_properties.pNext = &subgroup_properties;
        ctx->physical_device.getProperties2(&query_properties);

        const uint32_t reported = subgroup_properties.subgroupSize;
        const bool compute_supported =
                static_cast<bool>(control_properties.requiredSubgroupSizeStages &
                                  vk::ShaderStageFlagBits::eCompute);
        if (subgroup_control_features.subgroupSizeControl && compute_supported &&
            control_properties.minSubgroupSize <= reported &&
            reported <= control_properties.maxSubgroupSize) {
            subgroup_control_features.computeFullSubgroups = VK_FALSE;
            subgroup_control_features.pNext =
                    const_cast<void*>(features->pNext);
            features->pNext = &subgroup_control_features;
            device_extensions.push_back(
                    VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
            ctx->forced_subgroup_size = reported;
        }
    }
    // Windows arms a ~2 s GPU watchdog (TDR) per submission; Intel iGPUs
    // are slow enough that an 8-iteration training batch exceeds it once
    // MCMC grows the Gaussian count, losing the device mid-run.  Splitting
    // the batch keeps every submission short.  Intel-only: discrete GPUs
    // keep the single-submit path.
    if (ctx->physical_device.getProperties().vendorID == kIntelVendorId) {
        ctx->split_batch_submits = true;
    }
    ctx->device = vkw::CreateDevice(ctx->queue_family_idx,
                                    ctx->physical_device, 1,
                                    /*swapchain_support=*/false,
                                    features, device_extensions);
    ctx->queue = vkw::GetQueue(ctx->device, ctx->queue_family_idx);
    ctx->limits = vkw::GetPhysicalProperties(ctx->physical_device)->limits;
    ctx->glsl_compiler = std::make_unique<vkw::GLSLCompiler>(true, false);
    ctx->glsl_size_compiler = std::make_unique<vkw::GLSLCompiler>(true, true);
    return ctx;
}

void RunOneShot(
        const VkContext& ctx,
        const std::function<void(const vk::UniqueCommandBuffer&)>& record) {
    if (!ctx.oneshot_cmds) {
        ctx.oneshot_cmds = vkw::CreateCommandBuffersPack(
                ctx.device, ctx.queue_family_idx, 1);
        ctx.oneshot_fence = vkw::CreateFence(ctx.device);
    }
    const auto& cmd = ctx.oneshot_cmds->cmd_bufs[0];
    vkw::ResetCommand(cmd);
    vkw::BeginCommand(cmd, /*one_time_submit=*/true);
    record(cmd);
    vkw::EndCommand(cmd);
    vkw::ResetFence(ctx.device, ctx.oneshot_fence);
    vkw::QueueSubmit(ctx.queue, cmd, ctx.oneshot_fence);
    const vk::Result result =
            vkw::WaitForFence(ctx.device, ctx.oneshot_fence);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error(
                "one-shot command failed (" + vk::to_string(result) + ")");
    }
}

}  // namespace cross3dgs::gpu

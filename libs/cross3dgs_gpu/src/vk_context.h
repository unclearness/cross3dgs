#ifndef CROSS3DGS_GPU_VK_CONTEXT_H
#define CROSS3DGS_GPU_VK_CONTEXT_H

#include <vkw/vkw.h>

#include <functional>
#include <memory>

namespace cross3dgs::gpu {

// Headless Vulkan bring-up shared by every CustomGpuProgram on one device.
struct VkContext {
    vkw::InstancePackPtr instance;
    vk::PhysicalDevice physical_device;
    uint32_t queue_family_idx = 0;
    vk::UniqueDevice device;
    vk::Queue queue;
    vk::PhysicalDeviceLimits limits;
    // Non-zero when compute pipelines must pin their subgroup size to the
    // value reported in device properties (VK_EXT_subgroup_size_control).
    // Set only for drivers whose compilers may pick a narrower SIMD width
    // than reported (Intel Xe); zero everywhere else keeps prior behavior.
    uint32_t forced_subgroup_size = 0;
    // Submit batched training iterations one repetition at a time instead of
    // as a single 8x command buffer.  Set for Windows iGPUs (Intel) where a
    // multi-second submission trips the OS GPU watchdog (TDR, ~2 s) and
    // yields VK_ERROR_DEVICE_LOST once the Gaussian count grows; false
    // everywhere else keeps the single-submit fast path.
    bool split_batch_submits = false;
    std::unique_ptr<vkw::GLSLCompiler> glsl_compiler;
    std::unique_ptr<vkw::GLSLCompiler> glsl_size_compiler;

    // Reused one-shot command buffer + fence for synchronous transfers.
    mutable vkw::CommandBuffersPackPtr oneshot_cmds;
    mutable vkw::FencePtr oneshot_fence;
};
using VkContextPtr = std::shared_ptr<VkContext>;

VkContextPtr CreateVkContext(bool debug_enable = false);

// Runs `record` into a one-shot command buffer, submits, and waits.
void RunOneShot(
        const VkContext& ctx,
        const std::function<void(const vk::UniqueCommandBuffer&)>& record);

}  // namespace cross3dgs::gpu

#endif  // CROSS3DGS_GPU_VK_CONTEXT_H

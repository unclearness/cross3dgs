#include <vkw/vkw.h>

#include <cstdio>
#include <string>

#include "cross3dgs/gpu/gpu_program.h"

namespace cross3dgs_android {

std::string QueryDeviceCapsJson() {
    try {
        auto instance = vkw::CreateInstance(
                "cross-3dgs-caps", 1, "cross-3dgs-caps", 1,
                /*debug_enable=*/false, /*surface_enable=*/false);
        const auto physical = vkw::GetFirstPhysicalDevice(instance);
        const auto properties = vkw::GetPhysicalProperties(physical);
        cross3dgs::gpu::CustomGpuProgram probe;
        const auto& caps = probe.deviceCaps();
        char json[1024];
        std::snprintf(
                json, sizeof(json),
                R"json({"deviceName":"%s",)json"
                R"json("apiVersion":"%u.%u.%u",)json"
                R"json("vendorID":%u,)json"
                R"json("subgroupSize":%u,)json"
                R"json("subgroupOperations":%u,)json"
                R"json("subgroupStages":%u,)json"
                R"json("float32AtomicAdd":%s,)json"
                R"json("maxComputeWorkGroupInvocations":%u,)json"
                R"json("maxSharedMemory":%u,)json"
                R"json("maxStorageBufferRange":%u})json",
                properties->deviceName.data(),
                VK_API_VERSION_MAJOR(properties->apiVersion),
                VK_API_VERSION_MINOR(properties->apiVersion),
                VK_API_VERSION_PATCH(properties->apiVersion),
                properties->vendorID, caps.subgroup_size,
                caps.subgroup_supported_operations,
                caps.subgroup_supported_stages,
                caps.float32_atomic_add ? "true" : "false",
                caps.max_workgroup_invocations, caps.max_shared_memory,
                static_cast<uint32_t>(caps.max_storage_buffer_range));
        return json;
    } catch (const std::exception& error) {
        return std::string(R"json({"error":")json") + error.what() +
                R"json("})json";
    }
}

}  // namespace cross3dgs_android

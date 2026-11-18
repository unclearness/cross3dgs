#ifndef CROSS3DGS_GPU_GPU_PROGRAM_H
#define CROSS3DGS_GPU_GPU_PROGRAM_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cross3dgs::gpu {

struct VkContext;
using VkContextPtr = std::shared_ptr<VkContext>;

// Creates a headless Vulkan context that several programs can share by
// passing it to their constructors. `debug` (or CROSS3DGS_VK_DEBUG=1)
// enables the validation layers.
VkContextPtr CreateContext(bool debug = false);

enum class StorageClass {
    StorageBuffer,
    UniformBuffer,
    IndirectBuffer,
};

enum class ScalarType {
    F32,
    F16,
    U32,
    I32,
};

struct TensorShape {
    uint32_t rank = 1;
    std::array<uint32_t, 4> dims{1, 1, 1, 1};

    size_t elementCount() const;
};

struct TensorDesc {
    StorageClass storage = StorageClass::StorageBuffer;
    ScalarType scalar_type = ScalarType::F32;
    uint32_t lanes = 1;
    TensorShape logical_shape{};
    size_t capacity_bytes = 0;
    size_t alignment = 4;
    // Host-coherent storage is useful on unified-memory GPUs for buffers that
    // the CPU replaces between synchronous submissions.
    bool host_visible = false;

    size_t elementBytes() const;
};

class BufferVariable {
public:
    BufferVariable() = default;

    explicit operator bool() const { return m_id != 0; }
    bool operator==(const BufferVariable& other) const {
        return m_id == other.m_id;
    }
    uint64_t id() const { return m_id; }
    const TensorDesc& desc() const { return m_desc; }
    size_t logicalCount() const { return m_logical_count; }
    size_t capacityBytes() const { return m_desc.capacity_bytes; }

private:
    friend class CustomGpuProgram;
    BufferVariable(uint64_t id, TensorDesc desc, size_t logical_count)
        : m_id(id), m_desc(std::move(desc)), m_logical_count(logical_count) {}

    uint64_t m_id = 0;
    TensorDesc m_desc{};
    size_t m_logical_count = 0;
};

enum class PipelineStage {
    Compute,
    Indirect,
    Transfer,  // vkCmdFillBuffer and other transfer-stage operations
};

enum class AccessMode {
    Read,
    Write,
    ReadWrite,
};

struct BufferRange {
    size_t offset = 0;
    size_t size = 0;  // zero means the remainder of the buffer
};

struct ResourceUse {
    BufferVariable resource;
    PipelineStage stage = PipelineStage::Compute;
    AccessMode access = AccessMode::Read;
    BufferRange range{};
};

struct DispatchSize {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

struct DispatchDesc {
    enum class Mode {
        Direct,
        Indirect,
    };

    static DispatchDesc Direct(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    static DispatchDesc Indirect(const BufferVariable& buffer,
                                 uint32_t offset = 0);

    Mode mode = Mode::Direct;
    DispatchSize direct{};
    BufferVariable indirect;
    uint32_t indirect_offset = 0;
};

enum class ShaderOptimization {
    Throughput,
    Size,
};

struct ComputePassDesc {
    std::string label;
    std::string shader_source;
    // Descriptor binding i is buffers[i]. StorageClass selects storage vs
    // uniform descriptor type. Indirect buffers may also be storage bindings.
    std::vector<BufferVariable> buffers;
    std::vector<ResourceUse> resources;
    DispatchDesc dispatch{};
    ShaderOptimization optimization = ShaderOptimization::Throughput;
    // When true this pass is not a compute dispatch: every buffer in `buffers`
    // is zeroed with vkCmdFillBuffer (a transfer-stage op, no shader). Use for
    // gradient/scratch clears; avoids a large trivial memset dispatch that some
    // drivers (e.g. Adreno) mishandle. `shader_source` and `dispatch` are
    // ignored; `resources` still drives hazard barriers.
    bool fill_zero = false;
};

struct DeviceCaps {
    uint32_t subgroup_size = 0;
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
    uint32_t max_workgroup_invocations = 0;
    uint32_t max_shared_memory = 0;
    uint32_t max_storage_buffer_bindings = 0;
    uint64_t max_storage_buffer_range = 0;
    bool f16_arithmetic = false;
    bool f16_storage = false;
    bool shader_int64 = false;
    bool float32_atomic_add = false;
    bool subgroup_size_control = false;
    uint32_t subgroup_supported_operations = 0;
    uint32_t subgroup_supported_stages = 0;
    bool dispatch_indirect = true;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
};

struct PassTiming {
    std::string label;
    double gpu_microseconds = 0.0;
};

// Host wall time from queue submission through fence completion for one pass
// in executeFenced(). This is intentionally separate from GPU timestamps.
struct HostPassTiming {
    std::string label;
    double host_microseconds = 0.0;
};

// Opaque, pre-recorded Vulkan compute island. Buffer allocation, shader
// compilation, descriptors, and barriers happen in build(), never execute().
class CustomGpuProgram {
public:
    explicit CustomGpuProgram(VkContextPtr context = nullptr);
    ~CustomGpuProgram();
    CustomGpuProgram(const CustomGpuProgram&) = delete;
    CustomGpuProgram& operator=(const CustomGpuProgram&) = delete;
    CustomGpuProgram(CustomGpuProgram&&) noexcept;
    CustomGpuProgram& operator=(CustomGpuProgram&&) noexcept;

    BufferVariable createBuffer(const TensorDesc& desc,
                                size_t logical_count = 0);
    void setLogicalCount(BufferVariable& buffer, size_t logical_count);
    void upload(const BufferVariable& buffer, const void* data, size_t bytes,
                size_t offset = 0);
    std::vector<std::byte> download(const BufferVariable& buffer,
                                    size_t bytes = 0, size_t offset = 0);

    void addPass(const ComputePassDesc& pass);
    void setProfilingEnabled(bool enabled);
    // When enabled, a full compute+transfer memory barrier is inserted before
    // every pass in the batched command buffer, in addition to the
    // resource-derived barriers. This is a conservative schedule for drivers
    // with a weak memory model (observed on Adreno) where the minimal
    // per-resource barriers leave a latent hazard; must be set before build().
    void setConservativeBarriers(bool enabled);
    void build();
    void execute();
    // Submit the pre-recorded one-iteration command buffer and wait for its
    // fence without reading timestamp-query results on the host. This keeps
    // API-boundary host timing independent of profiling readback; GPU work is
    // otherwise identical to execute().
    void executeWithoutTimingReadback();
    void executeBatch(uint32_t repeat_count);
    // Runs one iteration as one vkQueueSubmit + fence per pass, draining the
    // GPU between passes. Slower than execute(), but required on drivers where
    // a single large multi-pass submit hangs even with full barriers (observed
    // on Adreno: the batched submit faults while fully drained passes do not).
    void executeFenced();
    // Qualcomm-only training path: cache one command buffer per pass, submit
    // the ordered set in one queue submission, and wait on one fence. Keeping
    // pass boundaries avoids the Adreno fault seen with one monolithic command
    // buffer while removing per-pass recording/submission/fence overhead.
    void executeAdrenoFencedBatch();

    // Diagnostic: submit each recorded pass in its own command buffer with a
    // full barrier, waiting on a per-pass fence with `timeout_ns`. Returns the
    // label of the first pass whose fence does not signal in time (i.e. the
    // pass that hangs the GPU), or an empty string if every pass completed.
    // Used to localize device-specific hangs; not part of the training path.
    std::string firstHangingPass(uint64_t timeout_ns);

    bool isBuilt() const;
    const DeviceCaps& deviceCaps() const;
    const std::vector<PassTiming>& lastTimings() const;
    const std::vector<HostPassTiming>& lastFencedHostTimings() const;
    // Total capacity of unique Vulkan buffers owned by this program.
    size_t allocatedBytes() const;
    VkContextPtr context() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace cross3dgs::gpu

#endif  // CROSS3DGS_GPU_GPU_PROGRAM_H

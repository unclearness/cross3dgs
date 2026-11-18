#include "gaussian/program_impl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace cross3dgs {
using detail::CameraUniform;

namespace {

size_t ReorderedShIndex(size_t gaussian,size_t packed) {
    return (gaussian&~size_t{31})*12+packed*32+(gaussian&31);
}

std::vector<glm::vec4> ReorderSh(const std::vector<glm::vec4>& linear,
                                 size_t gaussian_count) {
    std::vector<glm::vec4> reordered(((gaussian_count+31)/32)*32*12);
    for(size_t gaussian=0;gaussian<gaussian_count;gaussian++)
        for(size_t packed=0;packed<12;packed++)
            reordered[ReorderedShIndex(gaussian,packed)]=
                    linear[gaussian*12+packed];
    return reordered;
}

std::vector<glm::vec4> UnorderSh(const std::byte* reordered,
                                 size_t gaussian_count) {
    const auto* values=reinterpret_cast<const glm::vec4*>(reordered);
    std::vector<glm::vec4> linear(gaussian_count*12);
    for(size_t gaussian=0;gaussian<gaussian_count;gaussian++)
        for(size_t packed=0;packed<12;packed++)
            linear[gaussian*12+packed]=
                    values[ReorderedShIndex(gaussian,packed)];
    return linear;
}

std::pair<std::vector<glm::vec4>,std::vector<glm::vec4>> SplitShRest(
        const std::vector<glm::vec4>& linear,size_t gaussian_count) {
    std::pair<std::vector<glm::vec4>,std::vector<glm::vec4>> shards;
    shards.first.resize(gaussian_count*6);
    shards.second.resize(gaussian_count*6);
    for(size_t gaussian=0;gaussian<gaussian_count;gaussian++) {
        std::copy_n(linear.data()+gaussian*12,6,
                    shards.first.data()+gaussian*6);
        std::copy_n(linear.data()+gaussian*12+6,6,
                    shards.second.data()+gaussian*6);
    }
    return shards;
}

std::vector<glm::vec4> JoinShRest(const std::byte* first,
                                  const std::byte* second,
                                  size_t gaussian_count) {
    const auto* a=reinterpret_cast<const glm::vec4*>(first);
    const auto* b=reinterpret_cast<const glm::vec4*>(second);
    std::vector<glm::vec4> linear(gaussian_count*12);
    for(size_t gaussian=0;gaussian<gaussian_count;gaussian++) {
        std::copy_n(a+gaussian*6,6,linear.data()+gaussian*12);
        std::copy_n(b+gaussian*6,6,linear.data()+gaussian*12+6);
    }
    return linear;
}

}  // namespace

GaussianForwardProgram::GaussianForwardProgram(
        const GaussianForwardConfig& config, gpu::VkContextPtr context)
    : m_impl(std::make_unique<Impl>(config, std::move(context))) {}

GaussianForwardProgram::~GaussianForwardProgram() = default;

void GaussianForwardProgram::uploadState(const GaussianStateCpu& state) {
    auto& impl = *m_impl;
    if (state.size() > impl.config.gaussian_capacity ||
        state.scales_opacity.size() != state.size() ||
        state.rotations.size() != state.size() ||
        state.sh_dc.size() != state.size() ||
        state.sh_rest.size() != state.size() * 12) {
        throw std::invalid_argument("invalid GaussianStateCpu shape/capacity");
    }
    impl.gaussian_count = static_cast<uint32_t>(state.size());
    impl.program.upload(impl.means, state.means.data(),
                        state.means.size() * sizeof(glm::vec4));
    impl.program.upload(impl.scales_opacity, state.scales_opacity.data(),
                        state.scales_opacity.size() * sizeof(glm::vec4));
    impl.program.upload(impl.rotations, state.rotations.data(),
                        state.rotations.size() * sizeof(glm::vec4));
    impl.program.upload(impl.sh_dc, state.sh_dc.data(),
                        state.sh_dc.size() * sizeof(glm::vec4));
    if (impl.split_sh_rest_storage) {
        const auto shards=SplitShRest(state.sh_rest,state.size());
        impl.program.upload(impl.sh_rest,shards.first.data(),
                            shards.first.size()*sizeof(glm::vec4));
        impl.program.upload(impl.sh_rest_b,shards.second.data(),
                            shards.second.size()*sizeof(glm::vec4));
    } else if(impl.reorder_sh) {
        const auto reordered=ReorderSh(state.sh_rest,state.size());
        impl.program.upload(impl.sh_rest,reordered.data(),
                            reordered.size()*sizeof(glm::vec4));
    } else {
        impl.program.upload(impl.sh_rest, state.sh_rest.data(),
                            state.sh_rest.size() * sizeof(glm::vec4));
    }
    if (impl.config.enable_backward) {
        const glm::uvec4 meta(0, impl.gaussian_count, 1, 0);
        impl.program.upload(impl.optimizer_meta, &meta, sizeof(meta));
    }
}

namespace {

// Matches GLSL unpackUnorm4x8: byte c decodes to c/255.0, so images loaded
// from 8-bit sources survive the round trip bitwise unchanged.
std::vector<uint32_t> PackTargetRgba8(const std::vector<float>& pixels) {
    std::vector<uint32_t> packed(pixels.size() / 4);
    for (size_t index = 0; index < packed.size(); index++) {
        uint32_t value = 0;
        for (uint32_t channel = 0; channel < 4; channel++) {
            const float clamped = std::min(
                    1.0f, std::max(0.0f, pixels[index * 4 + channel]));
            value |= static_cast<uint32_t>(std::lround(clamped * 255.0f))
                     << (8 * channel);
        }
        packed[index] = value;
    }
    return packed;
}

// IGS+ Edge-Aware Score: replace the target alpha channel with a thinned edge
// map following the paper's distilled Canny pipeline -- grayscale, 5x5 Gaussian
// blur, Sobel gradients, non-maximum suppression along the gradient direction,
// then median-normalization. NMS yields a one-pixel-wide structural backbone so
// the edge gate concentrates splits on true boundaries, not planar surfaces.
// The loss/backward read only target.xyz, so repurposing alpha is free.
void ComputeEdgeAlpha(FloatImage& image) {
    const int w = image.width, h = image.height, c = image.channels;
    if (w < 3 || h < 3 || c < 4) return;
    const size_t pixels = static_cast<size_t>(w) * h;
    const auto at = [w](int x, int y) { return static_cast<size_t>(y) * w + x; };
    const auto clampx = [w](int x) { return x < 0 ? 0 : (x >= w ? w - 1 : x); };
    const auto clampy = [h](int y) { return y < 0 ? 0 : (y >= h ? h - 1 : y); };
    // 1. Grayscale (luminance).
    std::vector<float> gray(pixels);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const size_t p = at(x, y) * c;
            gray[at(x, y)] = 0.299f * image.pixels[p] +
                    0.587f * image.pixels[p + 1] + 0.114f * image.pixels[p + 2];
        }
    }
    // 2. 5x5 Gaussian blur (sigma ~1, kernel/273) to suppress sensor noise.
    static const int kernel[5][5] = {{1, 4, 7, 4, 1}, {4, 16, 26, 16, 4},
            {7, 26, 41, 26, 7}, {4, 16, 26, 16, 4}, {1, 4, 7, 4, 1}};
    std::vector<float> blur(pixels);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ky++)
                for (int kx = -2; kx <= 2; kx++)
                    acc += kernel[ky + 2][kx + 2] *
                           gray[at(clampx(x + kx), clampy(y + ky))];
            blur[at(x, y)] = acc / 273.0f;
        }
    }
    // 3. Sobel gradient magnitude + orientation.
    std::vector<float> magnitude(pixels, 0.0f), angle(pixels, 0.0f);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const int xm = clampx(x - 1), xp = clampx(x + 1);
            const int ym = clampy(y - 1), yp = clampy(y + 1);
            const float gx = blur[at(xp, ym)] + 2.0f * blur[at(xp, y)] +
                    blur[at(xp, yp)] - blur[at(xm, ym)] -
                    2.0f * blur[at(xm, y)] - blur[at(xm, yp)];
            const float gy = blur[at(xm, yp)] + 2.0f * blur[at(x, yp)] +
                    blur[at(xp, yp)] - blur[at(xm, ym)] -
                    2.0f * blur[at(x, ym)] - blur[at(xp, ym)];
            magnitude[at(x, y)] = std::sqrt(gx * gx + gy * gy);
            angle[at(x, y)] = std::atan2(gy, gx);
        }
    }
    // 4. Non-maximum suppression: keep a pixel only if it is a local maximum
    // along the (quantized) gradient direction.
    std::vector<float> thinned(pixels, 0.0f);
    const float pi = 3.14159265358979323846f;
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float a = angle[at(x, y)];
            if (a < 0.0f) a += pi;
            int dx, dy;
            if (a < pi / 8.0f || a >= 7.0f * pi / 8.0f) { dx = 1; dy = 0; }
            else if (a < 3.0f * pi / 8.0f) { dx = 1; dy = 1; }
            else if (a < 5.0f * pi / 8.0f) { dx = 0; dy = 1; }
            else { dx = -1; dy = 1; }
            const float m = magnitude[at(x, y)];
            if (m >= magnitude[at(x + dx, y + dy)] &&
                m >= magnitude[at(x - dx, y - dy)]) {
                thinned[at(x, y)] = m;
            }
        }
    }
    // 5. Median-normalization: scale by the median of the surviving ridge
    // magnitudes so strong boundaries are preserved without being crushed by a
    // few global peaks. Graded into [0,1]; planar regions stay exactly 0.
    std::vector<float> nonzero;
    nonzero.reserve(pixels / 8);
    for (float value : thinned)
        if (value > 0.0f) nonzero.push_back(value);
    float scale = 1e-6f;
    if (!nonzero.empty()) {
        const size_t mid = nonzero.size() / 2;
        std::nth_element(nonzero.begin(), nonzero.begin() + mid, nonzero.end());
        scale = 2.0f * nonzero[mid] + 1e-6f;
    }
    for (size_t i = 0; i < pixels; i++)
        image.pixels[i * c + 3] = std::min(1.0f, thinned[i] / scale);
}

}  // namespace

void GaussianForwardProgram::uploadTarget(const FloatImage& target) {
    auto& impl = *m_impl;
    if (!impl.config.enable_backward || target.width != impl.config.width ||
        target.height != impl.config.height || target.channels != 4 ||
        target.pixels.size() !=
                static_cast<size_t>(target.width) * target.height * 4) {
        throw std::invalid_argument("invalid target or backward is disabled");
    }
    if (impl.config.byte_targets) {
        const auto packed = PackTargetRgba8(target.pixels);
        impl.program.upload(impl.target, packed.data(),
                            packed.size() * sizeof(uint32_t));
    } else {
        impl.program.upload(impl.target, target.pixels.data(),
                            target.pixels.size() * sizeof(float));
    }
}

void GaussianForwardProgram::uploadTrainingSet(
        const std::vector<Camera>& cameras,
        const std::vector<FloatImage>& targets) {
    if (cameras.size() != targets.size()) {
        throw std::invalid_argument("invalid resident training set");
    }
    uploadTrainingSet(cameras, [&](size_t index) { return targets[index]; });
}

void GaussianForwardProgram::uploadTrainingSet(
        const std::vector<Camera>& cameras,
        const std::function<FloatImage(size_t)>& load_target) {
    auto& impl = *m_impl;
    if (impl.config.resident_view_capacity == 0 || cameras.empty() ||
        !load_target ||
        cameras.size() > impl.config.resident_view_capacity) {
        throw std::invalid_argument("invalid resident training set");
    }
    std::vector<CameraUniform> uniforms;
    uniforms.reserve(cameras.size());
    const size_t pixels =
            static_cast<size_t>(impl.config.width) * impl.config.height;
    if (impl.stream_resident_target) {
        impl.resident_target_stride = impl.config.byte_targets
                ? pixels * sizeof(uint32_t)
                : pixels * sizeof(glm::vec4);
        impl.resident_target_bytes.resize(
                cameras.size() * impl.resident_target_stride);
    }
    for (size_t index = 0; index < cameras.size(); index++) {
        const auto& camera = cameras[index];
        FloatImage target = load_target(index);
        if (camera.width != impl.config.width ||
            camera.height != impl.config.height ||
            camera.model != CameraModel::Pinhole ||
            target.width != impl.config.width ||
            target.height != impl.config.height || target.channels != 4 ||
            target.pixels.size() != pixels * 4) {
            throw std::invalid_argument("resident view shape/model mismatch");
        }
        if (impl.igs_edge_enabled) ComputeEdgeAlpha(target);
        CameraUniform uniform;
        uniform.world_to_camera = camera.world_to_camera;
        uniform.intrinsics = {camera.fx, camera.fy, camera.cx, camera.cy};
        uniform.dims = {impl.grid_width, impl.grid_height, impl.config.width,
                        impl.config.height};
        uniform.state = {impl.gaussian_count, 0, 0, 0};
        uniforms.push_back(uniform);
        if (impl.config.byte_targets) {
            const auto packed = PackTargetRgba8(target.pixels);
            if (impl.stream_resident_target) {
                std::memcpy(impl.resident_target_bytes.data() +
                                    index * impl.resident_target_stride,
                            packed.data(), impl.resident_target_stride);
            } else {
                impl.program.upload(impl.target, packed.data(),
                                    packed.size() * sizeof(uint32_t),
                                    index * pixels * sizeof(uint32_t));
            }
        } else {
            if (impl.stream_resident_target) {
                std::memcpy(impl.resident_target_bytes.data() +
                                    index * impl.resident_target_stride,
                            target.pixels.data(), impl.resident_target_stride);
            } else {
                impl.program.upload(impl.target, target.pixels.data(),
                                    target.pixels.size() * sizeof(float),
                                    index * pixels * sizeof(glm::vec4));
            }
        }
    }
    impl.program.upload(impl.camera_atlas, uniforms.data(),
                        uniforms.size() * sizeof(CameraUniform));
    impl.host_optimizer_step = 0;
    impl.resident_view_count = static_cast<uint32_t>(cameras.size());
    const glm::uvec4 meta(0, impl.gaussian_count,
                          static_cast<uint32_t>(cameras.size()), 0);
    impl.program.upload(impl.optimizer_meta, &meta, sizeof(meta));
}

void GaussianForwardProgram::setCamera(const Camera& camera) {
    auto& impl = *m_impl;
    if (camera.width != impl.config.width || camera.height != impl.config.height ||
        camera.model != CameraModel::Pinhole) {
        throw std::invalid_argument("forward M2 requires matching pinhole camera");
    }
    CameraUniform uniform;
    uniform.world_to_camera = camera.world_to_camera;
    uniform.intrinsics = {camera.fx, camera.fy, camera.cx, camera.cy};
    // dims: grid width, grid height, image width, image height.
    uniform.dims = {impl.grid_width, impl.grid_height, impl.config.width,
                    impl.config.height};
    // state: active Gaussian count; remaining lanes are reserved.
    uniform.state = {impl.gaussian_count, 0, 0, 0};
    impl.program.upload(impl.camera, &uniform, sizeof(uniform));
}

void GaussianForwardProgram::execute() {
    if (m_impl->use_host_step_fill) {
        const uint32_t step = m_impl->host_optimizer_step++;
        if (m_impl->stream_resident_target &&
            m_impl->resident_view_count > 0) {
            const auto hash32 = [](uint32_t value) {
                value ^= value >> 16u; value *= 0x7feb352du;
                value ^= value >> 15u; value *= 0x846ca68bu;
                return value ^ (value >> 16u);
            };
            size_t view = step % m_impl->resident_view_count;
            if (m_impl->config.randomize_training_views) {
                view = hash32(step ^ m_impl->config.training_view_seed) %
                       m_impl->resident_view_count;
            }
            m_impl->program.upload(
                    m_impl->target,
                    m_impl->resident_target_bytes.data() +
                    view * m_impl->resident_target_stride,
                    m_impl->resident_target_stride);
        }
    }
    if (m_impl->use_fenced_execution) {
        if (m_impl->program.deviceCaps().vendor_id == 0x5143) {
            m_impl->program.executeAdrenoFencedBatch();
        } else {
            m_impl->program.executeFenced();
        }
    } else {
        m_impl->program.execute();
    }
}

void GaussianForwardProgram::executeBatch(uint32_t repeat_count) {
    // A streamed target must be replaced before every optimizer step. It
    // therefore cannot reuse a pre-recorded multi-step batch containing one
    // target binding; preserve the exact cyclic view order with individual
    // executions.
    if (m_impl->use_host_step_fill) {
        for (uint32_t index = 0; index < repeat_count; index++) execute();
        return;
    }
    m_impl->program.executeBatch(repeat_count);
}

std::string GaussianForwardProgram::firstHangingPass(uint64_t timeout_ns) {
    return m_impl->program.firstHangingPass(timeout_ns);
}

FloatImage GaussianForwardProgram::downloadColor() {
    const auto& config = m_impl->config;
    const auto bytes = m_impl->program.download(m_impl->color);
    FloatImage image;
    image.width = config.width;
    image.height = config.height;
    image.pixels.resize(static_cast<size_t>(config.width) * config.height * 4);
    std::memcpy(image.pixels.data(), bytes.data(), bytes.size());
    return image;
}

GaussianGradientCpu GaussianForwardProgram::downloadGradients() {
    auto& impl = *m_impl;
    if (!impl.config.enable_backward) {
        throw std::logic_error("backward is disabled");
    }
    GaussianGradientCpu gradients;
    const auto copy = [&](const gpu::BufferVariable& buffer,
                          std::vector<glm::vec4>& output) {
        const auto bytes = impl.program.download(buffer);
        output.resize(impl.gaussian_count);
        std::memcpy(output.data(), bytes.data(),
                    output.size() * sizeof(glm::vec4));
    };
    copy(impl.means_grad, gradients.means);
    copy(impl.scale_grad, gradients.scales_opacity);
    copy(impl.rotation_grad, gradients.rotations);
    copy(impl.sh_grad, gradients.sh_dc);
    {
        const auto bytes = impl.program.download(impl.sh_rest_grad);
        if (impl.split_sh_rest_storage) {
            const auto second=impl.program.download(impl.sh_rest_grad_b);
            gradients.sh_rest=JoinShRest(bytes.data(),second.data(),
                                         impl.gaussian_count);
        } else if(impl.reorder_sh) {
            gradients.sh_rest=UnorderSh(bytes.data(),impl.gaussian_count);
        } else {
            gradients.sh_rest.resize(impl.gaussian_count * 12);
            std::memcpy(gradients.sh_rest.data(), bytes.data(),
                        gradients.sh_rest.size() * sizeof(glm::vec4));
        }
    }
    return gradients;
}

GaussianStateCpu GaussianForwardProgram::downloadState() {
    auto& impl = *m_impl;
    if (impl.config.enable_optimizer) {
        const auto meta_bytes = impl.program.download(impl.optimizer_meta);
        glm::uvec4 meta{};
        std::memcpy(&meta, meta_bytes.data(), sizeof(meta));
        impl.gaussian_count = meta.y;
    }
    GaussianStateCpu state;
    const auto copy = [&](const gpu::BufferVariable& buffer,
                          std::vector<glm::vec4>& output) {
        const auto bytes = impl.program.download(buffer);
        output.resize(impl.gaussian_count);
        std::memcpy(output.data(), bytes.data(),
                    output.size() * sizeof(glm::vec4));
    };
    copy(impl.means, state.means);
    copy(impl.scales_opacity, state.scales_opacity);
    copy(impl.rotations, state.rotations);
    copy(impl.sh_dc, state.sh_dc);
    {
        const auto bytes = impl.program.download(impl.sh_rest);
        if (impl.split_sh_rest_storage) {
            const auto second=impl.program.download(impl.sh_rest_b);
            state.sh_rest=JoinShRest(bytes.data(),second.data(),
                                     impl.gaussian_count);
        } else if(impl.reorder_sh) {
            state.sh_rest=UnorderSh(bytes.data(),impl.gaussian_count);
        } else {
            state.sh_rest.resize(impl.gaussian_count * 12);
            std::memcpy(state.sh_rest.data(), bytes.data(),
                        state.sh_rest.size() * sizeof(glm::vec4));
        }
    }
    return state;
}

glm::uvec4 GaussianForwardProgram::debugMeta() {
    const auto bytes = m_impl->program.download(m_impl->optimizer_meta);
    glm::uvec4 meta{};
    std::memcpy(&meta, bytes.data(), sizeof(meta));
    return meta;
}

void GaussianForwardProgram::debugSetStep(uint32_t step) {
    auto& impl = *m_impl;
    if (!impl.config.enable_backward) {
        throw std::logic_error("backward is disabled");
    }
    const auto bytes = impl.program.download(impl.optimizer_meta);
    glm::uvec4 meta{};
    std::memcpy(&meta, bytes.data(), sizeof(meta));
    meta.x = step;
    impl.host_optimizer_step = step;
    impl.program.upload(impl.optimizer_meta, &meta, sizeof(meta));
    const auto camera_bytes = impl.program.download(impl.camera);
    CameraUniform camera{};
    std::memcpy(&camera, camera_bytes.data(), sizeof(camera));
    camera.state.z = step;
    impl.program.upload(impl.camera, &camera, sizeof(camera));
}

glm::uvec4 GaussianForwardProgram::debugForwardCounts() {
    auto& impl = *m_impl;
    // Sum of per-Gaussian tile coverage (projection output) and the total pair
    // count the scan produced (radix_meta[0].x). Isolates where the forward
    // binning collapses to zero.
    const auto counts = impl.program.download(impl.gaussian_tile_counts);
    const auto* values = reinterpret_cast<const uint32_t*>(counts.data());
    uint64_t coverage = 0;
    for (uint32_t i = 0; i < impl.gaussian_count; i++) coverage += values[i];
    const auto meta_bytes = impl.program.download(impl.radix_meta);
    glm::uvec4 radix{};
    std::memcpy(&radix, meta_bytes.data(), sizeof(radix));
    return glm::uvec4(static_cast<uint32_t>(coverage & 0xffffffffu),
                      static_cast<uint32_t>(coverage >> 32), radix.x, 0);
}

glm::uvec4 GaussianForwardProgram::debugPairIntegrity() {
    auto& impl = *m_impl;
    const auto meta_bytes = impl.program.download(impl.radix_meta);
    glm::uvec4 meta{};
    std::memcpy(&meta, meta_bytes.data(), sizeof(meta));
    const uint32_t count = std::min<uint32_t>(
            meta.x, static_cast<uint32_t>(impl.pair_capacity));
    const auto key_bytes = impl.program.download(
            impl.radix_keys_a, static_cast<size_t>(count) * sizeof(uint32_t));
    const auto value_bytes = impl.program.download(
            impl.radix_values_a,
            static_cast<size_t>(count) * sizeof(uint32_t));
    const auto rect_bytes = impl.program.download(
            impl.tile_rect,
            static_cast<size_t>(impl.gaussian_count) * sizeof(glm::uvec4));
    const auto projected_bytes = impl.program.download(
            impl.projected,
            static_cast<size_t>(impl.gaussian_count) * sizeof(glm::vec4));
    uint32_t order_errors = 0;
    uint32_t key_value_errors = 0;
    uint32_t rectangle_errors = 0;
    for (uint32_t index = 0; index < count; index++) {
        uint32_t key = 0;
        uint32_t id = 0;
        std::memcpy(&key, key_bytes.data() +
                                  static_cast<size_t>(index) * sizeof(key),
                    sizeof(key));
        std::memcpy(&id, value_bytes.data() +
                                 static_cast<size_t>(index) * sizeof(id),
                    sizeof(id));
        if (index > 0) {
            uint32_t previous = 0;
            std::memcpy(&previous,
                        key_bytes.data() +
                                static_cast<size_t>(index - 1) *
                                        sizeof(previous),
                        sizeof(previous));
            if (key < previous) order_errors++;
        }
        if (id >= impl.gaussian_count) {
            key_value_errors++;
            rectangle_errors++;
            continue;
        }
        glm::uvec4 rectangle{};
        std::memcpy(&rectangle,
                    rect_bytes.data() + static_cast<size_t>(id) *
                                                sizeof(rectangle),
                    sizeof(rectangle));
        const uint32_t tile = key >> impl.depth_bits;
        const uint32_t x = tile % impl.grid_width;
        const uint32_t y = tile / impl.grid_width;
        if (x < rectangle.x || x >= rectangle.z || y < rectangle.y ||
            y >= rectangle.w) {
            rectangle_errors++;
        }
        glm::vec4 projected{};
        std::memcpy(&projected,
                    projected_bytes.data() + static_cast<size_t>(id) *
                                                     sizeof(projected),
                    sizeof(projected));
        uint32_t depth = 0;
        std::memcpy(&depth, &projected.z, sizeof(depth));
        const uint32_t expected = impl.depth_bits == 0
                ? tile
                : (tile << impl.depth_bits) |
                        (depth >> (32u - impl.depth_bits));
        if (key != expected) key_value_errors++;
    }
    return {count, order_errors, key_value_errors, rectangle_errors};
}

float GaussianForwardProgram::downloadMeanL1() {
    auto& impl = *m_impl;
    if (!impl.config.enable_backward) {
        throw std::logic_error("backward is disabled");
    }
    const size_t count =
            static_cast<size_t>(impl.config.width) * impl.config.height;
    const auto bytes = impl.program.download(
            impl.loss_value, count * sizeof(float));
    const auto* values = reinterpret_cast<const float*>(bytes.data());
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

TrainingPreview GaussianForwardProgram::downloadTrainingPreview() {
    auto& impl = *m_impl;
    if (!impl.config.enable_backward) {
        throw std::logic_error("backward is disabled");
    }
    const size_t pixels =
            static_cast<size_t>(impl.config.width) * impl.config.height;
    TrainingPreview preview;
    CameraUniform camera;
    const auto camera_bytes =
            impl.program.download(impl.camera, sizeof(camera));
    std::memcpy(&camera, camera_bytes.data(), sizeof(camera));
    preview.view = camera.state.y;

    // A streamed target holds only the current view.
    const size_t slot = impl.stream_resident_target ||
                                impl.config.resident_view_capacity == 0
                                ? 0
                                : preview.view;
    preview.target.width = impl.config.width;
    preview.target.height = impl.config.height;
    preview.target.pixels.resize(pixels * 4);
    if (impl.config.byte_targets) {
        const size_t stride = pixels * sizeof(uint32_t);
        const auto bytes =
                impl.program.download(impl.target, stride, slot * stride);
        const auto* packed = reinterpret_cast<const uint8_t*>(bytes.data());
        for (size_t i = 0; i < preview.target.pixels.size(); i++) {
            preview.target.pixels[i] = static_cast<float>(packed[i]) / 255.0f;
        }
    } else {
        const size_t stride = pixels * sizeof(glm::vec4);
        const auto bytes =
                impl.program.download(impl.target, stride, slot * stride);
        std::memcpy(preview.target.pixels.data(), bytes.data(), stride);
    }

    preview.rendered = downloadColor();

    const auto loss_bytes =
            impl.program.download(impl.loss_value, pixels * sizeof(float));
    preview.loss.resize(pixels);
    std::memcpy(preview.loss.data(), loss_bytes.data(), loss_bytes.size());
    return preview;
}

GaussianForwardStats GaussianForwardProgram::stats() {
    GaussianForwardStats result;
    const auto bytes = m_impl->program.download(m_impl->overflow);
    uint32_t counters[3] = {};
    std::memcpy(counters, bytes.data(), sizeof(counters));
    result.overflow_count = counters[0];
    result.overflow_update_count = counters[1];
    result.peak_pair_count = counters[2];
    const auto count_bytes = m_impl->program.download(m_impl->tile_counts);
    const auto* counts =
            reinterpret_cast<const uint32_t*>(count_bytes.data());
    for (uint32_t tile = 0; tile < m_impl->tile_count; tile++) {
        const uint32_t tile_total=counts[tile];
        result.max_tile_count = std::max(result.max_tile_count, tile_total);
        result.tile_reference_count += tile_total;
    }
    result.pair_capacity = m_impl->pair_capacity;
    if (m_impl->config.atomic_free_exact_backward) {
        constexpr size_t kPairPartialBytes = 3 * sizeof(glm::vec4);
        result.atomic_free_pair_scratch_bytes =
                result.pair_capacity * kPairPartialBytes;
        result.atomic_free_pair_scratch_used_bytes =
                static_cast<size_t>(result.tile_reference_count) *
                kPairPartialBytes;
    }
    result.timings = m_impl->program.lastTimings();
    result.fenced_host_timings =
            m_impl->program.lastFencedHostTimings();
    result.current_memory_bytes = m_impl->program.allocatedBytes();
    result.peak_memory_bytes = result.current_memory_bytes;
    return result;
}

GaussianScheduleIR GaussianForwardProgram::schedule() const {
    return m_impl->schedule_ir;
}

std::string GaussianForwardProgram::candidateLedgerJson() const {
    return m_impl->candidate_ledger_json;
}

gpu::DeviceCaps GaussianForwardProgram::deviceCaps() const {
    return m_impl->program.deviceCaps();
}

}  // namespace cross3dgs

#include "cross3dgs/training_preview.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

#include <glm/glm.hpp>

namespace cross3dgs {
namespace {

constexpr uint32_t kDivider = 4;

void SetPixel(FloatImage& image, uint32_t x, uint32_t y,
              const glm::vec3& color) {
    float* pixel = &image.pixels[
            (static_cast<size_t>(y) * image.width + x) * image.channels];
    pixel[0] = color.r;
    pixel[1] = color.g;
    pixel[2] = color.b;
    pixel[3] = 1.0f;
}

// Every source pixel the panel keeps is `stride` apart.
uint32_t PanelStride(uint32_t view_width, uint32_t view_height) {
    const uint32_t side = std::max(view_width, view_height);
    return std::max(1u, (side + kTrainingPreviewMaxPanelSide - 1) /
                                kTrainingPreviewMaxPanelSide);
}

void CopyPanel(const FloatImage& source, FloatImage& output,
               const TrainingPreviewRect& panel, uint32_t stride) {
    for (uint32_t y = 0; y < panel.height; y++) {
        for (uint32_t x = 0; x < panel.width; x++) {
            const float* pixel = &source.pixels[
                    (static_cast<size_t>(y) * stride * source.width +
                     static_cast<size_t>(x) * stride) *
                    source.channels];
            SetPixel(output, panel.x + x, panel.y + y,
                     {pixel[0], pixel[1], pixel[2]});
        }
    }
}

// The mean over each block, so thin high-loss edges survive the reduction.
float BlockMean(const std::vector<float>& values, uint32_t width,
                uint32_t height, uint32_t left, uint32_t top,
                uint32_t stride) {
    const uint32_t right = std::min(left + stride, width);
    const uint32_t bottom = std::min(top + stride, height);
    float sum = 0.0f;
    for (uint32_t y = top; y < bottom; y++) {
        for (uint32_t x = left; x < right; x++) {
            sum += values[static_cast<size_t>(y) * width + x];
        }
    }
    return sum / static_cast<float>((right - left) * (bottom - top));
}

// Dark to bright through purple and orange, so brightness alone also reads.
glm::vec3 HeatColor(float value) {
    static const std::array<glm::vec3, 5> kStops = {
            glm::vec3(0.00f, 0.00f, 0.02f), glm::vec3(0.34f, 0.06f, 0.43f),
            glm::vec3(0.74f, 0.22f, 0.33f), glm::vec3(0.98f, 0.56f, 0.04f),
            glm::vec3(0.99f, 1.00f, 0.64f)};
    const float position = std::clamp(value, 0.0f, 1.0f) *
                           static_cast<float>(kStops.size() - 1);
    const size_t low = std::min(static_cast<size_t>(position),
                                kStops.size() - 2);
    return glm::mix(kStops[low], kStops[low + 1],
                    position - static_cast<float>(low));
}

}  // namespace

TrainingPreviewLayout MakeTrainingPreviewLayout(uint32_t view_width,
                                                uint32_t view_height) {
    const uint32_t stride = PanelStride(view_width, view_height);
    const uint32_t panel_width = (view_width + stride - 1) / stride;
    const uint32_t panel_height = (view_height + stride - 1) / stride;
    const bool stacked = view_width >= view_height;
    const uint32_t step_x = stacked ? 0 : panel_width + kDivider;
    const uint32_t step_y = stacked ? panel_height + kDivider : 0;
    TrainingPreviewLayout layout;
    layout.width = panel_width + 2 * step_x;
    layout.height = panel_height + 2 * step_y;
    for (uint32_t panel = 0; panel < layout.panels.size(); panel++) {
        layout.panels[panel] = {panel * step_x, panel * step_y, panel_width,
                                panel_height};
    }
    return layout;
}

std::array<std::string, 3> TrainingPreviewLabels() {
    return {"Training photo", "Rendering", "Loss"};
}

FloatImage ComposeTrainingPreview(const TrainingPreview& preview) {
    const FloatImage& target = preview.target;
    const FloatImage& rendered = preview.rendered;
    const uint32_t width = target.width;
    const uint32_t height = target.height;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (width == 0 || height == 0 || target.channels < 3 ||
        rendered.channels < 3 || rendered.width != width ||
        rendered.height != height || preview.loss.size() != pixels) {
        throw std::invalid_argument("training preview images do not match");
    }
    const TrainingPreviewLayout layout =
            MakeTrainingPreviewLayout(width, height);

    FloatImage output;
    output.width = layout.width;
    output.height = layout.height;
    output.channels = 4;
    // White dividers show through between the panels.
    output.pixels.assign(
            static_cast<size_t>(output.width) * output.height * 4, 1.0f);

    const uint32_t stride = PanelStride(width, height);
    CopyPanel(target, output, layout.panels[0], stride);
    CopyPanel(rendered, output, layout.panels[1], stride);
    const TrainingPreviewRect& heat = layout.panels[2];
    for (uint32_t y = 0; y < heat.height; y++) {
        for (uint32_t x = 0; x < heat.width; x++) {
            const float loss = BlockMean(preview.loss, width, height,
                                         x * stride, y * stride, stride);
            SetPixel(output, heat.x + x, heat.y + y,
                     HeatColor(loss / kTrainingPreviewLossMax));
        }
    }
    return output;
}

}  // namespace cross3dgs

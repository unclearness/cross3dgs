#ifndef CROSS3DGS_TRAINING_PREVIEW_H
#define CROSS3DGS_TRAINING_PREVIEW_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cross3dgs/dataset.h"

namespace cross3dgs {

// What the optimizer compared at its last step, for live display. All three
// belong to the same training view.
struct TrainingPreview {
    uint32_t view = 0;
    FloatImage target;
    FloatImage rendered;
    // Per-pixel training loss (weighted L1 plus SSIM), width * height values.
    std::vector<float> loss;
};

// Loss shown at the top of the heat map. Fixed rather than scaled to each
// frame, so the map darkens as training converges.
inline constexpr float kTrainingPreviewLossMax = 0.25f;

struct TrainingPreviewRect {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Longest panel side in the composed image. A panel fills a fraction of a phone
// screen, and full-size panels cost training time and memory at every report.
inline constexpr uint32_t kTrainingPreviewMaxPanelSide = 800;

// Where the panels sit in the composed image: photo, render, loss. Landscape
// views are stacked and portrait views sit side by side, which fills a
// portrait phone screen either way.
struct TrainingPreviewLayout {
    uint32_t width = 0;
    uint32_t height = 0;
    std::array<TrainingPreviewRect, 3> panels;
};

TrainingPreviewLayout MakeTrainingPreviewLayout(uint32_t view_width,
                                                uint32_t view_height);

// Panel captions in layout order, as UTF-8. The apps draw them in their own
// UI font over the surface, so they stay sharp at any display scale.
std::array<std::string, 3> TrainingPreviewLabels();

// One image with the photo, the render and the loss heat map.
FloatImage ComposeTrainingPreview(const TrainingPreview& preview);

}  // namespace cross3dgs

#endif  // CROSS3DGS_TRAINING_PREVIEW_H

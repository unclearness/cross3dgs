#include <iostream>
#include <stdexcept>
#include <string>

#include "cross3dgs/training_preview.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        g_failures++;
    }
}

cross3dgs::FloatImage Solid(uint32_t width, uint32_t height, float red) {
    cross3dgs::FloatImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(static_cast<size_t>(width) * height * 4, 0.0f);
    for (size_t i = 0; i < image.pixels.size(); i += 4) {
        image.pixels[i] = red;
        image.pixels[i + 3] = 1.0f;
    }
    return image;
}

const float* Pixel(const cross3dgs::FloatImage& image, uint32_t x, uint32_t y) {
    return &image.pixels[(static_cast<size_t>(y) * image.width + x) * 4];
}

float Brightness(const float* pixel) {
    return pixel[0] + pixel[1] + pixel[2];
}

cross3dgs::TrainingPreview Preview(uint32_t width, uint32_t height) {
    cross3dgs::TrainingPreview preview;
    preview.target = Solid(width, height, 0.25f);
    preview.rendered = Solid(width, height, 0.75f);
    preview.loss.assign(static_cast<size_t>(width) * height, 0.0f);
    return preview;
}

void TestLandscapeStacks() {
    constexpr uint32_t kWidth = 240;
    constexpr uint32_t kHeight = 120;
    auto preview = Preview(kWidth, kHeight);
    // Loss rises along the row and overshoots the top of the scale at the end.
    for (uint32_t y = 0; y < kHeight; y++) {
        for (uint32_t x = 0; x < kWidth; x++) {
            preview.loss[static_cast<size_t>(y) * kWidth + x] =
                    2.0f * cross3dgs::kTrainingPreviewLossMax *
                    static_cast<float>(x) / static_cast<float>(kWidth - 1);
        }
    }
    const auto image = cross3dgs::ComposeTrainingPreview(preview);
    const uint32_t step = kHeight + 4;
    Check(image.width == kWidth && image.height == kHeight + 2 * step,
          "landscape views stack vertically");
    Check(Pixel(image, 200, 60)[0] == 0.25f, "photo is the first panel");
    Check(Pixel(image, 200, step + 60)[0] == 0.75f,
          "render is the second panel");
    Check(Pixel(image, 200, kHeight + 1)[0] == 1.0f, "divider is white");

    float previous = -1.0f;
    for (uint32_t x = 0; x < kWidth / 2; x += 10) {
        const float value = Brightness(Pixel(image, x, 2 * step + 60));
        Check(value > previous, "heat map brightens with loss");
        previous = value;
    }
    Check(Brightness(Pixel(image, kWidth - 1, 2 * step + 60)) ==
                  Brightness(Pixel(image, kWidth * 3 / 4, 2 * step + 60)),
          "loss above the scale saturates");
}

void TestPortraitSitsSideBySide() {
    constexpr uint32_t kWidth = 120;
    constexpr uint32_t kHeight = 240;
    const auto image =
            cross3dgs::ComposeTrainingPreview(Preview(kWidth, kHeight));
    const uint32_t step = kWidth + 4;
    Check(image.width == kWidth + 2 * step && image.height == kHeight,
          "portrait views sit side by side");
    Check(Pixel(image, 60, 200)[0] == 0.25f, "photo is on the left");
    Check(Pixel(image, step + 60, 200)[0] == 0.75f, "render is in the middle");
}

void TestLayoutMatchesComposition() {
    const auto layout = cross3dgs::MakeTrainingPreviewLayout(100, 60);
    const auto image = cross3dgs::ComposeTrainingPreview(Preview(100, 60));
    Check(layout.width == image.width && layout.height == image.height,
          "layout reports the composed size");
    Check(Pixel(image, layout.panels[1].x, layout.panels[1].y)[0] == 0.75f,
          "layout finds the render panel");
    Check(layout.panels[2].y == 128 && layout.panels[2].height == 60,
          "layout finds the loss panel");
}

void TestLargeViewsAreReduced() {
    constexpr uint32_t kWidth = 2001;
    constexpr uint32_t kHeight = 1000;
    auto preview = Preview(kWidth, kHeight);
    // A one-pixel line of high loss every third column, inside each 3x3 block.
    for (uint32_t y = 0; y < kHeight; y++) {
        for (uint32_t x = 1; x < kWidth; x += 3) {
            preview.loss[static_cast<size_t>(y) * kWidth + x] = 3.0f *
                    cross3dgs::kTrainingPreviewLossMax;
        }
    }
    const auto layout = cross3dgs::MakeTrainingPreviewLayout(kWidth, kHeight);
    Check(layout.panels[0].width == 667 && layout.panels[0].height == 334,
          "a panel shrinks to the size limit");
    const auto image = cross3dgs::ComposeTrainingPreview(preview);
    Check(image.width == layout.width && image.height == layout.height,
          "a reduced image matches its layout");
    const auto& heat = layout.panels[2];
    const float full = Brightness(Pixel(image, heat.x + 300, heat.y + 100));
    auto saturated = Preview(16, 16);
    saturated.loss.assign(saturated.loss.size(),
                          cross3dgs::kTrainingPreviewLossMax);
    const auto top = cross3dgs::ComposeTrainingPreview(saturated);
    const auto top_layout = cross3dgs::MakeTrainingPreviewLayout(16, 16);
    Check(full == Brightness(Pixel(top, top_layout.panels[2].x,
                                   top_layout.panels[2].y)),
          "the loss panel averages each block rather than sampling it");
}

void TestMismatchIsRejected() {
    auto preview = Preview(16, 16);
    preview.loss.pop_back();
    bool rejected = false;
    try {
        cross3dgs::ComposeTrainingPreview(preview);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    Check(rejected, "a loss map of the wrong size is rejected");
}

}  // namespace

int main() {
    TestLandscapeStacks();
    TestPortraitSitsSideBySide();
    TestLayoutMatchesComposition();
    TestLargeViewsAreReduced();
    TestMismatchIsRejected();
    if (g_failures != 0) return 1;
    std::cout << "training preview composition validated\n";
    return 0;
}

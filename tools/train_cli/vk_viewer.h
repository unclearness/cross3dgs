#ifndef CROSS3DGS_TRAIN_CLI_VK_VIEWER_H
#define CROSS3DGS_TRAIN_CLI_VK_VIEWER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cross3dgs_viewer {

// CPU-side float image shown by VkViewer (row-major, tightly packed
// channels).
struct ViewerImage {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    std::vector<float> data;

    ViewerImage() = default;
    ViewerImage(uint32_t w, uint32_t h, uint32_t c, float fill = 0.f)
        : width(w), height(h), channels(c), data(size_t(w) * h * c, fill) {}
    float& at(uint32_t x, uint32_t y, uint32_t c) {
        return data[(size_t(y) * width + x) * channels + c];
    }
    float at(uint32_t x, uint32_t y, uint32_t c) const {
        return data[(size_t(y) * width + x) * channels + c];
    }
};

// Minimal live-view window built on GLFW with its own Vulkan instance,
// device and function table, fully separate from the training Vulkan
// context: each update uploads a ViewerImage through a staging buffer and
// copies it to the swapchain (nearest-scaled on the CPU to the window size).
// Intended as a debug viewer; keep its cost out of optimization timing by
// measuring around update() separately.
class VkViewer {
public:
    VkViewer(uint32_t width, uint32_t height, const std::string& title);
    ~VkViewer();
    VkViewer(const VkViewer&) = delete;
    VkViewer& operator=(const VkViewer&) = delete;

    // False if the window could not be created (e.g. headless environment)
    bool valid() const;

    // Shows the image (1 or 3 channels, float [0,1]); returns false once
    // the window has been closed by the user.
    bool update(const ViewerImage& img);

    void setTitle(const std::string& title);

    // Moves the window (client area origin) to a fixed screen position --
    // deterministic layout for recording. No-op when headless.
    void setPosition(int x, int y);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace cross3dgs_viewer

#endif  // CROSS3DGS_TRAIN_CLI_VK_VIEWER_H

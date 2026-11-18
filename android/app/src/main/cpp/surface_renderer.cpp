#include "surface_renderer.h"

#include <algorithm>
#include <cstdint>

namespace cross3dgs_android {

SurfaceRenderer& SurfaceRenderer::instance() {
    static SurfaceRenderer renderer;
    return renderer;
}

void SurfaceRenderer::setSurface(ANativeWindow* window) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_window != nullptr) {
        ANativeWindow_release(m_window);
    }
    m_window = window;
    if (m_window == nullptr) {
        return;
    }
    ANativeWindow_setBuffersGeometry(m_window, 0, 0,
                                     WINDOW_FORMAT_RGBA_8888);
    if (m_latest.width != 0) {
        drawLocked(m_latest);
    } else {
        clearLocked();
    }
}

void SurfaceRenderer::show(const cross3dgs::FloatImage& image) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_latest = image;
    if (m_window != nullptr) {
        drawLocked(m_latest);
    }
}

void SurfaceRenderer::clearLocked() {
    if (m_window == nullptr) {
        return;
    }
    ANativeWindow_Buffer buffer;
    if (ANativeWindow_lock(m_window, &buffer, nullptr) != 0) {
        return;
    }
    auto* pixels = static_cast<uint32_t*>(buffer.bits);
    for (uint32_t y = 0; y < static_cast<uint32_t>(buffer.height); y++) {
        auto* row = pixels + static_cast<size_t>(y) *
                                     static_cast<uint32_t>(buffer.stride);
        std::fill(row, row + static_cast<uint32_t>(buffer.width),
                  0xff000000u);
    }
    ANativeWindow_unlockAndPost(m_window);
}

void SurfaceRenderer::drawLocked(
        const cross3dgs::FloatImage& image) {
    if (image.width == 0 || image.height == 0 || image.channels < 3) {
        return;
    }
    ANativeWindow_Buffer buffer;
    if (ANativeWindow_lock(m_window, &buffer, nullptr) != 0) {
        return;
    }
    auto* pixels = static_cast<uint32_t*>(buffer.bits);
    const uint32_t buffer_width = static_cast<uint32_t>(buffer.width);
    const uint32_t buffer_height = static_cast<uint32_t>(buffer.height);
    const float scale = std::min(
            static_cast<float>(buffer_width) / image.width,
            static_cast<float>(buffer_height) / image.height);
    const uint32_t draw_width =
            std::max(1u, static_cast<uint32_t>(image.width * scale));
    const uint32_t draw_height =
            std::max(1u, static_cast<uint32_t>(image.height * scale));
    const uint32_t offset_x = (buffer_width - draw_width) / 2;
    const uint32_t offset_y = (buffer_height - draw_height) / 2;
    const auto to_byte = [](float value) {
        return static_cast<uint32_t>(
                std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    for (uint32_t y = 0; y < buffer_height; y++) {
        auto* row = pixels + static_cast<size_t>(y) *
                                     static_cast<uint32_t>(buffer.stride);
        if (y < offset_y || y >= offset_y + draw_height) {
            std::fill(row, row + buffer_width, 0xff000000u);
            continue;
        }
        const uint32_t source_y = std::min(
                image.height - 1,
                static_cast<uint32_t>(
                        static_cast<float>(y - offset_y) / draw_height *
                        image.height));
        for (uint32_t x = 0; x < buffer_width; x++) {
            if (x < offset_x || x >= offset_x + draw_width) {
                row[x] = 0xff000000u;
                continue;
            }
            const uint32_t source_x = std::min(
                    image.width - 1,
                    static_cast<uint32_t>(
                            static_cast<float>(x - offset_x) / draw_width *
                            image.width));
            const size_t pixel =
                    (static_cast<size_t>(source_y) * image.width + source_x) *
                    image.channels;
            const uint32_t red = to_byte(image.pixels[pixel]);
            const uint32_t green = to_byte(image.pixels[pixel + 1]);
            const uint32_t blue = to_byte(image.pixels[pixel + 2]);
            row[x] = 0xff000000u | (blue << 16u) | (green << 8u) | red;
        }
    }
    ANativeWindow_unlockAndPost(m_window);
}

}  // namespace cross3dgs_android

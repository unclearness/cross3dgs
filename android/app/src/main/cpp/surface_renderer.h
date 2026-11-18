#ifndef CROSS3DGS_ANDROID_SURFACE_RENDERER_H
#define CROSS3DGS_ANDROID_SURFACE_RENDERER_H

#include <android/native_window.h>

#include <mutex>

#include "cross3dgs/dataset.h"

namespace cross3dgs_android {

class SurfaceRenderer {
public:
    static SurfaceRenderer& instance();

    // Takes ownership of an ANativeWindow acquired from a Java Surface.
    void setSurface(ANativeWindow* window);
    void show(const cross3dgs::FloatImage& image);

private:
    SurfaceRenderer() = default;
    void clearLocked();
    void drawLocked(const cross3dgs::FloatImage& image);

    std::mutex m_mutex;
    ANativeWindow* m_window = nullptr;
    cross3dgs::FloatImage m_latest;
};

}  // namespace cross3dgs_android

#endif

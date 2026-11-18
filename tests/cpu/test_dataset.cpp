#include <filesystem>
#include <iostream>

#include "cross3dgs/dataset.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "expected bicycle scene path\n";
        return 2;
    }
    const auto scene = cross3dgs::LoadColmapScene(
            std::filesystem::path(argv[1]), "images_4", "sparse/0");
    if (scene.cameras.size() != 194 || scene.trainCameraCount() != 169 ||
        scene.validationCameraCount() != 25 || scene.points.size() != 54275) {
        std::cerr << "unexpected bicycle dataset counts\n";
        return 1;
    }
    if (scene.cameras.front().width != 1237 ||
        scene.cameras.front().height != 822) {
        std::cerr << "unexpected images_4 dimensions\n";
        return 1;
    }
    const float scene_scale =
            cross3dgs::CameraExtentSceneScale(scene.cameras);
    if (std::abs(scene_scale - 1.304437f) > 1e-5f) {
        std::cerr << "scene scale mismatch: " << scene_scale << '\n';
        return 1;
    }
    const auto image = cross3dgs::LoadImageRgbaF32(
            scene.cameras.front().image_path);
    if (image.width != 1237 || image.height != 822 ||
        image.pixels.size() != static_cast<size_t>(1237 * 822 * 4)) {
        std::cerr << "image decode mismatch\n";
        return 1;
    }
    std::cout << "bicycle dataset validated\n";
    return 0;
}

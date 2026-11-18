#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cross3dgs/gaussian_state.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        g_failures++;
    }
}

float ShRest(const cross3dgs::GaussianStateCpu& state, int coefficient,
             int channel) {
    const int scalar = coefficient * 3 + channel;
    return state.sh_rest[scalar / 4][scalar % 4];
}

cross3dgs::GaussianStateCpu OneGaussian() {
    cross3dgs::GaussianStateCpu state;
    state.means = {glm::vec4(1.0f, 2.0f, 3.0f, 0.0f)};
    state.scales_opacity = {glm::vec4(0.1f, 0.2f, 0.3f, 0.5f)};
    state.rotations = {glm::normalize(glm::vec4(0.4f, 0.3f, -0.2f, 0.8f))};
    state.sh_dc = {glm::vec4(0.1f, 0.2f, 0.3f, 0.0f)};
    state.sh_rest.assign(12, glm::vec4(0.0f));
    for (int scalar = 0; scalar < 45; scalar++) {
        state.sh_rest[scalar / 4][scalar % 4] =
                0.01f * static_cast<float>(scalar + 1) *
                (scalar % 2 == 0 ? 1.0f : -1.0f);
    }
    return state;
}

// The renderer's basis (shaders_forward.cpp), written out independently of
// the library's copy so a sign or order slip there fails here.
std::array<float, 15> Basis(const glm::vec3& d) {
    const float x = d.x, y = d.y, z = d.z, xx = x * x, yy = y * y, zz = z * z;
    return {-0.48860251f * y,
            0.48860251f * z,
            -0.48860251f * x,
            1.09254843f * x * y,
            -1.09254843f * y * z,
            0.31539157f * (2.0f * zz - xx - yy),
            -1.09254843f * x * z,
            0.54627422f * (xx - yy),
            -0.59004359f * y * (3.0f * xx - yy),
            2.89061144f * x * y * z,
            -0.45704580f * y * (4.0f * zz - xx - yy),
            0.37317633f * z * (2.0f * zz - 3.0f * xx - 3.0f * yy),
            -0.45704580f * x * (4.0f * zz - xx - yy),
            1.44530572f * z * (xx - yy),
            -0.59004359f * x * (xx - 3.0f * yy)};
}

float Color(const cross3dgs::GaussianStateCpu& state, const glm::vec3& d,
             int channel) {
    const auto basis = Basis(d);
    float color = 0.0f;
    for (int coefficient = 0; coefficient < 15; coefficient++) {
        color += basis[coefficient] * ShRest(state, coefficient, channel);
    }
    return color;
}

glm::mat3 QuaternionMatrix(const glm::vec4& q) {
    const float r = q.x, x = q.y, y = q.z, z = q.w;
    return glm::mat3(
            glm::vec3(1 - 2 * (y * y + z * z), 2 * (x * y + r * z),
                      2 * (x * z - r * y)),
            glm::vec3(2 * (x * y - r * z), 1 - 2 * (x * x + z * z),
                      2 * (y * z + r * x)),
            glm::vec3(2 * (x * z + r * y), 2 * (y * z - r * x),
                      1 - 2 * (x * x + y * y)));
}

void TestStandardRestOrder() {
    const auto state = OneGaussian();
    const auto path =
            std::filesystem::temp_directory_path() / "cross3dgs_ply_export_test.ply";
    cross3dgs::SaveGaussianPly(path, state);
    std::ifstream input(path, std::ios::binary);
    std::string line;
    std::vector<std::string> properties;
    while (std::getline(input, line) && line != "end_header") {
        if (line.rfind("property float ", 0) == 0) {
            properties.push_back(line.substr(15));
        }
    }
    std::vector<float> row(properties.size());
    input.read(reinterpret_cast<char*>(row.data()),
               static_cast<std::streamsize>(row.size() * sizeof(float)));
    Check(input.good(), "PLY row is readable");
    input.close();
    std::filesystem::remove(path);
    Check(properties.size() == 59, "PLY has 59 float properties");
    const auto rest = static_cast<size_t>(
            std::find(properties.begin(), properties.end(), "f_rest_0") -
            properties.begin());
    for (int channel = 0; channel < 3; channel++) {
        for (int coefficient = 0; coefficient < 15; coefficient++) {
            Check(row[rest + channel * 15 + coefficient] ==
                          ShRest(state, coefficient, channel),
                  "f_rest is channel-major");
        }
    }
}

void TestDownRotation() {
    const glm::vec3 down = glm::normalize(glm::vec3(0.97f, 0.22f, -0.11f));
    const glm::mat3 rotation = cross3dgs::RotationToDownPositiveY(down);
    Check(glm::length(rotation * down - glm::vec3(0, 1, 0)) < 1e-5f,
          "world down maps to +Y");
    Check(std::abs(glm::determinant(rotation) - 1.0f) < 1e-5f,
          "down rotation is proper");
    const glm::mat3 flip =
            cross3dgs::RotationToDownPositiveY(glm::vec3(0, -1, 0));
    Check(glm::length(flip * glm::vec3(0, -1, 0) - glm::vec3(0, 1, 0)) < 1e-5f,
          "an upside-down world maps to +Y");
    Check(std::abs(glm::determinant(flip) - 1.0f) < 1e-5f,
          "upside-down rotation is proper");
}

void TestRotatedState() {
    const auto state = OneGaussian();
    const glm::mat3 rotation = cross3dgs::RotationToDownPositiveY(
            glm::normalize(glm::vec3(0.5f, -0.3f, 0.8f)));
    const auto rotated = cross3dgs::RotateGaussianState(state, rotation);

    Check(glm::length(glm::vec3(rotated.means[0]) -
                      rotation * glm::vec3(state.means[0])) < 1e-5f,
          "means rotate");
    Check(rotated.scales_opacity[0] == state.scales_opacity[0],
          "scales and opacity stay");
    Check(rotated.sh_dc[0] == state.sh_dc[0], "SH DC stays");

    const glm::mat3 expected = rotation * QuaternionMatrix(state.rotations[0]);
    const glm::mat3 actual = QuaternionMatrix(rotated.rotations[0]);
    for (int column = 0; column < 3; column++) {
        Check(glm::length(expected[column] - actual[column]) < 1e-5f,
              "orientations rotate");
    }

    const std::array<glm::vec3, 4> directions = {
            glm::normalize(glm::vec3(1, 2, 3)), glm::normalize(glm::vec3(-2, 1, 0.5f)),
            glm::normalize(glm::vec3(0.3f, -1, 2)), glm::vec3(0, 0, 1)};
    for (const auto& direction : directions) {
        for (int channel = 0; channel < 3; channel++) {
            Check(std::abs(Color(state, direction, channel) -
                           Color(rotated, rotation * direction, channel)) < 1e-4f,
                  "view-dependent color turns with the world");
        }
    }
}

}  // namespace

int main() {
    TestStandardRestOrder();
    TestDownRotation();
    TestRotatedState();
    if (g_failures != 0) {
        std::cerr << g_failures << " checks failed\n";
        return 1;
    }
    std::cout << "PLY export checks passed\n";
    return 0;
}

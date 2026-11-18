#ifndef CROSS3DGS_GPU_CHECK_H
#define CROSS3DGS_GPU_CHECK_H

#include <string>

namespace cross3dgs::gpu {

// Throws std::runtime_error("<msg> (<file>:<line>)").
[[noreturn]] void ThrowCheckFailure(const std::string& msg, const char* file,
                                    int line);

}  // namespace cross3dgs::gpu

#define CROSS3DGS_CHECK(cond, msg)                                           \
    do {                                                                     \
        if (!(cond)) {                                                       \
            ::cross3dgs::gpu::ThrowCheckFailure((msg), __FILE__, __LINE__); \
        }                                                                    \
    } while (false)

#endif  // CROSS3DGS_GPU_CHECK_H

#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string Replace(std::string source, const std::string& token,
                    const std::string& value) {
    size_t position = 0;
    while ((position = source.find(token, position)) != std::string::npos) {
        source.replace(position, token.size(), value);
        position += value.size();
    }
    return source;
}

// Resident targets are either RGBA32F vec4 or packed RGBA8 uints. The packed
// form is decoded with unpackUnorm4x8, whose c/255.0 conversion is identical
// to the CPU-side byte-to-float load, so 8-bit sources stay bitwise exact.
std::string ApplyTargetStorage(std::string source, bool packed,
                               bool streamed,
                               size_t target_pixels) {
    source = Replace(std::move(source), "TARGET_STORAGE_DECL",
                     packed ? "buffer Target { uint value[]; } target;"
                            : "buffer Target { vec4 value[]; } target;");
    const std::string index = streamed
            ? "((i)%" + std::to_string(target_pixels) + "u)"
            : "i";
    return Replace(std::move(source), "TARGET_FETCH_DEFINE",
                   packed ? "#define TARGET_FETCH(i) "
                            "unpackUnorm4x8(target.value[" + index + "])"
                          : "#define TARGET_FETCH(i) target.value[" + index + "]");
}


}  // namespace cross3dgs::detail

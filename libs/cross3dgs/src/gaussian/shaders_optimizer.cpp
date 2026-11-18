#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string AdamShader(uint32_t group, float learning_rate_xyz,
                       float learning_rate_w, float regularizer_xyz,
                       float regularizer_w, float learning_rate_final,
                       uint32_t max_steps, bool multistep,
                       uint32_t mu_start, uint32_t mu_stage2,
                       uint32_t mu_n1, uint32_t mu_n2,
                       bool fp32_moments) {
    std::string source = R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) buffer Parameter { vec4 value[]; } parameter;
layout(std430,binding=1) GRAD_QUAL buffer Gradient { vec4 value[]; } gradient;
layout(std430,binding=2) buffer FirstMoment { MOMENT_TYPE value[]; } first_moment;
layout(std430,binding=3) buffer SecondMoment { MOMENT_TYPE value[]; } second_moment;
layout(std430,binding=4) readonly buffer OptimizerMeta { uvec4 value; } meta;
vec4 load_bf16(uvec2 packed) {
    uvec4 half_bits=uvec4(packed.x&0xffffu,packed.x>>16u,
                          packed.y&0xffffu,packed.y>>16u);
    return uintBitsToFloat(half_bits<<16u);
}
uvec2 store_bf16(vec4 value) {
    uvec4 bits=floatBitsToUint(value);
    bits+=uvec4(0x7fffu)+((bits>>16u)&uvec4(1u));
    uvec4 half_bits=bits>>16u;
    return uvec2(half_bits.x|(half_bits.y<<16u),
                 half_bits.z|(half_bits.w<<16u));
}
MOMENT_HELPERS
void main() {
    uint i=gl_GlobalInvocationID.x;
    if(i>=(GROUP==4 ? meta.value.y*12u : meta.value.y)) return;
    MU_GATE
    vec4 physical=parameter.value[i];
    vec4 unconstrained=physical;
    vec4 grad=gradient.value[i];
    vec4 learning_rate=vec4(LR_XYZ,LR_XYZ,LR_XYZ,LR_W);
    if(GROUP==0) {
        grad.w=0.0; learning_rate.w=0.0;
        learning_rate.xyz*=pow(LR_RATIO,
            min(float(meta.value.x)/float(MAX_STEPS),1.0));
    } else if(GROUP==1) {
        learning_rate.xyz*=pow(LR_RATIO,
            min(float(meta.value.x)/float(MAX_STEPS),1.0));
        grad+=vec4(vec3(REG_XYZ/(3.0*float(meta.value.y))),
                   REG_W/float(meta.value.y));
        unconstrained=vec4(log(max(physical.xyz,vec3(1e-8))),
            log(max(physical.w,1e-8)/max(1.0-physical.w,1e-8)));
        grad=vec4(grad.xyz*physical.xyz,
                  grad.w*physical.w*(1.0-physical.w));
    }
    vec4 m=mix(load_moment(first_moment.value[i]),grad,vec4(0.1));
    vec4 v=mix(load_moment(second_moment.value[i]),grad*grad,vec4(0.001));
    float step=float(meta.value.x+1u);
    vec4 corrected_m=m/(1.0-pow(0.9,step));
    vec4 corrected_v=v/(1.0-pow(0.999,step));
    vec4 update=learning_rate*corrected_m/(sqrt(corrected_v)+vec4(1e-15));
    unconstrained-=update;
    if(GROUP==1) {
        physical=vec4(exp(unconstrained.xyz),
                      1.0/(1.0+exp(-unconstrained.w)));
    } else if(GROUP==2) {
        float length_squared=dot(unconstrained,unconstrained);
        physical=length_squared>1e-12 ? unconstrained/sqrt(length_squared)
                                     : vec4(1.0,0.0,0.0,0.0);
    } else {
        physical=unconstrained;
    }
    parameter.value[i]=physical;
    first_moment.value[i]=store_moment(m);
    second_moment.value[i]=store_moment(v);
    MU_ZERO
})";
    if (multistep) {
        source = Replace(std::move(source), "GRAD_QUAL", "");
        source = Replace(std::move(source), "MU_GATE",
            "uint mu_n=meta.value.x<" + std::to_string(mu_start) + "u ? 1u : "
            "(meta.value.x<" + std::to_string(mu_stage2) + "u ? " +
            std::to_string(mu_n1) + "u : " + std::to_string(mu_n2) + "u);"
            "if(((meta.value.x+1u)%mu_n)!=0u) return;");
        source = Replace(std::move(source), "MU_ZERO",
            "gradient.value[i]=vec4(0.0);");
    } else {
        source = Replace(std::move(source), "GRAD_QUAL", "readonly");
        source = Replace(std::move(source), "MU_GATE", "");
        source = Replace(std::move(source), "MU_ZERO", "");
    }
    source = Replace(std::move(source), "GROUP", std::to_string(group));
    source = Replace(std::move(source), "LR_XYZ",
                     std::to_string(learning_rate_xyz));
    source = Replace(std::move(source), "LR_W",
                     std::to_string(learning_rate_w));
    const float final_rate=learning_rate_final>0.0f ? learning_rate_final
                                                    : learning_rate_xyz;
    source = Replace(std::move(source), "LR_RATIO",
                     std::to_string(final_rate/learning_rate_xyz));
    source = Replace(std::move(source), "MAX_STEPS",
                     std::to_string(max_steps)+"u");
    source = Replace(std::move(source), "REG_XYZ",
                     std::to_string(regularizer_xyz));
    source = Replace(std::move(source), "MOMENT_TYPE",
                     fp32_moments ? "vec4" : "uvec2");
    source = Replace(std::move(source), "MOMENT_HELPERS",
                     fp32_moments
                         ? "vec4 load_moment(vec4 value){return value;}\n"
                           "vec4 store_moment(vec4 value){return value;}"
                         : "vec4 load_moment(uvec2 value){return load_bf16(value);}\n"
                           "uvec2 store_moment(vec4 value){return store_bf16(value);}");
    return Replace(std::move(source), "REG_W",
                   std::to_string(regularizer_w));
}

std::string IncrementStepShader(bool scalar_meta_store) {
    // Read and write all four components explicitly. A bare `meta.value.x+=1u`
    // is miscompiled on some Adreno drivers into a full uvec4 store that zeroes
    // y/z/w (the live Gaussian count lives in y), which then blanks training.
    if (scalar_meta_store) {
        return R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) buffer OptimizerMeta { uint value[]; } meta;
void main() {
    atomicAdd(meta.value[0],1u);
}
)";
    }
    return R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) buffer OptimizerMeta { uvec4 value; } meta;
void main() {
    uvec4 v=meta.value;
    meta.value=uvec4(v.x+1u,v.y,v.z,v.w);
}
)";
}


}  // namespace cross3dgs::detail

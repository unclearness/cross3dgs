#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string StochasticLossShader(bool byte_targets,
                                 bool stream_target,
                                 size_t target_pixels) {
    return ApplyTargetStorage(R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer Color { vec4 value[]; } color;
layout(std430,binding=2) readonly TARGET_STORAGE_DECL
layout(std430,binding=3) writeonly buffer Loss { float value[]; } loss_value;
layout(std430,binding=4) readonly buffer OptimizerMeta { uvec4 value; } meta;
TARGET_FETCH_DEFINE
void main() {
    uint i=gl_GlobalInvocationID.x;
    uint width=camera.dims.z, height=camera.dims.w;
    uint pixel_count=width*height;
    if(i>=pixel_count) return;
    uint view_index=camera.state.y;
    vec3 rendered=color.value[i].xyz;
    vec3 wanted=TARGET_FETCH(view_index*pixel_count+i).xyz;
    loss_value.value[i]=
        (abs(rendered.x-wanted.x)+abs(rendered.y-wanted.y)+
         abs(rendered.z-wanted.z))/3.0;
})", byte_targets, stream_target, target_pixels);
}

std::string StochasticRasterBackwardShader(uint32_t tile_capacity,
                                           uint32_t samples,
                                           bool byte_targets,
                                           bool stream_target,
                                           size_t target_pixels) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer Projected { vec4 value[]; } projected;
layout(std430,binding=2) readonly buffer ConicOpacity { vec4 value[]; } conic_opacity;
layout(std430,binding=3) readonly buffer Rgb { vec4 value[]; } rgb;
layout(std430,binding=4) readonly buffer TileCounts { uint value[]; } tile_counts;
layout(std430,binding=5) readonly buffer TileIds { uint value[]; } tile_ids;
layout(std430,binding=6) readonly buffer Color { vec4 value[]; } color;
layout(std430,binding=7) readonly TARGET_STORAGE_DECL
layout(std430,binding=8) writeonly buffer ProjectedGrad { vec4 value[]; } projected_grad;
layout(std430,binding=9) writeonly buffer ConicGrad { vec4 value[]; } conic_grad;
layout(std430,binding=10) writeonly buffer RgbGrad { vec4 value[]; } rgb_grad;
layout(std430,binding=11) readonly buffer OptimizerMeta { uvec4 value; } meta;
layout(std430,binding=12) readonly buffer TileMasks { uint value[]; } tile_masks;
layout(std430,binding=13) readonly buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430,binding=14) readonly buffer Contributors { uint value[]; } contributors;
TARGET_FETCH_DEFINE

uint sobol_hash(uint x) {
    x^=x>>16u; x*=0x7feb352du; x^=x>>15u;
    x*=0x846ca68bu; x^=x>>16u; return x;
}
float sobol_u01(uint x) {
    return float(x>>8u)*(1.0/16777216.0);
}
vec2 sobol_point(uint index) {
    uint dim0=bitfieldReverse(index);
    uint dim1=0u, direction=0x80000000u;
    for(uint bits=index;bits!=0u;bits>>=1u) {
        if((bits&1u)!=0u) dim1^=direction;
        direction^=direction>>1u;
    }
    return vec2(dim0,dim1)*(1.0/4294967296.0);
}
bool finite_vec4(vec4 value) {
    return !any(isnan(value)) && !any(isinf(value));
}
void main() {
    uint id=gl_GlobalInvocationID.x;
    uint gaussian_count=camera.state.x;
    if(id>=gaussian_count) return;
    uint width=camera.dims.z, height=camera.dims.w;
    vec4 projected_value=projected.value[id];
    vec2 center=projected_value.xy;
    float radius=projected_value.w;
    ivec2 minimum=clamp(ivec2(floor(center-vec2(radius))),
                        ivec2(0),ivec2(width,height));
    ivec2 maximum=clamp(ivec2(ceil(center+vec2(radius)))+ivec2(1),
                        ivec2(0),ivec2(width,height));
    ivec2 extent=max(maximum-minimum,ivec2(0));
    vec4 local_projected=vec4(0.0);
    vec4 local_conic=vec4(0.0);
    vec4 local_rgb=vec4(0.0);
    if(extent.x>0 && extent.y>0) {
        float sample_weight=float(extent.x*extent.y)/float(SAMPLE_COUNT);
        vec2 scramble=vec2(
            sobol_u01(sobol_hash(id*2u+0x9e3779b9u)),
            sobol_u01(sobol_hash(id*2u+0x243f6a88u)));
        uint view_index=camera.state.y;
        uint pixel_count=width*height;
        for(uint sample_index=0u;sample_index<SAMPLE_COUNT;sample_index++) {
            uint sequence_index=meta.value.x*SAMPLE_COUNT+sample_index+1u;
            vec2 q=fract(sobol_point(sequence_index)+scramble);
            ivec2 sampled=minimum+min(extent-ivec2(1),ivec2(q*vec2(extent)));
            uint pixel_x=uint(sampled.x), pixel_y=uint(sampled.y);
            uint pixel_index=pixel_y*width+pixel_x;
            uint tile=(pixel_y/16u)*camera.dims.x+(pixel_x/16u);
    uint count=tile_counts.value[tile];
            uint terminal=min(contributors.value[pixel_index],count);
            uint tile_offset=tile_offsets.value[tile];
            vec3 rendered=color.value[pixel_index].xyz;
            vec3 wanted=TARGET_FETCH(view_index*pixel_count+pixel_index).xyz;
            vec3 d_color=sign(rendered-wanted)/
                (float(width)*float(height)*3.0);
            vec3 accumulated_after=vec3(0.0);
            float transmittance_after=1.0-color.value[pixel_index].w;
            uint subtile=(pixel_y&15u)/4u*4u+(pixel_x&15u)/4u;
            for(uint reverse_index=terminal;reverse_index>0u;reverse_index--) {
                uint i=reverse_index-1u;
                uint other_id=tile_ids.value[tile_offset+i];
                if((tile_masks.value[tile_offset+i]&(1u<<subtile))==0u)
                    continue;
                vec2 d=vec2(pixel_x,pixel_y)-projected.value[other_id].xy;
                vec4 other_conic=conic_opacity.value[other_id];
                float power=-0.5*(other_conic.x*d.x*d.x+
                    2.0*other_conic.y*d.x*d.y+other_conic.z*d.y*d.y);
                if(power>0.0) continue;
                float exponential=exp(power);
                float raw_alpha=other_conic.w*exponential;
                float alpha=min(0.999,raw_alpha);
                if(alpha<0.00392156862745) continue;
                float one_minus=1.0-alpha;
                float transmittance_before=transmittance_after/one_minus;
                vec3 other_rgb=rgb.value[other_id].xyz;
                if(other_id==id) {
                    local_rgb.xyz+=sample_weight*d_color*
                        (transmittance_before*alpha);
                    float d_alpha=sample_weight*dot(d_color,
                        transmittance_before*other_rgb-
                        accumulated_after/one_minus);
                    if(raw_alpha<0.999) {
                        float d_power=d_alpha*alpha;
                        local_projected.xy+=d_power*vec2(
                            other_conic.x*d.x+other_conic.y*d.y,
                            other_conic.y*d.x+other_conic.z*d.y);
                        local_conic+=vec4(
                            d_power*(-0.5*d.x*d.x),
                            d_power*(-d.x*d.y),
                            d_power*(-0.5*d.y*d.y),
                            d_alpha*exponential);
                    }
                    break;
                }
                accumulated_after+=transmittance_before*alpha*other_rgb;
                transmittance_after=transmittance_before;
            }
        }
    }
    projected_grad.value[id]=finite_vec4(local_projected) ?
        local_projected : vec4(0.0);
    conic_grad.value[id]=finite_vec4(local_conic) ? local_conic : vec4(0.0);
    rgb_grad.value[id]=finite_vec4(local_rgb) ? local_rgb : vec4(0.0);
})";
    source=ApplyTargetStorage(std::move(source),byte_targets,stream_target,
                              target_pixels);
    source=Replace(std::move(source),"TILE_CAPACITY",
                   std::to_string(tile_capacity)+"u");
    return Replace(std::move(source),"SAMPLE_COUNT",
                   std::to_string(samples)+"u");
}

std::string ExactTileRasterBackwardShader(
        uint32_t tile_count, uint32_t tile_workgroups, uint32_t subgroup_size,
        bool byte_targets, bool stream_target, size_t target_pixels,
        bool external_pixel_gradient) {
    std::string source=R"(#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
layout(local_size_x=SUBGROUP_SIZE) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer Projected { vec4 value[]; } projected;
layout(std430,binding=2) readonly buffer ConicOpacity { vec4 value[]; } conic_opacity;
layout(std430,binding=3) readonly buffer Rgb { vec4 value[]; } rgb;
layout(std430,binding=4) readonly buffer TileCounts { uint value[]; } tile_counts;
layout(std430,binding=5) readonly buffer TileIds { uint value[]; } tile_ids;
layout(std430,binding=6) readonly buffer Color { vec4 value[]; } color;
layout(std430,binding=7) readonly TARGET_STORAGE_DECL
layout(std430,binding=8) writeonly buffer PairGrad { vec4 value[]; } pair_grad;
layout(std430,binding=9) readonly buffer TileMasks { uint value[]; } tile_masks;
layout(std430,binding=10) readonly buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430,binding=11) readonly buffer Contributors { uint value[]; } contributors;
layout(std430,binding=12) readonly buffer GaussianOffsets { uint value[]; } gaussian_offsets;
layout(std430,binding=13) readonly buffer TileRect { uvec4 value[]; } tile_rect;
TARGET_FETCH_DEFINE
bool finite_vec4(vec4 value) {
    return !any(isnan(value)) && !any(isinf(value));
}
void main() {
    uint lane=gl_LocalInvocationID.x;
    uint width=camera.dims.z, height=camera.dims.w;
    uint grid_width=camera.dims.x;
    uint view_index=camera.state.y;
    uint pixel_count=width*height;
    for(uint tile=gl_WorkGroupID.x;tile<TILE_COUNT;
        tile+=TILE_WORKGROUPS) {
        uint count=tile_counts.value[tile];
        uint tile_offset=tile_offsets.value[tile];
        uvec2 pixels[PIXELS_PER_LANE];
        uint terminals[PIXELS_PER_LANE];
        vec3 d_colors[PIXELS_PER_LANE];
        vec3 accumulated_after[PIXELS_PER_LANE];
        float transmittance_after[PIXELS_PER_LANE];
        for(uint p=0u;p<PIXELS_PER_LANE;p++) {
            uint local_pixel=lane+p*SUBGROUP_SIZE;
            uint pixel_x=(tile%grid_width)*16u+(local_pixel&15u);
            uint pixel_y=(tile/grid_width)*16u+(local_pixel>>4u);
            bool valid=pixel_x<width && pixel_y<height;
            uint pixel_index=valid ? pixel_y*width+pixel_x : 0u;
            pixels[p]=uvec2(pixel_x,pixel_y);
            terminals[p]=valid ?
                min(contributors.value[pixel_index],count) : 0u;
            vec3 rendered=valid ?
                color.value[pixel_index].xyz : vec3(0.0);
            vec3 wanted=valid ? TARGET_FETCH(EXTERNAL_GRADIENT ? pixel_index :
                view_index*pixel_count+pixel_index).xyz : vec3(0.0);
            d_colors[p]=valid ? (EXTERNAL_GRADIENT ? wanted :
                sign(rendered-wanted)/(float(width)*float(height)*3.0)) :
                vec3(0.0);
            accumulated_after[p]=vec3(0.0);
            transmittance_after[p]=valid ?
                1.0-color.value[pixel_index].w : 1.0;
        }
        for(uint reverse_index=count;reverse_index>0u;reverse_index--) {
            uint i=reverse_index-1u;
            uint pair_index=tile_offset+i;
            uint mask=tile_masks.value[pair_index];
            bool lane_active=false;
            for(uint p=0u;p<PIXELS_PER_LANE;p++) {
                uint subtile=(pixels[p].y&15u)/4u*4u+
                    (pixels[p].x&15u)/4u;
                lane_active=lane_active ||
                    (i<terminals[p] && (mask&(1u<<subtile))!=0u);
            }
            uint id=tile_ids.value[pair_index];
            vec4 local_projected=vec4(0.0);
            vec4 local_conic=vec4(0.0);
            vec4 local_rgb=vec4(0.0);
            if(subgroupAny(lane_active)) {
                vec3 splat_rgb=rgb.value[id].xyz;
                vec4 conic=conic_opacity.value[id];
                vec2 center=projected.value[id].xy;
                for(uint p=0u;p<PIXELS_PER_LANE;p++) {
                    uint pixel_x=pixels[p].x, pixel_y=pixels[p].y;
                    uint subtile=(pixel_y&15u)/4u*4u+
                        (pixel_x&15u)/4u;
                    if(i<terminals[p] && (mask&(1u<<subtile))!=0u) {
                        vec2 d=vec2(pixel_x,pixel_y)-center;
                        float power=-0.5*(conic.x*d.x*d.x+
                            2.0*conic.y*d.x*d.y+conic.z*d.y*d.y);
                        if(power<=0.0) {
                            float exponential=exp(power);
                            float raw_alpha=conic.w*exponential;
                            float alpha=min(0.999,raw_alpha);
                            if(alpha>=0.00392156862745) {
                                float one_minus=1.0-alpha;
                                float transmittance_before=
                                    transmittance_after[p]/one_minus;
                                local_rgb.xyz+=d_colors[p]*
                                    (transmittance_before*alpha);
                                float d_alpha=dot(d_colors[p],
                                    transmittance_before*splat_rgb-
                                    accumulated_after[p]/one_minus);
                                if(raw_alpha<0.999) {
                                    float d_power=d_alpha*alpha;
                                    local_projected.xy+=d_power*vec2(
                                        conic.x*d.x+conic.y*d.y,
                                        conic.y*d.x+conic.z*d.y);
                                    local_conic+=vec4(
                                        d_power*(-0.5*d.x*d.x),
                                        d_power*(-d.x*d.y),
                                        d_power*(-0.5*d.y*d.y),
                                        d_alpha*exponential);
                                }
                                accumulated_after[p]+=
                                    transmittance_before*alpha*splat_rgb;
                                transmittance_after[p]=transmittance_before;
                            }
                        }
                    }
                }
            }
            vec4 sum_projected=subgroupAdd(local_projected);
            vec4 sum_conic=subgroupAdd(local_conic);
            vec4 sum_rgb=subgroupAdd(local_rgb);
            if(subgroupElect()) {
                uint tile_x=tile%grid_width, tile_y=tile/grid_width;
                uvec4 rectangle=tile_rect.value[id];
                uint original=gaussian_offsets.value[id]+
                    (tile_y-rectangle.y)*(rectangle.z-rectangle.x)+
                    tile_x-rectangle.x;
                pair_grad.value[original*3u]=finite_vec4(sum_projected) ?
                    sum_projected : vec4(0.0);
                pair_grad.value[original*3u+1u]=finite_vec4(sum_conic) ?
                    sum_conic : vec4(0.0);
                pair_grad.value[original*3u+2u]=finite_vec4(sum_rgb) ?
                    sum_rgb : vec4(0.0);
            }
        }
    }
}
)";
    source=ApplyTargetStorage(std::move(source),
                              byte_targets && !external_pixel_gradient,
                              stream_target && !external_pixel_gradient,
                              target_pixels);
    source=Replace(std::move(source),"TILE_COUNT",
                   std::to_string(tile_count)+"u");
    source=Replace(std::move(source),"TILE_WORKGROUPS",
                   std::to_string(tile_workgroups)+"u");
    source=Replace(std::move(source),"PIXELS_PER_LANE",
                   std::to_string(256u/subgroup_size)+"u");
    source=Replace(std::move(source),"SUBGROUP_SIZE",
                   std::to_string(subgroup_size));
    return Replace(std::move(source),"EXTERNAL_GRADIENT",
                   external_pixel_gradient ? "true" : "false");
}

std::string ExactPairReduceShader(uint32_t gaussian_workgroups,
                                  bool drop_first_partial) {
    std::string source=R"(#version 450
layout(local_size_x=64) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer GaussianCounts { uint value[]; } gaussian_counts;
layout(std430,binding=2) readonly buffer GaussianOffsets { uint value[]; } gaussian_offsets;
layout(std430,binding=3) readonly buffer RadixMeta { uvec4 value[]; } radix_meta;
layout(std430,binding=4) readonly buffer PairGrad { vec4 value[]; } pair_grad;
layout(std430,binding=5) writeonly buffer ProjectedGrad { vec4 value[]; } projected_grad;
layout(std430,binding=6) writeonly buffer ConicGrad { vec4 value[]; } conic_grad;
layout(std430,binding=7) writeonly buffer RgbGrad { vec4 value[]; } rgb_grad;
bool finite_vec4(vec4 value) {
    return !any(isnan(value)) && !any(isinf(value));
}
void main() {
    uint gaussian_count=camera.state.x;
    uint total=radix_meta.value[0].x;
    uint stride=GAUSSIAN_WORKGROUPS*64u;
    for(uint id=gl_GlobalInvocationID.x;id<gaussian_count;id+=stride) {
        uint begin=min(gaussian_offsets.value[id],total);
        uint end=min(begin+gaussian_counts.value[id],total);
        vec4 sum_projected=vec4(0.0);
        vec4 sum_conic=vec4(0.0);
        vec4 sum_rgb=vec4(0.0);
        for(uint pair=begin;pair<end;pair++) {
            NEGATIVE_CONTROL_SKIP
            sum_projected+=pair_grad.value[pair*3u];
            sum_conic+=pair_grad.value[pair*3u+1u];
            sum_rgb+=pair_grad.value[pair*3u+2u];
        }
        projected_grad.value[id]=finite_vec4(sum_projected) ?
            sum_projected : vec4(0.0);
        conic_grad.value[id]=finite_vec4(sum_conic) ?
            sum_conic : vec4(0.0);
        rgb_grad.value[id]=finite_vec4(sum_rgb) ?
            sum_rgb : vec4(0.0);
    }
})";
    source=Replace(std::move(source),"GAUSSIAN_WORKGROUPS",
                   std::to_string(gaussian_workgroups)+"u");
    // Negative control: skip exactly one tile-Gaussian partial per
    // Gaussian so the conformance gate must report violations.
    source=Replace(std::move(source),"NEGATIVE_CONTROL_SKIP",
                   drop_first_partial
                           ? "if(pair==begin&&end>begin) { continue; }"
                           : "");
    return source;
}


}  // namespace cross3dgs::detail

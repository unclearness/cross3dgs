#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string RasterBackwardShader(uint32_t tile_capacity,
                                 bool native_float_atomic_add,
                                 bool external_pixel_gradient,
                                 bool byte_targets,
                                 bool stream_target,
                                 size_t target_pixels) {
    std::string source = R"(#version 450
ATOMIC_SUPPORT
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
layout(local_size_x=128) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer Projected { vec4 value[]; } projected;
layout(std430,binding=2) readonly buffer ConicOpacity { vec4 value[]; } conic_opacity;
layout(std430,binding=3) readonly buffer Rgb { vec4 value[]; } rgb;
layout(std430,binding=4) readonly buffer TileCounts { uint value[]; } tile_counts;
layout(std430,binding=5) readonly buffer TileIds { uint value[]; } tile_ids;
layout(std430,binding=6) readonly buffer Color { vec4 value[]; } color;
layout(std430,binding=7) readonly TARGET_STORAGE_DECL
layout(std430,binding=8) buffer ProjectedGrad { GRAD_VEC4 value[]; } projected_grad;
layout(std430,binding=9) buffer ConicGrad { GRAD_VEC4 value[]; } conic_grad;
layout(std430,binding=10) buffer RgbGrad { GRAD_VEC4 value[]; } rgb_grad;
layout(std430,binding=11) writeonly buffer Loss { float value[]; } loss_value;
layout(std430,binding=12) readonly buffer OptimizerMeta { uvec4 value; } meta;
layout(std430,binding=13) readonly buffer TileMasks { uint value[]; } tile_masks;
layout(std430,binding=14) readonly buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430,binding=15) readonly buffer Contributors { uint value[]; } contributors;
TARGET_FETCH_DEFINE

ATOMIC_ADD_IMPLEMENTATION

void main() {
    uint tile=gl_WorkGroupID.x, lane=gl_LocalInvocationID.x;
    uint grid_width=camera.dims.x;
    uint count=tile_counts.value[tile];
    uint tile_offset=tile_offsets.value[tile];
    uint width=camera.dims.z, height=camera.dims.w;
    uint view_index=camera.state.y;
    uvec2 pixels[2];
    uint terminals[2];
    vec3 d_colors[2];
    vec3 accumulated_after[2];
    float transmittance_after[2];
    uint thread_terminal=0u;
    for(uint p=0u;p<2u;p++) {
        uint local_pixel=lane+p*128u;
        uint pixel_x=(tile%grid_width)*16u+(local_pixel&15u);
        uint pixel_y=(tile/grid_width)*16u+(local_pixel>>4u);
        bool valid=pixel_x<width && pixel_y<height;
        uint pixel_index=valid ? pixel_y*width+pixel_x : 0u;
        pixels[p]=uvec2(pixel_x,pixel_y);
        terminals[p]=valid ? contributors.value[pixel_index] : 0u;
        thread_terminal=max(thread_terminal,terminals[p]);
        vec3 rendered=valid ? color.value[pixel_index].xyz : vec3(0.0);
        uint target_index=view_index*(width*height)+pixel_index;
        vec3 wanted=valid ? TARGET_FETCH(EXTERNAL_GRADIENT ? pixel_index : target_index).xyz : vec3(0.0);
        d_colors[p]=valid ? (EXTERNAL_GRADIENT ? wanted :
            sign(rendered-wanted)/(float(width)*float(height)*3.0)) :
            vec3(0.0);
        accumulated_after[p]=vec3(0.0);
        transmittance_after[p]=valid ?
            1.0-color.value[pixel_index].w : 1.0;
        if(valid && !EXTERNAL_GRADIENT) loss_value.value[pixel_index]=
            (abs(rendered.x-wanted.x)+abs(rendered.y-wanted.y)+
             abs(rendered.z-wanted.z))/3.0;
    }
    uint subgroup_terminal=subgroupMax(thread_terminal);
    for(uint reverse_index=subgroup_terminal;reverse_index>0u;reverse_index--) {
        uint i=reverse_index-1u;
        uint mask=tile_masks.value[tile_offset+i];
        bool lane_active=false;
        for(uint p=0u;p<2u;p++) {
            uint subtile=(pixels[p].y&15u)/4u*4u+(pixels[p].x&15u)/4u;
            lane_active=lane_active ||
                (i<terminals[p] && (mask&(1u<<subtile))!=0u);
        }
        if(!subgroupAny(lane_active)) continue;
        uint id=tile_ids.value[tile_offset+i];
        vec4 local_projected=vec4(0.0);
        vec4 local_conic=vec4(0.0);
        vec4 local_rgb=vec4(0.0);
        vec3 splat_rgb=rgb.value[id].xyz;
        vec3 conic=conic_opacity.value[id].xyz;
        vec2 center=projected.value[id].xy;
        for(uint p=0u;p<2u;p++) {
            uint pixel_x=pixels[p].x;
            uint pixel_y=pixels[p].y;
            uint subtile=(pixel_y&15u)/4u*4u+(pixel_x&15u)/4u;
            if(i<terminals[p] && (mask&(1u<<subtile))!=0u) {
                vec2 d=vec2(pixel_x,pixel_y)-center;
                float power=-0.5*(conic.x*d.x*d.x+
                    2.0*conic.y*d.x*d.y+conic.z*d.y*d.y);
                if(power<=0.0) {
                    float exponential=exp(power);
                    float raw_alpha=conic_opacity.value[id].w*exponential;
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
        vec4 sum_projected=subgroupAdd(local_projected);
        vec4 sum_conic=subgroupAdd(local_conic);
        vec4 sum_rgb=subgroupAdd(local_rgb);
        if(subgroupElect()) {
            ATOMIC_ADD(projected_grad.value[id].x,sum_projected.x);
            ATOMIC_ADD(projected_grad.value[id].y,sum_projected.y);
            ATOMIC_ADD(conic_grad.value[id].x,sum_conic.x);
            ATOMIC_ADD(conic_grad.value[id].y,sum_conic.y);
            ATOMIC_ADD(conic_grad.value[id].z,sum_conic.z);
            ATOMIC_ADD(conic_grad.value[id].w,sum_conic.w);
            ATOMIC_ADD(rgb_grad.value[id].x,sum_rgb.x);
            ATOMIC_ADD(rgb_grad.value[id].y,sum_rgb.y);
            ATOMIC_ADD(rgb_grad.value[id].z,sum_rgb.z);
        }
    }
})";
    source=ApplyTargetStorage(std::move(source),
                              byte_targets && !external_pixel_gradient,
                              stream_target && !external_pixel_gradient,
                              target_pixels);
    source=Replace(std::move(source),"TILE_CAPACITY",
                   std::to_string(tile_capacity));
    source=Replace(std::move(source),"EXTERNAL_GRADIENT",
                   external_pixel_gradient ? "true" : "false");
    if (native_float_atomic_add) {
        source=Replace(std::move(source), "ATOMIC_SUPPORT",
                       "#extension GL_EXT_shader_atomic_float : require");
        source=Replace(std::move(source), "GRAD_VEC4", "vec4");
        source=Replace(std::move(source), "ATOMIC_ADD_IMPLEMENTATION", R"(
#define ATOMIC_ADD(address, expression) { \
    float atomic_value=(expression); \
    if(atomic_value!=0.0 && !isnan(atomic_value) && !isinf(atomic_value)) \
        atomicAdd(address,atomic_value); \
}
)");
    } else {
        source=Replace(std::move(source), "ATOMIC_SUPPORT", "");
        source=Replace(std::move(source), "GRAD_VEC4", "uvec4");
        source=Replace(std::move(source), "ATOMIC_ADD_IMPLEMENTATION", R"(
#define ATOMIC_ADD(address, expression) { \
    float atomic_value=(expression); \
    if(atomic_value!=0.0 && !isnan(atomic_value) && !isinf(atomic_value)) { \
        uint atomic_expected=address; \
        for(;;) { \
            float atomic_current=uintBitsToFloat(atomic_expected); \
            uint atomic_desired=floatBitsToUint(atomic_current+atomic_value); \
            uint atomic_observed=atomicCompSwap( \
                    address,atomic_expected,atomic_desired); \
            if(atomic_observed==atomic_expected) break; \
            atomic_expected=atomic_observed; \
        } \
    } \
}
)");
    }
    return source;
}

std::string PerSplatRasterBackwardShader(bool native_float_atomic_add,
                                         bool external_pixel_gradient,
                                         bool byte_targets,
                                         bool stream_target,
                                         size_t target_pixels,
                                         bool pair_output) {
    std::string source=R"(#version 450
ATOMIC_SUPPORT
layout(local_size_x=128) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer Projected { vec4 value[]; } projected;
layout(std430,binding=2) readonly buffer ConicOpacity { vec4 value[]; } conic_opacity;
layout(std430,binding=3) readonly buffer Rgb { vec4 value[]; } rgb;
layout(std430,binding=4) readonly buffer TileCounts { uint value[]; } tile_counts;
layout(std430,binding=5) readonly buffer TileIds { uint value[]; } tile_ids;
layout(std430,binding=6) readonly buffer Color { vec4 value[]; } color;
layout(std430,binding=7) readonly TARGET_STORAGE_DECL
GRADIENT_BINDINGS
TARGET_FETCH_DEFINE

shared vec4 pixel_state[256];
shared vec4 pixel_gradient[256];
shared uint pixel_terminal[256];

ATOMIC_ADD_IMPLEMENTATION

void main() {
    uint tile=gl_WorkGroupID.x, lane=gl_LocalInvocationID.x;
    uint width=camera.dims.z, height=camera.dims.w;
    uint grid_width=camera.dims.x;
    uint tile_x=tile%grid_width, tile_y=tile/grid_width;
    uint view_index=camera.state.y;
    uint count=tile_counts.value[tile], offset=tile_offsets.value[tile];
    for(uint p=0u;p<2u;p++) {
        uint local_pixel=lane+p*128u;
        uint pixel_x=tile_x*16u+(local_pixel&15u);
        uint pixel_y=tile_y*16u+(local_pixel>>4u);
        bool valid=pixel_x<width&&pixel_y<height;
        uint pixel_index=valid?pixel_y*width+pixel_x:0u;
        vec3 rendered=valid?color.value[pixel_index].xyz:vec3(0.0);
        uint target_index=view_index*(width*height)+pixel_index;
        vec3 wanted=valid?TARGET_FETCH(EXTERNAL_GRADIENT?pixel_index:target_index).xyz:
            vec3(0.0);
        vec3 d_color=valid?(EXTERNAL_GRADIENT?wanted:
            sign(rendered-wanted)/(float(width)*float(height)*3.0)):vec3(0.0);
        pixel_state[local_pixel]=vec4(0.0,0.0,0.0,
            valid?1.0-color.value[pixel_index].w:1.0);
        pixel_gradient[local_pixel]=vec4(d_color,0.0);
        pixel_terminal[local_pixel]=valid?
            min(contributors.value[pixel_index],count):0u;
        LOSS_WRITE
    }
    barrier();

    uint rounds=(count+127u)/128u;
    for(uint batch=0u;batch<rounds;batch++) {
        uint progress=batch*128u+lane;
        bool splat_active=progress<count;
        uint reverse_index=splat_active?count-1u-progress:0u;
        uint pair_index=splat_active?offset+reverse_index:offset;
        uint id=splat_active?tile_ids.value[pair_index]:0u;
        uint mask=splat_active?tile_masks.value[pair_index]:0u;
        vec2 center=splat_active?projected.value[id].xy:vec2(0.0);
        vec4 conic=splat_active?conic_opacity.value[id]:vec4(0.0);
        vec3 splat_rgb=splat_active?rgb.value[id].xyz:vec3(0.0);
        vec4 local_projected=vec4(0.0);
        vec4 local_conic=vec4(0.0);
        vec4 local_rgb=vec4(0.0);
        uint num_splats=min(128u,count-batch*128u);
        for(uint wave=0u;wave<num_splats+255u;wave++) {
            int pixel_rank=int(wave)-int(lane);
            if(splat_active&&pixel_rank>=0&&pixel_rank<256&&
               reverse_index<pixel_terminal[uint(pixel_rank)]) {
                uint local_pixel=uint(pixel_rank);
                uint pixel_x=tile_x*16u+(local_pixel&15u);
                uint pixel_y=tile_y*16u+(local_pixel>>4u);
                uint subtile=((pixel_y&15u)>>2u)*4u+((pixel_x&15u)>>2u);
                if((mask&(1u<<subtile))!=0u) {
                    vec2 d=vec2(pixel_x,pixel_y)-center;
                    float power=-0.5*(conic.x*d.x*d.x+
                        2.0*conic.y*d.x*d.y+conic.z*d.y*d.y);
                    if(power<=0.0) {
                        float exponential=exp(power);
                        float raw_alpha=conic.w*exponential;
                        float alpha=min(0.999,raw_alpha);
                        if(alpha>=0.00392156862745) {
                            vec4 state=pixel_state[local_pixel];
                            float one_minus=1.0-alpha;
                            float transmittance_before=state.w/one_minus;
                            vec3 d_color=pixel_gradient[local_pixel].xyz;
                            local_rgb.xyz+=d_color*(transmittance_before*alpha);
                            float d_alpha=dot(d_color,
                                transmittance_before*splat_rgb-
                                state.xyz/one_minus);
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
                            state.xyz+=transmittance_before*alpha*splat_rgb;
                            state.w=transmittance_before;
                            pixel_state[local_pixel]=state;
                        }
                    }
                }
            }
            barrier();
        }
        GRADIENT_WRITE
        barrier();
    }
})";
    source=ApplyTargetStorage(std::move(source),
                              byte_targets&&!external_pixel_gradient,
                              stream_target&&!external_pixel_gradient,
                              target_pixels);
    source=Replace(std::move(source),"EXTERNAL_GRADIENT",
                   external_pixel_gradient?"true":"false");
    if(pair_output) {
        source=Replace(std::move(source),"GRADIENT_BINDINGS",R"(
layout(std430,binding=8) writeonly buffer PairGrad { vec4 value[]; } pair_grad;
layout(std430,binding=9) readonly buffer TileMasks { uint value[]; } tile_masks;
layout(std430,binding=10) readonly buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430,binding=11) readonly buffer Contributors { uint value[]; } contributors;
layout(std430,binding=12) readonly buffer GaussianOffsets { uint value[]; } gaussian_offsets;
layout(std430,binding=13) readonly buffer TileRect { uvec4 value[]; } tile_rect;
bool finite_vec4(vec4 value) {
    return !any(isnan(value)) && !any(isinf(value));
}
)");
        source=Replace(std::move(source),"GRADIENT_WRITE",R"(
if(splat_active) {
    uvec4 rectangle=tile_rect.value[id];
    uint original=gaussian_offsets.value[id]+
        (tile_y-rectangle.y)*(rectangle.z-rectangle.x)+
        tile_x-rectangle.x;
    pair_grad.value[original*3u]=finite_vec4(local_projected) ?
        local_projected : vec4(0.0);
    pair_grad.value[original*3u+1u]=finite_vec4(local_conic) ?
        local_conic : vec4(0.0);
    pair_grad.value[original*3u+2u]=finite_vec4(local_rgb) ?
        local_rgb : vec4(0.0);
}
)");
        source=Replace(std::move(source),"ATOMIC_SUPPORT","");
        source=Replace(std::move(source),"ATOMIC_ADD_IMPLEMENTATION","");
        source=Replace(std::move(source),"LOSS_WRITE","");
        return source;
    }
    source=Replace(std::move(source),"GRADIENT_BINDINGS",R"(
layout(std430,binding=8) buffer ProjectedGrad { GRAD_VEC4 value[]; } projected_grad;
layout(std430,binding=9) buffer ConicGrad { GRAD_VEC4 value[]; } conic_grad;
layout(std430,binding=10) buffer RgbGrad { GRAD_VEC4 value[]; } rgb_grad;
layout(std430,binding=11) writeonly buffer Loss { float value[]; } loss_value;
layout(std430,binding=12) readonly buffer OptimizerMeta { uvec4 value; } meta;
layout(std430,binding=13) readonly buffer TileMasks { uint value[]; } tile_masks;
layout(std430,binding=14) readonly buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430,binding=15) readonly buffer Contributors { uint value[]; } contributors;
)");
    source=Replace(std::move(source),"LOSS_WRITE",R"(
if(valid&&!EXTERNAL_GRADIENT) loss_value.value[pixel_index]=
    (abs(rendered.x-wanted.x)+abs(rendered.y-wanted.y)+
     abs(rendered.z-wanted.z))/3.0;
)");
    source=Replace(std::move(source),"EXTERNAL_GRADIENT",
                   external_pixel_gradient?"true":"false");
    source=Replace(std::move(source),"GRADIENT_WRITE",R"(
if(splat_active) {
    ATOMIC_ADD(projected_grad.value[id].x,local_projected.x);
    ATOMIC_ADD(projected_grad.value[id].y,local_projected.y);
    ATOMIC_ADD(conic_grad.value[id].x,local_conic.x);
    ATOMIC_ADD(conic_grad.value[id].y,local_conic.y);
    ATOMIC_ADD(conic_grad.value[id].z,local_conic.z);
    ATOMIC_ADD(conic_grad.value[id].w,local_conic.w);
    ATOMIC_ADD(rgb_grad.value[id].x,local_rgb.x);
    ATOMIC_ADD(rgb_grad.value[id].y,local_rgb.y);
    ATOMIC_ADD(rgb_grad.value[id].z,local_rgb.z);
}
)");
    if(native_float_atomic_add) {
        source=Replace(std::move(source),"ATOMIC_SUPPORT",
                       "#extension GL_EXT_shader_atomic_float : require");
        source=Replace(std::move(source),"GRAD_VEC4","vec4");
        source=Replace(std::move(source),"ATOMIC_ADD_IMPLEMENTATION",R"(
#define ATOMIC_ADD(address, expression) { \
    float atomic_value=(expression); \
    if(atomic_value!=0.0 && !isnan(atomic_value) && !isinf(atomic_value)) \
        atomicAdd(address,atomic_value); \
}
)");
    } else {
        source=Replace(std::move(source),"ATOMIC_SUPPORT","");
        source=Replace(std::move(source),"GRAD_VEC4","uvec4");
        source=Replace(std::move(source),"ATOMIC_ADD_IMPLEMENTATION",R"(
#define ATOMIC_ADD(address, expression) { \
    float atomic_value=(expression); \
    if(atomic_value!=0.0 && !isnan(atomic_value) && !isinf(atomic_value)) { \
        uint atomic_expected=address; \
        for(;;) { \
            float atomic_current=uintBitsToFloat(atomic_expected); \
            uint atomic_desired=floatBitsToUint(atomic_current+atomic_value); \
            uint atomic_observed=atomicCompSwap(address,atomic_expected,atomic_desired); \
            if(atomic_observed==atomic_expected) break; \
            atomic_expected=atomic_observed; \
        } \
    } \
}
)");
    }
    return source;
}

std::string TiledRasterBackwardShader(bool native_float_atomic_add,
                                       bool external_pixel_gradient,
                                       bool byte_targets,
                                       bool stream_target,
                                       size_t target_pixels) {
std::string source=R"(#version 450
ATOMIC_SUPPORT
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
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
layout(std430,binding=8) buffer ProjectedGrad { GRAD_VEC4 value[]; } projected_grad;
layout(std430,binding=9) buffer ConicGrad { GRAD_VEC4 value[]; } conic_grad;
layout(std430,binding=10) buffer RgbGrad { GRAD_VEC4 value[]; } rgb_grad;
layout(std430,binding=11) writeonly buffer Loss { float value[]; } loss_value;
layout(std430,binding=12) readonly buffer OptimizerMeta { uvec4 value; } meta;
TARGET_FETCH_DEFINE
layout(std430,binding=13) readonly buffer TileMasks { uint value[]; } tile_masks;
layout(std430,binding=14) readonly buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430,binding=15) readonly buffer Contributors { uint value[]; } contributors;

const uint BATCH=8u;
const uint PIXEL_BATCHES=32u;
shared uint collected_id[256];
shared vec2 collected_center[256];
shared vec4 collected_conic[256];
shared vec4 collected_rgb[256];
shared uint collected_mask[256];
shared float splat_embed[256][6];
shared vec2 alpha_gradient[BATCH][256];
shared vec4 pixel_gradient[256];
shared float reduce_splat[PIXEL_BATCHES][BATCH][6];
shared vec4 reduce_rgb[256];

ATOMIC_ADD_IMPLEMENTATION

void embed_splat(vec2 center,vec4 conic,vec2 tile_center,out float e[6]) {
    vec2 c=center-tile_center;
    e[0]=log(conic.w)-0.5*conic.x*c.x*c.x-conic.y*c.x*c.y-
         0.5*conic.z*c.y*c.y;
    e[1]=conic.x*c.x+conic.y*c.y;
    e[2]=conic.y*c.x+conic.z*c.y;
    e[3]=-0.5*conic.x;e[4]=-conic.y;e[5]=-0.5*conic.z;
}
void embed_pixel(uint pixel,out float e[6]) {
    vec2 p=vec2(float(pixel&15u)-8.0,float(pixel>>4u)-8.0);
    e[0]=1.0;e[1]=p.x;e[2]=p.y;e[3]=p.x*p.x;e[4]=p.x*p.y;e[5]=p.y*p.y;
}
void embed_backward(vec2 center,vec4 conic,vec2 tile_center,float g[6],
                    out vec2 center_grad,out vec4 conic_grad_value) {
    vec2 c=center-tile_center;
    center_grad=vec2((-conic.x*c.x-conic.y*c.y)*g[0]+conic.x*g[1]+conic.y*g[2],
                     (-conic.y*c.x-conic.z*c.y)*g[0]+conic.y*g[1]+conic.z*g[2]);
    conic_grad_value=vec4(-0.5*c.x*c.x*g[0]+c.x*g[1]-0.5*g[3],
        -c.x*c.y*g[0]+c.y*g[1]+c.x*g[2]-g[4],
        -0.5*c.y*c.y*g[0]+c.y*g[2]-0.5*g[5],g[0]/conic.w);
}

void main() {
    uint tile=gl_WorkGroupID.x,lane=gl_LocalInvocationID.x;
    uint width=camera.dims.z,height=camera.dims.w,grid_width=camera.dims.x;
    uint tile_x=tile%grid_width,tile_y=tile/grid_width;
    uint pixel_x=tile_x*16u+(lane&15u),pixel_y=tile_y*16u+(lane>>4u);
    bool valid=pixel_x<width&&pixel_y<height;
    uint pixel_index=valid?pixel_y*width+pixel_x:0u;
    uint terminal=valid?contributors.value[pixel_index]:0u;
    vec3 rendered=valid?color.value[pixel_index].xyz:vec3(0.0);
    uint target_index=camera.state.y*(width*height)+pixel_index;
    vec3 wanted=valid?TARGET_FETCH(EXTERNAL_GRADIENT?pixel_index:target_index).xyz:vec3(0.0);
    vec3 d_color=valid?(EXTERNAL_GRADIENT?wanted:
        sign(rendered-wanted)/(float(width)*float(height)*3.0)):vec3(0.0);
    if(valid&&!EXTERNAL_GRADIENT) loss_value.value[pixel_index]=
        (abs(rendered.x-wanted.x)+abs(rendered.y-wanted.y)+abs(rendered.z-wanted.z))/3.0;
    float transmittance=valid?1.0-color.value[pixel_index].w:1.0;
    float d_transmittance=0.0;
    pixel_gradient[lane]=vec4(d_color,0.0);

    uint count=tile_counts.value[tile],offset=tile_offsets.value[tile];
    vec2 tile_center=vec2(tile_x*16u+8u,tile_y*16u+8u);
    uint rounds=(count+BATCH-1u)/BATCH;
    uint splat_lane=lane&(BATCH-1u),pixel_batch=lane/BATCH;
    float pixel_embedding[6];embed_pixel(lane,pixel_embedding);
    uint subtile=(lane>>6u)*4u+((lane&15u)>>2u);
    for(uint batch=0u;batch<rounds;batch++) {
        int reverse_base=int(count)-1-int(batch*BATCH);
        if((batch%PIXEL_BATCHES)==0u) {
            if(batch>0u) barrier();
            int reverse_index=reverse_base-int(lane);
            vec2 center=vec2(0.0);vec4 conic=vec4(0.0,0.0,0.0,1.0);
            vec4 color_value=vec4(0.0);
            uint id=0u,mask=0u;
            if(reverse_index>=0) {
                id=tile_ids.value[offset+uint(reverse_index)];
                center=projected.value[id].xy;conic=conic_opacity.value[id];
                color_value=rgb.value[id];
                mask=tile_masks.value[offset+uint(reverse_index)];
            }
            uint slot=(batch%PIXEL_BATCHES)*BATCH+lane;
            collected_id[slot]=id;collected_center[slot]=center;
            collected_conic[slot]=conic;collected_rgb[slot]=color_value;
            collected_mask[slot]=mask;
            float embedding[6];embed_splat(center,conic,tile_center,embedding);
            [[unroll]] for(uint k=0u;k<6u;k++)splat_embed[slot][k]=embedding[k];
            barrier();
        }
        uint base=(batch%PIXEL_BATCHES)*BATCH;
        uint num_splats=min(BATCH,count-batch*BATCH);
        for(uint splat=0u;splat<num_splats;splat++) {
            bool overlaps=(collected_mask[base+splat]&(1u<<subtile))!=0u;
            int reverse_index=reverse_base-int(splat);
            vec2 local_alpha_gradient=vec2(0.0);
            if(overlaps&&valid&&reverse_index>=0&&
               uint(reverse_index)<terminal) {
                float power=0.0;
                [[unroll]] for(uint k=0u;k<6u;k++)
                    power+=pixel_embedding[k]*splat_embed[base+splat][k];
                float raw_alpha=exp(power);
                if(raw_alpha>=0.00392156862745) {
                    float alpha=min(0.999,raw_alpha);
                    transmittance/=1.0-alpha;
                    vec3 splat_rgb=collected_rgb[base+splat].xyz;
                    float alpha_transmittance=alpha*transmittance;
                    float color_dot=dot(d_color,splat_rgb);
                    float d_log_alpha=alpha_transmittance*
                        (color_dot-d_transmittance)*float(raw_alpha<0.999);
                    d_transmittance=d_transmittance*(1.0-alpha)+alpha*color_dot;
                    local_alpha_gradient=vec2(alpha_transmittance,d_log_alpha);
                }
            }
            alpha_gradient[splat][lane]=local_alpha_gradient;
        }
        for(uint splat=num_splats;splat<BATCH;splat++)
            alpha_gradient[splat][lane]=vec2(0.0);
        subgroupBarrier();

        float gradient_embed[6]={0.0,0.0,0.0,0.0,0.0,0.0};
        vec3 local_rgb=vec3(0.0);
        [[unroll]] for(uint q=0u;q<8u;q++) {
            uint rotated=(q+splat_lane)&7u;
            uint pixel=pixel_batch*8u+rotated;
            vec2 ag=alpha_gradient[splat_lane][pixel];
            float e[6];embed_pixel(pixel,e);
            [[unroll]] for(uint k=0u;k<6u;k++)gradient_embed[k]+=ag.y*e[k];
            local_rgb+=ag.x*pixel_gradient[pixel].xyz;
        }
        vec2 center_gradient;vec4 conic_gradient;
        embed_backward(collected_center[base+splat_lane],
                       collected_conic[base+splat_lane],tile_center,
                       gradient_embed,center_gradient,conic_gradient);
        if(splat_lane>=num_splats) {
            center_gradient=vec2(0.0);conic_gradient=vec4(0.0);local_rgb=vec3(0.0);
        }
        float packed[6]={center_gradient.x,center_gradient.y,conic_gradient.x,
                         conic_gradient.y,conic_gradient.z,conic_gradient.w};
        [[unroll]] for(uint k=0u;k<6u;k++)reduce_splat[pixel_batch][splat_lane][k]=packed[k];
        reduce_rgb[lane]=vec4(local_rgb,0.0);
        barrier();
        if(pixel_batch<8u&&splat_lane<num_splats) {
            [[unroll]] for(uint k=0u;k<6u;k++)packed[k]=
                (reduce_splat[pixel_batch][splat_lane][k]+
                 reduce_splat[pixel_batch+8u][splat_lane][k])+
                (reduce_splat[pixel_batch+16u][splat_lane][k]+
                 reduce_splat[pixel_batch+24u][splat_lane][k]);
            local_rgb=(reduce_rgb[lane].xyz+reduce_rgb[lane+8u*BATCH].xyz)+
                      (reduce_rgb[lane+16u*BATCH].xyz+
                       reduce_rgb[lane+24u*BATCH].xyz);
            uint id=collected_id[base+splat_lane];
            ATOMIC_ADD(projected_grad.value[id].x,packed[0]);
            ATOMIC_ADD(projected_grad.value[id].y,packed[1]);
            ATOMIC_ADD(conic_grad.value[id].x,packed[2]);
            ATOMIC_ADD(conic_grad.value[id].y,packed[3]);
            ATOMIC_ADD(conic_grad.value[id].z,packed[4]);
            ATOMIC_ADD(conic_grad.value[id].w,packed[5]);
            ATOMIC_ADD(rgb_grad.value[id].x,local_rgb.x);
            ATOMIC_ADD(rgb_grad.value[id].y,local_rgb.y);
            ATOMIC_ADD(rgb_grad.value[id].z,local_rgb.z);
        }
    }
})";
    source=ApplyTargetStorage(std::move(source),
                              byte_targets && !external_pixel_gradient,
                              stream_target && !external_pixel_gradient,
                              target_pixels);
    source=Replace(std::move(source),"EXTERNAL_GRADIENT",
                   external_pixel_gradient?"true":"false");
    if(native_float_atomic_add) {
        source=Replace(std::move(source),"ATOMIC_SUPPORT",
                       "#extension GL_EXT_shader_atomic_float : require");
        source=Replace(std::move(source),"GRAD_VEC4","vec4");
        return Replace(std::move(source),"ATOMIC_ADD_IMPLEMENTATION",R"(
#define ATOMIC_ADD(address,expression) { float value=(expression); \
 if(value!=0.0&&!isnan(value)&&!isinf(value)) atomicAdd(address,value); }
)");
    }
    source=Replace(std::move(source),"ATOMIC_SUPPORT","");
    source=Replace(std::move(source),"GRAD_VEC4","uvec4");
    return Replace(std::move(source),"ATOMIC_ADD_IMPLEMENTATION",R"(
#define ATOMIC_ADD(address,expression) { float value=(expression); \
 if(value!=0.0&&!isnan(value)&&!isinf(value)){uint expected=address;for(;;){ \
 float current=uintBitsToFloat(expected);uint desired=floatBitsToUint(current+value); \
 uint observed=atomicCompSwap(address,expected,desired);if(observed==expected)break;expected=observed;}}}
)");
}


}  // namespace cross3dgs::detail

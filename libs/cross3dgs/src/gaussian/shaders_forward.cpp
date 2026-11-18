#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string ProjectionShader(size_t gaussian_capacity, uint32_t sh_degree,
                             bool training, bool reorder_sh,
                             bool flatten_camera_block,
                             bool enable_igs_edge,
                             bool byte_targets,
                             bool split_sh_rest,
                             bool enable_mip3d) {
    std::string source = R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430, binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430, binding=1) readonly buffer Means { vec4 value[]; } means;
layout(std430, binding=2) readonly buffer ScalesOpacity { vec4 value[]; } scale_opacity;
layout(std430, binding=3) readonly buffer Rotations { vec4 value[]; } rotations;
layout(std430, binding=4) readonly buffer ShDc { vec4 value[]; } sh_dc;
layout(std430, binding=5) readonly buffer ShRest { vec4 value[]; } sh_rest;
layout(std430, binding=6) writeonly buffer Projected { vec4 value[]; } projected;
layout(std430, binding=7) writeonly buffer ConicOpacity { vec4 value[]; } conic_opacity;
layout(std430, binding=8) writeonly buffer Rgb { vec4 value[]; } rgb;
layout(std430, binding=9) writeonly buffer TileRect { uvec4 value[]; } tile_rect;
layout(std430, binding=10) writeonly buffer GaussianTileCounts { uint value[]; } gaussian_tile_counts;
IGS_EDGE_BINDINGS

uint SH_LINEAR_INDEX(uint gaussian,uint packed){return gaussian*12u+packed;}
uint SH_REORDER_INDEX(uint gaussian,uint packed){
    return (gaussian&~31u)*12u+packed*32u+(gaussian&31u);
}
float sh_scalar(uint gaussian, uint scalar) {
    return sh_rest.value[SH_INDEX(gaussian,scalar/4u)][scalar%4u];
}
vec3 sh_coeff(uint gaussian, uint coefficient) {
    uint scalar=(coefficient-1u)*3u;
    return vec3(sh_scalar(gaussian,scalar),sh_scalar(gaussian,scalar+1u),
                sh_scalar(gaussian,scalar+2u));
}
void sh_basis(vec3 d, out float b[15]) {
    float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z;
    b[0]=-0.4886025119029199*y;
    b[1]= 0.4886025119029199*z;
    b[2]=-0.4886025119029199*x;
    b[3]= 1.0925484305920792*x*y;
    b[4]=-1.0925484305920792*y*z;
    b[5]= 0.31539156525252005*(2.0*zz-xx-yy);
    b[6]=-1.0925484305920792*x*z;
    b[7]= 0.5462742152960396*(xx-yy);
    b[8]=-0.5900435899266435*y*(3.0*xx-yy);
    b[9]= 2.890611442640554*x*y*z;
    b[10]=-0.4570457994644658*y*(4.0*zz-xx-yy);
    b[11]= 0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy);
    b[12]=-0.4570457994644658*x*(4.0*zz-xx-yy);
    b[13]= 1.445305721320277*z*(xx-yy);
    b[14]=-0.5900435899266435*x*(xx-3.0*yy);
}

mat3 quaternion_rotation(vec4 q) {
    q = normalize(q);
    float r=q.x, x=q.y, y=q.z, z=q.w;
    float r00=1.0-2.0*(y*y+z*z), r01=2.0*(x*y-r*z), r02=2.0*(x*z+r*y);
    float r10=2.0*(x*y+r*z), r11=1.0-2.0*(x*x+z*z), r12=2.0*(y*z-r*x);
    float r20=2.0*(x*z-r*y), r21=2.0*(y*z+r*x), r22=1.0-2.0*(x*x+y*y);
    return mat3(vec3(r00,r10,r20), vec3(r01,r11,r21), vec3(r02,r12,r22));
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint count = camera.state.x;
    if (index >= GAUSSIAN_CAPACITY) return;
    if (index >= count) {
        gaussian_tile_counts.value[index]=0u;
        return;
    }
    vec3 mean = means.value[index].xyz;
    vec3 view = (camera.world_to_camera * vec4(mean, 1.0)).xyz;
    if (view.z <= 0.01) {
        projected.value[index] = vec4(0.0);
        tile_rect.value[index] = uvec4(0);
        gaussian_tile_counts.value[index]=0u;
        return;
    }
    float fx=camera.intrinsics.x, fy=camera.intrinsics.y;
    vec2 screen = vec2(fx*view.x/view.z + camera.intrinsics.z,
                       fy*view.y/view.z + camera.intrinsics.w) - vec2(0.5);
    vec3 scale = scale_opacity.value[index].xyz;
    mat3 rotation = quaternion_rotation(rotations.value[index]);
    mat3 covariance = rotation * mat3(scale.x*scale.x,0,0, 0,scale.y*scale.y,0,
                                      0,0,scale.z*scale.z) * transpose(rotation);
    mat3 world_view_rotation = mat3(camera.world_to_camera);
    vec3 row0=vec3(world_view_rotation[0][0],world_view_rotation[1][0],world_view_rotation[2][0]);
    vec3 row1=vec3(world_view_rotation[0][1],world_view_rotation[1][1],world_view_rotation[2][1]);
    vec3 row2=vec3(world_view_rotation[0][2],world_view_rotation[1][2],world_view_rotation[2][2]);
    float inv_z=1.0/view.z;
    float tan_fovx=0.5*float(camera.dims.z)/fx;
    float tan_fovy=0.5*float(camera.dims.w)/fy;
    float limit_x_positive=(float(camera.dims.z)-camera.intrinsics.z)/fx+
                           0.3*tan_fovx;
    float limit_x_negative=camera.intrinsics.z/fx+0.3*tan_fovx;
    float limit_y_positive=(float(camera.dims.w)-camera.intrinsics.w)/fy+
                           0.3*tan_fovy;
    float limit_y_negative=camera.intrinsics.w/fy+0.3*tan_fovy;
    float covariance_x=clamp(view.x*inv_z,-limit_x_negative,
                             limit_x_positive)*view.z;
    float covariance_y=clamp(view.y*inv_z,-limit_y_negative,
                             limit_y_positive)*view.z;
    vec3 t0=(fx*inv_z)*row0+
            (-fx*covariance_x*inv_z*inv_z)*row2;
    vec3 t1=(fy*inv_z)*row1+
            (-fy*covariance_y*inv_z*inv_z)*row2;
    vec3 cov_t0 = covariance * t0;
    vec3 cov_t1 = covariance * t1;
    float xx=dot(t0,cov_t0)+0.3, xy=dot(t0,cov_t1), yy=dot(t1,cov_t1)+0.3;
    float determinant=xx*yy-xy*xy;
    if (!(determinant > 0.0) || isnan(determinant) || isinf(determinant)) {
        projected.value[index] = vec4(0.0);
        tile_rect.value[index] = uvec4(0);
        gaussian_tile_counts.value[index]=0u;
        return;
    }
    vec3 inverse=vec3(yy,-xy,xx)/determinant;
    float opacity=scale_opacity.value[index].w;
    if(opacity<=0.00392156862745) {
        projected.value[index]=vec4(0.0);tile_rect.value[index]=uvec4(0);
        gaussian_tile_counts.value[index]=0u;return;
    }
    float extend=sqrt(2.0*log(min(opacity,1.0)/0.00392156862745));
    vec2 radii=ceil(sqrt(max(vec2(xx,yy),vec2(0.0)))*extend);
    uint grid_width=camera.dims.x, grid_height=camera.dims.y;
    ivec2 minimum=ivec2(floor((screen-radii)/16.0));
    ivec2 maximum=ivec2(ceil((screen+radii)/16.0));
    minimum=clamp(minimum,ivec2(0),ivec2(grid_width,grid_height));
    maximum=clamp(maximum,ivec2(0),ivec2(grid_width,grid_height));
    if (minimum.x>=maximum.x || minimum.y>=maximum.y) {
        projected.value[index] = vec4(0.0);
        tile_rect.value[index] = uvec4(0);
        gaussian_tile_counts.value[index]=0u;
        return;
    }
    projected.value[index]=vec4(screen,view.z,max(radii.x,radii.y));
    conic_opacity.value[index]=vec4(inverse,scale_opacity.value[index].w);
    vec3 color=vec3(0.5)+0.28209479177387814*sh_dc.value[index].xyz;
    uint active_degree=ACTIVE_SH;
    if (active_degree>0u) {
        mat3 world_rotation=mat3(camera.world_to_camera);
        vec3 camera_position=-transpose(world_rotation)*camera.world_to_camera[3].xyz;
        float basis[15]; sh_basis(normalize(mean-camera_position),basis);
        uint coefficient_count=active_degree==1u ? 3u : (active_degree==2u ? 8u : 15u);
        for(uint coefficient=1u;coefficient<=coefficient_count;coefficient++)
            color+=basis[coefficient-1u]*sh_coeff(index,coefficient);
    }
    rgb.value[index]=vec4(max(vec3(0.0),color),1.0);
    IGS_EDGE_ACCUM
    tile_rect.value[index]=uvec4(minimum,maximum);
    gaussian_tile_counts.value[index]=uint(
            (maximum.x-minimum.x)*(maximum.y-minimum.y));
})";
    if (enable_mip3d) {
        source=Replace(std::move(source),
            "vec3 scale = scale_opacity.value[index].xyz;",
            "vec3 scale = scale_opacity.value[index].xyz;\n"
            "    float mip_f2=means.value[index].w*means.value[index].w;\n"
            "    vec3 mip_s2=scale*scale;\n"
            "    float mip_comp=sqrt(max(mip_s2.x*mip_s2.y*mip_s2.z,0.0)/\n"
            "        max((mip_s2.x+mip_f2)*(mip_s2.y+mip_f2)*(mip_s2.z+mip_f2),1e-30));");
        source=Replace(std::move(source),
            "mat3 covariance = rotation * mat3(scale.x*scale.x,0,0, 0,scale.y*scale.y,0,\n"
            "                                      0,0,scale.z*scale.z) * transpose(rotation);",
            "mat3 covariance = rotation * mat3(scale.x*scale.x+mip_f2,0,0, 0,scale.y*scale.y+mip_f2,0,\n"
            "                                      0,0,scale.z*scale.z+mip_f2) * transpose(rotation);");
        source=Replace(std::move(source),
            "float opacity=scale_opacity.value[index].w;",
            "float opacity=scale_opacity.value[index].w*mip_comp;");
        source=Replace(std::move(source),
            "conic_opacity.value[index]=vec4(inverse,scale_opacity.value[index].w);",
            "conic_opacity.value[index]=vec4(inverse,opacity);");
    }
    if (flatten_camera_block) {
        // Adreno's shader compiler has already miscompiled partial accesses to
        // an SSBO struct in this pipeline (increment_optimizer_step). Keep the
        // nested declaration byte-identical on all other vendors, but expose
        // the same std430 layout as direct block members on Qualcomm.
        source=Replace(std::move(source),
            "struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };\n"
            "layout(std430, binding=0) readonly buffer CameraBlock { CameraData camera; };",
            "layout(std430, binding=0) readonly buffer CameraBlock {\n"
            "    mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state;\n"
            "} camera;");
    }
    source=Replace(std::move(source),"GAUSSIAN_CAPACITY",
                   std::to_string(gaussian_capacity)+"u");
    const std::string degree=std::to_string(sh_degree)+"u";
    source=Replace(std::move(source),"ACTIVE_SH",training ?
            "min("+degree+",camera.state.z/1000u)" : degree);
    if (enable_igs_edge) {
        source=Replace(std::move(source),"IGS_EDGE_BINDINGS",R"(
layout(std430,binding=11) readonly TARGET_STORAGE_DECL
layout(std430,binding=12) buffer IgsScore { vec4 value[]; } igs_score;
TARGET_FETCH_DEFINE)");
        source=Replace(std::move(source),"IGS_EDGE_ACCUM",R"(
    {
        uint igs_w=camera.dims.z, igs_h=camera.dims.w;
        ivec2 igs_px=clamp(ivec2(round(screen)),ivec2(0),
                           ivec2(int(igs_w)-1,int(igs_h)-1));
        uint igs_pidx=camera.state.y*(igs_w*igs_h)+
                      uint(igs_px.y)*igs_w+uint(igs_px.x);
        float igs_omega=TARGET_FETCH(igs_pidx).w;
        vec4 igs_acc=igs_score.value[index];
        igs_acc.zw+=vec2(igs_omega*opacity,1.0);
        igs_score.value[index]=igs_acc;
    })");
        source=ApplyTargetStorage(std::move(source),byte_targets);
    } else {
        source=Replace(std::move(source),"IGS_EDGE_BINDINGS","");
        source=Replace(std::move(source),"IGS_EDGE_ACCUM","");
    }
    if (split_sh_rest) {
        source=Replace(std::move(source),
            "layout(std430, binding=5) readonly buffer ShRest { vec4 value[]; } sh_rest;",
            "layout(std430, binding=5) readonly buffer ShRestA { vec4 value[]; } sh_rest;\n"
            "layout(std430, binding=11) readonly buffer ShRestB { vec4 value[]; } sh_rest_b;");
        source=Replace(std::move(source),
            "return sh_rest.value[SH_INDEX(gaussian,scalar/4u)][scalar%4u];",
            "uint packed=scalar/4u;\n"
            "    uint index=gaussian*6u+(packed%6u);\n"
            "    return packed<6u?sh_rest.value[index][scalar%4u]:"
            "sh_rest_b.value[index][scalar%4u];");
    }
    return Replace(std::move(source),"SH_INDEX",
        reorder_sh ? "SH_REORDER_INDEX" : "SH_LINEAR_INDEX");
}

std::string RasterShader(uint32_t tile_capacity) {
    std::string source = R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430, binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430, binding=1) readonly buffer Projected { vec4 value[]; } projected;
layout(std430, binding=2) readonly buffer ConicOpacity { vec4 value[]; } conic_opacity;
layout(std430, binding=3) readonly buffer Rgb { vec4 value[]; } rgb;
layout(std430, binding=4) readonly buffer TileCounts { uint value[]; } tile_counts;
layout(std430, binding=5) readonly buffer TileIds { uint value[]; } tile_ids;
layout(std430, binding=6) writeonly buffer Color { vec4 value[]; } color;
layout(std430, binding=7) writeonly buffer Contributors { uint value[]; } contributors;
layout(std430, binding=8) readonly buffer TileMasks { uint value[]; } tile_masks;
layout(std430, binding=9) readonly buffer TileOffsets { uint value[]; } tile_offsets;
shared vec4 chunk_projected[256];
shared vec4 chunk_conic[256];
shared vec4 chunk_rgb[256];
shared uint chunk_mask[256];
shared uint active_pixel_count;

bool ellipse_overlaps_rect(vec3 q,float x0,float x1,float y0,float y1) {
    if(x0<=0.0 && x1>=0.0 && y0<=0.0 && y1>=0.0) return true;
    float minimum=1e30;
    float x=clamp(-q.y*y0/q.x,x0,x1);
    minimum=min(minimum,q.x*x*x+2.0*q.y*x*y0+q.z*y0*y0);
    x=clamp(-q.y*y1/q.x,x0,x1);
    minimum=min(minimum,q.x*x*x+2.0*q.y*x*y1+q.z*y1*y1);
    float y=clamp(-q.y*x0/q.z,y0,y1);
    minimum=min(minimum,q.x*x0*x0+2.0*q.y*x0*y+q.z*y*y);
    y=clamp(-q.y*x1/q.z,y0,y1);
    minimum=min(minimum,q.x*x1*x1+2.0*q.y*x1*y+q.z*y*y);
    return minimum<=1.0001;
}

uint compute_subtile_mask(uint tile, uint grid_width, vec4 projected_value,
                          vec4 conic_value) {
    float extend_squared=2.0*log(max(conic_value.w,0.00392156862745)/
                                  0.00392156862745);
    vec3 normalized=conic_value.xyz/max(extend_squared,1e-6);
    float determinant=normalized.x*normalized.z-normalized.y*normalized.y;
    if(!(determinant>0.0)) return 0xffu;
    vec2 half_extent=sqrt(max(vec2(0.0),normalized.zx/determinant));
    vec2 tile_origin=vec2(tile%grid_width,tile/grid_width)*16.0;
    vec2 center=projected_value.xy-tile_origin;
    uint mask=0u;
    for(uint sy=0u;sy<4u;sy++) {
        for(uint sx=0u;sx<4u;sx++) {
            float x0=float(sx*4u)-center.x, x1=x0+3.0;
            float y0=float(sy*4u)-center.y, y1=y0+3.0;
            bool overlap=ellipse_overlaps_rect(normalized,x0,x1,y0,y1);
            if(overlap) mask|=1u<<(sy*4u+sx);
        }
    }
    return mask;
}

void main() {
    uint tile=gl_WorkGroupID.x, thread_id=gl_LocalInvocationID.x;
    uint grid_width=camera.dims.x;
    uint count=tile_counts.value[tile];
    uint tile_offset=tile_offsets.value[tile];
    uint tile_x=tile%grid_width, tile_y=tile/grid_width;
    uint pixel_x=tile_x*16u+(thread_id&15u);
    uint pixel_y=tile_y*16u+(thread_id>>4u);
    uint image_width=camera.dims.z;
    uint image_height=camera.dims.w;
    bool valid=pixel_x<image_width && pixel_y<image_height;
    vec3 accumulated=vec3(0.0);
    float transmittance=1.0;
    bool pixel_active=valid;
    uint terminal_index=0u;
    if(thread_id==0u) {
        uint valid_width=min(16u,image_width-min(image_width,tile_x*16u));
        uint valid_height=min(16u,image_height-min(image_height,tile_y*16u));
        active_pixel_count=valid_width*valid_height;
    }
    for (uint base=0u; base<count; base+=256u) {
        uint source=base+thread_id;
        if (source<count) {
            uint id=tile_ids.value[tile_offset+source];
            chunk_projected[thread_id]=projected.value[id];
            chunk_conic[thread_id]=conic_opacity.value[id];
            chunk_rgb[thread_id]=rgb.value[id];
            chunk_mask[thread_id]=tile_masks.value[tile_offset+source];
        }
        barrier();
        bool was_active=pixel_active;
        if (was_active) {
            uint n=min(256u,count-base);
            vec2 pixel=vec2(pixel_x,pixel_y);
            uint subtile=(pixel_y&15u)/4u*4u+(pixel_x&15u)/4u;
            for (uint i=0u;i<n;i++) {
                if((chunk_mask[i]&(1u<<subtile))==0u) continue;
                vec2 d=pixel-chunk_projected[i].xy;
                vec3 conic=chunk_conic[i].xyz;
                float power=-0.5*(conic.x*d.x*d.x+2.0*conic.y*d.x*d.y+conic.z*d.y*d.y);
                if (power>0.0) continue;
                float alpha=min(0.999,chunk_conic[i].w*exp(power));
                if (alpha<0.00392156862745) continue;
                float next_transmittance=transmittance*(1.0-alpha);
                if(next_transmittance<0.0001) {
                    terminal_index=base+i;
                    pixel_active=false;
                    atomicAdd(active_pixel_count,0xffffffffu);
                    break;
                }
                float weight=transmittance*alpha;
                accumulated+=weight*chunk_rgb[i].xyz;
                transmittance=next_transmittance;
                terminal_index=base+i+1u;
            }
        }
        barrier();
        if(active_pixel_count==0u) break;
    }
    if (valid) {
        uint pixel_index=pixel_y*image_width+pixel_x;
        color.value[pixel_index]=vec4(accumulated,1.0-transmittance);
        contributors.value[pixel_index]=terminal_index;
    }
})";
    source=Replace(std::move(source),"TILE_CAPACITY",
                   std::to_string(tile_capacity));
    return source;
}

std::string ClearShader() {
    return R"(#version 450
layout(local_size_x=256) in;
layout(std430, binding=0) buffer TileCounts { uint value[]; } tile_counts;
layout(std430, binding=1) buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430, binding=2) buffer Overflow { uint value; } overflow;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i<TILE_COUNT) {
        tile_counts.value[i]=0u;
        tile_offsets.value[i]=0u;
    }
    if (i==0u) overflow.value=0u;
})";
}

std::string EmitDepthKeysShader(size_t gaussian_capacity,
                                size_t padded_capacity) {
    // Keys for the global splat depth argsort: positive-float view.z bits
    // (order-preserving for z > 0) for visible splats, 0xFFFFFFFF for
    // culled/inactive/padding lanes so they sort last.
    std::string source=R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer GaussianCounts { uint value[]; } gaussian_counts;
layout(std430,binding=2) readonly buffer Projected { vec4 value[]; } projected;
layout(std430,binding=3) writeonly buffer Keys { uint value[]; } keys;
layout(std430,binding=4) writeonly buffer Values { uint value[]; } values;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if(i>=PADDED_CAPACITY) return;
    uint key=0xFFFFFFFFu;
    if(i<GAUSSIAN_CAPACITY && i<camera.state.x &&
       gaussian_counts.value[i]>0u)
        key=floatBitsToUint(projected.value[i].z);
    keys.value[i]=key;
    values.value[i]=i;
})";
    source=Replace(std::move(source),"GAUSSIAN_CAPACITY",
                   std::to_string(gaussian_capacity)+"u");
    return Replace(std::move(source),"PADDED_CAPACITY",
                   std::to_string(padded_capacity)+"u");
}

std::string DepthMetaShader(size_t padded_capacity) {
    return Replace(R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) writeonly buffer DepthMeta { uvec4 value[]; } meta;
void main() {
    for(uint pass=0u;pass<4u;pass++)
        meta.value[pass]=uvec4(PADDED_CAPACITY,0u,0u,0u);
})", "PADDED_CAPACITY", std::to_string(padded_capacity)+"u");
}

std::string GatherCountsByRankShader(size_t gaussian_capacity,
                                     size_t padded_capacity) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer Values { uint value[]; } values;
layout(std430,binding=2) readonly buffer GaussianCounts { uint value[]; } gaussian_counts;
layout(std430,binding=3) writeonly buffer RankCounts { uint value[]; } rank_counts;
void main() {
    uint rank=gl_GlobalInvocationID.x;
    if(rank>=PADDED_CAPACITY) return;
    uint id=values.value[rank];
    rank_counts.value[rank]=
        (id<GAUSSIAN_CAPACITY && id<camera.state.x)
            ? gaussian_counts.value[id] : 0u;
})";
    source=Replace(std::move(source),"GAUSSIAN_CAPACITY",
                   std::to_string(gaussian_capacity)+"u");
    return Replace(std::move(source),"PADDED_CAPACITY",
                   std::to_string(padded_capacity)+"u");
}

std::string ScatterOffsetsByIdShader(size_t gaussian_capacity,
                                     size_t padded_capacity) {
    // Rewrites the id-indexed pair offsets so every per-splat consumer (the
    // pair-partial backward, the exact reduce) keeps its contiguous block,
    // now positioned at the splat's global depth rank.
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Values { uint value[]; } values;
layout(std430,binding=1) readonly buffer RankOffsets { uint value[]; } rank_offsets;
layout(std430,binding=2) writeonly buffer GaussianOffsets { uint value[]; } gaussian_offsets;
void main() {
    uint rank=gl_GlobalInvocationID.x;
    if(rank>=PADDED_CAPACITY) return;
    uint id=values.value[rank];
    if(id<GAUSSIAN_CAPACITY) gaussian_offsets.value[id]=rank_offsets.value[rank];
})";
    source=Replace(std::move(source),"GAUSSIAN_CAPACITY",
                   std::to_string(gaussian_capacity)+"u");
    return Replace(std::move(source),"PADDED_CAPACITY",
                   std::to_string(padded_capacity)+"u");
}

std::string PreparePairsShader(size_t gaussian_capacity,
                               size_t pair_capacity,
                               bool indirect_pairs,
                               size_t rank_padded_capacity) {
    std::string source=R"(#version 450
layout(local_size_x=1) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer GaussianCounts { uint value[]; } gaussian_counts;
layout(std430,binding=2) readonly buffer GaussianOffsets { uint value[]; } gaussian_offsets;
layout(std430,binding=3) writeonly buffer RadixMeta { uvec4 value[]; } radix_meta;
layout(std430,binding=4) buffer Overflow { uint value; uint updates; uint peak; } overflow;
PAIR_DISPATCH_BINDING
void main() {
    uint count=min(camera.state.x,GAUSSIAN_CAPACITY);
    uint total=count==0u ? 0u :
        gaussian_offsets.value[count-1u]+gaussian_counts.value[count-1u];
    uint clamped=min(total,PAIR_CAPACITY);
    for(uint pass=0u;pass<4u;pass++)
        radix_meta.value[pass]=uvec4(clamped,0u,0u,0u);
    overflow.value=total-clamped;
    overflow.updates+=total>clamped ? 1u : 0u;
    overflow.peak=max(overflow.peak,total);
    PAIR_DISPATCH_WRITE
})";
    if (rank_padded_capacity > 0) {
        // Full-depth-sort mode: bindings 1/2 receive the rank-ordered
        // count/offset arrays whose padded tail is all zero, so the total
        // is read from the last padded lane instead of the last splat id.
        source=Replace(std::move(source),
            "uint count=min(camera.state.x,GAUSSIAN_CAPACITY);\n"
            "    uint total=count==0u ? 0u :\n"
            "        gaussian_offsets.value[count-1u]+gaussian_counts.value[count-1u];",
            "uint total=gaussian_offsets.value["+
            std::to_string(rank_padded_capacity-1)+"u]+"
            "gaussian_counts.value["+
            std::to_string(rank_padded_capacity-1)+"u];");
    }
    source=Replace(std::move(source),"GAUSSIAN_CAPACITY",
                   std::to_string(gaussian_capacity)+"u");
    source=Replace(std::move(source),"PAIR_DISPATCH_BINDING",
                   indirect_pairs
                       ? "layout(std430,binding=5) writeonly buffer "
                         "PairDispatch { uint value[]; } pair_dispatch;"
                       : "");
    source=Replace(std::move(source),"PAIR_DISPATCH_WRITE",
                   indirect_pairs
                       ? "pair_dispatch.value[0]=(clamped+255u)/256u;"
                         "pair_dispatch.value[1]=1u;pair_dispatch.value[2]=1u;"
                       : "");
    return Replace(std::move(source),"PAIR_CAPACITY",
                   std::to_string(pair_capacity)+"u");
}

std::string EmitPairsShader(size_t gaussian_capacity,
                            size_t pair_capacity,
                            uint32_t depth_bits,
                            bool depth_ordered) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer TileRect { uvec4 value[]; } tile_rect;
layout(std430,binding=2) readonly buffer Projected { vec4 value[]; } projected;
layout(std430,binding=3) readonly buffer GaussianOffsets { uint value[]; } gaussian_offsets;
layout(std430,binding=4) writeonly buffer Keys { uint value[]; } keys;
layout(std430,binding=5) writeonly buffer Values { uint value[]; } values;
layout(std430,binding=6) readonly buffer RadixMeta { uvec4 value[]; } radix_meta;
void main() {
    uint pair_index=gl_GlobalInvocationID.x;
    if(pair_index>=radix_meta.value[0].x) return;
    uint low=0u;
    uint high=min(camera.state.x,GAUSSIAN_CAPACITY);
    while(low<high) {
        uint middle=low+(high-low)/2u;
        if(gaussian_offsets.value[middle]<=pair_index) low=middle+1u;
        else high=middle;
    }
    uint index=low-1u;
    uvec4 rectangle=tile_rect.value[index];
    uint rectangle_width=rectangle.z-rectangle.x;
    uint local_index=pair_index-gaussian_offsets.value[index];
    uint x=rectangle.x+local_index%rectangle_width;
    uint y=rectangle.y+local_index/rectangle_width;
    uint tile=y*camera.dims.x+x;
    uint depth=floatBitsToUint(projected.value[index].z);
    keys.value[pair_index]=(tile<<DEPTH_BITS)|
        (depth>>(32u-DEPTH_BITS));
    values.value[pair_index]=index;
})";
    if (depth_ordered) {
        // Binding 3 receives the rank-ordered offsets; the binary search
        // yields a depth rank which depth_vals maps back to the splat id.
        // Pairs inherit the global depth order, so the tile-only key plus
        // the sort's LSD stability reproduces a full-precision depth order.
        source=Replace(std::move(source),
            "layout(std430,binding=6) readonly buffer RadixMeta { uvec4 value[]; } radix_meta;",
            "layout(std430,binding=6) readonly buffer RadixMeta { uvec4 value[]; } radix_meta;\n"
            "layout(std430,binding=7) readonly buffer DepthVals { uint value[]; } depth_vals;");
        source=Replace(std::move(source),
            "uint index=low-1u;",
            "uint rank=low-1u;\n    uint index=depth_vals.value[rank];");
        source=Replace(std::move(source),
            "uint local_index=pair_index-gaussian_offsets.value[index];",
            "uint local_index=pair_index-gaussian_offsets.value[rank];");
        source=Replace(std::move(source),
            "uint depth=floatBitsToUint(projected.value[index].z);\n"
            "    keys.value[pair_index]=(tile<<DEPTH_BITS)|\n"
            "        (depth>>(32u-DEPTH_BITS));",
            "keys.value[pair_index]=tile;");
    }
    source=Replace(std::move(source),"GAUSSIAN_CAPACITY",
                   std::to_string(gaussian_capacity)+"u");
    (void)pair_capacity;
    return Replace(std::move(source),"DEPTH_BITS",
                   std::to_string(depth_bits)+"u");
}

std::string RangeStartsShader(uint32_t depth_bits) {
    return Replace(R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Keys { uint value[]; } keys;
layout(std430,binding=1) readonly buffer RadixMeta { uvec4 value[]; } radix_meta;
layout(std430,binding=2) writeonly buffer TileOffsets { uint value[]; } tile_offsets;
void main() {
    uint index=gl_GlobalInvocationID.x;
    uint count=radix_meta.value[0].x;
    if(index>=count) return;
    uint tile=keys.value[index]>>DEPTH_BITS;
    if(index==0u || (keys.value[index-1u]>>DEPTH_BITS)!=tile)
        tile_offsets.value[tile]=index;
})", "DEPTH_BITS", std::to_string(depth_bits)+"u");
}

std::string RangeEndsShader(uint32_t depth_bits, uint32_t tile_capacity,
                            uint32_t grid_width) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Keys { uint value[]; } keys;
layout(std430,binding=1) readonly buffer RadixMeta { uvec4 value[]; } radix_meta;
layout(std430,binding=2) readonly buffer TileOffsets { uint value[]; } tile_offsets;
layout(std430,binding=3) writeonly buffer TileCounts { uint value[]; } tile_counts;
layout(std430,binding=4) buffer Overflow { uint value; } overflow;
layout(std430,binding=5) readonly buffer Values { uint value[]; } values;
layout(std430,binding=6) readonly buffer Projected { vec4 value[]; } projected;
layout(std430,binding=7) readonly buffer ConicOpacity { vec4 value[]; } conic_opacity;
layout(std430,binding=8) writeonly buffer TileMasks { uint value[]; } tile_masks;

bool ellipse_overlaps_rect(vec3 q,float x0,float x1,float y0,float y1) {
    if(x0<=0.0 && x1>=0.0 && y0<=0.0 && y1>=0.0) return true;
    float minimum=1e30;
    float x=clamp(-q.y*y0/q.x,x0,x1);
    minimum=min(minimum,q.x*x*x+2.0*q.y*x*y0+q.z*y0*y0);
    x=clamp(-q.y*y1/q.x,x0,x1);
    minimum=min(minimum,q.x*x*x+2.0*q.y*x*y1+q.z*y1*y1);
    float y=clamp(-q.y*x0/q.z,y0,y1);
    minimum=min(minimum,q.x*x0*x0+2.0*q.y*x0*y+q.z*y*y);
    y=clamp(-q.y*x1/q.z,y0,y1);
    minimum=min(minimum,q.x*x1*x1+2.0*q.y*x1*y+q.z*y*y);
    return minimum<=1.0001;
}

uint compute_subtile_mask(uint tile, vec4 projected_value,
                          vec4 conic_value) {
    float extend_squared=2.0*log(max(conic_value.w,0.00392156862745)/
                                  0.00392156862745);
    vec3 normalized=conic_value.xyz/max(extend_squared,1e-6);
    float determinant=normalized.x*normalized.z-normalized.y*normalized.y;
    if(!(determinant>0.0)) return 0xffffu;
    vec2 half_extent=sqrt(max(vec2(0.0),normalized.zx/determinant));
    vec2 tile_origin=vec2(tile%GRID_WIDTH,tile/GRID_WIDTH)*16.0;
    vec2 center=projected_value.xy-tile_origin;
    uint mask=0u;
    for(uint sy=0u;sy<4u;sy++) {
        for(uint sx=0u;sx<4u;sx++) {
            float x0=float(sx*4u)-center.x, x1=x0+3.0;
            float y0=float(sy*4u)-center.y, y1=y0+3.0;
            bool overlap=ellipse_overlaps_rect(normalized,x0,x1,y0,y1);
            if(overlap) mask|=1u<<(sy*4u+sx);
        }
    }
    return mask;
}
void main() {
    uint index=gl_GlobalInvocationID.x;
    uint count=radix_meta.value[0].x;
    if(index>=count) return;
    uint tile=keys.value[index]>>DEPTH_BITS;
    uint id=values.value[index];
    tile_masks.value[index]=compute_subtile_mask(
            tile,projected.value[id],conic_opacity.value[id]);
    if(index+1u==count || (keys.value[index+1u]>>DEPTH_BITS)!=tile) {
        uint tile_count=index+1u-tile_offsets.value[tile];
        tile_counts.value[tile]=tile_count;
    }
})";
    source=Replace(std::move(source),"DEPTH_BITS",
                   std::to_string(depth_bits)+"u");
    source=Replace(std::move(source),"GRID_WIDTH",
                   std::to_string(grid_width)+"u");
    return Replace(std::move(source),"TILE_CAPACITY",
                   std::to_string(tile_capacity)+"u");
}

std::string SelectCameraShader(bool explicit_copy, bool randomize_views,
                               bool epoch_shuffle_views, uint32_t view_seed) {
    std::string source = R"(#version 450
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(local_size_x=1) in;
layout(std430,binding=0) writeonly buffer ActiveCamera { CameraData camera; } selected;
layout(std430,binding=1) readonly buffer CameraAtlas { CameraData camera[]; } atlas;
layout(std430,binding=2) readonly buffer OptimizerMeta { uvec4 value; } meta;
uint hash32(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}
uint gcd32(uint a,uint b) { while(b!=0u){uint r=a%b;a=b;b=r;}return a; }
uint shuffled_view(uint step,uint count) {
    uint n=max(count,1u),position=step%n,epoch=step/n;
    uint a=hash32(epoch^VIEW_SEED)|1u;
    while(gcd32(a,n)!=1u) a+=2u;
    uint b=hash32(epoch^OFFSET_SEED)%n;
    return (position*a+b)%n;
}
void main() {
    uint view=VIEW_INDEX;
    CAMERA_COPY
    selected.camera.state=uvec4(meta.value.y,view,meta.value.x,0u);
})";
    const std::string random_index =
            "hash32(meta.value.x ^ " + std::to_string(view_seed) +
            "u) % max(meta.value.z, 1u)";
    source=Replace(std::move(source),"VIEW_SEED",std::to_string(view_seed)+"u");
    source=Replace(std::move(source),"OFFSET_SEED",
                   std::to_string(view_seed ^ 0x9e3779b9u)+"u");
    const std::string epoch_index =
            "shuffled_view(meta.value.x,meta.value.z)";
    source = Replace(std::move(source), "VIEW_INDEX",
        epoch_shuffle_views ? epoch_index : randomize_views
            ? random_index
            : "meta.value.x % max(meta.value.z, 1u)");
    return Replace(std::move(source), "CAMERA_COPY",
        explicit_copy
            ? "selected.camera.world_to_camera[0]=atlas.camera[view].world_to_camera[0];\n"
              "    selected.camera.world_to_camera[1]=atlas.camera[view].world_to_camera[1];\n"
              "    selected.camera.world_to_camera[2]=atlas.camera[view].world_to_camera[2];\n"
              "    selected.camera.world_to_camera[3]=atlas.camera[view].world_to_camera[3];\n"
              "    selected.camera.intrinsics=atlas.camera[view].intrinsics;\n"
              "    selected.camera.dims=atlas.camera[view].dims;\n"
              "    selected.camera.state=atlas.camera[view].state;"
            : "selected.camera=atlas.camera[view];");
}

}  // namespace cross3dgs::detail

#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string ProjectionBackwardShader(uint32_t sh_degree,
                                     const GaussianForwardConfig* optimizer,
                                     bool reorder_sh, bool enable_igs,
                                     bool multistep_update,
                                     bool split_sh_rest,
                                     bool enable_mip3d,
                                     bool zero_sh_rest_shard_b) {
    // IGS+ accumulates the per-Gaussian screen-space gradient magnitude in
    // classic 3DGS NDC units so the split threshold is scene-independent.
    const std::string igs_accum = enable_igs ? R"(
    {
        vec4 igs_acc=igs_score.value[i];
        if(any(notEqual(d_screen,vec2(0.0)))||any(notEqual(d_conic,vec3(0.0)))){
            float igs_gx=d_screen.x*(0.5*float(camera.dims.z));
            float igs_gy=d_screen.y*(0.5*float(camera.dims.w));
            igs_acc.xy+=vec2(length(vec2(igs_gx,igs_gy)),1.0);
        }
        igs_score.value[i]=igs_acc;
    })" : "";
    std::string source=R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraBlock { CameraData camera; };
layout(std430,binding=1) readonly buffer Means { vec4 value[]; } means;
layout(std430,binding=2) readonly buffer ScalesOpacity { vec4 value[]; } scale_opacity;
layout(std430,binding=3) readonly buffer Rotations { vec4 value[]; } rotations;
layout(std430,binding=4) readonly buffer ShDc { vec4 value[]; } sh_dc;
layout(std430,binding=5) readonly buffer ShRest { vec4 value[]; } sh_rest;
layout(std430,binding=6) readonly buffer ProjectedGrad { vec4 value[]; } projected_grad;
layout(std430,binding=7) readonly buffer ConicGrad { vec4 value[]; } conic_grad;
layout(std430,binding=8) readonly buffer RgbGrad { vec4 value[]; } rgb_grad;
layout(std430,binding=9) writeonly buffer MeansGrad { vec4 value[]; } means_grad;
layout(std430,binding=10) writeonly buffer ScaleGrad { vec4 value[]; } scale_grad;
layout(std430,binding=11) writeonly buffer RotationGrad { vec4 value[]; } rotation_grad;
layout(std430,binding=12) writeonly buffer ShGrad { vec4 value[]; } sh_grad;
layout(std430,binding=13) writeonly buffer ShRestGrad { vec4 value[]; } sh_rest_grad;
FUSED_BINDINGS
IGS_BINDINGS

uint SH_LINEAR_INDEX(uint gaussian,uint packed){return gaussian*12u+packed;}
uint SH_REORDER_INDEX(uint gaussian,uint packed){
    return (gaussian&~31u)*12u+packed*32u+(gaussian&31u);
}
float sh_scalar(uint gaussian,uint scalar) {
    return sh_rest.value[SH_INDEX(gaussian,scalar/4u)][scalar%4u];
}
vec3 sh_coeff(uint gaussian,uint coefficient) {
    uint scalar=(coefficient-1u)*3u;
    return vec3(sh_scalar(gaussian,scalar),sh_scalar(gaussian,scalar+1u),
                sh_scalar(gaussian,scalar+2u));
}
void sh_basis(vec3 d,out float b[15]) {
    float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z;
    b[0]=-0.4886025119029199*y; b[1]=0.4886025119029199*z;
    b[2]=-0.4886025119029199*x; b[3]=1.0925484305920792*x*y;
    b[4]=-1.0925484305920792*y*z;
    b[5]=0.31539156525252005*(2.0*zz-xx-yy);
    b[6]=-1.0925484305920792*x*z; b[7]=0.5462742152960396*(xx-yy);
    b[8]=-0.5900435899266435*y*(3.0*xx-yy);
    b[9]=2.890611442640554*x*y*z;
    b[10]=-0.4570457994644658*y*(4.0*zz-xx-yy);
    b[11]=0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy);
    b[12]=-0.4570457994644658*x*(4.0*zz-xx-yy);
    b[13]=1.445305721320277*z*(xx-yy);
    b[14]=-0.5900435899266435*x*(xx-3.0*yy);
}
void sh_basis_gradient(vec3 d,out vec3 g[15]) {
    float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z;
    const float c1=0.4886025119029199;
    const float c2=1.0925484305920792;
    const float c3=0.31539156525252005;
    const float c4=0.5462742152960396;
    const float c5=0.5900435899266435;
    const float c6=2.890611442640554;
    const float c7=0.4570457994644658;
    const float c8=0.3731763325901154;
    const float c9=1.445305721320277;
    g[0]=vec3(0.0,-c1,0.0);
    g[1]=vec3(0.0,0.0,c1);
    g[2]=vec3(-c1,0.0,0.0);
    g[3]=vec3(c2*y,c2*x,0.0);
    g[4]=vec3(0.0,-c2*z,-c2*y);
    g[5]=vec3(-2.0*c3*x,-2.0*c3*y,4.0*c3*z);
    g[6]=vec3(-c2*z,0.0,-c2*x);
    g[7]=vec3(2.0*c4*x,-2.0*c4*y,0.0);
    g[8]=vec3(-6.0*c5*x*y,-3.0*c5*(xx-yy),0.0);
    g[9]=vec3(c6*y*z,c6*x*z,c6*x*y);
    g[10]=vec3(2.0*c7*x*y,-c7*(4.0*zz-xx-3.0*yy),
                   -8.0*c7*y*z);
    g[11]=vec3(-6.0*c8*x*z,-6.0*c8*y*z,
                   3.0*c8*(2.0*zz-xx-yy));
    g[12]=vec3(-c7*(4.0*zz-3.0*xx-yy),2.0*c7*x*y,
                   -8.0*c7*x*z);
    g[13]=vec3(2.0*c9*x*z,-2.0*c9*y*z,c9*(xx-yy));
    g[14]=vec3(-3.0*c5*(xx-yy),6.0*c5*x*y,0.0);
}

mat3 quaternion_rotation(vec4 q) {
    q=normalize(q);
    float r=q.x,x=q.y,y=q.z,z=q.w;
    return mat3(
      vec3(1.0-2.0*(y*y+z*z),2.0*(x*y+r*z),2.0*(x*z-r*y)),
      vec3(2.0*(x*y-r*z),1.0-2.0*(x*x+z*z),2.0*(y*z+r*x)),
      vec3(2.0*(x*z+r*y),2.0*(y*z-r*x),1.0-2.0*(x*x+y*y)));
}

struct Projection { vec2 screen; vec3 conic; };
Projection project_value(vec3 mean,vec3 scale,vec4 rotation) {
    vec3 view=(camera.world_to_camera*vec4(mean,1.0)).xyz;
    float fx=camera.intrinsics.x,fy=camera.intrinsics.y;
    Projection result;
    result.screen=vec2(fx*view.x/view.z+camera.intrinsics.z,
                       fy*view.y/view.z+camera.intrinsics.w)-vec2(0.5);
    mat3 r=quaternion_rotation(rotation);
    mat3 covariance=r*mat3(scale.x*scale.x,0,0,0,scale.y*scale.y,0,
                           0,0,scale.z*scale.z)*transpose(r);
    mat3 w=mat3(camera.world_to_camera);
    vec3 row0=vec3(w[0][0],w[1][0],w[2][0]);
    vec3 row1=vec3(w[0][1],w[1][1],w[2][1]);
    vec3 row2=vec3(w[0][2],w[1][2],w[2][2]);
    float iz=1.0/view.z;
    float tan_fovx=0.5*float(camera.dims.z)/fx;
    float tan_fovy=0.5*float(camera.dims.w)/fy;
    float limit_x_positive=(float(camera.dims.z)-camera.intrinsics.z)/fx+
                           0.3*tan_fovx;
    float limit_x_negative=camera.intrinsics.z/fx+0.3*tan_fovx;
    float limit_y_positive=(float(camera.dims.w)-camera.intrinsics.w)/fy+
                           0.3*tan_fovy;
    float limit_y_negative=camera.intrinsics.w/fy+0.3*tan_fovy;
    float covariance_x=clamp(view.x*iz,-limit_x_negative,
                             limit_x_positive)*view.z;
    float covariance_y=clamp(view.y*iz,-limit_y_negative,
                             limit_y_positive)*view.z;
    vec3 t0=(fx*iz)*row0+(-fx*covariance_x*iz*iz)*row2;
    vec3 t1=(fy*iz)*row1+(-fy*covariance_y*iz*iz)*row2;
    float xx=dot(t0,covariance*t0)+0.3;
    float xy=dot(t0,covariance*t1);
    float yy=dot(t1,covariance*t1)+0.3;
    result.conic=vec3(yy,-xy,xx)/(xx*yy-xy*xy);
    return result;
}
float central_difference(Projection plus,Projection minus,
                         vec2 d_screen,vec3 d_conic,float epsilon) {
    return (dot(plus.screen-minus.screen,d_screen)+
            dot(plus.conic-minus.conic,d_conic))/(2.0*epsilon);
}
FUSED_HELPERS
void main() {
    uint i=gl_GlobalInvocationID.x;
    if(i>=camera.state.x) return;
    vec3 mean=means.value[i].xyz;
    vec3 scale=scale_opacity.value[i].xyz;
    vec4 rotation=rotations.value[i];
    vec2 d_screen=projected_grad.value[i].xy;
    vec3 d_conic=conic_grad.value[i].xyz;
    vec3 d_rgb=rgb_grad.value[i].xyz;
    vec3 d_mean=vec3(0.0), d_scale=vec3(0.0);
    vec4 d_rotation=vec4(0.0);
    if(any(notEqual(d_screen,vec2(0.0))) ||
       any(notEqual(d_conic,vec3(0.0)))) {
        vec3 view=(camera.world_to_camera*vec4(mean,1.0)).xyz;
        float fx=camera.intrinsics.x,fy=camera.intrinsics.y;
        float iz=1.0/view.z,iz2=iz*iz,iz3=iz2*iz;
        mat3 world_to_view=mat3(camera.world_to_camera);
        vec3 row0=vec3(world_to_view[0][0],world_to_view[1][0],
                       world_to_view[2][0]);
        vec3 row1=vec3(world_to_view[0][1],world_to_view[1][1],
                       world_to_view[2][1]);
        vec3 row2=vec3(world_to_view[0][2],world_to_view[1][2],
                       world_to_view[2][2]);
        float tan_fovx=0.5*float(camera.dims.z)/fx;
        float tan_fovy=0.5*float(camera.dims.w)/fy;
        float limit_x_positive=
                (float(camera.dims.z)-camera.intrinsics.z)/fx+
                0.3*tan_fovx;
        float limit_x_negative=camera.intrinsics.z/fx+0.3*tan_fovx;
        float limit_y_positive=
                (float(camera.dims.w)-camera.intrinsics.w)/fy+
                0.3*tan_fovy;
        float limit_y_negative=camera.intrinsics.w/fy+0.3*tan_fovy;
        float ratio_x=view.x*iz,ratio_y=view.y*iz;
        float clamped_x=clamp(ratio_x,-limit_x_negative,limit_x_positive);
        float clamped_y=clamp(ratio_y,-limit_y_negative,limit_y_positive);
        bool interior_x=ratio_x>-limit_x_negative&&ratio_x<limit_x_positive;
        bool interior_y=ratio_y>-limit_y_negative&&ratio_y<limit_y_positive;
        vec3 t0=(fx*iz)*row0+(-fx*clamped_x*iz)*row2;
        vec3 t1=(fy*iz)*row1+(-fy*clamped_y*iz)*row2;
        mat3 rotation_matrix=quaternion_rotation(rotation);
        mat3 covariance=rotation_matrix*
            mat3(scale.x*scale.x,0,0,0,scale.y*scale.y,0,
                 0,0,scale.z*scale.z)*transpose(rotation_matrix);

        Projection current=project_value(mean,scale,rotation);
        mat2 inverse_covariance=mat2(
                vec2(current.conic.x,current.conic.y),
                vec2(current.conic.y,current.conic.z));
        mat2 inverse_gradient=mat2(
                vec2(d_conic.x,0.5*d_conic.y),
                vec2(0.5*d_conic.y,d_conic.z));
        mat2 covariance_gradient=
                -inverse_covariance*inverse_gradient*inverse_covariance;
        float d_xx=covariance_gradient[0][0];
        float d_xy=covariance_gradient[0][1]+covariance_gradient[1][0];
        float d_yy=covariance_gradient[1][1];

        vec3 covariance_t0=covariance*t0;
        vec3 covariance_t1=covariance*t1;
        vec3 d_t0=2.0*d_xx*covariance_t0+d_xy*covariance_t1;
        vec3 d_t1=d_xy*covariance_t0+2.0*d_yy*covariance_t1;
        vec3 d_view=vec3(
                d_screen.x*fx*iz,
                d_screen.y*fy*iz,
                -d_screen.x*fx*view.x*iz2-
                 d_screen.y*fy*view.y*iz2);
        if(interior_x) d_view.x+=dot(d_t0,-fx*iz2*row2);
        if(interior_y) d_view.y+=dot(d_t1,-fy*iz2*row2);
        d_view.z+=dot(d_t0,-fx*iz2*row0+
                (interior_x ? 2.0*fx*view.x*iz3 :
                              fx*clamped_x*iz2)*row2);
        d_view.z+=dot(d_t1,-fy*iz2*row1+
                (interior_y ? 2.0*fy*view.y*iz3 :
                              fy*clamped_y*iz2)*row2);
        d_mean=transpose(world_to_view)*d_view;

        mat3 covariance_world_gradient=
                d_xx*outerProduct(t0,t0)+
                0.5*d_xy*(outerProduct(t0,t1)+outerProduct(t1,t0))+
                d_yy*outerProduct(t1,t1);
        for(int axis=0;axis<3;axis++) {
            vec3 direction=rotation_matrix[axis];
            d_scale[axis]=2.0*scale[axis]*
                    dot(direction,covariance_world_gradient*direction);
        }
        mat3 scale_squared=mat3(scale.x*scale.x,0,0,
                                0,scale.y*scale.y,0,
                                0,0,scale.z*scale.z);
        mat3 d_rotation_matrix=2.0*covariance_world_gradient*
                               rotation_matrix*scale_squared;
        vec4 q=normalize(rotation);
        float r=q.x,x=q.y,y=q.z,z=q.w;
        mat3 derivative_r=mat3(vec3(0.0,2.0*z,-2.0*y),
                               vec3(-2.0*z,0.0,2.0*x),
                               vec3(2.0*y,-2.0*x,0.0));
        mat3 derivative_x=mat3(vec3(0.0,2.0*y,2.0*z),
                               vec3(2.0*y,-4.0*x,2.0*r),
                               vec3(2.0*z,-2.0*r,-4.0*x));
        mat3 derivative_y=mat3(vec3(-4.0*y,2.0*x,-2.0*r),
                               vec3(2.0*x,0.0,2.0*z),
                               vec3(2.0*r,2.0*z,-4.0*y));
        mat3 derivative_z=mat3(vec3(-4.0*z,2.0*r,2.0*x),
                               vec3(-2.0*r,-4.0*z,2.0*y),
                               vec3(2.0*x,2.0*y,0.0));
        vec4 d_q=vec4(
            dot(d_rotation_matrix[0],derivative_r[0])+
            dot(d_rotation_matrix[1],derivative_r[1])+
            dot(d_rotation_matrix[2],derivative_r[2]),
            dot(d_rotation_matrix[0],derivative_x[0])+
            dot(d_rotation_matrix[1],derivative_x[1])+
            dot(d_rotation_matrix[2],derivative_x[2]),
            dot(d_rotation_matrix[0],derivative_y[0])+
            dot(d_rotation_matrix[1],derivative_y[1])+
            dot(d_rotation_matrix[2],derivative_y[2]),
            dot(d_rotation_matrix[0],derivative_z[0])+
            dot(d_rotation_matrix[1],derivative_z[1])+
            dot(d_rotation_matrix[2],derivative_z[2]));
        float rotation_length=length(rotation);
        if(rotation_length>1e-12)
            d_rotation=(d_q-q*dot(q,d_q))/rotation_length;
    }
    if(any(isnan(d_mean)) || any(isinf(d_mean))) d_mean=vec3(0.0);
    if(any(isnan(d_scale)) || any(isinf(d_scale))) d_scale=vec3(0.0);
    if(any(isnan(d_rotation)) || any(isinf(d_rotation))) d_rotation=vec4(0.0);
    mat3 world_rotation=mat3(camera.world_to_camera);
    vec3 camera_position=-transpose(world_rotation)*camera.world_to_camera[3].xyz;
    vec3 view_direction=mean-camera_position;
    float direction_length=length(view_direction);
    vec3 direction=direction_length>1e-12 ? view_direction/direction_length
                                         : vec3(0.0,0.0,1.0);
    float basis[15]; sh_basis(direction,basis);
    uint active_degree=min(SH_DEGREE,camera.state.z/1000u);
    uint coefficient_count=active_degree==0u ? 0u : (active_degree==1u ? 3u : (active_degree==2u ? 8u : 15u));
    vec3 unclamped=vec3(0.5)+0.28209479177387814*sh_dc.value[i].xyz;
    for(uint coefficient=1u;coefficient<=coefficient_count;coefficient++)
        unclamped+=basis[coefficient-1u]*sh_coeff(i,coefficient);
    vec3 d_color=d_rgb*vec3(greaterThan(unclamped,vec3(0.0)));
    vec3 d_sh=d_color*0.28209479177387814;
    if(coefficient_count>0u && direction_length>1e-12) {
        vec3 basis_gradient[15]; sh_basis_gradient(direction,basis_gradient);
        vec3 d_direction=vec3(0.0);
        for(uint coefficient=1u;coefficient<=coefficient_count;coefficient++)
            d_direction+=dot(d_color,sh_coeff(i,coefficient))*
                         basis_gradient[coefficient-1u];
        d_mean+=(d_direction-direction*dot(direction,d_direction))/
                direction_length;
    }
    GRADIENT_SINK
    IGS_ACCUM
})";
    source=Replace(std::move(source),"SH_DEGREE",std::to_string(sh_degree)+"u");
    if (enable_mip3d) {
        source=Replace(std::move(source),
            "struct Projection { vec2 screen; vec3 conic; };",
            "float mip_f2=0.0;\n"
            "struct Projection { vec2 screen; vec3 conic; };");
        source=Replace(std::move(source),
            "    mat3 covariance=r*mat3(scale.x*scale.x,0,0,0,scale.y*scale.y,0,\n"
            "                           0,0,scale.z*scale.z)*transpose(r);",
            "    mat3 covariance=r*mat3(scale.x*scale.x+mip_f2,0,0,0,scale.y*scale.y+mip_f2,0,\n"
            "                           0,0,scale.z*scale.z+mip_f2)*transpose(r);");
        source=Replace(std::move(source),
            "    vec3 scale=scale_opacity.value[i].xyz;",
            "    vec3 scale=scale_opacity.value[i].xyz;\n"
            "    mip_f2=means.value[i].w*means.value[i].w;\n"
            "    vec3 mip_s2=scale*scale;\n"
            "    float mip_comp=sqrt(max(mip_s2.x*mip_s2.y*mip_s2.z,0.0)/\n"
            "        max((mip_s2.x+mip_f2)*(mip_s2.y+mip_f2)*(mip_s2.z+mip_f2),1e-30));");
        source=Replace(std::move(source),
            "        mat3 covariance=rotation_matrix*\n"
            "            mat3(scale.x*scale.x,0,0,0,scale.y*scale.y,0,\n"
            "                 0,0,scale.z*scale.z)*transpose(rotation_matrix);",
            "        mat3 covariance=rotation_matrix*\n"
            "            mat3(scale.x*scale.x+mip_f2,0,0,0,scale.y*scale.y+mip_f2,0,\n"
            "                 0,0,scale.z*scale.z+mip_f2)*transpose(rotation_matrix);");
        source=Replace(std::move(source),
            "        mat3 scale_squared=mat3(scale.x*scale.x,0,0,\n"
            "                                0,scale.y*scale.y,0,\n"
            "                                0,0,scale.z*scale.z);",
            "        mat3 scale_squared=mat3(scale.x*scale.x+mip_f2,0,0,\n"
            "                                0,scale.y*scale.y+mip_f2,0,\n"
            "                                0,0,scale.z*scale.z+mip_f2);");
        // Opacity compensation chain: d/ds_k of m=sqrt(prod s^2/(s^2+f^2))
        // is m*f^2/(s_k*(s_k^2+f^2)); the raster backward differentiated the
        // effective opacity o*m, so the raw-opacity gradient scales by m.
        source=Replace(std::move(source),
            "    if(any(isnan(d_mean)) || any(isinf(d_mean))) d_mean=vec3(0.0);",
            "    if(mip_f2>0.0) {\n"
            "        float d_oe=conic_grad.value[i].w*scale_opacity.value[i].w*mip_comp;\n"
            "        d_scale+=d_oe*mip_f2/max(scale*(mip_s2+vec3(mip_f2)),vec3(1e-12));\n"
            "    }\n"
            "    if(any(isnan(d_mean)) || any(isinf(d_mean))) d_mean=vec3(0.0);");
    }
    if(optimizer) {
        source=Replace(std::move(source),"layout(local_size_x=256) in;",
                       "layout(local_size_x=32) in;");
    }
    if (!optimizer) {
        source=Replace(std::move(source),"FUSED_BINDINGS","");
        source=Replace(std::move(source),"FUSED_HELPERS","");
        source=Replace(std::move(source),"IGS_BINDINGS", enable_igs
            ? "layout(std430,binding=" +
                    std::to_string(split_sh_rest ? 16u : 14u) +
                    ") buffer IgsScore { vec4 value[]; } igs_score;"
            : "");
        source=Replace(std::move(source),"IGS_ACCUM",igs_accum);
        // Multi-step Update accumulates gradients across N backward passes, so
        // the sink adds into the grad buffers (and skips the per-step zeroing of
        // inactive SH coefficients); the Adam pass zeroes them when it steps.
        const char* op = multistep_update ? "+=" : "=";
        std::string sink = std::string(R"(
    means_grad.value[i])") + op + R"(vec4(d_mean,0.0);
    scale_grad.value[i])" + op + R"(vec4(d_scale,conic_grad.value[i].w);
    rotation_grad.value[i])" + op + R"(d_rotation;
    sh_grad.value[i])" + op + R"(vec4(d_sh,0.0);
)";
        if (split_sh_rest) {
            sink += R"(    for(uint packed=0u;packed<6u;packed++) {
        sh_rest_grad.value[i*6u+packed]=vec4(0.0);
        sh_rest_grad_b.value[i*6u+packed]=vec4(0.0);
    }
    for(uint coefficient=1u;coefficient<=coefficient_count;coefficient++) {
        uint scalar=(coefficient-1u)*3u;
        vec3 value=d_color*basis[coefficient-1u];
        store_sh_rest_grad(i,scalar,value.x);
        store_sh_rest_grad(i,scalar+1u,value.y);
        store_sh_rest_grad(i,scalar+2u,value.z);
    })";
        } else {
            if (!multistep_update) {
                sink += R"(    for(uint packed=0u;packed<12u;packed++) sh_rest_grad.value[SH_INDEX(i,packed)]=vec4(0.0);
)";
            }
            sink += std::string(R"(    for(uint coefficient=1u;coefficient<=coefficient_count;coefficient++) {
        uint scalar=(coefficient-1u)*3u;
        vec3 value=d_color*basis[coefficient-1u];
        sh_rest_grad.value[SH_INDEX(i,scalar/4u)][scalar%4u])") + op + R"(value.x;
        sh_rest_grad.value[SH_INDEX(i,(scalar+1u)/4u)][(scalar+1u)%4u])" + op + R"(value.y;
        sh_rest_grad.value[SH_INDEX(i,(scalar+2u)/4u)][(scalar+2u)%4u])" + op + R"(value.z;
    })";
        }
        source=Replace(std::move(source),"GRADIENT_SINK",sink);
        if (enable_mip3d) {
            source=Replace(std::move(source),
                "vec4(d_scale,conic_grad.value[i].w)",
                "vec4(d_scale,mip_comp*conic_grad.value[i].w)");
        }
        if (split_sh_rest) {
            source=Replace(std::move(source),
                "layout(std430,binding=5) readonly buffer ShRest { vec4 value[]; } sh_rest;",
                "layout(std430,binding=5) readonly buffer ShRestA { vec4 value[]; } sh_rest;\n"
                "layout(std430,binding=14) readonly buffer ShRestB { vec4 value[]; } sh_rest_b;");
            source=Replace(std::move(source),
                "layout(std430,binding=13) writeonly buffer ShRestGrad { vec4 value[]; } sh_rest_grad;",
                "layout(std430,binding=13) writeonly buffer ShRestGradA { vec4 value[]; } sh_rest_grad;\n"
                "layout(std430,binding=15) writeonly buffer ShRestGradB { vec4 value[]; } sh_rest_grad_b;");
            source=Replace(std::move(source),
                "return sh_rest.value[SH_INDEX(gaussian,scalar/4u)][scalar%4u];",
                "uint packed=scalar/4u; uint index=gaussian*6u+(packed%6u);\n"
                "    return packed<6u?sh_rest.value[index][scalar%4u]:"
                "sh_rest_b.value[index][scalar%4u];");
            source=Replace(std::move(source),"void sh_basis(vec3 d,out float b[15]) {",
                zero_sh_rest_shard_b ? R"(
void store_sh_rest_grad(uint gaussian,uint scalar,float value) {
    uint packed=scalar/4u; uint index=gaussian*6u+(packed%6u);
    // Negative control: the shard-B write is dropped on purpose.
    if(packed<6u) sh_rest_grad.value[index][scalar%4u]=value;
}
void sh_basis(vec3 d,out float b[15]) {
)" : R"(
void store_sh_rest_grad(uint gaussian,uint scalar,float value) {
    uint packed=scalar/4u; uint index=gaussian*6u+(packed%6u);
    if(packed<6u) sh_rest_grad.value[index][scalar%4u]=value;
    else sh_rest_grad_b.value[index][scalar%4u]=value;
}
void sh_basis(vec3 d,out float b[15]) {
)");
        }
        return Replace(std::move(source),"SH_INDEX",
            reorder_sh ? "SH_REORDER_INDEX" : "SH_LINEAR_INDEX");
    }
    source=Replace(std::move(source),
                   "layout(std430,binding=1) readonly buffer Means",
                   "layout(std430,binding=1) buffer Means");
    source=Replace(std::move(source),
                   "layout(std430,binding=2) readonly buffer ScalesOpacity",
                   "layout(std430,binding=2) buffer ScalesOpacity");
    source=Replace(std::move(source),
                   "layout(std430,binding=3) readonly buffer Rotations",
                   "layout(std430,binding=3) buffer Rotations");
    source=Replace(std::move(source),
                   "layout(std430,binding=4) readonly buffer ShDc",
                   "layout(std430,binding=4) buffer ShDc");
    source=Replace(std::move(source),
                   "layout(std430,binding=5) readonly buffer ShRest",
                   "layout(std430,binding=5) buffer ShRest");
    source=Replace(std::move(source),"FUSED_BINDINGS",R"(
layout(std430,binding=14) buffer MeansM { MOMENT_TYPE value[]; } means_m;
layout(std430,binding=15) buffer MeansV { MOMENT_TYPE value[]; } means_v;
layout(std430,binding=16) buffer ScaleM { MOMENT_TYPE value[]; } scale_m;
layout(std430,binding=17) buffer ScaleV { MOMENT_TYPE value[]; } scale_v;
layout(std430,binding=18) buffer RotationM { MOMENT_TYPE value[]; } rotation_m;
layout(std430,binding=19) buffer RotationV { MOMENT_TYPE value[]; } rotation_v;
layout(std430,binding=20) buffer ShM { MOMENT_TYPE value[]; } sh_m;
layout(std430,binding=21) buffer ShV { MOMENT_TYPE value[]; } sh_v;
layout(std430,binding=22) buffer ShRestM { MOMENT_TYPE value[]; } sh_rest_m;
layout(std430,binding=23) buffer ShRestV { MOMENT_TYPE value[]; } sh_rest_v;
layout(std430,binding=24) readonly buffer OptimizerMeta { uvec4 value; } meta;
)");
    source=Replace(std::move(source),"MOMENT_TYPE",
                   optimizer->fp32_adam_moments ? "vec4" : "uvec2");
    const std::string moment_helpers = optimizer->fp32_adam_moments ? R"(
vec4 load_moment(vec4 value) { return value; }
vec4 store_moment(vec4 value) { return value; }
)" : R"(
vec4 load_moment(uvec2 packed) {
    uvec4 half_bits=uvec4(packed.x&0xffffu,packed.x>>16u,
                          packed.y&0xffffu,packed.y>>16u);
    return uintBitsToFloat(half_bits<<16u);
}
uvec2 store_moment(vec4 value) {
    uvec4 bits=floatBitsToUint(value);
    bits+=uvec4(0x7fffu)+((bits>>16u)&uvec4(1u));
    uvec4 half_bits=bits>>16u;
    return uvec2(half_bits.x|(half_bits.y<<16u),
                 half_bits.z|(half_bits.w<<16u));
}
)";
    source=Replace(std::move(source),"FUSED_HELPERS",std::string(R"(
void adam_values(vec4 x,vec4 g,vec4 old_m,vec4 old_v,vec4 lr,
                 float inverse_bias_m,float inverse_bias_v,
                 out vec4 new_x,out vec4 new_m,out vec4 new_v) {
    new_m=mix(old_m,g,vec4(0.1));
    new_v=mix(old_v,g*g,vec4(0.001));
    vec4 corrected_m=new_m*inverse_bias_m;
    vec4 corrected_v=new_v*inverse_bias_v;
    new_x=x-lr*corrected_m/(sqrt(corrected_v)+vec4(1e-15));
}
)") + moment_helpers);
    source=Replace(std::move(source),"GRADIENT_SINK",R"(
    float step=float(meta.value.x+1u);
    float inverse_bias_m=1.0/(1.0-pow(0.9,step));
    float inverse_bias_v=1.0/(1.0-pow(0.999,step));
    vec4 x,m,v;
    vec4 means_lr=vec4(vec3(MEANS_LR*pow(MEANS_LR_RATIO,
        min(float(meta.value.x)/float(MAX_STEPS),1.0))),0.0);
    adam_values(means.value[i],vec4(d_mean,0.0),load_moment(means_m.value[i]),
                load_moment(means_v.value[i]),means_lr,inverse_bias_m,
                inverse_bias_v,x,m,v);
    means.value[i]=x;means_m.value[i]=store_moment(m);
    means_v.value[i]=store_moment(v);

    vec4 physical=scale_opacity.value[i];
    vec4 scale_gradient=vec4(d_scale,conic_grad.value[i].w)+
        vec4(vec3(SCALE_REG/(3.0*float(meta.value.y))),
             OPACITY_REG/float(meta.value.y));
    vec4 unconstrained=vec4(log(max(physical.xyz,vec3(1e-8))),
        log(max(physical.w,1e-8)/max(1.0-physical.w,1e-8)));
    scale_gradient=vec4(scale_gradient.xyz*physical.xyz,
        scale_gradient.w*physical.w*(1.0-physical.w));
    vec4 scale_learning_rate=vec4(vec3(SCALES_LR*pow(SCALES_LR_RATIO,
        min(float(meta.value.x)/float(MAX_STEPS),1.0))),OPACITY_LR);
    adam_values(unconstrained,scale_gradient,load_moment(scale_m.value[i]),
                load_moment(scale_v.value[i]),
                scale_learning_rate,inverse_bias_m,inverse_bias_v,
                x,m,v);
    scale_opacity.value[i]=vec4(exp(x.xyz),1.0/(1.0+exp(-x.w)));
    scale_m.value[i]=store_moment(m);scale_v.value[i]=store_moment(v);

    adam_values(rotations.value[i],d_rotation,load_moment(rotation_m.value[i]),
                load_moment(rotation_v.value[i]),vec4(ROTATION_LR),inverse_bias_m,
                inverse_bias_v,x,m,v);
    float rotation_length_squared=dot(x,x);
    rotations.value[i]=rotation_length_squared>1e-12
        ? x*inversesqrt(rotation_length_squared) : vec4(1.0,0.0,0.0,0.0);
    rotation_m.value[i]=store_moment(m);rotation_v.value[i]=store_moment(v);

    adam_values(sh_dc.value[i],vec4(d_sh,0.0),load_moment(sh_m.value[i]),
                load_moment(sh_v.value[i]),
                vec4(SH_DC_LR,SH_DC_LR,SH_DC_LR,0.0),inverse_bias_m,
                inverse_bias_v,x,m,v);
    sh_dc.value[i]=x;sh_m.value[i]=store_moment(m);sh_v.value[i]=store_moment(v);

    vec4 rest_gradient[12];
    for(uint packed=0u;packed<12u;packed++) rest_gradient[packed]=vec4(0.0);
    for(uint coefficient=1u;coefficient<=coefficient_count;coefficient++) {
        uint scalar=(coefficient-1u)*3u;
        vec3 value=d_color*basis[coefficient-1u];
        rest_gradient[scalar/4u][scalar%4u]=value.x;
        rest_gradient[(scalar+1u)/4u][(scalar+1u)%4u]=value.y;
        rest_gradient[(scalar+2u)/4u][(scalar+2u)%4u]=value.z;
    }
    for(uint packed=0u;packed<12u;packed++) {
        uint index=SH_INDEX(i,packed);
        adam_values(sh_rest.value[index],rest_gradient[packed],
                    load_moment(sh_rest_m.value[index]),
                    load_moment(sh_rest_v.value[index]),
                    vec4(SH_REST_LR),inverse_bias_m,inverse_bias_v,x,m,v);
        sh_rest.value[index]=x;sh_rest_m.value[index]=store_moment(m);
        sh_rest_v.value[index]=store_moment(v);
    })");
    source=Replace(std::move(source),"MEANS_LR_RATIO",
                   std::to_string(optimizer->means_lr_final/optimizer->means_lr));
    source=Replace(std::move(source),"MEANS_LR",std::to_string(optimizer->means_lr));
    source=Replace(std::move(source),"MAX_STEPS",std::to_string(optimizer->max_steps)+"u");
    source=Replace(std::move(source),"SCALE_REG",std::to_string(optimizer->scale_regularizer));
    source=Replace(std::move(source),"OPACITY_REG",std::to_string(optimizer->opacity_regularizer));
    source=Replace(std::move(source),"SCALES_LR_RATIO",
                   std::to_string(optimizer->scales_lr_final>0.0f
                       ? optimizer->scales_lr_final/optimizer->scales_lr
                       : 1.0f));
    source=Replace(std::move(source),"SCALES_LR",std::to_string(optimizer->scales_lr));
    source=Replace(std::move(source),"OPACITY_LR",std::to_string(optimizer->opacity_lr));
    source=Replace(std::move(source),"ROTATION_LR",std::to_string(optimizer->rotation_lr));
    source=Replace(std::move(source),"SH_DC_LR",std::to_string(optimizer->sh_dc_lr));
    source=Replace(std::move(source),"SH_REST_LR",std::to_string(optimizer->sh_rest_lr));
    source=Replace(std::move(source),"IGS_BINDINGS", enable_igs
        ? "layout(std430,binding=25) buffer IgsScore { vec4 value[]; } igs_score;"
        : "");
    source=Replace(std::move(source),"IGS_ACCUM",igs_accum);
    if (enable_mip3d) {
        source=Replace(std::move(source),
            "vec4(d_scale,conic_grad.value[i].w)",
            "vec4(d_scale,mip_comp*conic_grad.value[i].w)");
    }
    return Replace(std::move(source),"SH_INDEX",
        reorder_sh ? "SH_REORDER_INDEX" : "SH_LINEAR_INDEX");
}


}  // namespace cross3dgs::detail

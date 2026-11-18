#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string SsimForwardShader() {
    return R"(#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer CameraBlock {
    mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state;
} camera;
layout(std430,binding=1) readonly buffer Color { vec4 value[]; } color;
layout(std430,binding=2) readonly buffer Target { vec4 value[]; } target;
layout(std430,binding=3) writeonly buffer SsimMap { vec4 value[]; } ssim_map;
layout(std430,binding=4) readonly buffer OptimizerMeta { uvec4 value; } meta;
const float weights[11]=float[](0.0010283801,0.0075987582,0.0360007733,
  0.1093606874,0.2130055279,0.2660117149,0.2130055279,0.1093606874,
  0.0360007733,0.0075987582,0.0010283801);
void main() {
    uvec2 p=gl_GlobalInvocationID.xy;
    uint width=camera.dims.z,height=camera.dims.w;
    if(p.x>=width||p.y>=height) return;
    uint pixel=p.y*width+p.x;
    uint view=camera.state.y;
    for(uint channel=0u;channel<3u;channel++) {
        float mx=0.0,my=0.0,mxx=0.0,myy=0.0,mxy=0.0;
        for(int dy=-5;dy<=5;dy++) for(int dx=-5;dx<=5;dx++) {
            int x=int(p.x)+dx,y=int(p.y)+dy;
            if(x<0||y<0||x>=int(width)||y>=int(height)) continue;
            float weight=weights[dx+5]*weights[dy+5];
            uint index=uint(y)*width+uint(x);
            float a=color.value[index][channel];
            float b=target.value[view*width*height+index][channel];
            mx+=weight*a; my+=weight*b; mxx+=weight*a*a;
            myy+=weight*b*b; mxy+=weight*a*b;
        }
        float vx=mxx-mx*mx,vy=myy-my*my,cov=mxy-mx*my;
        float A=mx*mx+my*my+0.0001;
        float B=vx+vy+0.0009;
        float C=2.0*mx*my+0.0001;
        float D=2.0*cov+0.0009;
        float ssim=C*D/(A*B);
        float d_mu=(2.0*my*D)/(A*B)-(2.0*my*C)/(A*B)-
          (2.0*mx*C*D)/(A*A*B)+(2.0*mx*C*D)/(A*B*B);
        float d_var=-C*D/(A*B*B);
        float d_cov=2.0*C/(A*B);
        ssim_map.value[pixel*3u+channel]=vec4(ssim,d_mu,d_var,d_cov);
    }
})";
}

std::string SsimBackwardShader(float l1_weight,float ssim_weight) {
    std::string source=R"(#version 450
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer CameraBlock {
    mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state;
} camera;
layout(std430,binding=1) readonly buffer Color { vec4 value[]; } color;
layout(std430,binding=2) readonly buffer Target { vec4 value[]; } target;
layout(std430,binding=3) readonly buffer SsimMap { vec4 value[]; } ssim_map;
layout(std430,binding=4) writeonly buffer PixelGrad { vec4 value[]; } pixel_grad;
layout(std430,binding=5) writeonly buffer Loss { float value[]; } loss_value;
layout(std430,binding=6) readonly buffer OptimizerMeta { uvec4 value; } meta;
const float weights[11]=float[](0.0010283801,0.0075987582,0.0360007733,
  0.1093606874,0.2130055279,0.2660117149,0.2130055279,0.1093606874,
  0.0360007733,0.0075987582,0.0010283801);
void main() {
    uvec2 p=gl_GlobalInvocationID.xy;
    uint width=camera.dims.z,height=camera.dims.w;
    if(p.x>=width||p.y>=height) return;
    uint pixel=p.y*width+p.x;
    uint view=camera.state.y;
    vec3 rendered=color.value[pixel].xyz;
    vec3 wanted=target.value[view*width*height+pixel].xyz;
    vec3 gradient=vec3(0.0),center_ssim=vec3(0.0);
    for(uint channel=0u;channel<3u;channel++) {
        float sum_mu=0.0,sum_var=0.0,sum_cov=0.0;
        for(int dy=-5;dy<=5;dy++) for(int dx=-5;dx<=5;dx++) {
            int x=int(p.x)+dx,y=int(p.y)+dy;
            if(x<5||y<5||x>=int(width)-5||y>=int(height)-5) continue;
            float weight=weights[dx+5]*weights[dy+5];
            vec4 derivative=ssim_map.value[(uint(y)*width+uint(x))*3u+channel];
            sum_mu+=weight*derivative.y;
            sum_var+=weight*derivative.z;
            sum_cov+=weight*derivative.w;
        }
        float l1=sign(rendered[channel]-wanted[channel])*
          L1_WEIGHT/(float(width)*float(height)*3.0);
        float ssim=(sum_mu+2.0*rendered[channel]*sum_var+
                    wanted[channel]*sum_cov)*
          (-SSIM_WEIGHT/(3.0*float((width-10u)*(height-10u))));
        float clamp_derivative=rendered[channel]>0.0 ? 1.0 : 0.0;
        gradient[channel]=(l1+ssim)*clamp_derivative;
        center_ssim[channel]=ssim_map.value[pixel*3u+channel].x;
    }
    pixel_grad.value[pixel]=vec4(gradient,0.0);
    float ssim_loss=0.0;
    if(p.x>=5u&&p.y>=5u&&p.x<width-5u&&p.y<height-5u)
      ssim_loss=SSIM_WEIGHT*(1.0-(center_ssim.x+center_ssim.y+center_ssim.z)/3.0)*
        float(width*height)/float((width-10u)*(height-10u));
    loss_value.value[pixel]=L1_WEIGHT*(abs(rendered.x-wanted.x)+
      abs(rendered.y-wanted.y)+abs(rendered.z-wanted.z))/3.0+ssim_loss;
})";
    source=Replace(std::move(source),"L1_WEIGHT",std::to_string(l1_weight));
    return Replace(std::move(source),"SSIM_WEIGHT",std::to_string(ssim_weight));
}

std::string LegacySeparableSsimForwardShader(bool byte_targets,
                                             bool stream_target,
                                             size_t target_pixels) {
    return ApplyTargetStorage(R"(#version 450
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer CameraBlock {mat4 w;vec4 i;uvec4 dims;uvec4 state;} camera;
layout(std430,binding=1) readonly buffer Color {vec4 value[];} color;
layout(std430,binding=2) readonly TARGET_STORAGE_DECL
layout(std430,binding=3) writeonly buffer SsimMap {vec4 value[];} ssim_map;
layout(std430,binding=4) readonly buffer OptimizerMeta {uvec4 value;} meta;
TARGET_FETCH_DEFINE
const float weights[6]=float[](0.0010283801,0.0075987582,0.0360007733,
  0.1093606874,0.2130055279,0.2660117149);
shared float tile_data[26][26][2];
shared float horizontal[26][16][5];
void main(){
 uint lx=gl_LocalInvocationID.x,ly=gl_LocalInvocationID.y;
 uint gx=gl_WorkGroupID.x*16u+lx,gy=gl_WorkGroupID.y*16u+ly;
 uint width=camera.dims.z,height=camera.dims.w;
 uint view=camera.state.y,lane=ly*16u+lx;
 for(uint channel=0u;channel<3u;channel++){
  [[unroll]] for(uint step=0u;step<3u;step++){
   uint tid=step*256u+lane;if(tid<676u){uint y=tid/26u,x=tid%26u;
    int px=int(gl_WorkGroupID.x*16u+x)-5,py=int(gl_WorkGroupID.y*16u+y)-5;
    float a=0.0,b=0.0;if(px>=0&&py>=0&&px<int(width)&&py<int(height)){
     uint index=uint(py)*width+uint(px);a=color.value[index][channel];
     b=TARGET_FETCH(view*width*height+index)[channel];}
    tile_data[y][x][0]=a;tile_data[y][x][1]=b;}}
  barrier();
  [[unroll]] for(uint pass=0u;pass<2u;pass++){uint row=ly+pass*16u;if(row<26u){
   uint x=lx+5u;float sx=0,sy=0,sxx=0,syy=0,sxy=0;
   [[unroll]] for(uint d=1u;d<=5u;d++){float weight=weights[5u-d];
    float xl=tile_data[row][x-d][0],xr=tile_data[row][x+d][0];
    float yl=tile_data[row][x-d][1],yr=tile_data[row][x+d][1];
    sx+=(xl+xr)*weight;sy+=(yl+yr)*weight;sxx+=(xl*xl+xr*xr)*weight;
    syy+=(yl*yl+yr*yr)*weight;sxy+=(xl*yl+xr*yr)*weight;}
   float wc=weights[5],xc=tile_data[row][x][0],yc=tile_data[row][x][1];
   horizontal[row][lx][0]=sx+xc*wc;horizontal[row][lx][1]=sxx+xc*xc*wc;
   horizontal[row][lx][2]=sy+yc*wc;horizontal[row][lx][3]=syy+yc*yc*wc;
   horizontal[row][lx][4]=sxy+xc*yc*wc;}}
  barrier();
  if(gx<width&&gy<height){uint row=ly+5u;float o[5]=float[](0,0,0,0,0);
   [[unroll]] for(uint d=1u;d<=5u;d++){float weight=weights[5u-d];
    [[unroll]] for(uint k=0u;k<5u;k++)o[k]+=(horizontal[row-d][lx][k]+horizontal[row+d][lx][k])*weight;}
   [[unroll]] for(uint k=0u;k<5u;k++)o[k]+=horizontal[row][lx][k]*weights[5];
   float vx=o[1]-o[0]*o[0],vy=o[3]-o[2]*o[2],cov=o[4]-o[0]*o[2];
   float A=o[0]*o[0]+o[2]*o[2]+0.0001,B=vx+vy+0.0009;
   float C=2.0*o[0]*o[2]+0.0001,D=2.0*cov+0.0009;
   float ssim=C*D/(A*B);
   float dmu=(2.0*o[2]*D)/(A*B)-(2.0*o[2]*C)/(A*B)-
     (2.0*o[0]*C*D)/(A*A*B)+(2.0*o[0]*C*D)/(A*B*B);
   ssim_map.value[(gy*width+gx)*3u+channel]=vec4(ssim,dmu,-C*D/(A*B*B),2.0*C/(A*B));}
  barrier();
 }
})", byte_targets, stream_target, target_pixels);
}

std::string LegacySeparableSsimBackwardShader(
        float l1_weight, float ssim_weight, bool byte_targets,
        bool stream_target, size_t target_pixels,
        bool full_image_ssim) {
 std::string source=ApplyTargetStorage(R"(#version 450
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer CameraBlock {mat4 w;vec4 i;uvec4 dims;uvec4 state;} camera;
layout(std430,binding=1) readonly buffer Color {vec4 value[];} color;
layout(std430,binding=2) readonly TARGET_STORAGE_DECL
layout(std430,binding=3) readonly buffer SsimMap {vec4 value[];} ssim_map;
layout(std430,binding=4) buffer PixelGrad {vec4 value[];} pixel_grad;
layout(std430,binding=5) writeonly buffer Loss {float value[];} loss_value;
layout(std430,binding=6) readonly buffer OptimizerMeta {uvec4 value;} meta;
TARGET_FETCH_DEFINE
const float weights[6]=float[](0.0010283801,0.0075987582,0.0360007733,
  0.1093606874,0.2130055279,0.2660117149);
shared float derivatives[3][26][26];shared float horizontal[26][16][3];
void main(){uint lx=gl_LocalInvocationID.x,ly=gl_LocalInvocationID.y;
 uint gx=gl_WorkGroupID.x*16u+lx,gy=gl_WorkGroupID.y*16u+ly;
 uint width=camera.dims.z,height=camera.dims.w,view=camera.state.y;
 uint lane=ly*16u+lx;vec3 gradient=vec3(0),center_ssim=vec3(0);
 for(uint channel=0u;channel<3u;channel++){
  [[unroll]] for(uint step=0u;step<3u;step++){uint tid=step*256u+lane;if(tid<676u){
   uint y=tid/26u,x=tid%26u;int px=int(gl_WorkGroupID.x*16u+x)-5;
   int py=int(gl_WorkGroupID.y*16u+y)-5;vec4 d=vec4(0);
   if(SSIM_GATHER_PRED)
    d=ssim_map.value[(uint(py)*width+uint(px))*3u+channel];
   derivatives[0][y][x]=d.y;derivatives[1][y][x]=d.z;derivatives[2][y][x]=d.w;}}
  barrier();
  [[unroll]] for(uint pass=0u;pass<2u;pass++){uint row=ly+pass*16u;if(row<26u){uint x=lx+5u;
   float a=0,b=0,c=0;[[unroll]] for(uint d=1u;d<=5u;d++){float w=weights[5u-d];
    a+=(derivatives[0][row][x-d]+derivatives[0][row][x+d])*w;
    b+=(derivatives[1][row][x-d]+derivatives[1][row][x+d])*w;
    c+=(derivatives[2][row][x-d]+derivatives[2][row][x+d])*w;}
   horizontal[row][lx][0]=a+derivatives[0][row][x]*weights[5];
   horizontal[row][lx][1]=b+derivatives[1][row][x]*weights[5];
   horizontal[row][lx][2]=c+derivatives[2][row][x]*weights[5];}}
  barrier();
  if(gx<width&&gy<height){uint row=ly+5u,index=gy*width+gx;vec3 sum=vec3(0);
   [[unroll]] for(uint d=1u;d<=5u;d++){float w=weights[5u-d];
    sum+=(vec3(horizontal[row-d][lx][0],horizontal[row-d][lx][1],horizontal[row-d][lx][2])+
          vec3(horizontal[row+d][lx][0],horizontal[row+d][lx][1],horizontal[row+d][lx][2]))*w;}
   sum+=vec3(horizontal[row][lx][0],horizontal[row][lx][1],horizontal[row][lx][2])*weights[5];
   float rendered=color.value[index][channel];
   float wanted=TARGET_FETCH(view*width*height+index)[channel];
   float l1=sign(rendered-wanted)*L1_WEIGHT/(float(width)*float(height)*3.0);
   float ssim=(sum.x+2.0*rendered*sum.y+wanted*sum.z)*
    (-SSIM_WEIGHT/(3.0*SSIM_NORM));
   gradient[channel]=(l1+ssim)SSIM_GRAD_GATE;
   center_ssim[channel]=ssim_map.value[index*3u+channel].x;}
  barrier();}
 if(gx<width&&gy<height){uint index=gy*width+gx;pixel_grad.value[index]=vec4(gradient,0);
  vec3 rendered=color.value[index].xyz,wanted=TARGET_FETCH(view*width*height+index).xyz;
  SSIM_LOSS_STMT
  loss_value.value[index]=L1_WEIGHT*(abs(rendered.x-wanted.x)+abs(rendered.y-wanted.y)+abs(rendered.z-wanted.z))/3.0+sl;}
})", byte_targets, stream_target, target_pixels);
 if (full_image_ssim) {
  source=Replace(std::move(source),"SSIM_GATHER_PRED",
    "px>=0&&py>=0&&px<int(width)&&py<int(height)");
  source=Replace(std::move(source),"SSIM_NORM",
    "(float(width)*float(height))");
  source=Replace(std::move(source),"SSIM_GRAD_GATE","");
  source=Replace(std::move(source),"SSIM_LOSS_STMT",
    "float sl=SSIM_WEIGHT*(1.0-(center_ssim.x+center_ssim.y+center_ssim.z)/3.0);");
 } else {
  source=Replace(std::move(source),"SSIM_GATHER_PRED",
    "px>=5&&py>=5&&px<int(width)-5&&py<int(height)-5");
  source=Replace(std::move(source),"SSIM_NORM",
    "float((width-10u)*(height-10u))");
  source=Replace(std::move(source),"SSIM_GRAD_GATE","*float(rendered>0.0)");
  source=Replace(std::move(source),"SSIM_LOSS_STMT",
    "float sl=0;if(gx>=5u&&gy>=5u&&gx<width-5u&&gy<height-5u)"
    "sl=SSIM_WEIGHT*(1.0-(center_ssim.x+center_ssim.y+center_ssim.z)/3.0)*"
    "float(width*height)/float((width-10u)*(height-10u));");
 }
 source=Replace(std::move(source),"L1_WEIGHT",std::to_string(l1_weight));
 return Replace(std::move(source),"SSIM_WEIGHT",std::to_string(ssim_weight));
}

std::string AdrenoSeparableSsimForwardShader(bool byte_targets,
                                       bool stream_target,
                                       size_t target_pixels,
                                       bool adreno_channel_split) {
    std::string source = ApplyTargetStorage(R"(#version 450
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer CameraBlock {mat4 w;vec4 i;uvec4 dims;uvec4 state;} camera;
layout(std430,binding=1) readonly buffer Color {vec4 value[];} color;
layout(std430,binding=2) readonly TARGET_STORAGE_DECL
layout(std430,binding=3) writeonly buffer SsimMap {vec4 value[];} ssim_map;
layout(std430,binding=4) readonly buffer OptimizerMeta {uvec4 value;} meta;
TARGET_FETCH_DEFINE
const float weights[6]=float[](0.0010283801,0.0075987582,0.0360007733,
  0.1093606874,0.2130055279,0.2660117149);
shared float tile_color[26][26];
shared float tile_target[26][26];
shared float horizontal[5][26][16];
void main(){
 uint lx=gl_LocalInvocationID.x,ly=gl_LocalInvocationID.y;
 uint gx=gl_WorkGroupID.x*16u+lx,gy=gl_WorkGroupID.y*16u+ly;
 uint width=camera.dims.z,height=camera.dims.w;
 uint view=camera.state.y,lane=ly*16u+lx;
 CHANNEL_OPEN
  [[unroll]] for(uint step=0u;step<3u;step++){
   uint tid=step*256u+lane;if(tid<676u){uint y=tid/26u,x=tid%26u;
    int px=int(gl_WorkGroupID.x*16u+x)-5,py=int(gl_WorkGroupID.y*16u+y)-5;
    float a=0.0,b=0.0;if(px>=0&&py>=0&&px<int(width)&&py<int(height)){
     uint index=uint(py)*width+uint(px);a=color.value[index][channel];
     b=TARGET_FETCH(view*width*height+index)[channel];}
    tile_color[y][x]=a;tile_target[y][x]=b;}}
  barrier();
  [[unroll]] for(uint pass=0u;pass<2u;pass++){uint row=ly+pass*16u;if(row<26u){
   uint x=lx+5u;float sx=0,sy=0,sxx=0,syy=0,sxy=0;
   [[unroll]] for(uint d=1u;d<=5u;d++){float weight=weights[5u-d];
    float xl=tile_color[row][x-d],xr=tile_color[row][x+d];
    float yl=tile_target[row][x-d],yr=tile_target[row][x+d];
    sx+=(xl+xr)*weight;sy+=(yl+yr)*weight;sxx+=(xl*xl+xr*xr)*weight;
    syy+=(yl*yl+yr*yr)*weight;sxy+=(xl*yl+xr*yr)*weight;}
   float wc=weights[5],xc=tile_color[row][x],yc=tile_target[row][x];
   horizontal[0][row][lx]=sx+xc*wc;horizontal[1][row][lx]=sxx+xc*xc*wc;
   horizontal[2][row][lx]=sy+yc*wc;horizontal[3][row][lx]=syy+yc*yc*wc;
   horizontal[4][row][lx]=sxy+xc*yc*wc;}}
  barrier();
  if(gx<width&&gy<height){uint row=ly+5u;float o[5]=float[](0,0,0,0,0);
   [[unroll]] for(uint d=1u;d<=5u;d++){float weight=weights[5u-d];
    [[unroll]] for(uint k=0u;k<5u;k++)o[k]+=(horizontal[k][row-d][lx]+horizontal[k][row+d][lx])*weight;}
   [[unroll]] for(uint k=0u;k<5u;k++)o[k]+=horizontal[k][row][lx]*weights[5];
   float vx=o[1]-o[0]*o[0],vy=o[3]-o[2]*o[2],cov=o[4]-o[0]*o[2];
   float A=o[0]*o[0]+o[2]*o[2]+0.0001,B=vx+vy+0.0009;
   float C=2.0*o[0]*o[2]+0.0001,D=2.0*cov+0.0009;
   float ssim=C*D/(A*B);
   float dmu=(2.0*o[2]*D)/(A*B)-(2.0*o[2]*C)/(A*B)-
     (2.0*o[0]*C*D)/(A*A*B)+(2.0*o[0]*C*D)/(A*B*B);
   ssim_map.value[(gy*width+gx)*3u+channel]=vec4(ssim,dmu,-C*D/(A*B*B),2.0*C/(A*B));}
 CHANNEL_CLOSE
})", byte_targets, stream_target, target_pixels);
    if (adreno_channel_split) {
        source=Replace(std::move(source),"CHANNEL_OPEN",
            "uint channel=gl_WorkGroupID.z;\n if(channel<3u){");
        return Replace(std::move(source),"CHANNEL_CLOSE","}");
    }
    source=Replace(std::move(source),"CHANNEL_OPEN",
        "[[unroll]] for(uint channel=0u;channel<3u;channel++){");
    return Replace(std::move(source),"CHANNEL_CLOSE","  barrier();\n }");
}

std::string AdrenoSeparableSsimBackwardShader(float l1_weight,float ssim_weight,
                                        bool byte_targets,
                                        bool stream_target,
                                        size_t target_pixels,
                                        bool full_image_ssim,
                                        bool adreno_channel_split) {
 std::string source=ApplyTargetStorage(R"(#version 450
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x=16,local_size_y=16) in;
layout(std430,binding=0) readonly buffer CameraBlock {mat4 w;vec4 i;uvec4 dims;uvec4 state;} camera;
layout(std430,binding=1) readonly buffer Color {vec4 value[];} color;
layout(std430,binding=2) readonly TARGET_STORAGE_DECL
layout(std430,binding=3) readonly buffer SsimMap {vec4 value[];} ssim_map;
layout(std430,binding=4) buffer PixelGrad {vec4 value[];} pixel_grad;
layout(std430,binding=5) writeonly buffer Loss {float value[];} loss_value;
layout(std430,binding=6) readonly buffer OptimizerMeta {uvec4 value;} meta;
TARGET_FETCH_DEFINE
const float weights[6]=float[](0.0010283801,0.0075987582,0.0360007733,
  0.1093606874,0.2130055279,0.2660117149);
shared float derivatives[3][26][26];shared float horizontal[3][26][16];
void main(){uint lx=gl_LocalInvocationID.x,ly=gl_LocalInvocationID.y;
 uint gx=gl_WorkGroupID.x*16u+lx,gy=gl_WorkGroupID.y*16u+ly;
 uint width=camera.dims.z,height=camera.dims.w,view=camera.state.y;
 uint lane=ly*16u+lx;CHANNEL_OPEN
  [[unroll]] for(uint step=0u;step<3u;step++){uint tid=step*256u+lane;if(tid<676u){
   uint y=tid/26u,x=tid%26u;int px=int(gl_WorkGroupID.x*16u+x)-5;
   int py=int(gl_WorkGroupID.y*16u+y)-5;vec4 d=vec4(0);
   if(SSIM_GATHER_PRED)
    d=ssim_map.value[(uint(py)*width+uint(px))*3u+channel];
   derivatives[0][y][x]=d.y;derivatives[1][y][x]=d.z;derivatives[2][y][x]=d.w;}}
  barrier();
  [[unroll]] for(uint pass=0u;pass<2u;pass++){uint row=ly+pass*16u;if(row<26u){uint x=lx+5u;
   float a=0,b=0,c=0;[[unroll]] for(uint d=1u;d<=5u;d++){float w=weights[5u-d];
    a+=(derivatives[0][row][x-d]+derivatives[0][row][x+d])*w;
    b+=(derivatives[1][row][x-d]+derivatives[1][row][x+d])*w;
    c+=(derivatives[2][row][x-d]+derivatives[2][row][x+d])*w;}
   horizontal[0][row][lx]=a+derivatives[0][row][x]*weights[5];
   horizontal[1][row][lx]=b+derivatives[1][row][x]*weights[5];
   horizontal[2][row][lx]=c+derivatives[2][row][x]*weights[5];}}
  barrier();
  if(gx<width&&gy<height){uint row=ly+5u,index=gy*width+gx;vec3 sum=vec3(0);
   [[unroll]] for(uint d=1u;d<=5u;d++){float w=weights[5u-d];
    sum+=(vec3(horizontal[0][row-d][lx],horizontal[1][row-d][lx],horizontal[2][row-d][lx])+
          vec3(horizontal[0][row+d][lx],horizontal[1][row+d][lx],horizontal[2][row+d][lx]))*w;}
   sum+=vec3(horizontal[0][row][lx],horizontal[1][row][lx],horizontal[2][row][lx])*weights[5];
   float rendered=color.value[index][channel];
   float wanted=TARGET_FETCH(view*width*height+index)[channel];
   float l1=sign(rendered-wanted)*L1_WEIGHT/(float(width)*float(height)*3.0);
   float ssim=(sum.x+2.0*rendered*sum.y+wanted*sum.z)*
    (-SSIM_WEIGHT/(3.0*SSIM_NORM));
   pixel_grad.value[index][channel]=(l1+ssim)SSIM_GRAD_GATE;
   if(channel==2u)SSIM_LOSS_STMT_WRITE;}
 CHANNEL_CLOSE
})", byte_targets, stream_target, target_pixels);
    if (adreno_channel_split) {
        source=Replace(std::move(source),"CHANNEL_OPEN",
            "uint channel=gl_WorkGroupID.z;\n if(channel<3u){");
        source=Replace(std::move(source),"CHANNEL_CLOSE","}");
    } else {
        source=Replace(std::move(source),"CHANNEL_OPEN",
            "[[unroll]] for(uint channel=0u;channel<3u;channel++){");
        source=Replace(std::move(source),"CHANNEL_CLOSE","  barrier();\n }");
    }
 // full_image_ssim=true switches the DSSIM term from the legacy 5-px-cropped
 // "valid" mean to the fused-ssim/Brush zero-padded full-image mean and drops
 // the composited-pixel gradient gate (the per-Gaussian max(0,color) clamp
 // derivative in the projection backward is the correct one).
 if (full_image_ssim) {
  source=Replace(std::move(source),"SSIM_GATHER_PRED",
    "px>=0&&py>=0&&px<int(width)&&py<int(height)");
  source=Replace(std::move(source),"SSIM_NORM",
    "(float(width)*float(height))");
  source=Replace(std::move(source),"SSIM_GRAD_GATE","");
  source=Replace(std::move(source),"SSIM_LOSS_STMT_WRITE",
    "{vec3 lr=color.value[index].xyz,lw=TARGET_FETCH(view*width*height+index).xyz;"
    "float cs0=ssim_map.value[index*3u].x,cs1=ssim_map.value[index*3u+1u].x,"
    "cs2=ssim_map.value[index*3u+2u].x;"
    "float sl=SSIM_WEIGHT*(1.0-(cs0+cs1+cs2)/3.0);"
    "loss_value.value[index]=L1_WEIGHT*(abs(lr.x-lw.x)+abs(lr.y-lw.y)+"
    "abs(lr.z-lw.z))/3.0+sl;}");
 } else {
  source=Replace(std::move(source),"SSIM_GATHER_PRED",
    "px>=5&&py>=5&&px<int(width)-5&&py<int(height)-5");
  source=Replace(std::move(source),"SSIM_NORM",
    "float((width-10u)*(height-10u))");
  source=Replace(std::move(source),"SSIM_GRAD_GATE","*float(rendered>0.0)");
  source=Replace(std::move(source),"SSIM_LOSS_STMT_WRITE",
    "{vec3 lr=color.value[index].xyz,lw=TARGET_FETCH(view*width*height+index).xyz;"
    "float cs0=ssim_map.value[index*3u].x,cs1=ssim_map.value[index*3u+1u].x,"
    "cs2=ssim_map.value[index*3u+2u].x;"
    "float sl=0;if(gx>=5u&&gy>=5u&&gx<width-5u&&gy<height-5u)"
    "sl=SSIM_WEIGHT*(1.0-(cs0+cs1+cs2)/3.0)*"
    "float(width*height)/float((width-10u)*(height-10u));"
    "loss_value.value[index]=L1_WEIGHT*(abs(lr.x-lw.x)+abs(lr.y-lw.y)+"
    "abs(lr.z-lw.z))/3.0+sl;}");
 }
 source=Replace(std::move(source),"L1_WEIGHT",std::to_string(l1_weight));
 return Replace(std::move(source),"SSIM_WEIGHT",std::to_string(ssim_weight));
}


}  // namespace cross3dgs::detail

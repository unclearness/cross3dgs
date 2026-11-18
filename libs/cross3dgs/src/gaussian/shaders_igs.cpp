#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {


// --- ImprovedGS+ (IGS+) gradient-driven Long-Axis-Split densification ---
// A refine step fires when start < step < stop and step % every == 0. Every
// pass is dispatched directly and self-gates on that predicate, mirroring the
// MCMC block but selecting a data-dependent set of split candidates by their
// accumulated screen-space gradient (no opacity sampling, no injected noise).

std::string IgsMarkShader(size_t capacity,uint32_t start,uint32_t stop,
                          uint32_t every,float grad_threshold,
                          float edge_threshold,uint32_t warmup_steps) {
    // xy = gradient accumulator, zw = Edge-Aware Score accumulator. A negative
    // edge_threshold disables the edge gate (Phase B). During the first
    // warmup_steps refine steps the gradient gate is suspended so low-gradient
    // surfaces still populate from the edge score alone.
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer IgsScore { vec4 value[]; } score;
layout(std430,binding=1) writeonly buffer Selected { uint value[]; } selected;
layout(std430,binding=2) readonly buffer OptimizerMeta { uvec4 value; } meta;
void main(){
 uint i=gl_GlobalInvocationID.x; if(i>=CAPACITY)return;
 selected.value[i]=0u;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 if(!refine||i>=meta.value.y)return;
 vec4 s=score.value[i];
 float avg_grad=s.y>0.0?s.x/s.y:0.0;
 float avg_edge=s.w>0.0?s.z/s.w:0.0;
 bool edge_ok=avg_edge>EDGE_THRESHOLD;
 bool grad_ok=avg_grad>GRAD_THRESHOLD;
 bool warmup=meta.value.x<=START+WARMUP*EVERY;
 selected.value[i]=((warmup?edge_ok:(grad_ok&&edge_ok))?1u:0u);
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    source=Replace(std::move(source),"WARMUP",std::to_string(warmup_steps)+"u");
    source=Replace(std::move(source),"GRAD_THRESHOLD",
                   std::to_string(grad_threshold));
    return Replace(std::move(source),"EDGE_THRESHOLD",
                   std::to_string(edge_threshold));
}

// Growth Control budget expression: N_max*sqrt((step-start)/(stop-start)),
// clamped to the capacity. Built with the constants inlined so it can be
// substituted into a shader as a plain uint expression over meta.value.x.
std::string GrowthControlCapExpr(size_t capacity,bool growth_control,
                                 uint32_t gc_start,uint32_t gc_stop) {
    if (!growth_control) return std::to_string(capacity)+"u";
    const std::string span=std::to_string(
            static_cast<double>(gc_stop>gc_start ? gc_stop-gc_start : 1u));
    return "min("+std::to_string(capacity)+"u,uint("+
           std::to_string(static_cast<double>(capacity))+
           "*sqrt(clamp((float(meta.value.x)-"+
           std::to_string(static_cast<double>(gc_start))+")/"+span+
           ",0.0,1.0))))";
}

std::string IgsSplitShader(size_t capacity,uint32_t start,uint32_t stop,
                           uint32_t every,float alpha,float gamma,float beta,
                           bool reorder_sh,bool growth_control) {
    std::string source=R"(#version 450
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Selected { uint value[]; } selected;
layout(std430,binding=1) readonly buffer Offsets { uint value[]; } offsets;
layout(std430,binding=2) buffer Means { vec4 value[]; } means;
layout(std430,binding=3) buffer Scales { vec4 value[]; } scales;
layout(std430,binding=4) buffer Rotations { vec4 value[]; } rotations;
layout(std430,binding=5) buffer ShDc { vec4 value[]; } sh_dc;
layout(std430,binding=6) buffer ShRest { vec4 value[]; } sh_rest;
layout(std430,binding=7) readonly buffer OptimizerMeta { uvec4 value; } meta;
uint SH_LINEAR_INDEX(uint gaussian,uint packed){return gaussian*12u+packed;}
uint SH_REORDER_INDEX(uint gaussian,uint packed){return (gaussian&~31u)*12u+packed*32u+(gaussian&31u);}
mat3 quaternion_rotation(vec4 q){q=normalize(q);float r=q.x,x=q.y,y=q.z,z=q.w;
 return mat3(vec3(1.0-2.0*(y*y+z*z),2.0*(x*y+r*z),2.0*(x*z-r*y)),
  vec3(2.0*(x*y-r*z),1.0-2.0*(x*x+z*z),2.0*(y*z+r*x)),
  vec3(2.0*(x*z+r*y),2.0*(y*z-r*x),1.0-2.0*(x*x+y*y)));}
void main(){
 uint i=gl_GlobalInvocationID.x; if(i>=CAPACITY)return;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 if(!refine)return;
 uint count=meta.value.y;
 if(i>=count||selected.value[i]==0u)return;
 uint dest=count+offsets.value[i];
 if(dest>=EFFECTIVE_CAP)return;
 vec4 so=scales.value[i];
 vec3 sc=so.xyz; float op=so.w;
 vec4 q=rotations.value[i];
 vec3 m=means.value[i].xyz;
 uint lidx=0u; if(sc.y>sc.x)lidx=1u; if(sc.z>sc[lidx])lidx=2u;
 float offset_mag=sc[lidx]*ALPHA;
 mat3 rmat=quaternion_rotation(q);
 vec3 voff=rmat[lidx]*offset_mag;
 vec3 nsc=sc*GAMMA; nsc[lidx]=sc[lidx]*ALPHA;
 vec4 nso=vec4(nsc,op*BETA);
 means.value[i]=vec4(m+voff,0.0); scales.value[i]=nso;
 means.value[dest]=vec4(m-voff,0.0); scales.value[dest]=nso;
 rotations.value[dest]=q; sh_dc.value[dest]=sh_dc.value[i];
 for(uint j=0u;j<12u;j++)sh_rest.value[SH_INDEX(dest,j)]=sh_rest.value[SH_INDEX(i,j)];
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    source=Replace(std::move(source),"EFFECTIVE_CAP",
        GrowthControlCapExpr(capacity,growth_control,start,stop));
    source=Replace(std::move(source),"ALPHA",std::to_string(alpha));
    source=Replace(std::move(source),"GAMMA",std::to_string(gamma));
    source=Replace(std::move(source),"BETA",std::to_string(beta));
    return Replace(std::move(source),"SH_INDEX",
        reorder_sh ? "SH_REORDER_INDEX" : "SH_LINEAR_INDEX");
}

std::string IgsResetMomentsShader(size_t capacity,uint32_t start,uint32_t stop,
                                  uint32_t every,bool reorder_sh,
                                  bool growth_control) {
    std::string source=R"(#version 450
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Selected {uint value[];} selected;
layout(std430,binding=1) readonly buffer Offsets {uint value[];} offsets;
layout(std430,binding=2) buffer MeansM {uvec2 value[];} means_m;
layout(std430,binding=3) buffer MeansV {uvec2 value[];} means_v;
layout(std430,binding=4) buffer ScaleM {uvec2 value[];} scale_m;
layout(std430,binding=5) buffer ScaleV {uvec2 value[];} scale_v;
layout(std430,binding=6) buffer RotationM {uvec2 value[];} rotation_m;
layout(std430,binding=7) buffer RotationV {uvec2 value[];} rotation_v;
layout(std430,binding=8) buffer ShM {uvec2 value[];} sh_m;
layout(std430,binding=9) buffer ShV {uvec2 value[];} sh_v;
layout(std430,binding=10) buffer ShRestM {uvec2 value[];} sh_rest_m;
layout(std430,binding=11) buffer ShRestV {uvec2 value[];} sh_rest_v;
layout(std430,binding=12) readonly buffer OptimizerMeta {uvec4 value;} meta;
uint SH_LINEAR_INDEX(uint gaussian,uint packed){return gaussian*12u+packed;}
uint SH_REORDER_INDEX(uint gaussian,uint packed){return (gaussian&~31u)*12u+packed*32u+(gaussian&31u);}
void main(){
 uint i=gl_GlobalInvocationID.x; if(i>=CAPACITY)return;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 if(!refine)return;
 uint count=meta.value.y;
 uint total=offsets.value[CAPACITY-1u]+selected.value[CAPACITY-1u];
 uint next=min(count+total,EFFECTIVE_CAP);
 bool reset=(i<count&&selected.value[i]!=0u)||(i>=count&&i<next);
 if(!reset)return;
 means_m.value[i]=uvec2(0);means_v.value[i]=uvec2(0);
 scale_m.value[i]=uvec2(0);scale_v.value[i]=uvec2(0);
 rotation_m.value[i]=uvec2(0);rotation_v.value[i]=uvec2(0);
 sh_m.value[i]=uvec2(0);sh_v.value[i]=uvec2(0);
 for(uint j=0u;j<12u;j++){sh_rest_m.value[SH_INDEX(i,j)]=uvec2(0);
                           sh_rest_v.value[SH_INDEX(i,j)]=uvec2(0);}
})";
    source=Replace(std::move(source),"EFFECTIVE_CAP",
        GrowthControlCapExpr(capacity,growth_control,start,stop));
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    return Replace(std::move(source),"SH_INDEX",
        reorder_sh ? "SH_REORDER_INDEX" : "SH_LINEAR_INDEX");
}

std::string IgsFinalizeShader(size_t capacity,uint32_t start,uint32_t stop,
                              uint32_t every,bool growth_control) {
    std::string source=R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer Selected {uint value[];} selected;
layout(std430,binding=1) readonly buffer Offsets {uint value[];} offsets;
layout(std430,binding=2) buffer OptimizerMeta {uvec4 value;} meta;
void main(){
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 if(!refine)return;
 uint count=meta.value.y;
 uint total=offsets.value[CAPACITY-1u]+selected.value[CAPACITY-1u];
 meta.value.y=min(count+total,EFFECTIVE_CAP);
})";
    source=Replace(std::move(source),"EFFECTIVE_CAP",
        GrowthControlCapExpr(capacity,growth_control,start,stop));
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    return Replace(std::move(source),"EVERY",std::to_string(every)+"u");
}

std::string IgsResetAccumulatorShader(size_t capacity,uint32_t start,
                                      uint32_t stop,uint32_t every) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) writeonly buffer IgsScore {vec4 value[];} score;
layout(std430,binding=1) readonly buffer OptimizerMeta {uvec4 value;} meta;
void main(){
 uint i=gl_GlobalInvocationID.x; if(i>=CAPACITY)return;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 if(!refine)return;
 score.value[i]=vec4(0.0);
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    return Replace(std::move(source),"EVERY",std::to_string(every)+"u");
}

// --- Recovery-Aware Pruning (soft prune) ---
// At step_a and step_b, cull the lowest-opacity prune_fraction of Gaussians by
// driving their opacity below the projection cull threshold. Culled Gaussians
// receive no gradient, so they stay dead and their budget is re-densified by
// later Long-Axis-Splits under Growth Control. A 256-bin opacity histogram
// yields the prune threshold without a sort or a full SoA compaction.

std::string IgsRapPredicate() {
    return "meta.value.x==STEP_A||meta.value.x==STEP_B";
}

std::string IgsRapHistZeroShader(uint32_t step_a,uint32_t step_b) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) writeonly buffer Histogram { uint value[]; } hist;
layout(std430,binding=1) readonly buffer OptimizerMeta { uvec4 value; } meta;
void main(){
 uint i=gl_GlobalInvocationID.x; if(i>=256u)return;
 if(RAP)hist.value[i]=0u;
})";
    source=Replace(std::move(source),"RAP",IgsRapPredicate());
    source=Replace(std::move(source),"STEP_A",std::to_string(step_a)+"u");
    return Replace(std::move(source),"STEP_B",std::to_string(step_b)+"u");
}

std::string IgsRapHistBuildShader(size_t capacity,uint32_t step_a,
                                  uint32_t step_b) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer ScalesOpacity { vec4 value[]; } scales;
layout(std430,binding=1) buffer Histogram { uint value[]; } hist;
layout(std430,binding=2) readonly buffer OptimizerMeta { uvec4 value; } meta;
void main(){
 uint i=gl_GlobalInvocationID.x; if(i>=CAPACITY)return;
 if(!(RAP)||i>=meta.value.y)return;
 float opacity=clamp(scales.value[i].w,0.0,1.0);
 atomicAdd(hist.value[uint(opacity*255.0)],1u);
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"RAP",IgsRapPredicate());
    source=Replace(std::move(source),"STEP_A",std::to_string(step_a)+"u");
    return Replace(std::move(source),"STEP_B",std::to_string(step_b)+"u");
}

std::string IgsRapThresholdShader(uint32_t step_a,uint32_t step_b,
                                  float fraction) {
    std::string source=R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer Histogram { uint value[]; } hist;
layout(std430,binding=1) writeonly buffer Threshold { float value[]; } threshold;
layout(std430,binding=2) readonly buffer OptimizerMeta { uvec4 value; } meta;
void main(){
 if(!(RAP))return;
 uint target=uint(FRACTION*float(meta.value.y));
 uint cumulative=0u; uint bin=0u;
 for(bin=0u;bin<256u;bin++){cumulative+=hist.value[bin];
   if(cumulative>=target)break;}
 threshold.value[0]=float(bin)/255.0;
})";
    source=Replace(std::move(source),"RAP",IgsRapPredicate());
    source=Replace(std::move(source),"STEP_A",std::to_string(step_a)+"u");
    source=Replace(std::move(source),"STEP_B",std::to_string(step_b)+"u");
    return Replace(std::move(source),"FRACTION",std::to_string(fraction));
}

std::string IgsRapPruneShader(size_t capacity,uint32_t step_a,uint32_t step_b) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) buffer ScalesOpacity { vec4 value[]; } scales;
layout(std430,binding=1) readonly buffer Threshold { float value[]; } threshold;
layout(std430,binding=2) readonly buffer OptimizerMeta { uvec4 value; } meta;
void main(){
 uint i=gl_GlobalInvocationID.x; if(i>=CAPACITY)return;
 if(!(RAP)||i>=meta.value.y)return;
 if(scales.value[i].w<=threshold.value[0]) scales.value[i].w=0.0009765625;
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"RAP",IgsRapPredicate());
    source=Replace(std::move(source),"STEP_A",std::to_string(step_a)+"u");
    return Replace(std::move(source),"STEP_B",std::to_string(step_b)+"u");
}

std::string Mip3dFilterShader(size_t capacity, uint32_t every,
                              uint32_t freeze_step) {
    // Refresh the per-splat Mip-Splatting 3D filter std-dev in means.w:
    // f = sqrt(0.1) * min over resident views of (view depth / focal_px),
    // taken over views whose projection lands within a 15% guard band of the
    // image. Splats visible in no view get 0 (filter inert). Runs every
    // `every` steps and freezes at `freeze_step` (Brush semantics).
    std::string source=R"(#version 450
layout(local_size_x=256) in;
struct CameraData { mat4 world_to_camera; vec4 intrinsics; uvec4 dims; uvec4 state; };
layout(std430,binding=0) readonly buffer CameraAtlas { CameraData camera[]; } atlas;
layout(std430,binding=1) buffer Means { vec4 value[]; } means;
layout(std430,binding=2) readonly buffer OptimizerMeta { uvec4 value; } meta;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=CAPACITY)return;
 if(!(meta.value.x%EVERY==0u&&meta.value.x<FREEZE))return;
 if(i>=meta.value.y){means.value[i].w=0.0;return;}
 vec3 mean=means.value[i].xyz;float best=3.402823e38;
 for(uint v=0u;v<meta.value.z;v++){
  vec3 view=(atlas.camera[v].world_to_camera*vec4(mean,1.0)).xyz;
  if(view.z<=0.01)continue;
  float fx=atlas.camera[v].intrinsics.x,fy=atlas.camera[v].intrinsics.y;
  float sx=fx*view.x/view.z+atlas.camera[v].intrinsics.z;
  float sy=fy*view.y/view.z+atlas.camera[v].intrinsics.w;
  float w=float(atlas.camera[v].dims.z),h=float(atlas.camera[v].dims.w);
  if(sx<-0.15*w||sx>1.15*w||sy<-0.15*h||sy>1.15*h)continue;
  best=min(best,view.z/fx);}
 means.value[i].w=best<3.0e38 ? 0.31622776601683794*best : 0.0;
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    return Replace(std::move(source),"FREEZE",
                   std::to_string(freeze_step)+"u");
}


}  // namespace cross3dgs::detail

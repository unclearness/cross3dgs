#include "shaders.h"

#include <utility>

namespace cross3dgs::detail {

std::string McmcPrepareShader(size_t capacity,uint32_t start,uint32_t stop,
                              uint32_t every,float min_opacity,
                              float gradient_sampling,
                              float gradient_reference) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer ScalesOpacity { vec4 value[]; } scales;
layout(std430,binding=1) writeonly buffer Probabilities { uint value[]; } probs;
layout(std430,binding=2) writeonly buffer CloneCounts { uint value[]; } clone_counts;
layout(std430,binding=3) writeonly buffer RelocationCounts { uint value[]; } relocation_counts;
layout(std430,binding=4) readonly buffer OptimizerMeta { uvec4 value; } meta;
GRADIENT_BINDING
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=CAPACITY)return;
 clone_counts.value[i]=0u; relocation_counts.value[i]=0u; probs.value[i]=0u;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 if(!refine||i>=meta.value.y)return;
 float opacity=clamp(scales.value[i].w,1e-6,1.0-1e-6);
 float weight=opacity;
 GRADIENT_WEIGHT
 float maximum=float(0x70000000u)/float(meta.value.y);
 probs.value[i]=(opacity<=MIN_OPACITY||isnan(weight)||isinf(weight))?0u:
   max(uint(round(clamp(weight,0.0,1.0)*maximum)),1u);
})";
    if (gradient_sampling>0.0f) {
        // The accumulator holds (sum |dL/d(NDC pos)|, contribution count) since
        // the last refine. Scaling by the reference keeps the weight
        // scene-independent, and the cap bounds how far one Gaussian can
        // dominate the multinomial. The result is renormalized by 1+w*cap so
        // it stays in [0,1] like the plain opacity weight.
        source=Replace(std::move(source),"GRADIENT_BINDING",
            "layout(std430,binding=5) readonly buffer IgsScore { vec4 value[]; } score;");
        source=Replace(std::move(source),"GRADIENT_WEIGHT",
            "{vec4 s=score.value[i];\n"
            "  float mean_grad=s.y>0.0?s.x/s.y:0.0;\n"
            "  float boost=1.0+GRADIENT_W*min(mean_grad/GRADIENT_REF,8.0);\n"
            "  weight=opacity*boost/(1.0+GRADIENT_W*8.0);}");
        source=Replace(std::move(source),"GRADIENT_W",
                       std::to_string(gradient_sampling));
        source=Replace(std::move(source),"GRADIENT_REF",
                       std::to_string(gradient_reference));
    } else {
        source=Replace(std::move(source),"GRADIENT_BINDING","");
        source=Replace(std::move(source),"GRADIENT_WEIGHT","");
    }
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    return Replace(std::move(source),"MIN_OPACITY",std::to_string(min_opacity));
}

std::string McmcDispatchShader(size_t capacity,uint32_t start,uint32_t stop,
                              uint32_t every) {
    std::vector<size_t> scan_counts;
    for (size_t count=capacity;;) {
        scan_counts.push_back(count);
        const size_t group_count=(count+511)/512;
        if (group_count<=1) break;
        count=group_count;
    }
    std::vector<size_t> command_groups;
    command_groups.push_back((capacity+255)/256);
    for (const size_t count : scan_counts) {
        command_groups.push_back((count+511)/512);
    }
    if (scan_counts.size()>1) {
        for (size_t level=scan_counts.size()-1;level>0;--level) {
            command_groups.push_back((scan_counts[level-1]+255)/256);
        }
    }
    command_groups.push_back((capacity+255)/256);
    command_groups.push_back((capacity+63)/64);
    command_groups.push_back((capacity+63)/64);
    command_groups.push_back((capacity+63)/64);
    command_groups.push_back(1);
    std::string group_cases;
    for (size_t command=0;command<command_groups.size();++command) {
        group_cases += " if(command=="+std::to_string(command)+"u)return "+
                std::to_string(command_groups[command])+"u;\n";
    }
    std::string source=R"(#version 450
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer OptimizerMeta {uvec4 value;} meta;
layout(std430,binding=1) writeonly buffer DispatchArgs {uint value[];} args;
uint group_count(uint command) {
 GROUP_CASES
 return 1u;
}
void main(){
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 for(uint command=0u;command<COMMAND_COUNT;command++) {
  args.value[3u*command]=refine?group_count(command):0u;
  args.value[3u*command+1u]=1u;args.value[3u*command+2u]=1u;
 }
})";
    source=Replace(std::move(source),"GROUP_CASES",group_cases);
    source=Replace(std::move(source),"COMMAND_COUNT",
                   std::to_string(command_groups.size())+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    return Replace(std::move(source),"EVERY",std::to_string(every)+"u");
}

std::string McmcMapShader(size_t capacity,uint32_t start,uint32_t stop,
                          uint32_t every,float grow_factor) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Probabilities { uint value[]; } probs;
layout(std430,binding=1) readonly buffer ProbabilityOffsets { uint value[]; } offsets;
layout(std430,binding=2) writeonly buffer IndexMap { uint value[]; } index_map;
layout(std430,binding=3) buffer CloneCounts { uint value[]; } clone_counts;
layout(std430,binding=4) buffer RelocationCounts { uint value[]; } relocation_counts;
layout(std430,binding=5) readonly buffer OptimizerMeta { uvec4 value; } meta;
uint hash(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
uint sample_source(uint key,uint count){
 uint total=offsets.value[count-1u]+probs.value[count-1u];
 if(total==0u)return key%count;
 uint wanted=hash(key^meta.value.x*0x9e3779b9u)%total+1u;
 uint lo=0u,hi=count-1u;while(lo<hi){uint mid=lo+(hi-lo)/2u;
   if(offsets.value[mid]+probs.value[mid]<wanted)lo=mid+1u;else hi=mid;}
 return lo;
}
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=CAPACITY)return;
 uint count=meta.value.y; index_map.value[i]=i;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 if(!refine)return;
 uint next=min(uint(float(count)*GROW_FACTOR),CAPACITY);
 if(i<count&&probs.value[i]==0u){uint source=sample_source(i,count);
   index_map.value[i]=source;atomicAdd(clone_counts.value[source],1u);
   atomicAdd(relocation_counts.value[source],1u);}
 if(i>=count&&i<next){uint source=sample_source(i,count);
   index_map.value[i]=source;atomicAdd(clone_counts.value[source],1u);}
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    return Replace(std::move(source),"GROW_FACTOR",std::to_string(grow_factor));
}

std::string McmcRelocateSourceShader(size_t capacity,float min_opacity) {
    std::string source=R"(#version 450
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer CloneCounts { uint value[]; } clones;
layout(std430,binding=1) buffer ScalesOpacity { vec4 value[]; } scales;
layout(std430,binding=2) readonly buffer OptimizerMeta { uvec4 value; } meta;
float log_factorial(float x){return (0.996763591291*log(x+2.73507778369)-0.978531458767+
 (-0.0400131099548*x-0.596969060666)/(((0.00251628691783*x+0.0959712063178)*x+0.803207449544)*x+1.0))*x;}
float binom(float n,float k){return exp(log_factorial(n)-log_factorial(k)-log_factorial(n-k));}
void main(){uint i=gl_GlobalInvocationID.x;if(i>=meta.value.y||i>=CAPACITY)return;
 uint n=min(clones.value[i]+1u,50u);if(n==1u)return;
 vec4 state=scales.value[i];float new_opacity=1.0-pow(1.0-state.w,1.0/float(n));
 float denominator=0.0;for(uint j=1u;j<=n;j++)for(uint k=0u;k<j;k++)
   denominator+=binom(float(j-1u),float(k))*((k&1u)==0u?1.0:-1.0)/sqrt(float(k+1u))*pow(new_opacity,float(k+1u));
 state.xyz*=state.w/max(denominator,1e-8);state.w=clamp(new_opacity,MIN_OPACITY,0.999999);
 scales.value[i]=state;
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    return Replace(std::move(source),"MIN_OPACITY",std::to_string(min_opacity));
}

std::string McmcCopyShader(size_t capacity,uint32_t start,uint32_t stop,
                           uint32_t every,float grow_factor,bool reorder_sh,
                           bool split_sh_rest,float clone_offset) {
    std::string source=R"(#version 450
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer IndexMap { uint value[]; } map;
layout(std430,binding=1) buffer Means { vec4 value[]; } means;
layout(std430,binding=2) buffer Scales { vec4 value[]; } scales;
layout(std430,binding=3) buffer Rotations { vec4 value[]; } rotations;
layout(std430,binding=4) buffer ShDc { vec4 value[]; } sh_dc;
layout(std430,binding=5) buffer ShRest { vec4 value[]; } sh_rest;
layout(std430,binding=6) readonly buffer OptimizerMeta { uvec4 value; } meta;
uint SH_LINEAR_INDEX(uint gaussian,uint packed){return gaussian*12u+packed;}
uint SH_REORDER_INDEX(uint gaussian,uint packed){return (gaussian&~31u)*12u+packed*32u+(gaussian&31u);}
CLONE_OFFSET_HELPERS
void main(){uint i=gl_GlobalInvocationID.x;if(i>=CAPACITY)return;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 uint next=min(uint(float(meta.value.y)*GROW_FACTOR),CAPACITY);
 if(!refine||i>=next||map.value[i]==i)return;uint source=map.value[i];
 means.value[i]=means.value[source];scales.value[i]=scales.value[source];
 rotations.value[i]=rotations.value[source];sh_dc.value[i]=sh_dc.value[source];
 for(uint j=0u;j<12u;j++)sh_rest.value[SH_INDEX(i,j)]=sh_rest.value[SH_INDEX(source,j)];
 CLONE_OFFSET_STMT
})";
    if (clone_offset>0.0f) {
        source=Replace(std::move(source),"CLONE_OFFSET_HELPERS",
R"(uint hash(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
float uniform_value(uint x){return (float(hash(x))+1.0)/4294967297.0;}
vec3 normal3(uint i){float u1=uniform_value(i*4u+meta.value.x*29u);
 float u2=uniform_value(i*4u+1u+meta.value.x*29u);
 float u3=uniform_value(i*4u+2u+meta.value.x*29u);
 float u4=uniform_value(i*4u+3u+meta.value.x*29u);
 float r0=sqrt(-2.0*log(u1)),r1=sqrt(-2.0*log(u3));
 return vec3(r0*cos(6.28318530718*u2),r0*sin(6.28318530718*u2),
             r1*cos(6.28318530718*u4));}
mat3 rotation_matrix(vec4 q){q=normalize(q);float w=q.x,x=q.y,y=q.z,z=q.w;
 return mat3(vec3(1.0-2.0*(y*y+z*z),2.0*(x*y+w*z),2.0*(x*z-w*y)),
  vec3(2.0*(x*y-w*z),1.0-2.0*(x*x+z*z),2.0*(y*z+w*x)),
  vec3(2.0*(x*z+w*y),2.0*(y*z-w*x),1.0-2.0*(x*x+y*y)));})");
        source=Replace(std::move(source),"CLONE_OFFSET_STMT",
            "means.value[i].xyz+=rotation_matrix(rotations.value[i])*"
            "(scales.value[i].xyz*normal3(i))*CLONE_OFFSET;");
        source=Replace(std::move(source),"CLONE_OFFSET",
                       std::to_string(clone_offset));
    } else {
        source=Replace(std::move(source),"CLONE_OFFSET_HELPERS","");
        source=Replace(std::move(source),"CLONE_OFFSET_STMT","");
    }
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    source=Replace(std::move(source),"GROW_FACTOR",std::to_string(grow_factor));
    if (split_sh_rest) {
        source=Replace(std::move(source),
            "layout(std430,binding=5) buffer ShRest { vec4 value[]; } sh_rest;",
            "layout(std430,binding=5) buffer ShRestA { vec4 value[]; } sh_rest;\n"
            "layout(std430,binding=7) buffer ShRestB { vec4 value[]; } sh_rest_b;");
        source=Replace(std::move(source),
            "for(uint j=0u;j<12u;j++)sh_rest.value[SH_INDEX(i,j)]=sh_rest.value[SH_INDEX(source,j)];",
            "for(uint j=0u;j<6u;j++){sh_rest.value[i*6u+j]=sh_rest.value[source*6u+j];"
            "sh_rest_b.value[i*6u+j]=sh_rest_b.value[source*6u+j];}");
    }
    return Replace(std::move(source),"SH_INDEX",
        reorder_sh ? "SH_REORDER_INDEX" : "SH_LINEAR_INDEX");
}

std::string McmcFinalizeShader(size_t capacity,uint32_t start,uint32_t stop,
                               uint32_t every,float grow_factor) {
    std::string source=R"(#version 450
layout(local_size_x=1) in;layout(std430,binding=0) buffer OptimizerMeta {uvec4 value;} meta;
void main(){if(meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u)
 meta.value.y=min(uint(float(meta.value.y)*GROW_FACTOR),CAPACITY);})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    return Replace(std::move(source),"GROW_FACTOR",std::to_string(grow_factor));
}

std::string McmcOpacityDecayShader(size_t capacity,uint32_t start,uint32_t stop,
                                   uint32_t every,float decay,
                                   uint32_t max_steps) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) buffer ScalesOpacity {vec4 value[];} scales;
layout(std430,binding=1) readonly buffer OptimizerMeta {uvec4 value;} meta;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=meta.value.y||i>=CAPACITY)return;
 if(!(meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u))return;
 float progress=min(float(meta.value.x)/float(MAX_STEPS),1.0);
 float o=scales.value[i].w-DECAY*(1.0-progress);
 scales.value[i].w=clamp(o,1e-6,1.0-1e-6);
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    source=Replace(std::move(source),"DECAY",std::to_string(decay));
    return Replace(std::move(source),"MAX_STEPS",std::to_string(max_steps)+"u");
}

std::string McmcNoiseShader(size_t capacity,float noise_lr,float means_lr,
                            float means_lr_final,uint32_t max_steps) {
    std::string source=R"(#version 450
layout(local_size_x=256) in;
layout(std430,binding=0) buffer Means {vec4 value[];} means;
layout(std430,binding=1) readonly buffer Scales {vec4 value[];} scales;
layout(std430,binding=2) readonly buffer Rotations {vec4 value[];} rotations;
layout(std430,binding=3) readonly buffer OptimizerMeta {uvec4 value;} meta;
uint hash(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
float uniform_value(uint x){return (float(hash(x))+1.0)/4294967297.0;}
vec3 normal3(uint i){float u1=uniform_value(i*4u+meta.value.x*17u);
 float u2=uniform_value(i*4u+1u+meta.value.x*17u);
 float u3=uniform_value(i*4u+2u+meta.value.x*17u);
 float u4=uniform_value(i*4u+3u+meta.value.x*17u);
 float r0=sqrt(-2.0*log(u1)),r1=sqrt(-2.0*log(u3));
 return vec3(r0*cos(6.28318530718*u2),r0*sin(6.28318530718*u2),
             r1*cos(6.28318530718*u4));}
mat3 rotation_matrix(vec4 q){q=normalize(q);float w=q.x,x=q.y,y=q.z,z=q.w;
 return mat3(vec3(1.0-2.0*(y*y+z*z),2.0*(x*y+w*z),2.0*(x*z-w*y)),
  vec3(2.0*(x*y-w*z),1.0-2.0*(x*x+z*z),2.0*(y*z+w*x)),
  vec3(2.0*(x*z+w*y),2.0*(y*z-w*x),1.0-2.0*(x*x+y*y)));}
void main(){uint i=gl_GlobalInvocationID.x;if(i>=meta.value.y||i>=CAPACITY)return;
 vec4 state=scales.value[i];mat3 r=rotation_matrix(rotations.value[i]);
 mat3 covariance=r*mat3(state.x*state.x,0,0,0,state.y*state.y,0,
                         0,0,state.z*state.z)*transpose(r);
 float gate=1.0/(1.0+exp(-100.0*((1.0-state.w)-0.995)));
 float progress=min(float(meta.value.x)/float(MAX_STEPS),1.0);
 float rate=NOISE_LR*MEANS_LR*pow(LR_RATIO,progress);
 means.value[i].xyz+=covariance*normal3(i)*(gate*rate);
})";
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"NOISE_LR",std::to_string(noise_lr));
    source=Replace(std::move(source),"MEANS_LR",std::to_string(means_lr));
    source=Replace(std::move(source),"LR_RATIO",
                   std::to_string(means_lr_final/means_lr));
    return Replace(std::move(source),"MAX_STEPS",std::to_string(max_steps)+"u");
}

std::string McmcResetMomentsShader(size_t capacity,uint32_t start,uint32_t stop,
                                   uint32_t every,float grow_factor,bool reorder_sh,
                                   bool fp32_moments,bool split_sh_rest) {
    std::string source=R"(#version 450
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer RelocationCounts {uint value[];} relocations;
layout(std430,binding=1) buffer MeansM {MOMENT_TYPE value[];} means_m;
layout(std430,binding=2) buffer MeansV {MOMENT_TYPE value[];} means_v;
layout(std430,binding=3) buffer ScaleM {MOMENT_TYPE value[];} scale_m;
layout(std430,binding=4) buffer ScaleV {MOMENT_TYPE value[];} scale_v;
layout(std430,binding=5) buffer RotationM {MOMENT_TYPE value[];} rotation_m;
layout(std430,binding=6) buffer RotationV {MOMENT_TYPE value[];} rotation_v;
layout(std430,binding=7) buffer ShM {MOMENT_TYPE value[];} sh_m;
layout(std430,binding=8) buffer ShV {MOMENT_TYPE value[];} sh_v;
layout(std430,binding=9) buffer ShRestM {MOMENT_TYPE value[];} sh_rest_m;
layout(std430,binding=10) buffer ShRestV {MOMENT_TYPE value[];} sh_rest_v;
layout(std430,binding=11) readonly buffer OptimizerMeta {uvec4 value;} meta;
uint SH_LINEAR_INDEX(uint gaussian,uint packed){return gaussian*12u+packed;}
uint SH_REORDER_INDEX(uint gaussian,uint packed){return (gaussian&~31u)*12u+packed*32u+(gaussian&31u);}
void main(){uint i=gl_GlobalInvocationID.x;if(i>=CAPACITY)return;
 bool refine=meta.value.x>START&&meta.value.x<STOP&&meta.value.x%EVERY==0u;
 uint next=min(uint(float(meta.value.y)*GROW_FACTOR),CAPACITY);
 bool reset_relocation_source=i<meta.value.y&&relocations.value[i]>0u;
 bool reset_new_destination=i>=meta.value.y&&i<next;
 if(!refine||(!reset_relocation_source&&!reset_new_destination))return;
 means_m.value[i]=MOMENT_ZERO;means_v.value[i]=MOMENT_ZERO;
 scale_m.value[i]=MOMENT_ZERO;scale_v.value[i]=MOMENT_ZERO;
 rotation_m.value[i]=MOMENT_ZERO;rotation_v.value[i]=MOMENT_ZERO;
 sh_m.value[i]=MOMENT_ZERO;sh_v.value[i]=MOMENT_ZERO;
 for(uint j=0u;j<12u;j++){sh_rest_m.value[SH_INDEX(i,j)]=MOMENT_ZERO;
                           sh_rest_v.value[SH_INDEX(i,j)]=MOMENT_ZERO;}
})";
    if (split_sh_rest) {
        source=Replace(std::move(source),
            "layout(std430,binding=9) buffer ShRestM {MOMENT_TYPE value[];} sh_rest_m;\n"
            "layout(std430,binding=10) buffer ShRestV {MOMENT_TYPE value[];} sh_rest_v;",
            "layout(std430,binding=9) buffer ShRestMA {MOMENT_TYPE value[];} sh_rest_m;\n"
            "layout(std430,binding=10) buffer ShRestVA {MOMENT_TYPE value[];} sh_rest_v;\n"
            "layout(std430,binding=12) buffer ShRestMB {MOMENT_TYPE value[];} sh_rest_m_b;\n"
            "layout(std430,binding=13) buffer ShRestVB {MOMENT_TYPE value[];} sh_rest_v_b;");
        source=Replace(std::move(source),
            "for(uint j=0u;j<12u;j++){sh_rest_m.value[SH_INDEX(i,j)]=MOMENT_ZERO;\n"
            "                           sh_rest_v.value[SH_INDEX(i,j)]=MOMENT_ZERO;}",
            "for(uint j=0u;j<6u;j++){uint index=i*6u+j;"
            "sh_rest_m.value[index]=MOMENT_ZERO;sh_rest_v.value[index]=MOMENT_ZERO;"
            "sh_rest_m_b.value[index]=MOMENT_ZERO;sh_rest_v_b.value[index]=MOMENT_ZERO;}");
    }
    source=Replace(std::move(source),"CAPACITY",std::to_string(capacity)+"u");
    source=Replace(std::move(source),"MOMENT_TYPE",
                   fp32_moments ? "vec4" : "uvec2");
    source=Replace(std::move(source),"MOMENT_ZERO",
                   fp32_moments ? "vec4(0.0)" : "uvec2(0u)");
    source=Replace(std::move(source),"START",std::to_string(start)+"u");
    source=Replace(std::move(source),"STOP",std::to_string(stop)+"u");
    source=Replace(std::move(source),"EVERY",std::to_string(every)+"u");
    source=Replace(std::move(source),"GROW_FACTOR",std::to_string(grow_factor));
    return Replace(std::move(source),"SH_INDEX",
        reorder_sh ? "SH_REORDER_INDEX" : "SH_LINEAR_INDEX");
}

}  // namespace cross3dgs::detail

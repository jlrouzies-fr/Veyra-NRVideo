// Adaptive sharpening on every output frame, after frame generation (before motion blur and
// deband). Unsharp detail from the 4-neighbour cross, weakened where the local range is
// already large (strong edges, so they do not ring), and clamped to the 3x3 min/max so no
// halo is ever brighter or darker than its neighbourhood. Works on the output signal; an scRGB
// target is sharpened in PQ. The user's protected regions are left untouched.
#include "HdrColor.hlsli"
#include "NrProtection.hlsli"
Texture2D<float4> frameTex : register(t0);
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float amount;      // 0..1
    float outputMode;  // 0 sRGB, 1 scRGB (sharpened in PQ), 2 PQ
    float exclusionOn, feather;
    float pad0, pad1;
    float4 regions[4];
};

float3 toSignal(float3 v) { return outputMode>0.5&&outputMode<1.5 ? HdrEncodePq(v) : v; }
float3 fromSignal(float3 v) { return outputMode>0.5&&outputMode<1.5 ? HdrDecodePq(v) : v; }
float3 load(int2 p) { return toSignal(frameTex[clamp(p,int2(0,0),int2(width,height)-1)].rgb); }

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    if(amount<=0||(exclusionOn>0&&NrProtection(float2(id.xy)+0.5,uint2(width,height),1,feather,regions)>0)){
        outputTex[id.xy]=centre;return;
    }
    const int2 p=int2(id.xy);
    const float3 c=toSignal(centre.rgb);
    const float3 n=load(p+int2(0,-1)),s=load(p+int2(0,1)),w=load(p+int2(-1,0)),e=load(p+int2(1,0));
    const float3 nw=load(p+int2(-1,-1)),ne=load(p+int2(1,-1)),sw=load(p+int2(-1,1)),se=load(p+int2(1,1));
    const float3 low=min(min(min(c,n),min(s,w)),min(min(e,nw),min(ne,min(sw,se))));
    const float3 high=max(max(max(c,n),max(s,w)),max(max(e,nw),max(ne,max(sw,se))));
    const float3 detail=c-(n+s+w+e)*0.25;
    // Range 0.1 of the signal (about 100 10-bit codes) halves the gain: fine texture gets the
    // most, hard edges little.
    const float range=max(high.r-low.r,max(high.g-low.g,high.b-low.b));
    const float gain=2.0*amount/(1.0+range*10.0);
    const float3 sharpened=clamp(c+detail*gain,low,high);
    outputTex[id.xy]=float4(fromSignal(sharpened),centre.a);
}

// Edge vignette compensation, step 1 of 3: log luminance of the edge band (top, or bottom with
// fromBottom) and of the whole frame.
// One sample per 4x4 pixels; each 16x16 group writes (sum log top, top count, sum log all,
// all count) for its 64x64 pixel block.
#include "HdrColor.hlsli"
Texture2D<float4> frameTex : register(t0);
RWTexture2D<float4> partialTex : register(u0);   // groupsX x groupsY
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 0 sRGB, 1 scRGB, 2 PQ
    float bandRows;     // rows counted as the edge band
    float sdrWhiteNits;
    float fromBottom;   // 1: the band is counted from the bottom edge
    float pad1, pad2;
};

float nits(float3 encoded) {
    float3 l;
    if(outputMode>1.5)l=HdrDecodePq(encoded)*80.0;
    else if(outputMode>0.5)l=encoded*80.0;
    else l=select(encoded<=0.04045,encoded/12.92,pow((max(encoded,0)+0.055)/1.055,2.4))*sdrWhiteNits;
    return dot(l,float3(0.2126,0.7152,0.0722));
}

groupshared float4 sums[256];

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    float4 v=0;
    const uint2 pixel=id.xy*4+2;
    if(pixel.x<width&&pixel.y<height){
        const float l=log(max(nits(frameTex[pixel].rgb),0.5));
        v.zw=float2(l,1);
        const float row=fromBottom>0.5?float(height-1-pixel.y):float(pixel.y);
        if(row<bandRows)v.xy=float2(l,1);
    }
    sums[index]=v;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for(uint stride=128;stride>0;stride>>=1){
        if(index<stride)sums[index]+=sums[index+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if(index==0)partialTex[group.xy]=sums[0];
}

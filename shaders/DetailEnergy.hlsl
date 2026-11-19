// Detail match, measuring step: fine-detail energy of one frame per 16x16 block (mean square of
// luma minus its 3x3 mean, in the output signal; an scRGB target is measured in PQ).
// Frame generation softens fine detail on slow motion, so generated and real frames alternate
// soft / sharp at the output rate: a pulsing flicker. These energies drive DetailBoost (generated
// frames) and DetailSoften (real frames).
#include "HdrColor.hlsli"
Texture2D<float4> frameTex : register(t0);
RWTexture2D<float> energyTex : register(u0);   // groupsX x groupsY
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 0 sRGB, 1 scRGB, 2 PQ
    float pad0, pad1, pad2, pad3, pad4;
};
float luma(int2 p) {
    p=clamp(p,int2(0,0),int2(width,height)-1);
    float3 c=frameTex[p].rgb;
    if(outputMode>0.5&&outputMode<1.5)c=HdrEncodePq(c);
    return dot(c,float3(0.2126,0.7152,0.0722));
}
groupshared float sums[256];
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    float e=0;
    if(id.x<width&&id.y<height){
        const int2 p=int2(id.xy);
        float mean=0;
        [unroll] for(int y=-1;y<=1;++y)[unroll] for(int x=-1;x<=1;++x)mean+=luma(p+int2(x,y));
        const float hp=luma(p)-mean/9.0;
        e=hp*hp;
    }
    sums[index]=e;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for(uint stride=128;stride>0;stride>>=1){
        if(index<stride)sums[index]+=sums[index+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if(index==0)energyTex[group.xy]=sums[0]/256.0;
}

// Detail match, generated frames: restore the fine detail frame generation softened. Per block,
// gain = sqrt(min(real energies) / generated energy), 1..3, applied as an unsharp boost of the
// frame's own 3x3 detail, clamped to the light / dark extremes the generated frame and the two
// real frames hold around the pixel (no halos, nothing the real frames do not have).
#include "HdrColor.hlsli"
Texture2D<float4> generatedTex : register(t0);
Texture2D<float4> currentTex : register(t1);
Texture2D<float4> previousTex : register(t2);
Texture2D<float> generatedEnergy : register(t3);
Texture2D<float> currentEnergy : register(t4);
Texture2D<float> previousEnergy : register(t5);
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 0 sRGB, 1 scRGB (worked in PQ), 2 PQ
    uint groupsX, groupsY;
    float maxGain;
    float pad0, pad1;
};
float3 toSignal(float3 v) { return outputMode>0.5&&outputMode<1.5 ? HdrEncodePq(v) : v; }
float3 fromSignal(float3 v) { return outputMode>0.5&&outputMode<1.5 ? HdrDecodePq(v) : v; }
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=generatedTex[id.xy];
    const float2 uv=(float2(id.xy)+0.5)/16.0/float2(groupsX,groupsY);
    const float eg=generatedEnergy.SampleLevel(linearClamp,uv,0);
    const float er=min(currentEnergy.SampleLevel(linearClamp,uv,0),previousEnergy.SampleLevel(linearClamp,uv,0));
    const float gain=clamp(sqrt((er+1e-7)/(eg+1e-7)),1.0,maxGain);
    if(!(gain>1.01)){outputTex[id.xy]=centre;return;}
    const int2 p=int2(id.xy),last=int2(width,height)-1;
    const float3 c=toSignal(centre.rgb);
    float3 mean=0,low=c,high=c;
    [unroll] for(int y=-1;y<=1;++y)[unroll] for(int x=-1;x<=1;++x){
        const int2 q=clamp(p+int2(x,y),int2(0,0),last);
        const float3 g=toSignal(generatedTex[q].rgb);
        const float3 a=toSignal(currentTex[q].rgb);
        const float3 b=toSignal(previousTex[q].rgb);
        mean+=g;
        low=min(low,min(g,min(a,b)));high=max(high,max(g,max(a,b)));
    }
    mean/=9.0;
    const float3 boosted=clamp(c+(gain-1.0)*(c-mean),low,high);
    outputTex[id.xy]=float4(fromSignal(boosted),centre.a);
}

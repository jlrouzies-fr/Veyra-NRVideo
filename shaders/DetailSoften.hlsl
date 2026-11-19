// Detail match, real frames: meet the generated frames halfway. Where the last generated frame
// (after its boost) still had less fine detail than this real frame, remove that share of the
// real frame's 3x3 detail, up to 60%, times `strength`. Blocks where they already match are
// untouched, so a still or fast-moving picture keeps its full sharpness.
#include "HdrColor.hlsli"
Texture2D<float4> frameTex : register(t0);
Texture2D<float> realEnergy : register(t1);
Texture2D<float> generatedEnergy : register(t2);
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;
    uint groupsX, groupsY;
    float strength;     // 0 copies the frame
    float pad0, pad1;
};
float3 toSignal(float3 v) { return outputMode>0.5&&outputMode<1.5 ? HdrEncodePq(v) : v; }
float3 fromSignal(float3 v) { return outputMode>0.5&&outputMode<1.5 ? HdrDecodePq(v) : v; }
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    if(strength<=0){outputTex[id.xy]=centre;return;}
    const float2 uv=(float2(id.xy)+0.5)/16.0/float2(groupsX,groupsY);
    const float er=realEnergy.SampleLevel(linearClamp,uv,0);
    const float eg=generatedEnergy.SampleLevel(linearClamp,uv,0);
    const float amount=clamp(1.0-sqrt((eg+1e-7)/(er+1e-7)),0.0,0.6)*strength;
    if(!(amount>0.01)){outputTex[id.xy]=centre;return;}
    const int2 p=int2(id.xy),last=int2(width,height)-1;
    const float3 c=toSignal(centre.rgb);
    float3 mean=0;
    [unroll] for(int y=-1;y<=1;++y)[unroll] for(int x=-1;x<=1;++x)mean+=toSignal(frameTex[clamp(p+int2(x,y),int2(0,0),last)].rgb);
    mean/=9.0;
    outputTex[id.xy]=float4(fromSignal(c-amount*(c-mean)),centre.a);
}

// Anti-vignette, step 3 of 3: brighten every output frame by the learned per-tile gain
// (bilinear between tile centres), in linear light so colours keep their ratios. Dark areas
// stay dark relative to their surroundings: the gain only undoes screen-fixed darkening.
#include "HdrColor.hlsli"
#include "NrProtection.hlsli"
Texture2D<float4> frameTex : register(t0);
Texture2D<float> gainTex : register(t1);
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 0 sRGB, 1 scRGB, 2 PQ
    float tilePixels;
    uint tilesX, tilesY;
    float exclusionOn, feather;
    float4 regions[4];
};
float3 decode(float3 v) {
    if(outputMode>1.5)return HdrDecodePq(v);
    if(outputMode>0.5)return v;
    return select(v<=0.04045,v/12.92,pow((max(v,0)+0.055)/1.055,2.4));
}
float3 encode(float3 v) {
    if(outputMode>1.5)return HdrEncodePq(v);
    if(outputMode>0.5)return v;
    v=max(v,0);
    return select(v<=0.0031308,v*12.92,1.055*pow(v,1.0/2.4)-0.055);
}
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    if(tilePixels<=0){outputTex[id.xy]=centre;return;}   // no map solved yet
    const float2 uv=(float2(id.xy)+0.5)/tilePixels/float2(tilesX,tilesY);
    const float gain=gainTex.SampleLevel(linearClamp,uv,0);
    if(!(gain>1.002)||(exclusionOn>0&&NrProtection(float2(id.xy)+0.5,uint2(width,height),1,feather,regions)>0)){
        outputTex[id.xy]=centre;return;
    }
    outputTex[id.xy]=float4(encode(decode(centre.rgb)*gain),centre.a);
}

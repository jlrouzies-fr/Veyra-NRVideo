// Top vignette compensation, step 3 of 3: brighten the top rows by the eased lift, strongest at
// the top edge and fading as (1 - y/depth)^rolloff to nothing at `depth` of the height. Bright pixels (HUD, sun, clouds
// far above the scene level) are left alone. Only the top `depth` rows are dispatched.
#include "HdrColor.hlsli"
Texture2D<float4> frameTex : register(t0);
Texture2D<float4> stateTex : register(t1);
RWTexture2D<float4> outputTex : register(u0);
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 0 sRGB, 1 scRGB, 2 PQ
    float depthRows;
    float sdrWhiteNits;
    float rolloff;      // exponent of (1 - y/depth)
    float pad1, pad2;
};

float3 toLinear(float3 e) {
    if(outputMode>1.5)return HdrDecodePq(e)*80.0;
    if(outputMode>0.5)return e*80.0;
    return select(e<=0.04045,e/12.92,pow((max(e,0)+0.055)/1.055,2.4))*sdrWhiteNits;
}
float3 fromLinear(float3 l) {
    if(outputMode>1.5)return HdrEncodePq(l/80.0);
    if(outputMode>0.5)return l/80.0;
    const float3 x=saturate(l/sdrWhiteNits);
    return select(x<=0.0031308,x*12.92,1.055*pow(x,1.0/2.4)-0.055);
}

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    const float4 state=stateTex[uint2(0,0)];
    const float t=saturate(1.0-(float(id.y)+0.5)/depthRows);
    const float lift=state.x*pow(t,rolloff);
    if(lift<1e-4||!(state.y>0)){outputTex[id.xy]=centre;return;}
    const float3 l=toLinear(centre.rgb);
    const float y=dot(l,float3(0.2126,0.7152,0.0722));
    const float h=saturate((y/state.y-4.0)/12.0);
    const float keep=h*h*(3.0-2.0*h);
    const float gain=lerp(exp2(lift),1.0,keep);
    outputTex[id.xy]=float4(fromLinear(l*gain),centre.a);
}

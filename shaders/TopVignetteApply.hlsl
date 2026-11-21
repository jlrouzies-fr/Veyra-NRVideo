// Edge vignette compensation, step 3 of 3: brighten the rows of one edge by its eased lift,
// strongest at the edge and fading as (1 - distance/depth)^rolloff to nothing at `depth` of the
// height. The lifted luminance rolls off between 4x and 16x the scene level and never drops
// below the original, so the HUD, the sun and bright clouds stay as they are and brightness
// order is kept at any lift. Only the `depth` rows of the edge are dispatched.
#include "HdrColor.hlsli"
Texture2D<float4> frameTex : register(t0);
Texture2D<float4> stateTex : register(t1);
RWTexture2D<float4> outputTex : register(u0);
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 0 sRGB, 1 scRGB, 2 PQ
    float depthRows;
    float sdrWhiteNits;
    float rolloff;      // exponent of (1 - distance/depth)
    float fromBottom;   // 1: the bottom edge; output row r is frame row height-depth+r
    float pad2;
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
    const uint rows=uint(depthRows);
    if(id.x>=width||id.y>=rows||rows>height)return;
    const bool bottom=fromBottom>0.5;
    const uint frameRow=bottom?height-rows+id.y:id.y;
    const float4 centre=frameTex[uint2(id.x,frameRow)];
    const float4 state=stateTex[uint2(bottom?1:0,0)];
    const float distance=float(bottom?height-1-frameRow:frameRow);
    const float t=saturate(1.0-(distance+0.5)/depthRows);
    const float lift=state.x*pow(t,rolloff);
    if(lift<1e-4||!(state.y>0)){outputTex[id.xy]=centre;return;}
    const float3 l=toLinear(centre.rgb);
    const float y=dot(l,float3(0.2126,0.7152,0.0722));
    if(!(y>1e-6)){outputTex[id.xy]=centre;return;}
    // Monotonic in y: full gain up to the knee, a shoulder toward the ceiling above it, and
    // never darker than the original.
    const float knee=4.0*state.y, ceiling=16.0*state.y;
    float lifted=y*exp2(lift);
    if(lifted>knee){const float over=lifted-knee;lifted=knee+over/(1.0+over/(ceiling-knee));}
    const float gain=max(lifted,y)/y;
    outputTex[id.xy]=float4(fromLinear(l*gain),centre.a);
}

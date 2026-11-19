// Deband on every output frame (after frame generation and motion blur). A 10-bit source
// cleaned by NR shows smooth gradients (skies) as one-code steps: visible contours. Where the
// neighbourhood is flat to within `threshold` (in 10-bit/8-bit codes of the output signal),
// the pixel is replaced by an average of samples spread over growing rings at full float
// precision, then a per-frame triangular dither carries the sub-code value through the
// 10-bit (or 8-bit) write. Edges and texture stronger than the threshold stop the averaging.
// Runs last, after sharpening, so neither the bands nor the dither get sharpened.
#include "HdrColor.hlsli"
Texture2D<float4> frameTex : register(t0);
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float radius;      // outermost ring, pixels
    float threshold;   // largest difference still averaged, in output-signal units
    float ditherStep;  // quantisation step of the target (1/1023, 1/255); 0 = float target
    float outputMode;  // 0 sRGB, 1 scRGB (compare in PQ), 2 PQ
    uint frameIndex;   // varies the sample pattern and dither every frame
    float pad0;
};

uint hash(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
    return x;
}
float random(inout uint state) {
    state = hash(state);
    return float(state & 0xFFFFFFu) / 16777215.0;
}
float3 signal(float3 value) { return outputMode>0.5&&outputMode<1.5 ? HdrEncodePq(value) : value; }
float difference(float3 a, float3 b) { const float3 d=abs(a-b); return max(d.r,max(d.g,d.b)); }

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    uint state=hash(id.x*73856093u ^ id.y*19349663u ^ frameIndex*83492791u);
    const float2 size=float2(width,height);
    const float2 uv=(float2(id.xy)+0.5)/size;
    const float3 centreSignal=signal(centre.rgb);
    float3 result=centre.rgb;
    // Three rings of two opposite pairs. A ring is accepted when each pair's mean stays within
    // the threshold of the centre (the area is a straight gradient through it: steep skies
    // pass, edges fail) and no pair spans more than four thresholds (real texture); the first
    // rejected ring stops the search. Tighter limits for the inner rings.
    [unroll] for(uint ring=1;ring<=3;++ring){
        const float distance=radius*(float(ring)/3.0)*(0.5+0.5*random(state));
        const float angle=6.2831853*random(state);
        const float2 a=float2(cos(angle),sin(angle))*distance;
        const float2 b=float2(-a.y,a.x);
        const float3 s0=frameTex.SampleLevel(linearClamp,uv+a/size,0).rgb;
        const float3 s1=frameTex.SampleLevel(linearClamp,uv-a/size,0).rgb;
        const float3 s2=frameTex.SampleLevel(linearClamp,uv+b/size,0).rgb;
        const float3 s3=frameTex.SampleLevel(linearClamp,uv-b/size,0).rgb;
        const float limit=threshold*(0.5+0.5*float(ring)/3.0);
        const float3 p0=signal(s0),p1=signal(s1),p2=signal(s2),p3=signal(s3);
        if(!(difference((p0+p1)*0.5,centreSignal)<=limit&&difference((p2+p3)*0.5,centreSignal)<=limit))break;
        if(!(max(difference(p0,p1),difference(p2,p3))<=threshold*4.0))break;
        result=(s0+s1+s2+s3)*0.25;
    }
    if(ditherStep>0){
        const float r1=random(state),r2=random(state);
        result+=(r1-r2)*ditherStep;
    }
    outputTex[id.xy]=float4(result,centre.a);
}

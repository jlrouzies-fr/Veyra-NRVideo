// Motion blur along the optical flow, after frame generation: every output frame (real and
// generated) is averaged along its motion over the share of one output frame a camera
// shutter would stay open. The flow points from the current picture to the previous one, in
// work pixels, over one whole source frame; `length` is shutter / output frames per source frame.
Texture2D<float4> frameTex : register(t0);
Texture2D<float2> flowTex : register(t1);
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float length;     // 0 copies the frame
    float maxPixels;  // the longest blur, in pixels (bad flow never smears the whole picture)
    float minPixels;  // shorter than this: left sharp
    float pad0, pad1, pad2;
};
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    if(length<=0){outputTex[id.xy]=centre;return;}
    const float2 size=float2(width,height);
    const float2 uv=(float2(id.xy)+0.5)/size;
    float2 streak=flowTex.SampleLevel(linearClamp,uv,0)*length;
    const float pixels=sqrt(dot(streak,streak));
    if(!(pixels>minPixels)){outputTex[id.xy]=centre;return;}   // also catches NaN flow
    if(pixels>maxPixels)streak*=maxPixels/pixels;
    const uint taps=clamp(uint(ceil(min(pixels,maxPixels)*0.5)),2u,16u);
    // Centred on the pixel: half the streak toward the previous picture, half away from it.
    float3 sum=0;
    [loop] for(uint i=0;i<taps;++i){
        const float t=(i+0.5)/taps-0.5;
        sum+=frameTex.SampleLevel(linearClamp,uv+streak*t/size,0).rgb;
    }
    outputTex[id.xy]=float4(sum/taps,centre.a);
}

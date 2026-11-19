// Motion blur along the optical flow, after frame generation: every output frame (real and
// generated) is averaged along its motion over the share of one output frame a camera
// shutter would stay open. The flow points from the current picture to the previous one, in
// work pixels, over one whole source frame; `blurLength` is shutter / output frames per source frame.
//
// HUD guard: the optical flow is coarse and bleeds the scene's motion onto static overlays
// (HUD, subtitles, menus). A pixel whose 3x3 neighbourhood is the same in the current and the
// previous real frame is "still": it is left sharp, and taps gathered for moving pixels skip
// still pixels, so HUD edges neither blur nor smear into the scene around them.
#include "NrProtection.hlsli"
Texture2D<float4> frameTex : register(t0);
Texture2D<float2> flowTex : register(t1);
Texture2D<float4> currentTex : register(t2);   // current real frame (sharp)
Texture2D<float4> previousTex : register(t3);  // previous real frame (sharp)
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float blurLength;         // 0 copies the frame
    float maxPixels;      // the longest blur, in pixels (bad flow never smears the whole picture)
    float minPixels;      // shorter than this: left sharp
    float stillTolerance; // largest per-channel change (encoded) still counted as unchanged; 0 disables
    float exclusionOn, feather;
    float4 regions[4];    // the user's protected rectangles: never blurred
};

float change(int2 p) {
    p=clamp(p,int2(0,0),int2(width,height)-1);
    const float3 d=abs(currentTex[p].rgb-previousTex[p].rgb);
    return max(d.r,max(d.g,d.b));
}
bool still(int2 p) {
    if(stillTolerance<=0)return false;
    float m=0;
    [unroll] for(int y=-1;y<=1;++y)
        [unroll] for(int x=-1;x<=1;++x)m=max(m,change(p+int2(x,y)));
    return m<=stillTolerance;
}

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    if(blurLength<=0){outputTex[id.xy]=centre;return;}
    if(exclusionOn>0&&NrProtection(float2(id.xy)+0.5,uint2(width,height),1,feather,regions)>0){outputTex[id.xy]=centre;return;}
    if(still(int2(id.xy))){outputTex[id.xy]=centre;return;}
    const float2 size=float2(width,height);
    const float2 uv=(float2(id.xy)+0.5)/size;
    float2 streak=flowTex.SampleLevel(linearClamp,uv,0)*blurLength;
    const float pixels=sqrt(dot(streak,streak));
    if(!(pixels>minPixels)){outputTex[id.xy]=centre;return;}   // also catches NaN flow
    if(pixels>maxPixels)streak*=maxPixels/pixels;
    const uint taps=clamp(uint(ceil(min(pixels,maxPixels)*0.5)),2u,16u);
    // Centred on the pixel: half the streak toward the previous picture, half away from it.
    float3 sum=0;
    float count=0;
    [loop] for(uint i=0;i<taps;++i){
        const float t=(i+0.5)/taps-0.5;
        const float2 at=uv+streak*t/size;
        // Changed pixels only: a still neighbour (HUD) is not smeared into the scene.
        if(stillTolerance>0&&change(int2(at*size))<=stillTolerance)continue;
        sum+=frameTex.SampleLevel(linearClamp,at,0).rgb;
        count+=1;
    }
    outputTex[id.xy]=count>0?float4(sum/count,centre.a):centre;
}

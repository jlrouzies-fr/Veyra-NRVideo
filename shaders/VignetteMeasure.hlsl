// Anti-vignette, step 1 of 3: measure screen-fixed darkening from motion.
// Content keeps its brightness as the camera moves it across the screen; a vignette does not
// move. For a pixel that came from p+f in the previous real frame (f: optical flow, current ->
// previous, work pixels) the log-luminance change is
//     d = log L(p) - log L'(p+f) = logV(p) - logV(p+f) + c ~= -grad(logV).f + c
// (c: a whole-frame exposure change). Each 16x16 group writes its weighted sums for the
// per-tile least-squares fit of grad(logV): three float4 texels per group.
#include "HdrColor.hlsli"
Texture2D<float4> currentTex : register(t0);
Texture2D<float4> previousTex : register(t1);
Texture2D<float2> flowTex : register(t2);
RWTexture2D<float4> partialTex : register(u0);   // (groupsX*3) x groupsY
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 0 sRGB, 1 scRGB, 2 PQ
    float darkFloor;    // linear luminance below which a pixel is too noisy to use
    float pad0, pad1, pad2, pad3;
};

float luminance(float3 encoded) {
    float3 l;
    if(outputMode>1.5)l=HdrDecodePq(encoded);
    else if(outputMode>0.5)l=encoded;
    else l=select(encoded<=0.04045,encoded/12.92,pow((max(encoded,0)+0.055)/1.055,2.4));
    return dot(l,float3(0.2126,0.7152,0.0722));
}

static const uint kValues=9;
groupshared float sums[kValues][256];

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint index : SV_GroupIndex) {
    float v[kValues];
    [unroll] for(uint k=0;k<kValues;++k)v[k]=0;
    if(id.x<width&&id.y<height){
        const float2 size=float2(width,height);
        const float2 pixel=float2(id.xy)+0.5;
        float2 f=flowTex.SampleLevel(pointClamp,pixel/size,0);
        const float speed=length(f);
        const float2 from=pixel+f;
        if(speed>0.5&&speed<64.0&&all(from>2.0)&&all(from<size-2.0)){
            const float lc=luminance(currentTex[id.xy].rgb);
            const float lp=luminance(previousTex.SampleLevel(linearClamp,from/size,0).rgb);
            if(min(lc,lp)>darkFloor){
                const float d=log(lc)-log(lp);
                if(abs(d)<0.3){
                    // Robust weight: moving objects, reflections and flow errors give large d.
                    const float r=d/0.04;
                    const float w=1.0/(1.0+r*r);
                    v[0]=w; v[1]=w*f.x; v[2]=w*f.y; v[3]=w*d;
                    v[4]=w*f.x*f.x; v[5]=w*f.x*f.y; v[6]=w*f.y*f.y; v[7]=w*f.x*d; v[8]=w*f.y*d;
                }
            }
        }
    }
    [unroll] for(uint k=0;k<kValues;++k)sums[k][index]=v[k];
    GroupMemoryBarrierWithGroupSync();
    [unroll] for(uint stride=128;stride>0;stride>>=1){
        if(index<stride){
            [unroll] for(uint k=0;k<kValues;++k)sums[k][index]+=sums[k][index+stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if(index==0){
        const uint2 at=uint2(group.x*3,group.y);
        partialTex[at]=float4(sums[0][0],sums[1][0],sums[2][0],sums[3][0]);
        partialTex[at+uint2(1,0)]=float4(sums[4][0],sums[5][0],sums[6][0],sums[7][0]);
        partialTex[at+uint2(2,0)]=float4(sums[8][0],0,0,0);
    }
}

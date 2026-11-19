// Highlight clamp for generated frames: small bright details (sky through leaves, glints)
// are where frame generation errs most, and in HDR a generated highlight brighter, dimmer or
// displaced against the real frames around it reads as a shimmer at the output rate.
// Each bright generated pixel is clamped to what the two real frames hold at the matching
// place, found along the optical flow (current -> previous, work pixels, one source frame):
//   loose range: per-channel min/max of the 3x3 neighbourhoods in both real frames;
//   tight range: between the two real frames' values at the pixel itself.
// `strength` blends loose (0) to tight (1). Dark and mid-tone pixels are left untouched.
Texture2D<float4> generatedTex : register(t0);
Texture2D<float2> flowTex : register(t1);
Texture2D<float4> currentTex : register(t2);   // real frame after the generated one
Texture2D<float4> previousTex : register(t3);  // real frame before it
RWTexture2D<float4> outputTex : register(u0);
SamplerState pointClamp : register(s0);
SamplerState linearClamp : register(s1);
cbuffer Constants : register(b0) {
    uint width, height;
    float phase;      // the generated frame's time between previous (0) and current (1)
    float flowValid;  // 0: no flow this frame, the neighbourhoods are taken in place
    float brightLow, brightHigh; // brightness (max channel, output encoding) ramp where the clamp starts / is full
    float strength;
    float pad0;
};

[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 generated=generatedTex[id.xy];
    const float2 size=float2(width,height);
    const float2 pixel=float2(id.xy)+0.5;
    float2 flow=flowValid>0?flowTex.SampleLevel(linearClamp,pixel/size,0):0;
    if(any(isnan(flow)))flow=0;
    // What sits here at `phase` was at pixel + flow*phase before and pixel - flow*(1-phase) after.
    const float2 atPrevious=pixel+flow*phase;
    const float2 atCurrent=pixel-flow*(1-phase);
    const float3 previousCentre=previousTex.SampleLevel(linearClamp,atPrevious/size,0).rgb;
    const float3 currentCentre=currentTex.SampleLevel(linearClamp,atCurrent/size,0).rgb;
    float3 low=min(previousCentre,currentCentre),high=max(previousCentre,currentCentre);
    const float3 tightLow=low,tightHigh=high;
    const int2 last=int2(width,height)-1;
    const int2 p0=int2(floor(atPrevious)),c0=int2(floor(atCurrent));
    [unroll] for(int y=-1;y<=1;++y)
        [unroll] for(int x=-1;x<=1;++x){
            const float3 a=previousTex[clamp(p0+int2(x,y),int2(0,0),last)].rgb;
            const float3 b=currentTex[clamp(c0+int2(x,y),int2(0,0),last)].rgb;
            low=min(low,min(a,b));high=max(high,max(a,b));
        }
    const float3 g=generated.rgb;
    // Bright where the generated pixel or the real frames around it are bright: catches a
    // highlight the generator invented as well as one it dimmed.
    const float bright=max(max(g.r,max(g.g,g.b)),max(tightHigh.r,max(tightHigh.g,tightHigh.b)));
    const float weight=smoothstep(brightLow,brightHigh,bright);
    if(!(weight>0)){outputTex[id.xy]=generated;return;}
    const float3 clamped=clamp(g,lerp(low,tightLow,strength),lerp(high,tightHigh,strength));
    outputTex[id.xy]=float4(lerp(g,clamped,weight),generated.a);
}

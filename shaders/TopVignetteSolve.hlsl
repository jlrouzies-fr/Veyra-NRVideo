// Edge vignette compensation, step 2 of 3: one group sums the partials and updates the state of
// one edge (texel 0 top, 1 bottom). The edge band is "abnormally dark" by how many stops its geometric mean sits below the whole
// frame's, beyond an allowance (a sky is often a little darker at the top). The lift is that
// excess, capped at maxStops, faded out for dark scenes, and eased in over about a second so a
// scene change never jumps it.
Texture2D<float4> partialTex : register(t0);
RWTexture2D<float4> stateTex : register(u0);   // 2x1: per edge (lift stops, scene geometric mean nits, release frames left, -)
cbuffer Constants : register(b0) {
    uint groupsX, groupsY;
    float maxStops;
    float allowanceStops;
    float ease;          // share of the way to a higher target per real frame
    uint reset;
    float darkNits, fullNits;   // scene geometric mean: off below darkNits, full above fullNits
    uint side;                  // state texel: 0 top, 1 bottom
    float rolloffMean;          // mean of the roll-off weight over the band rows (0..1]
    float easeDown;             // the same while releasing after a scene change
    uint pad2;
};

groupshared float4 sums[1024];

[numthreads(1024,1,1)]
void main(uint index : SV_GroupIndex) {
    float4 v=0;
    const uint count=groupsX*groupsY;
    for(uint i=index;i<count;i+=1024)v+=partialTex[uint2(i%groupsX,i/groupsX)];
    sums[index]=v;
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=512;stride>0;stride>>=1){
        if(index<stride)sums[index]+=sums[index+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if(index!=0)return;
    const float4 s=sums[0];
    if(s.y<1||s.w<1)return;
    const float top=s.x/s.y;
    // The reference: the rows beyond the corrected reach, so the darkened zone never drags its
    // own reference down.
    const float all=s.z/s.w;
    const float sceneNits=exp(all);
    const float deficit=(all-top)/log(2.0);
    const float active=saturate((sceneNits-darkNits)/max(fullNits-darkNits,1e-3));
    // The band gets the edge lift times the roll-off weight; divide by its mean so the band
    // as a whole reaches the reference (less the allowance).
    const float target=clamp((deficit-allowanceStops)/max(rolloffMean,0.05),0,maxStops)*active;
    float4 state=reset?float4(target,sceneNits,0,0):stateTex[uint2(side,0)];
    // Slow both ways, so a passing car or a camera turn never pumps the lift. Only a real
    // scene change (the reference jumps up by more than 1.5 stops against its eased level, as
    // when stepping from a dark interior into daylight) lets the lift go fast, for 10 frames.
    if(!reset&&state.y>0&&log2(sceneNits/state.y)>1.5)state.z=10;
    const bool release=state.z>0.5;
    state.x=lerp(state.x,target,reset?1.0:release&&target<state.x?easeDown:ease);
    state.y=lerp(state.y,sceneNits,reset?1.0:release?easeDown:ease);
    state.z=max(state.z-1,0);
    stateTex[uint2(side,0)]=state;
}

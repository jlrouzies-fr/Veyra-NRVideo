// Edge vignette compensation, step 2 of 3: one group turns the partials into the state of one
// edge (texel 0 top, 1 bottom).
// Each column of blocks gives its own gap: how many stops the edge band sits below the central
// reference rows in that column. A vignette darkens every column alike; a car, a building or
// the HUD only some. The MEDIAN of the column gaps is therefore the vignette estimate, and the
// median of the column references the scene level. The lift is the gap beyond the allowance,
// divided by the mean roll-off weight of the band, capped at maxStops, faded out for dark
// scenes and eased.
Texture2D<float4> partialTex : register(t0);
RWTexture2D<float4> stateTex : register(u0);   // 2x1: per edge (lift stops, scene geometric mean nits, release frames left, -)
cbuffer Constants : register(b0) {
    uint groupsX, groupsY;
    float maxStops;
    float allowanceStops;
    float ease;          // share of the way to the target per real frame
    uint reset;
    float darkNits, fullNits;   // scene level (median column reference): off below darkNits, full above fullNits
    uint side;                  // state texel: 0 top, 1 bottom
    float rolloffMean;          // mean of the roll-off weight over the band rows (0..1]
    float easeDown;             // the same while releasing after a scene change
    uint pad2;
};

groupshared float colGap[1024], colRef[1024];
groupshared uint colValid[1024];
groupshared float medianGap, medianRef;
groupshared uint validCount;

[numthreads(1024,1,1)]
void main(uint index : SV_GroupIndex) {
    float gap=0,ref=0;uint valid=0;
    if(index<groupsX){
        float4 v=0;
        for(uint j=0;j<groupsY;++j)v+=partialTex[uint2(index,j)];
        if(v.y>=1&&v.w>=1){ref=v.z/v.w;gap=(ref-v.x/v.y)/log(2.0);valid=1;}
    }
    colGap[index]=gap;colRef[index]=ref;colValid[index]=valid;
    if(index==0){medianGap=0;medianRef=0;validCount=0;}
    GroupMemoryBarrierWithGroupSync();
    if(valid){
        uint count=0,rankGap=0,rankRef=0;
        for(uint k=0;k<groupsX;++k){
            if(!colValid[k])continue;
            ++count;
            if(colGap[k]<gap||(colGap[k]==gap&&k<index))++rankGap;
            if(colRef[k]<ref||(colRef[k]==ref&&k<index))++rankRef;
        }
        if(rankGap==count/2){medianGap=gap;validCount=count;}
        if(rankRef==count/2)medianRef=ref;
    }
    GroupMemoryBarrierWithGroupSync();
    if(index!=0||validCount==0)return;
    const float sceneNits=exp(medianRef);
    const float active=saturate((sceneNits-darkNits)/max(fullNits-darkNits,1e-3));
    const float target=clamp((medianGap-allowanceStops)/max(rolloffMean,0.05),0,maxStops)*active;
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

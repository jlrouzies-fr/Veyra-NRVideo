// Edge vignette compensation, step 2 of 3: one group sums the partials and updates the state of
// one edge (texel 0 top, 1 bottom). The edge band is "abnormally dark" by how many stops its geometric mean sits below the whole
// frame's, beyond an allowance (a sky is often a little darker at the top). The lift is that
// excess, capped at maxStops, faded out for dark scenes, and eased in over about a second so a
// scene change never jumps it.
Texture2D<float4> partialTex : register(t0);
RWTexture2D<float4> stateTex : register(u0);   // 2x1: per edge (lift stops, scene geometric mean nits, -, -)
cbuffer Constants : register(b0) {
    uint groupsX, groupsY;
    float maxStops;
    float allowanceStops;
    float ease;          // share of the way to the target per real frame
    uint reset;
    float darkNits, fullNits;   // scene geometric mean: off below darkNits, full above fullNits
    uint side;                  // state texel: 0 top, 1 bottom
    uint pad0, pad1, pad2;
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
    const float top=s.x/s.y, all=s.z/s.w;
    const float sceneNits=exp(all);
    const float deficit=(all-top)/log(2.0);
    const float active=saturate((sceneNits-darkNits)/max(fullNits-darkNits,1e-3));
    const float target=clamp(deficit-allowanceStops,0,maxStops)*active;
    float4 state=reset?float4(target,sceneNits,0,0):stateTex[uint2(side,0)];
    state.x=lerp(state.x,target,reset?1.0:ease);
    state.y=lerp(state.y,sceneNits,reset?1.0:ease);
    stateTex[uint2(side,0)]=state;
}

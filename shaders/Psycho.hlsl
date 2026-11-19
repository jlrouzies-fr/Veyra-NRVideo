// Psycho tone mapping (RenoDX psychotm_test22) on each real frame before frame generation, for
// HDR output: compresses highlights toward a chosen peak with cone-space adaptation, hue
// restoration and gamut compression, keeping mid-grey where it is. The RenoDX shader library
// under renodx/ is MIT licensed (renodx/LICENSE).
// Working units: linear BT.709 with 1 = reference white (203 nits), as the library expects.
#include "HdrColor.hlsli"
#include "renodx/tonemap/psychov/test22.hlsl"
Texture2D<float4> frameTex : register(t0);
RWTexture2D<float4> outputTex : register(u0);
cbuffer Constants : register(b0) {
    uint width, height;
    float outputMode;   // 1 scRGB, 2 PQ (SDR output never runs this pass)
    float peakNits;
    float whiteNits;
    float pad0, pad1, pad2;
};
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    const float4 centre=frameTex[id.xy];
    // Output signal -> linear BT.709 scRGB (1 = 80 nits) -> reference-white units.
    const float3 scrgb=outputMode>1.5?HdrDecodePq(centre.rgb):centre.rgb;
    const float3 scene=scrgb*(80.0/whiteNits);
    const float3 mapped=renodx::tonemap::psychov::psychotm_test22(scene,peakNits/whiteNits);
    const float3 result=mapped*(whiteNits/80.0);
    outputTex[id.xy]=float4(outputMode>1.5?HdrEncodePq(result):result,centre.a);
}

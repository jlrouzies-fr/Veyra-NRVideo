// Anti-vignette, step 2 of 3: one group of 1024 threads, one thread per screen tile.
//  1. Sum the tile's group partials into its normal equations; remove the frame's exposure
//     change c (weighted mean of d over the whole frame).
//  2. Keep a decaying history of the equations (a few seconds), solve grad(logV) per tile.
//     Then remove the whole-screen tilt (median gradient of the confident tiles).
//  3. Integrate the gradient field into logV (weighted Jacobi, warm-started every frame):
//     confident tiles constrain their neighbours, unmeasured ones just follow.
//  4. gain = exp(brightest - logV) above a dead zone, capped, scaled by strength and by the
//     local confidence (tiles nobody measured are left alone).
Texture2D<float4> partialTex : register(t0);
RWTexture2D<float4> stateTex : register(u0);   // (TX*2) x TY: (Axx,Axy,Ayy,-) (bx,by,u,-)
RWTexture2D<float> gainTex : register(u1);     // TX x TY
cbuffer Constants : register(b0) {
    uint groupsX, groupsY, tileGroups, tilesX;
    uint tilesY, iterations, reset, pad0;
    float decay, tilePixels, strength, maxLog;
    float deadZone, confidenceScale, pad1, pad2;
};

groupshared float sumA[1024], sumB[1024];
groupshared float U[1024], GX[1024], GY[1024], C[1024];

uint tileAt(int x, int y) { return uint(y)*tilesX+uint(x); }

[numthreads(1024,1,1)]
void main(uint index : SV_GroupIndex) {
    const uint count=tilesX*tilesY;
    const bool valid=index<count;
    const int tx=int(index%tilesX), ty=int(index/tilesX);
    float w=0,wfx=0,wfy=0,wd=0,axx=0,axy=0,ayy=0,bx=0,by=0;
    if(valid){
        for(uint j=0;j<tileGroups;++j)for(uint i=0;i<tileGroups;++i){
            const uint gx=uint(tx)*tileGroups+i, gy=uint(ty)*tileGroups+j;
            if(gx>=groupsX||gy>=groupsY)continue;
            const float4 a=partialTex[uint2(gx*3,gy)];
            const float4 b=partialTex[uint2(gx*3+1,gy)];
            const float4 c=partialTex[uint2(gx*3+2,gy)];
            w+=a.x; wfx+=a.y; wfy+=a.z; wd+=a.w;
            axx+=b.x; axy+=b.y; ayy+=b.z; bx+=b.w; by+=c.x;
        }
    }
    // Whole-frame exposure change.
    sumA[index]=w; sumB[index]=wd;
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=512;stride>0;stride>>=1){
        if(index<stride){sumA[index]+=sumA[index+stride];sumB[index]+=sumB[index+stride];}
        GroupMemoryBarrierWithGroupSync();
    }
    const float exposure=sumA[0]>1?sumB[0]/sumA[0]:0;
    bx-=exposure*wfx; by-=exposure*wfy;

    float4 s0=0,s1=0;
    if(valid){
        if(reset==0){s0=stateTex[uint2(tx*2,ty)];s1=stateTex[uint2(tx*2+1,ty)];}
        s0.xyz=s0.xyz*decay+float3(axx,axy,ayy);
        s1.xy=s1.xy*decay+float2(bx,by);
        // Least squares for d = -g.f: g = -(A + lambda I)^-1 b.
        const float trace=s0.x+s0.z;
        const float lambda=1e-3*trace+1.0;
        const float a11=s0.x+lambda,a12=s0.y,a22=s0.z+lambda;
        const float det=a11*a22-a12*a12;
        const float2 g=det>0?-float2(a22*s1.x-a12*s1.y,a11*s1.y-a12*s1.x)/det:0;
        GX[index]=g.x*tilePixels; GY[index]=g.y*tilePixels;   // log change per tile step
        C[index]=trace/(trace+confidenceScale);
        U[index]=s1.z;
    } else {
        GX[index]=0; GY[index]=0; C[index]=0; U[index]=0;
    }
    GroupMemoryBarrierWithGroupSync();

    // A whole-screen tilt cannot be told from an exposure change (a pan moves every pixel the
    // same way), so the fit carries an arbitrary constant gradient. A vignette covers a
    // minority of the screen: take the median gradient of the confident tiles as that tilt.
    const bool counted=valid&&C[index]>0.3;
    sumA[index]=counted?1:0;
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=512;stride>0;stride>>=1){
        if(index<stride)sumA[index]+=sumA[index+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    const float confident=sumA[0];
    GroupMemoryBarrierWithGroupSync();
    float tilt[2]={0,0};
    if(confident>0){
        [unroll] for(uint axis=0;axis<2;++axis){
            float low=-4,high=4;   // log change per tile step
            for(uint step=0;step<20;++step){
                const float mid=0.5*(low+high);
                const float value=axis==0?GX[index]:GY[index];
                sumA[index]=counted&&value<mid?1:0;
                GroupMemoryBarrierWithGroupSync();
                for(uint stride=512;stride>0;stride>>=1){
                    if(index<stride)sumA[index]+=sumA[index+stride];
                    GroupMemoryBarrierWithGroupSync();
                }
                if(sumA[0]<confident*0.5)low=mid; else high=mid;
                GroupMemoryBarrierWithGroupSync();
            }
            tilt[axis]=0.5*(low+high);
        }
    }
    if(valid){GX[index]-=tilt[0];GY[index]-=tilt[1];}
    GroupMemoryBarrierWithGroupSync();

    // Weighted Jacobi: u_j - u_i ~= mean gradient of i and j along the step, weight min(C_i, C_j).
    for(uint it=0;it<iterations;++it){
        float next=0;
        if(valid){
            float num=0,den=1e-4;   // tiny pull to 0 keeps an unmeasured field flat
            const uint me=index;
            if(tx>0){const uint o=tileAt(tx-1,ty);const float k=min(C[me],C[o]);num+=k*(U[o]+0.5*(GX[me]+GX[o]));den+=k;}
            if(tx+1<int(tilesX)){const uint o=tileAt(tx+1,ty);const float k=min(C[me],C[o]);num+=k*(U[o]-0.5*(GX[me]+GX[o]));den+=k;}
            if(ty>0){const uint o=tileAt(tx,ty-1);const float k=min(C[me],C[o]);num+=k*(U[o]+0.5*(GY[me]+GY[o]));den+=k;}
            if(ty+1<int(tilesY)){const uint o=tileAt(tx,ty+1);const float k=min(C[me],C[o]);num+=k*(U[o]-0.5*(GY[me]+GY[o]));den+=k;}
            next=num/den;
        }
        GroupMemoryBarrierWithGroupSync();
        if(valid)U[index]=next;
        GroupMemoryBarrierWithGroupSync();
    }

    // Light centre-weighted smoothing (keeps the screen edges, where vignettes are strongest)
    // and the confidence spread by one tile.
    float smooth=0,cover=0;
    if(valid){
        float total=3;
        smooth=3*U[index];
        for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){
            const int px=tx+x,py=ty+y;
            if(px<0||py<0||px>=int(tilesX)||py>=int(tilesY))continue;
            const uint o=tileAt(px,py);
            smooth+=U[o];total+=1;cover=max(cover,C[o]);
        }
        smooth/=total;
    }
    // Brightest confident level: the reference nothing is brightened beyond.
    sumA[index]=valid&&C[index]>0.3?smooth:-1e9;
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=512;stride>0;stride>>=1){
        if(index<stride)sumA[index]=max(sumA[index],sumA[index+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if(!valid)return;
    const float brightest=sumA[0];
    float logGain=0;
    if(brightest>-1e8)logGain=clamp(brightest-smooth-deadZone,0,maxLog)*strength*saturate((cover-0.3)/0.4);
    s1.z=U[index];
    stateTex[uint2(tx*2,ty)]=s0;
    stateTex[uint2(tx*2+1,ty)]=s1;
    gainTex[uint2(tx,ty)]=exp(logGain);
}

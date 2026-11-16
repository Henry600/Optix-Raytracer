#ifndef OPTIX_RAYTRACER_SURFACE_CONTEXT_H
#define OPTIX_RAYTRACER_SURFACE_CONTEXT_H

#include "dev_common.h"
#include "path_payload.h"
#include "path_state.h"
#include "pbr.h"

// Per-vertex quantities shared by every lobe.
struct SurfaceCtx
{
    float3 Nf;        // front-facing normal (faces the incoming ray)
    float  cosV;      // dot(Nf, -rayDir)
    float  alpha;     // base GGX alpha = roughness^2
    float  cc_alpha;  // coat GGX alpha
    float3 cc_F;      // coat Fresnel (achromatic)
    float  p_coat;    // probability of sampling the coat specular lobe
};

static __forceinline__ __device__
SurfaceCtx makeSurfaceCtx(const PathVertex& vtx, const float3 rayDir)
{
    SurfaceCtx sc;
    sc.alpha = fmaxf(vtx.roughness * vtx.roughness, 1e-3f);

    // Front-facing normal: always faces the incoming ray.
    sc.Nf   = (devDot(vtx.N, -rayDir) >= 0.0f) ? vtx.N : -vtx.N;
    sc.cosV = devDot(sc.Nf, -rayDir);  // shared by clearcoat + base Fresnel

    // Clearcoat Fresnel (Schlick, F0 from the coat IOR; 1.5 -> 0.04)
    sc.cc_alpha = fmaxf(vtx.clearcoatRoughness * vtx.clearcoatRoughness, 1e-3f);
    const float cc_f0 = ((vtx.clearcoatIor - 1.0f) / (vtx.clearcoatIor + 1.0f))
                      * ((vtx.clearcoatIor - 1.0f) / (vtx.clearcoatIor + 1.0f));
    sc.cc_F    = devFresnel(sc.cosV, make_float3(cc_f0, cc_f0, cc_f0));
    sc.p_coat  = vtx.clearcoat * devLuminance(sc.cc_F);
    return sc;
}

#endif // OPTIX_RAYTRACER_SURFACE_CONTEXT_H

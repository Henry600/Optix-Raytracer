#ifndef OPTIX_RAYTRACER_SHADOW_CATCHER_H
#define OPTIX_RAYTRACER_SHADOW_CATCHER_H

#include "dev_common.h"
#include "shadow_ray.h"
#include "path_payload.h"
#include "path_state.h"
#include "rng_sampling.h"
#include "environment.h"
#include "emissive_lights.h"

// Invisible surface: shows the background along the incoming ray direction,
// darkened by how occluded this point's sky hemisphere is.  No BSDF, no
// further bounces - the caller terminates the path afterwards.
//
// The shadow-ray direction is importance-sampled toward the env map's bright
// regions (or an emissive light) so a blocker in front of the sun reads as a
// hard directional shadow.  If the sample falls below the local horizon it
// falls back to a cosine-weighted hemisphere sample, which is always valid -
// every launch must contribute some value to the accumulator, otherwise the
// average would never go fully dark under total occlusion.
static __forceinline__ __device__
void shadeShadowCatcher(const PathVertex& vtx, PathState& ps)
{
    float3& rayDir = ps.rayDir;
    float3& throughput = ps.throughput;
    float3& radiance = ps.radiance;
    uint32_t& seed = ps.seed;

    const float3 bg  = sampleBackground(rayDir);
    const float3 Nfc = (devDot(vtx.N, -rayDir) >= 0.0f) ? vtx.N : -vtx.N;

    float3 skyDir;
    float  skyTMax = 1e30f;
    bool   haveDir = false;

    if (optixLaunchParams.envMarginalCdf)
    {
        float3 neeDir; float neePdf;
        sampleEnvMapIS(rnd(seed), rnd(seed), neeDir, neePdf);
        if (neePdf > 0.0f && devDot(neeDir, Nfc) > 0.0f)
        {
            skyDir  = neeDir;
            haveDir = true;
        }
    }
    else if (optixLaunchParams.emissiveLightCount > 0)
    {
        float3 lightPos, lightN;
        const EmissiveLightData* selectedLight = nullptr;
        float lightPdf = 0.0f;
        if (sampleEmissiveLight(vtx.pos, rnd(seed), rnd(seed), rnd(seed), rnd(seed),
                lightPos, lightN, selectedLight, lightPdf))
        {
            const float3 toLight = lightPos - vtx.pos;
            const float  dist    = sqrtf(devDot(toLight, toLight));
            const float3 dir     = toLight * (1.0f / dist);
            if (devDot(dir, Nfc) > 0.0f)
            {
                skyDir  = dir;
                skyTMax = dist - 2e-3f;
                haveDir = true;
            }
        }
    }

    if (!haveDir)
    {
        float3 T, B;
        buildONB(Nfc, T, B);
        const float3 d = cosineSampleHemisphere(rnd(seed), rnd(seed));
        skyDir  = devNormalize(T * d.x + B * d.y + Nfc * d.z);
        skyTMax = 1e30f;
    }

    float3 shadowFilter;
    const bool shadowVis = traceShadow(vtx.pos + Nfc * 1e-3f, skyDir, skyTMax, shadowFilter);
    const float3 shadow = shadowVis ? shadowFilter : make_float3(0.0f, 0.0f, 0.0f);

    radiance += throughput * bg * shadow;
}

#endif // OPTIX_RAYTRACER_SHADOW_CATCHER_H

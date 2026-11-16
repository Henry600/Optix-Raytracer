#ifndef OPTIX_RAYTRACER_RAYGEN_RENDER_H
#define OPTIX_RAYTRACER_RAYGEN_RENDER_H

#include "dev_common.h"
#include "path_payload.h"
#include "path_state.h"
#include "rng_sampling.h"
#include "display.h"
#include "splat_gather.h"
#include "camera_ray.h"
#include "path_events.h"
#include "shadow_catcher.h"
#include "surface_context.h"
#include "coat_lobe.h"
#include "base_lobe.h"

// ─── Raygen — iterative path loop ────────────────────────────────────────────
// Drives the whole bounce loop; every stage lives in its own header:
//   camera_ray.h        primary ray (jitter + thin-lens DoF)
//   path_events.h       guide layers, background / env, absorption + emission
//   shadow_catcher.h    invisible shadow-only surface
//   surface_context.h   per-vertex values shared by the lobes
//   coat_lobe.h         clearcoat specular + coat darkening
//   base_lobe.h         base dispatch -> specular / diffuse / transmission lobes

extern "C" __global__ void __raygen__renderFrame()
{
    const uint3        idx   = optixGetLaunchIndex();
    const uint3        dim   = optixGetLaunchDimensions();
    const unsigned int fbIdx = idx.y * dim.x + idx.x;

    // ── Path tracing ──────────────────────────────────────────────────────────
    // traversable == 0 (no scene / everything hidden) never reaches this raygen —
    // application.cpp skips optixLaunch entirely in that case.

    // Seed: mix pixel index and sample index for uncorrelated sequences
    PathState ps;
    initPathState(ps, fbIdx * 1973u + optixLaunchParams.sampleIndex * 9277u + 127u);
    pcgNext(ps.seed);  // warm up

    MediumStack ms;
    initMediumStack(ms);

    generateCameraRay(idx, dim, ps.seed, ps.rayOrig, ps.rayDir);

    float3& rayOrig    = ps.rayOrig;
    float3& rayDir     = ps.rayDir;
    float3& throughput = ps.throughput;
    float3& radiance   = ps.radiance;

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        PathVertex vtx;
        vtx.hit        = 0;
        vtx.instanceId = ~0u;

        uint32_t p0, p1;
        packPointer(&vtx, p0, p1);

        // Splat-only scenes have nothing for this trace to hit — skip it.
        if (optixLaunchParams.hasGeometry)
        {
            optixTrace(
                optixLaunchParams.traversable,
                rayOrig, rayDir,
                1e-3f, 1e30f, 0.0f,
                OptixVisibilityMask(VIS_MASK_GEOMETRY),  // splats composite in a separate gather
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // anyhit only needed for shadow rays
                0, 1, 0,
                p0, p1);
        }

        // ── Gaussian splat volume segment ────────────────────────────────────
        // Composite splats between the ray origin and the surface hit (or to
        // infinity on miss).  Splats add in-scattered radiance and attenuate
        // everything behind them — including the env map on miss below.
        if (optixLaunchParams.hasSplats)
        {
            float3 splatC;
            float  splatT;
            traceSplatSegment(rayOrig, rayDir, 1e-3f,
                              vtx.hit ? vtx.t : 1e30f,
                              splatC, splatT);
            radiance   += throughput * splatC;
            throughput  = throughput * splatT;
            if (fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)) < 1e-4f)
            {
                break;
            }
        }

        if (bounce == 0)
        {
            writeGuideLayers(fbIdx, vtx);
        }

        if (!vtx.hit)
        {
            shadeBackground(ps);
            break;
        }

        if (vtx.shadowCatcher)
        {
            shadeShadowCatcher(vtx, ps);
            break;
        }

        applyAbsorptionAndEmission(vtx, ps);

        // ── Lobe selection: clearcoat -> base (specular / diffuse / refraction) ──
        // Clearcoat is sampled first: it's a dielectric layer on top of the base
        // material.  If not chosen, the base receives the energy the coat lets
        // through (see applyCoatDarkening).
        const SurfaceCtx sc = makeSurfaceCtx(vtx, rayDir);

        bool alive;
        if (sc.p_coat > 0.0f && rnd(ps.seed) < sc.p_coat)
        {
            alive = sampleCoat(vtx, sc, ps);
        }
        else
        {
            alive = sampleBase(vtx, sc, ps, ms);
        }
        if (!alive)
        {
            break;
        }

        // Russian roulette: stochastically terminate dim paths (all branches)
        if (bounce >= 3)
        {
            const float maxThr = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rnd(ps.seed) > maxThr)
            {
                break;
            }
            throughput *= 1.0f / fmaxf(maxThr, 1e-6f);
        }
    }

    // ── Accumulate ────────────────────────────────────────────────────────────
    float4& acc = optixLaunchParams.accumBuffer[fbIdx];
    acc.x += radiance.x;
    acc.y += radiance.y;
    acc.z += radiance.z;

    // ── HDR average for denoiser ──────────────────────────────────────────────
    const float  inv = 1.0f / (float)(optixLaunchParams.sampleIndex + 1);
    if (optixLaunchParams.hdrBuffer)
    {
        optixLaunchParams.hdrBuffer[fbIdx] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 1.0f);
    }

    // ── Display encode ────────────────────────────────────────────────────────
    // Reinhard (hdrDisplay=0) or unclamped (hdrDisplay=1) → linear float4 scRGB.
    writeDisplay(fbIdx, make_float3(acc.x * inv, acc.y * inv, acc.z * inv));
}

#endif // OPTIX_RAYTRACER_RAYGEN_RENDER_H

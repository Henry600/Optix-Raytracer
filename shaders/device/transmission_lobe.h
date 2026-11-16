#ifndef OPTIX_RAYTRACER_TRANSMISSION_LOBE_H
#define OPTIX_RAYTRACER_TRANSMISSION_LOBE_H

#include "dev_common.h"
#include "path_payload.h"
#include "path_state.h"
#include "surface_context.h"
#include "rng_sampling.h"
#include "pbr.h"

// Transmission: thin-walled pass-through, or thick refraction with volumetric
// scattering and the nested-dielectric medium stack.
static __forceinline__ __device__
void sampleTransmission(const PathVertex& vtx, const SurfaceCtx& sc, PathState& ps, MediumStack& ms)
{
    float3& rayOrig = ps.rayOrig;
    float3& rayDir = ps.rayDir;
    float3& throughput = ps.throughput;
    float& bsdfPdfForMis = ps.bsdfPdfForMis;
    float3& absorb = ps.absorb;
    float (&medIor)[MediumStack::kMaxDepth] = ms.medIor;
    float3 (&medAbsorb)[MediumStack::kMaxDepth] = ms.medAbsorb;
    int& medTop = ms.medTop;
    uint32_t& seed = ps.seed;
    constexpr int kMaxMediumDepth = MediumStack::kMaxDepth;
    const float3 Nf = sc.Nf;
    const float alpha = sc.alpha;

        bsdfPdfForMis = 0.0f;  // refraction/pass-through — no diffuse NEE on next hit

    if (vtx.thinWalled)
    {
        // Zero-thickness surface: front and back refractions cancel so
        // there is no medium interior and no Beer-Lambert absorption.
        // Apply albedo as a flat transmission tint — consistent with how
        // the any-hit shader filters NEE shadow rays through thin glass.
        // vtx.transmissionColor is already linear (linearised in applyMaterial).
        throughput.x *= vtx.transmissionColor.x;
        throughput.y *= vtx.transmissionColor.y;
        throughput.z *= vtx.transmissionColor.z;
        // Nf faces the incoming ray, so -Nf steps to the far side.
        rayOrig = vtx.pos - Nf * 1e-3f;
        // rayDir and absorb unchanged — medium is unaffected.
    }
    else
    {
    // ── 3. Refraction (rough Snell's law via GGX microfacet) ─────────
    // vtx.N is the true outward geometry normal — used only for the
    // entering/exiting test, not for the microfacet direction.
    const bool   entering = (devDot(rayDir, vtx.N) < 0.0f);
    const float3 faceN    = entering ? vtx.N : -vtx.N;
    const float  eta      = entering
                          ? (medIor[medTop] / vtx.ior)
                          : (vtx.ior / medIor[medTop > 0 ? medTop - 1 : 0]);

    // ── Volumetric scattering ─────────────────────────────────────────
    // Per-channel delta tracking: sample the collision event at σ_max
    // (the highest per-channel rate, typically blue for Rayleigh media).
    // When a scatter fires, weight throughput by σ_c/σ_max per channel.
    // Channels with lower σ carry reduced throughput along the scattered
    // direction, making the transmitted beam yellow when blue scatters more.
    // Beer-Lambert was already applied for vtx.t; undo the excess portion.
    const float3& sc       = vtx.scatteringCoeff;
    const float   sigmaMax = fmaxf(fmaxf(sc.x, sc.y), sc.z);
    bool scattered = false;
    if (!entering && sigmaMax > 0.0f)
    {
        const float t_scatter = -logf(fmaxf(1e-6f, rnd(seed))) / sigmaMax;
        if (t_scatter < vtx.t)
        {
            const float excess = vtx.t - t_scatter;
            throughput.x *= expf(absorb.x * excess);
            throughput.y *= expf(absorb.y * excess);
            throughput.z *= expf(absorb.z * excess);

            // Chromatic ratio weight — unbiased delta-tracking estimator.
            throughput.x *= sc.x / sigmaMax;
            throughput.y *= sc.y / sigmaMax;
            throughput.z *= sc.z / sigmaMax;

            const float3 scatterPos = rayOrig + rayDir * t_scatter;
            float3 Ts, Bs;
            buildONB(rayDir, Ts, Bs);
            const float3 localDir = sampleHenyeyGreenstein(
                vtx.scatteringAnisotropy, rnd(seed), rnd(seed));
            rayDir  = devNormalize(Ts * localDir.x + Bs * localDir.y + rayDir * localDir.z);
            rayOrig = scatterPos;
            scattered = true;
        }
    }

    if (!scattered)
    {
    // Sample a GGX microfacet normal (same VNDF as the specular lobe).
    // At alpha → 0 this converges to the macro normal → smooth glass.
    // At higher alpha the half-vector is scattered → frosted glass.
    float3 Tt, Bt;
    buildONB(Nf, Tt, Bt);
    const float3 Vt       = -rayDir;
    const float3 Vt_local = make_float3(
        devDot(Vt, Tt), devDot(Vt, Bt), fmaxf(1e-4f, devDot(Vt, Nf)));
    const float3 Ht_local = devSampleGGX_VNDF(Vt_local, alpha, rnd(seed), rnd(seed));
    const float3 Ht       = devNormalize(Tt * Ht_local.x + Bt * Ht_local.y + Nf * Ht_local.z);

    // Refract through the microfacet normal; TIR falls back to reflection.
    const bool refracted = devRefract(rayDir, Ht, eta, rayDir);

    // Smith G1 on the transmitted direction — mirrors the specular weight.
    // Skip at eta≈1: the refracted direction equals the incident direction, so
    // G1(cosT,α) would incorrectly darken grazing angles on an IOR=1 surface.
    if (refracted && fabsf(eta - 1.0f) > 1e-4f)
    {
        // cosine against the far-side normal (-faceN)
        const float cosT = fmaxf(1e-4f, devDot(rayDir, -faceN));
        throughput *= devSmithG1(cosT, alpha);
    }

    // Update medium stack and Beer-Lambert absorption.
    // Push on entry, pop on exit; absorb is always medAbsorb[medTop] after.
    // On TIR the stack is unchanged — ray stays in the same medium.
    if (refracted)
    {
        if (entering)
        {
            if (medTop + 1 < kMaxMediumDepth)
            {
                ++medTop;
                medIor[medTop] = vtx.ior;
                // Total extinction σ_t = σ_absorption + σ_scattering per channel.
                // Scattering removes photons from the direct beam just like absorption;
                // including it here makes the transmitted beam yellow when blue
                // scatters more (Rayleigh).  The scatter event code undoes the excess
                // σ_t beyond the actual scatter point, keeping the estimator correct.
                const float invDist = 1.0f / vtx.absorptionDistance;
                medAbsorb[medTop].x = -logf(fmaxf(vtx.transmissionColor.x, 1e-6f)) * invDist + vtx.scatteringCoeff.x;
                medAbsorb[medTop].y = -logf(fmaxf(vtx.transmissionColor.y, 1e-6f)) * invDist + vtx.scatteringCoeff.y;
                medAbsorb[medTop].z = -logf(fmaxf(vtx.transmissionColor.z, 1e-6f)) * invDist + vtx.scatteringCoeff.z;
            }
        }
        else
        {
            if (medTop > 0) { --medTop; }
        }
        absorb = medAbsorb[medTop];
    }

    // On TIR devRefract writes the reflected direction; offset stays on
    // the same side.  On true refraction offset to the far side.
    rayOrig = vtx.pos + faceN * (refracted ? -1e-3f : 1e-3f);
    }   // end if (!scattered)
    }   // end thick refraction else (not thin-walled)
}

#endif // OPTIX_RAYTRACER_TRANSMISSION_LOBE_H

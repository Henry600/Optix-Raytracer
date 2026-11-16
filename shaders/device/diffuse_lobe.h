#ifndef OPTIX_RAYTRACER_DIFFUSE_LOBE_H
#define OPTIX_RAYTRACER_DIFFUSE_LOBE_H

#include "dev_common.h"
#include "shadow_ray.h"
#include "path_payload.h"
#include "path_state.h"
#include "surface_context.h"
#include "rng_sampling.h"
#include "pbr.h"
#include "environment.h"
#include "emissive_lights.h"
#include "eon_diffuse.h"

// Diffuse (EON, cosine-weighted sampling) with env-map and emissive-light NEE.
static __forceinline__ __device__
void sampleDiffuse(const PathVertex& vtx, const SurfaceCtx& sc, PathState& ps,
                   const float p_spec, const float p_trans)
{
    float3& rayOrig = ps.rayOrig;
    float3& rayDir = ps.rayDir;
    float3& throughput = ps.throughput;
    float3& radiance = ps.radiance;
    float& bsdfPdfForMis = ps.bsdfPdfForMis;
    uint32_t& seed = ps.seed;
    const float3 Nf = sc.Nf;
    const float cosV = sc.cosV;
    const float p_coat = sc.p_coat;

    // ── 2. Diffuse (EON rough diffuse, cosine-weighted sampling) ──────
    // σ = 0 reduces to Lambertian; the roughness factor is applied
    // per outgoing direction (NEE below, path continuation at the end).
    throughput *= vtx.albedo;
    const float3 woD = -rayDir;

    float3 T, B;
    buildONB(Nf, T, B);

    // ── NEE: direct env-map lighting ──────────────────────────
    // Sample the env map IS distribution, fire a shadow ray to
    // check visibility, and add a MIS-weighted direct contribution.
    if (optixLaunchParams.envMarginalCdf)
    {
        float3 neeDir; float neePdf;
        float3 neeL = sampleEnvMapIS(rnd(seed), rnd(seed), neeDir, neePdf);

        const float cosL = devDot(neeDir, Nf);
        if (cosL > 0.0f && neePdf > 0.0f)
        {
            // Shadow ray: SBT miss index 1 → __miss__shadow sets p0=1.
            // p1/p2/p3 carry the accumulated thin-walled colour filter.
            // No TERMINATE_ON_FIRST_HIT — anyhit calls optixTerminateRay()
            // for opaque surfaces so thin-walled filter order is near-to-far.
            float3 shadowFilter;
            if (traceShadow(vtx.pos + Nf * 1e-3f, neeDir, 1e30f, shadowFilter))
            {
                // throughput already carries kNS = (1-F)*albedo/(1-p_spec)
                // Lambertian: f×cosθ = kNS * (cosθ/π) evaluated at neeDir
                const float kInvPi = 0.31830988618f;
                const float pBsdf  = (1.0f - p_coat) * (1.0f - p_spec)
                                   * (1.0f - p_trans) * cosL * kInvPi;
                const float wNee   = (neePdf * neePdf)
                                   / (neePdf * neePdf + pBsdf * pBsdf);
                const float3 fON   = eonFactor(cosL, cosV, devDot(neeDir, woD),
                                               vtx.diffuseRoughness, vtx.albedo);
                radiance += throughput * shadowFilter
                          * fON * (cosL * kInvPi / neePdf) * neeL * wNee;
            }
        }
    }

    // ── NEE: diffuse direct emissive-light sampling ───────────
    if (optixLaunchParams.emissiveLightCount > 0)
    {
        float3 lightPos, lightN;
        const EmissiveLightData* selectedLight = nullptr;
        float lightPdf = 0.0f;
        if (sampleEmissiveLight(vtx.pos,
                rnd(seed), rnd(seed), rnd(seed), rnd(seed),
                lightPos, lightN, selectedLight, lightPdf))
        {
            const float3 toLight = lightPos - vtx.pos;
            const float  dist2   = devDot(toLight, toLight);
            const float  dist    = sqrtf(dist2);
            const float3 neeDir  = toLight * (1.0f / dist);
            const float  cosL    = devDot(neeDir, Nf);

            if (cosL > 0.0f)
            {
                float3 shadowFilter;
                if (traceShadow(vtx.pos + Nf * 1e-3f, neeDir, dist - 2e-3f, shadowFilter))
                {
                    const float kInvPi = 0.31830988618f;
                    const float pBsdf  = (1.0f - p_coat)
                        * (1.0f - p_spec) * (1.0f - p_trans)
                        * cosL * kInvPi;
                    const float wNee   = (lightPdf * lightPdf)
                        / (lightPdf * lightPdf + pBsdf * pBsdf);
                    const float3 fON   = eonFactor(cosL, cosV, devDot(neeDir, woD),
                                                   vtx.diffuseRoughness, vtx.albedo);
                    radiance += throughput * shadowFilter
                              * fON * (cosL * kInvPi / lightPdf)
                              * selectedLight->emission * wNee;
                }
            }
        }
    }

    // ── Continue: cosine-sample the hemisphere for the path ────
    const float3 d = cosineSampleHemisphere(rnd(seed), rnd(seed));
    rayDir  = devNormalize(T * d.x + B * d.y + Nf * d.z);
    rayOrig = vtx.pos + Nf * 1e-3f;

    // EON weight for the sampled direction (pdf is unchanged).
    throughput *= eonFactor(devDot(rayDir, Nf), cosV, devDot(rayDir, woD),
                            vtx.diffuseRoughness, vtx.albedo);

    // Store BSDF PDF for MIS in next-iteration miss block
    const float kInvPi = 0.31830988618f;
    bsdfPdfForMis = devDot(rayDir, Nf) * kInvPi;
}

#endif // OPTIX_RAYTRACER_DIFFUSE_LOBE_H

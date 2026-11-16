#ifndef OPTIX_RAYTRACER_SPECULAR_LOBE_H
#define OPTIX_RAYTRACER_SPECULAR_LOBE_H

#include "dev_common.h"
#include "shadow_ray.h"
#include "path_payload.h"
#include "path_state.h"
#include "surface_context.h"
#include "rng_sampling.h"
#include "pbr.h"
#include "environment.h"
#include "emissive_lights.h"

// Specular reflection (GGX-VNDF) with env-map and emissive-light NEE.
// Returns false when the sampled direction is below the horizon.
static __forceinline__ __device__
bool sampleSpecular(const PathVertex& vtx, const SurfaceCtx& sc, PathState& ps, MediumStack& ms,
                    const float3 F, const float p_spec)
{
    float3& rayOrig = ps.rayOrig;
    float3& rayDir = ps.rayDir;
    float3& throughput = ps.throughput;
    float3& radiance = ps.radiance;
    float& bsdfPdfForMis = ps.bsdfPdfForMis;
    float (&medIor)[MediumStack::kMaxDepth] = ms.medIor;
    int& medTop = ms.medTop;
    uint32_t& seed = ps.seed;
    const float3 Nf = sc.Nf;
    const float alpha = sc.alpha;
    const float p_coat = sc.p_coat;

    float3 T, B;
    buildONB(Nf, T, B);

    const float3 V       = -rayDir;
    const float3 V_local = make_float3(
        devDot(V, T), devDot(V, B), fmaxf(1e-4f, devDot(V, Nf)));

    const float3 H_local = devSampleGGX_VNDF(V_local, alpha, rnd(seed), rnd(seed));
    const float3 H       = devNormalize(T * H_local.x + B * H_local.y + Nf * H_local.z);
    const float3 L       = devReflect(rayDir, H);

    if (devDot(L, Nf) <= 0.0f)
    {
        return false;
    }

    const float cosNL = fmaxf(1e-4f, devDot(Nf, L));
    const float cosO  = V_local.z;   // dot(V, Nf), already clamped ≥ 1e-4

    // Branch probability compensation — applied before NEE so both
    // the direct and indirect contributions are correctly scaled.
    throughput = throughput * (1.0f / fmaxf(1e-4f, p_spec));

    // ── NEE: specular direct env-map lighting ──────────────────────
    if (optixLaunchParams.envMarginalCdf)
    {
        float3 neeDir; float neePdf;
        float3 neeL = sampleEnvMapIS(rnd(seed), rnd(seed), neeDir, neePdf);

        const float cosI_nee = devDot(neeDir, Nf);
        if (cosI_nee > 0.0f && neePdf > 0.0f)
        {
            const float3 H_nee     = devNormalize(V + neeDir);
            const float  cosH_nee  = fmaxf(0.0f, devDot(H_nee, Nf));
            const float  cosVH_nee = fmaxf(0.0f, devDot(V, H_nee));
            const float  fNEE_diel = devFresnelDielectric(cosVH_nee, medIor[medTop] / vtx.ior);
            const float3 F_nee     = devMix(make_float3(fNEE_diel, fNEE_diel, fNEE_diel),
                                            devFresnel(cosVH_nee, vtx.albedo), vtx.metallic);
            const float  brdfCosI  = devGGX_BRDFCosI(cosI_nee, cosO, cosH_nee, alpha);

            if (brdfCosI > 0.0f)
            {
                // Shadow ray: p0=vis, p1/p2/p3=RGB colour filter (init 1.0).
                // No TERMINATE_ON_FIRST_HIT — anyhit handles termination
                // explicitly so thin-walled filter accumulation is in ray order.
                float3 shadowFilter;
                if (traceShadow(vtx.pos + Nf * 1e-3f, neeDir, 1e30f, shadowFilter))
                {
                    // MIS power heuristic: pBsdf = (1-p_coat)*p_spec * GGX_PDF(neeDir)
                    const float pGGX_nee = devGGX_PDF(cosO, cosH_nee, alpha);
                    const float pBsdf    = (1.0f - p_coat) * p_spec * pGGX_nee;
                    const float wNee     = neePdf * neePdf
                                         / fmaxf(neePdf * neePdf + pBsdf * pBsdf, 1e-12f);
                    radiance += throughput * shadowFilter
                              * F_nee * brdfCosI / neePdf * neeL * wNee;
                }
            }
        }
    }

    // ── NEE: specular direct emissive-light sampling ──────────────
    if (optixLaunchParams.emissiveLightCount > 0)
    {
        float3 lightPos, lightN;
        const EmissiveLightData* selectedLight = nullptr;
        float lightPdf = 0.0f;
        if (sampleEmissiveLight(vtx.pos,
                rnd(seed), rnd(seed), rnd(seed), rnd(seed),
                lightPos, lightN, selectedLight, lightPdf))
        {
            const float3 toLight  = lightPos - vtx.pos;
            const float  dist2    = devDot(toLight, toLight);
            const float  dist     = sqrtf(dist2);
            const float3 neeDir   = toLight * (1.0f / dist);
            const float  cosI_nee = devDot(neeDir, Nf);

            if (cosI_nee > 0.0f)
            {
                const float3 H_nee     = devNormalize(V + neeDir);
                const float  cosH_nee  = fmaxf(0.0f, devDot(H_nee, Nf));
                const float  cosVH_nee = fmaxf(0.0f, devDot(V, H_nee));
                const float  fNEE_diel = devFresnelDielectric(
                    cosVH_nee, medIor[medTop] / vtx.ior);
                const float3 F_nee     = devMix(
                    make_float3(fNEE_diel, fNEE_diel, fNEE_diel),
                    devFresnel(cosVH_nee, vtx.albedo), vtx.metallic);
                const float  brdfCosI  = devGGX_BRDFCosI(
                    cosI_nee, cosO, cosH_nee, alpha);

                if (brdfCosI > 0.0f)
                {
                    float3 shadowFilter;
                    if (traceShadow(vtx.pos + Nf * 1e-3f, neeDir, dist - 2e-3f, shadowFilter))
                    {
                        const float pGGX_nee = devGGX_PDF(
                            cosO, cosH_nee, alpha);
                        const float pBsdf    = (1.0f - p_coat)
                                             * p_spec * pGGX_nee;
                        const float wNee     = lightPdf * lightPdf
                            / fmaxf(lightPdf * lightPdf + pBsdf * pBsdf,
                                    1e-12f);
                        radiance += throughput * shadowFilter
                                  * F_nee * brdfCosI / lightPdf
                                  * selectedLight->emission * wNee;
                    }
                }
            }
        }
    }

    // ── Continue path with BSDF-sampled direction ──────────────────
    // p_spec already divided above; only the BRDF weight remains.
    throughput *= F * devSmithG1(cosNL, alpha);
    rayDir        = L;
    rayOrig       = vtx.pos + Nf * 1e-3f;
    // Store GGX PDF so miss block can MIS-weight env-map hits
    bsdfPdfForMis = devGGX_PDF(cosO, H_local.z, alpha);

    return true;
}

#endif // OPTIX_RAYTRACER_SPECULAR_LOBE_H

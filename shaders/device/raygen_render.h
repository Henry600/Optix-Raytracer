#ifndef OPTIX_RAYTRACER_RAYGEN_RENDER_H
#define OPTIX_RAYTRACER_RAYGEN_RENDER_H

#include "dev_common.h"
#include "path_payload.h"
#include "rng_sampling.h"
#include "pbr.h"
#include "environment.h"
#include "emissive_lights.h"
#include "display.h"
#include "eon_diffuse.h"
#include "splat_gather.h"

// ─── Raygen — iterative path loop ────────────────────────────────────────────

extern "C" __global__ void __raygen__renderFrame()
{
    const uint3        idx   = optixGetLaunchIndex();
    const uint3        dim   = optixGetLaunchDimensions();
    const unsigned int fbIdx = idx.y * dim.x + idx.x;

    // ── Path tracing ──────────────────────────────────────────────────────────
    // traversable == 0 (no scene / everything hidden) never reaches this raygen —
    // application.cpp skips optixLaunch entirely in that case.

    // Seed: mix pixel index and sample index for uncorrelated sequences
    uint32_t seed = fbIdx * 1973u + optixLaunchParams.sampleIndex * 9277u + 127u;
    pcgNext(seed);  // warm up

    // Sub-pixel jitter for anti-aliasing
    const float nx = ((float)idx.x + rnd(seed)) / (float)dim.x * 2.0f - 1.0f;
    const float ny = ((float)idx.y + rnd(seed)) / (float)dim.y * 2.0f - 1.0f;

    float3 rayOrig = optixLaunchParams.eye;
    float3 rayDir  = devNormalize(
        optixLaunchParams.U * nx +
        optixLaunchParams.V * ny +
        optixLaunchParams.W);

    // ── Thin-lens depth of field ──────────────────────────────────────────────
    // When lensRadius > 0, offset the ray origin over the aperture disk and
    // redirect the ray to pass through the focal point on the focal plane.
    if (optixLaunchParams.lensRadius > 0.0f)
    {
        // Focal point: intersection of the pinhole ray with the focal plane
        // (plane perpendicular to W at distance focusDistance along W)
        const float  t       = optixLaunchParams.focusDistance
                              / devDot(rayDir, optixLaunchParams.W);
        const float3 focalPt = optixLaunchParams.eye + rayDir * t;

        // Sample the lens disk with optional edge bias.
        // alpha = 0.5 * (1 - bias): alpha=0.5 → sqrt (uniform area), alpha→0 → rim ring.
        // r = R * u^alpha concentrates samples toward the edge as bias → 1.
        const float alpha = 0.5f * (1.0f - devClamp01(optixLaunchParams.bokehEdgeBias));
        const float r     = powf(rnd(seed), fmaxf(alpha, 1e-4f)) * optixLaunchParams.lensRadius;
        const float phi = 2.0f * 3.14159265358979f * rnd(seed);

        // Lens plane axes = normalised camera right (U) and up (V)
        const float3 lensU = devNormalize(optixLaunchParams.U);
        const float3 lensV = devNormalize(optixLaunchParams.V);

        rayOrig = optixLaunchParams.eye
                + lensU * (r * cosf(phi))
                + lensV * (r * sinf(phi));
        rayDir  = devNormalize(focalPt - rayOrig);
    }

    float3 throughput    = make_float3(1.0f, 1.0f, 1.0f);
    float3 radiance      = make_float3(0.0f, 0.0f, 0.0f);
    // PDF of the last BSDF-sampled direction (diffuse only); 0 = specular/glass/first ray.
    // Carried into the next iteration so the miss block can MIS-weight env map samples.
    float  bsdfPdfForMis = 0.0f;

    // Beer-Lambert absorption coefficient: -log(albedo) per unit distance.
    // Kept in sync with medAbsorb[medTop]; updated whenever the stack changes.
    float3 absorb = make_float3(0.0f, 0.0f, 0.0f);

    // Nested-dielectric medium stack.
    // Index 0 = air (always present, never popped).  Push on entry, pop on exit.
    constexpr int kMaxMediumDepth = 8;
    float  medIor   [kMaxMediumDepth];
    float3 medAbsorb[kMaxMediumDepth];
    int    medTop = 0;
    medIor[0]    = 1.0f;
    medAbsorb[0] = make_float3(0.0f, 0.0f, 0.0f);

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

        // ── Denoiser guide layers: write on first bounce only ────────────────
        if (bounce == 0)
        {
            if (optixLaunchParams.normalBuffer)
            {
                optixLaunchParams.normalBuffer[fbIdx] = vtx.hit
                    ? make_float4(vtx.N.x, vtx.N.y, vtx.N.z, 0.0f)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            if (optixLaunchParams.albedoBuffer)
            {
                optixLaunchParams.albedoBuffer[fbIdx] = vtx.hit
                    ? make_float4(vtx.albedo.x, vtx.albedo.y, vtx.albedo.z, 1.0f)
                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        if (!vtx.hit)
        {
            float3 bg = sampleBackground(rayDir);
            if (optixLaunchParams.envMarginalCdf && bsdfPdfForMis > 0.0f)
            {
                // MIS power heuristic: down-weight the BSDF path when the env map
                // IS would have sampled this direction with a higher density.
                const float pEnv  = evalEnvMapPdf(rayDir);
                const float wBsdf = (pEnv > 0.0f)
                    ? (bsdfPdfForMis * bsdfPdfForMis)
                      / (bsdfPdfForMis * bsdfPdfForMis + pEnv * pEnv)
                    : 1.0f;
                radiance += throughput * bg * wBsdf;
            }
            else
            {
                radiance += throughput * bg;
            }
            break;
        }

        // ── Shadow catcher ────────────────────────────────────────────────────
        // Invisible surface: shows the background along the incoming ray
        // direction, darkened by how occluded this point's sky hemisphere is.
        // No BSDF, no further bounces — shadows only, no GI or reflections
        // are contributed back into the scene from this hit.
        //
        // The shadow-ray direction is importance-sampled toward the env map's
        // bright regions (or an emissive light) so a blocker specifically in
        // front of the sun reads as a hard directional shadow rather than flat
        // ambient occlusion.  If that sample lands below the local horizon
        // (cosI <= 0) it falls back to a cosine-weighted hemisphere sample,
        // which is always valid — every launch must contribute *some* value to
        // the per-pixel accumulator, so treating an invalid sample as
        // "unshadowed" would bias the average away from ever going fully dark
        // under total occlusion.
        if (vtx.shadowCatcher)
        {
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

            uint32_t shadowVis = 0u;
            uint32_t shadowFR  = __float_as_uint(1.0f);
            uint32_t shadowFG  = __float_as_uint(1.0f);
            uint32_t shadowFB  = __float_as_uint(1.0f);
            optixTrace(
                optixLaunchParams.traversable,
                vtx.pos + Nfc * 1e-3f, skyDir,
                1e-3f, skyTMax, 0.0f,
                OptixVisibilityMask(VIS_MASK_GEOMETRY),
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                0, 1, 1,
                shadowVis, shadowFR, shadowFG, shadowFB);

            const float3 shadow = shadowVis
                ? make_float3(__uint_as_float(shadowFR), __uint_as_float(shadowFG), __uint_as_float(shadowFB))
                : make_float3(0.0f, 0.0f, 0.0f);

            radiance += throughput * bg * shadow;
            break;
        }

        // ── Beer-Lambert absorption ───────────────────────────────────────────
        // Apply exp(-σ·t) for each channel; σ = -log(albedo) set when entering glass.
        throughput.x *= expf(-absorb.x * vtx.t);
        throughput.y *= expf(-absorb.y * vtx.t);
        throughput.z *= expf(-absorb.z * vtx.t);

        // ── Emission ──────────────────────────────────────────────────────────
        // When emissive NEE is active and this hit was an implicit light, apply
        // the MIS power heuristic to avoid double-counting direct contributions:
        // the NEE strategy already sampled this light in the previous bounce.
        if (optixLaunchParams.emissiveLightCount > 0
            && vtx.instanceId != ~0u
            && bsdfPdfForMis > 0.0f)
        {
            const float pLight = evalEmissiveLightPdf(
                vtx.instanceId, rayOrig, vtx.pos, vtx.N, vtx.t);
            const float wBsdf  = (pLight > 0.0f)
                ? (bsdfPdfForMis * bsdfPdfForMis)
                  / (bsdfPdfForMis * bsdfPdfForMis + pLight * pLight)
                : 1.0f;
            radiance += throughput * vtx.emission * wBsdf;
        }
        else
        {
            radiance += throughput * vtx.emission;
        }

        // ── Lobe selection: clearcoat → specular → diffuse / refraction ─────────
        //
        // Clearcoat is sampled first: it's a dielectric layer (IOR 1.5, F0 = 0.04)
        // on top of the base material.  If not chosen, the base material receives
        // attenuated energy: throughput *= (1 − clearcoat · F_cc).
        //
        // Within the base material:
        //   1. specular   — p_spec (metallic / Fresnel)
        //   2. diffuse    — (1 − p_spec) · (1 − transmission)
        //   3. refraction — (1 − p_spec) ·      transmission

        const float alpha = fmaxf(vtx.roughness * vtx.roughness, 1e-3f);

        // Front-facing normal: always faces the incoming ray.
        const float3 Nf  = (devDot(vtx.N, -rayDir) >= 0.0f) ? vtx.N : -vtx.N;
        const float  cosV = devDot(Nf, -rayDir);  // shared by clearcoat + base Fresnel

        // ── Clearcoat Fresnel (Schlick, F0 from the coat IOR; 1.5 → 0.04) ────
        const float  cc_alpha = fmaxf(vtx.clearcoatRoughness * vtx.clearcoatRoughness, 1e-3f);
        const float  cc_f0    = ((vtx.clearcoatIor - 1.0f) / (vtx.clearcoatIor + 1.0f))
                              * ((vtx.clearcoatIor - 1.0f) / (vtx.clearcoatIor + 1.0f));
        const float3 cc_F     = devFresnel(cosV, make_float3(cc_f0, cc_f0, cc_f0));
        const float  p_coat   = vtx.clearcoat * devLuminance(cc_F);

        if (p_coat > 0.0f && rnd(seed) < p_coat)
        {
            // ── Clearcoat specular (GGX-VNDF) ────────────────────────────────
            float3 Tc, Bc;
            buildONB(Nf, Tc, Bc);
            const float3 Vc       = -rayDir;
            const float3 Vc_local = make_float3(
                devDot(Vc, Tc), devDot(Vc, Bc), fmaxf(1e-4f, devDot(Vc, Nf)));
            const float3 Hc_local = devSampleGGX_VNDF(Vc_local, cc_alpha, rnd(seed), rnd(seed));
            const float3 Hc       = devNormalize(Tc * Hc_local.x + Bc * Hc_local.y + Nf * Hc_local.z);
            const float3 Lc       = devReflect(rayDir, Hc);

            if (devDot(Lc, Nf) <= 0.0f)
            {
                break;
            }

            // cc_F is achromatic (scalar F0 from the coat IOR → Schlick stays grey), so
            // cc_F * clearcoat / p_coat = cc_F * clearcoat / (clearcoat * luminance(cc_F)) = 1.
            // The probability selection already carries the energy split; only G1 remains.
            const float cosNLc  = fmaxf(1e-4f, devDot(Nf, Lc));
            const float cosO_cc = Vc_local.z;   // dot(Vc, Nf), already clamped ≥ 1e-4
            throughput *= devSmithG1(cosNLc, cc_alpha);
            rayDir        = Lc;
            rayOrig       = vtx.pos + Nf * 1e-3f;
            bsdfPdfForMis = devGGX_PDF(cosO_cc, Hc_local.z, cc_alpha);
        }
        else
        {
            // ── Base material ─────────────────────────────────────────────────
            // The coat's Fresnel transmission is already carried by branch
            // selection: (1 - C·F) / (1 - p_coat) = 1 for achromatic cc_F.
            //
            // OpenPBR coat darkening (arXiv 2512.23696 §5.1, coat_darkening = 1):
            // light transmitted through the coat and diffusely reflected by the
            // base is partly trapped by total internal reflection at the coat
            // interface and re-bounces off the base, so saturated bases darken.
            //   Δ = (1 − K) / (1 − E_b·K·CC)        CC = coat_color
            //   K = lerp(K_s, K_r, r_b)             K_r = 1 − (1 − E_F)/η²
            // Spec weight of the base under the coat:
            //   (1 − C) + C·Δ·T_coat·(1 − F)
            // with T_coat = CC^(1/μ) the coat absorption tint along the refracted
            // path (entry and exit cosines approximated by the view's, since the
            // outgoing direction is not sampled yet).  The branch is chosen with
            // probability (1 − C·F), so scale by the ratio.  K_r is precomputed
            // on the host from the coat IOR (coat_math.h).
            if (vtx.clearcoat > 0.0f)
            {
                const float kCoatIor = vtx.clearcoatIor;
                const float kKr      = vtx.clearcoatKr;

                // Base normal-incidence dielectric reflectance under the coat
                const float eta_a = medIor[medTop] / vtx.ior;
                const float eta_c = kCoatIor / vtx.ior;
                const float eta_s = eta_a + (eta_c - eta_a) * vtx.clearcoat;
                const float F0    = devFresnelDielectric(1.0f, eta_s);

                // Rough/Lambertian bases scatter light isotropically (K_r); smooth
                // specular bases reflect at the coat's directional Fresnel (K_s).
                const float Fs = devClamp01(F0);
                const float rd = 1.0f + (vtx.roughness - 1.0f) * Fs;
                const float rb = rd + (vtx.roughness - rd) * vtx.metallic;
                const float K  = cc_F.x + (kKr - cc_F.x) * rb;

                // Whole-base albedo estimate E_b: diffuse colour, (1 − F0) for the
                // transmissive part, plain albedo for metals.
                const float3 E_diel = devMix(vtx.albedo,
                                             make_float3(1.0f - F0, 1.0f - F0, 1.0f - F0),
                                             vtx.transmission);
                const float3 E_b    = devMix(E_diel, vtx.albedo, vtx.metallic);

                const float3 CC = vtx.clearcoatColor;
                const float3 delta = make_float3(
                    fmaxf(1.0f - K, 0.0f) / fmaxf(1.0f - E_b.x * K * CC.x, 1e-4f),
                    fmaxf(1.0f - K, 0.0f) / fmaxf(1.0f - E_b.y * K * CC.y, 1e-4f),
                    fmaxf(1.0f - K, 0.0f) / fmaxf(1.0f - E_b.z * K * CC.z, 1e-4f));

                // Absorption tint along the refracted path: CC^(1/μ).
                const float mu  = sqrtf(fmaxf(1.0f - (1.0f - cosV * cosV) / (kCoatIor * kCoatIor), 1e-4f));
                const float tintExp = 1.0f / mu;
                const float3 tint = make_float3(powf(CC.x, tintExp), powf(CC.y, tintExp), powf(CC.z, tintExp));

                // coat_darkening blends the physical darkening in: lerp(1, Δ, δ).
                const float dk    = vtx.clearcoatDarkening;
                const float C     = vtx.clearcoat;
                const float F     = cc_F.x;
                const float pBase = fmaxf(1.0f - C * F, 1e-4f);
                throughput.x *= ((1.0f - C) + C * (1.0f + (delta.x - 1.0f) * dk) * tint.x * (1.0f - F)) / pBase;
                throughput.y *= ((1.0f - C) + C * (1.0f + (delta.y - 1.0f) * dk) * tint.y * (1.0f - F)) / pBase;
                throughput.z *= ((1.0f - C) + C * (1.0f + (delta.z - 1.0f) * dk) * tint.z * (1.0f - F)) / pBase;
            }

            // Base Fresnel — exact dielectric (handles IOR=1 correctly) blended with metal Schlick.
            // Schlick with F0=0 gives spurious (1−cosV)^5 at grazing for IOR=1; exact gives 0.
            const float  fDiel = devFresnelDielectric(cosV, medIor[medTop] / vtx.ior);
            const float3 F     = devMix(make_float3(fDiel, fDiel, fDiel),
                                        devFresnel(cosV, vtx.albedo), vtx.metallic);

            // Specular probability — metals always specular
            const float p_spec = devClamp01((1.0f - vtx.metallic) * devLuminance(F) + vtx.metallic);

            if (rnd(seed) < p_spec)
            {
                // ── 1. Specular reflection (GGX-VNDF) ─────────────────────────
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
                    break;
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
                            uint32_t shadowVis = 0u;
                            uint32_t shadowFR  = __float_as_uint(1.0f);
                            uint32_t shadowFG  = __float_as_uint(1.0f);
                            uint32_t shadowFB  = __float_as_uint(1.0f);
                            optixTrace(
                                optixLaunchParams.traversable,
                                vtx.pos + Nf * 1e-3f, neeDir,
                                1e-3f, 1e30f, 0.0f,
                                OptixVisibilityMask(VIS_MASK_GEOMETRY),
                                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                0, 1, 1,
                                shadowVis, shadowFR, shadowFG, shadowFB);

                            if (shadowVis)
                            {
                                const float3 shadowFilter = make_float3(
                                    __uint_as_float(shadowFR),
                                    __uint_as_float(shadowFG),
                                    __uint_as_float(shadowFB));
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
                                uint32_t shadowVis = 0u;
                                uint32_t shadowFR  = __float_as_uint(1.0f);
                                uint32_t shadowFG  = __float_as_uint(1.0f);
                                uint32_t shadowFB  = __float_as_uint(1.0f);
                                optixTrace(
                                    optixLaunchParams.traversable,
                                    vtx.pos + Nf * 1e-3f, neeDir,
                                    1e-3f, dist - 2e-3f, 0.0f,
                                    OptixVisibilityMask(VIS_MASK_GEOMETRY),
                                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                    0, 1, 1,
                                    shadowVis, shadowFR, shadowFG, shadowFB);

                                if (shadowVis)
                                {
                                    const float3 shadowFilter = make_float3(
                                        __uint_as_float(shadowFR),
                                        __uint_as_float(shadowFG),
                                        __uint_as_float(shadowFB));
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
            }
            else
            {
                // Non-specular weight — Fresnel transmission factor only.
                // albedo is applied inside the diffuse branch; the refraction branch
                // gets its color from Beer-Lambert absorption, not a surface multiply.
                const float3 kNS = (make_float3(1.0f, 1.0f, 1.0f) - F)
                                   * (1.0f / fmaxf(1e-4f, 1.0f - p_spec));
                throughput *= kNS;

                const float p_trans = devClamp01(vtx.transmission);

                if (rnd(seed) > p_trans)
                {
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
                            uint32_t shadowVis = 0u;
                            uint32_t shadowFR  = __float_as_uint(1.0f);
                            uint32_t shadowFG  = __float_as_uint(1.0f);
                            uint32_t shadowFB  = __float_as_uint(1.0f);
                            optixTrace(
                                optixLaunchParams.traversable,
                                vtx.pos + Nf * 1e-3f, neeDir,
                                1e-3f, 1e30f, 0.0f,
                                OptixVisibilityMask(VIS_MASK_GEOMETRY),
                                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                0, 1, 1,   // offset=0, stride=1, missIndex=1
                                shadowVis, shadowFR, shadowFG, shadowFB);

                            if (shadowVis)
                            {
                                const float3 shadowFilter = make_float3(
                                    __uint_as_float(shadowFR),
                                    __uint_as_float(shadowFG),
                                    __uint_as_float(shadowFB));
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
                                uint32_t shadowVis = 0u;
                                uint32_t shadowFR  = __float_as_uint(1.0f);
                                uint32_t shadowFG  = __float_as_uint(1.0f);
                                uint32_t shadowFB  = __float_as_uint(1.0f);
                                optixTrace(
                                    optixLaunchParams.traversable,
                                    vtx.pos + Nf * 1e-3f, neeDir,
                                    1e-3f, dist - 2e-3f, 0.0f,
                                    OptixVisibilityMask(VIS_MASK_GEOMETRY),
                                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                    0, 1, 1,
                                    shadowVis, shadowFR, shadowFG, shadowFB);

                                if (shadowVis)
                                {
                                    const float3 shadowFilter = make_float3(
                                        __uint_as_float(shadowFR),
                                        __uint_as_float(shadowFG),
                                        __uint_as_float(shadowFB));
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
                else
                {
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
                }   // end refraction else
            }   // end non-specular else (diffuse / refraction)
        }   // end base material else (specular / diffuse / refraction) = clearcoat else

        // Russian roulette: stochastically terminate dim paths (all branches)
        if (bounce >= 3)
        {
            const float maxThr = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rnd(seed) > maxThr)
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

#ifndef OPTIX_RAYTRACER_COAT_LOBE_H
#define OPTIX_RAYTRACER_COAT_LOBE_H

#include "dev_common.h"
#include "path_payload.h"
#include "path_state.h"
#include "surface_context.h"
#include "rng_sampling.h"
#include "pbr.h"

// Clearcoat specular reflection (GGX-VNDF).  Returns false when the sampled
// direction is below the horizon (the path terminates).
static __forceinline__ __device__
bool sampleCoat(const PathVertex& vtx, const SurfaceCtx& sc, PathState& ps)
{
    float3& rayOrig = ps.rayOrig;
    float3& rayDir = ps.rayDir;
    float3& throughput = ps.throughput;
    float& bsdfPdfForMis = ps.bsdfPdfForMis;
    uint32_t& seed = ps.seed;
    const float3 Nf = sc.Nf;
    const float cc_alpha = sc.cc_alpha;

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
        return false;
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

    return true;
}

// Base-branch throughput weight under the coat (OpenPBR coat darkening).
static __forceinline__ __device__
void applyCoatDarkening(const PathVertex& vtx, const SurfaceCtx& sc, PathState& ps, MediumStack& ms)
{
    float3& throughput = ps.throughput;
    float (&medIor)[MediumStack::kMaxDepth] = ms.medIor;
    int& medTop = ms.medTop;
    const float cosV = sc.cosV;
    const float3 cc_F = sc.cc_F;

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
}

#endif // OPTIX_RAYTRACER_COAT_LOBE_H

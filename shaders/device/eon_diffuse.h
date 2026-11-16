#ifndef OPTIX_RAYTRACER_EON_DIFFUSE_H
#define OPTIX_RAYTRACER_EON_DIFFUSE_H

#include "dev_common.h"

// ─── EON rough diffuse (OpenPBR base_diffuse_roughness) ──────────────────────
// Energy-preserving Oren-Nayar (Portsmouth, Kutz & Hill, arXiv 2410.18026):
// Fujii's Oren-Nayar (FON) single-scatter lobe plus a multiple-scattering
// compensation lobe, with r = roughness.
//   f_F    = ρ/π · (A + B·s/t)       A = 1/(1 + (1/2 − 2/(3π))·r),  B = r·A
//   s = ωi·ωo − cosθi·cosθo,         1/t = (s > 0) ? 1/max(cosθi, cosθo) : 1
//   f_ms   = ρ_ms/π · (1 − Ê(ωi))(1 − Ê(ωo)) / (1 − ⟨Ê⟩)
//   ρ_ms   = ρ²·⟨Ê⟩ / (1 − ρ(1 − ⟨Ê⟩))
//   Ê(μ)   = A + B·Σ g_k (1 − μ)^k   (k = 1..4, fit error < 0.1 %)
//   ⟨Ê⟩    = A + (2/3 − 28/(15π))·B
// Returns f·π/ρ per channel — the ratio to the Lambertian albedo weight that
// the throughput already carries.  Cosine-weighted sampling keeps pdf = cosθ/π,
// so the path throughput and both NEE contributions simply gain this factor.

static __forceinline__ __device__
float eonAlbedo(float mu, float A, float B)
{
    const float x = 1.0f - mu;
    return A + B * x * (0.0571085289f
                 + x * (0.491881867f
                 + x * (-0.332181442f
                 + x *  0.0714429953f)));
}

static __forceinline__ __device__
float3 eonFactor(float cosI, float cosO, float dotIO, float sigma, float3 rho)
{
    if (sigma <= 0.0f)
    {
        return make_float3(1.0f, 1.0f, 1.0f);
    }
    const float A = 1.0f / (1.0f + (0.5f - 0.2122065908f) * sigma);   // 2/(3π) = 0.2122065908
    const float B = sigma * A;

    // Single-scatter FON lobe
    const float s = dotIO - cosI * cosO;
    const float t = (s > 0.0f) ? fmaxf(fmaxf(cosI, cosO), 1e-4f) : 1.0f;
    const float single = A + B * s / t;

    // Multiple-scattering compensation
    const float avgE = A + 0.0724882125f * B;                           // 2/3 − 28/(15π)
    const float oneMinusI = 1.0f - eonAlbedo(cosI, A, B);
    const float oneMinusO = 1.0f - eonAlbedo(cosO, A, B);
    const float msShape   = oneMinusI * oneMinusO / fmaxf(1.0f - avgE, 1e-4f);

    // ρ_ms / ρ = ρ·⟨Ê⟩ / (1 − ρ(1 − ⟨Ê⟩)), per channel
    const float3 msWeight = make_float3(
        rho.x * avgE / (1.0f - rho.x * (1.0f - avgE)),
        rho.y * avgE / (1.0f - rho.y * (1.0f - avgE)),
        rho.z * avgE / (1.0f - rho.z * (1.0f - avgE)));

    return make_float3(single + msWeight.x * msShape,
                       single + msWeight.y * msShape,
                       single + msWeight.z * msShape);
}

#endif // OPTIX_RAYTRACER_EON_DIFFUSE_H

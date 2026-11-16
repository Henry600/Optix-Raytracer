#ifndef OPTIX_RAYTRACER_PBR_H
#define OPTIX_RAYTRACER_PBR_H

#include "dev_common.h"

// ─── PBR helpers ─────────────────────────────────────────────────────────────

// Schlick Fresnel approximation (for metals: F0 = albedo).
static __forceinline__ __device__
float3 devFresnel(float cosTheta, float3 F0)
{
    const float t  = 1.0f - devClamp01(cosTheta);
    const float t5 = t * t * t * t * t;
    return F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * t5;
}

// Exact unpolarised Fresnel for a dielectric interface.
// eta = n_incident / n_transmitted (same convention as devRefract).
// Correctly returns 0.0 at all angles when eta = 1.0 (IOR = 1).
static __forceinline__ __device__
float devFresnelDielectric(float cosI, float eta)
{
    const float sinT2 = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    if (sinT2 >= 1.0f) { return 1.0f; }  // total internal reflection
    const float cosT = sqrtf(1.0f - sinT2);
    const float rs   = (eta * cosI - cosT) / (eta * cosI + cosT);
    const float rp   = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rs * rs + rp * rp);
}

// Sample GGX visible normals (Heitz 2018).
// Standard hemisphere sampling produces many below-horizon reflections on rough
// surfaces (causing energy loss).  VNDF conditions on the view direction so the
// sampled half-vector is always visible, making sub-horizon outcomes nearly
// impossible and simplifying the MC weight to just F·G1(L).
//
// V_local: view direction in tangent space (z = N component, must be > 0).
// Returns: half-vector in tangent space.
static __forceinline__ __device__
float3 devSampleGGX_VNDF(float3 V_local, float alpha, float u1, float u2)
{
    // Stretch view vector into the hemisphere configuration
    const float3 Vh = devNormalize(
        make_float3(alpha * V_local.x, alpha * V_local.y, V_local.z));

    // Orthonormal basis around Vh (T1 ⊥ Vh in the XY plane)
    const float  len2 = Vh.x*Vh.x + Vh.y*Vh.y;
    const float3 T1   = (len2 > 1e-7f)
        ? make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(len2)
        : make_float3(1.0f, 0.0f, 0.0f);
    // T2 = cross(Vh, T1)
    const float3 T2 = make_float3(Vh.y*T1.z - Vh.z*T1.y,
                                   Vh.z*T1.x - Vh.x*T1.z,
                                   Vh.x*T1.y - Vh.y*T1.x);

    // Sample a point on the projected area disk
    const float r   = sqrtf(u1);
    const float phi = 2.0f * 3.14159265358979f * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    // Blend between uniform disk (s=0) and projected hemisphere (s=1)
    const float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;

    // Lift onto hemisphere, stretch back to GGX ellipsoid
    const float3 Nh = T1 * t1 + T2 * t2
                    + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    return devNormalize(
        make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(1e-6f, Nh.z)));
}

// Smith G1 shadowing-masking term for GGX.
static __forceinline__ __device__
float devSmithG1(float cosTheta, float alpha)
{
    const float a2 = alpha * alpha;
    const float c2 = cosTheta * cosTheta;
    return 2.0f * cosTheta / (cosTheta + sqrtf(a2 + (1.0f - a2) * c2));
}

// GGX normal distribution function: D(H) = α² / (π·(cosH²·(α²−1)+1)²)
static __forceinline__ __device__
float devGGX_D(float cosH, float alpha)
{
    const float a2 = alpha * alpha;
    const float d  = cosH * cosH * (a2 - 1.0f) + 1.0f;
    return a2 / (3.14159265358979f * d * d);
}

// PDF of a direction sampled by GGX VNDF:  p(ωo) = G1(cosV,α)·D(H) / (4·cosV)
static __forceinline__ __device__
float devGGX_PDF(float cosV, float cosH, float alpha)
{
    return devSmithG1(cosV, alpha) * devGGX_D(cosH, alpha)
           / (4.0f * fmaxf(cosV, 1e-5f));
}

// Smith height-correlated G2 for arbitrary light + view directions.
// G2 = 2·cosI·cosO / (cosO·√(α²+(1−α²)·cosI²) + cosI·√(α²+(1−α²)·cosO²))
static __forceinline__ __device__
float devGGX_G2(float cosI, float cosO, float alpha)
{
    const float a2 = alpha * alpha;
    const float lv = cosO * sqrtf(a2 + (1.0f - a2) * cosI * cosI);
    const float ll = cosI * sqrtf(a2 + (1.0f - a2) * cosO * cosO);
    return 2.0f * cosI * cosO / fmaxf(lv + ll, 1e-8f);
}

// GGX BRDF × cosI without Fresnel, for NEE evaluation.
// Returns D(H)·G2(cosI,cosO) / (4·cosO).  Caller multiplies by Fresnel(V·H, F0).
static __forceinline__ __device__
float devGGX_BRDFCosI(float cosI, float cosO, float cosH, float alpha)
{
    return devGGX_D(cosH, alpha) * devGGX_G2(cosI, cosO, alpha)
           / (4.0f * fmaxf(cosO, 1e-5f));
}

// Reflect incident direction v around normal n.
static __forceinline__ __device__
float3 devReflect(float3 v, float3 n)
{
    return v - n * (2.0f * devDot(v, n));
}

// Refract incident direction v through a surface with normal n (opposing v) and
// relative IOR eta = n1/n2.  Returns true on successful refraction and writes the
// transmitted direction to `out`.  Returns false on total internal reflection and
// writes the mirrored direction to `out` so the caller can continue the path.
static __forceinline__ __device__
bool devRefract(float3 v, float3 n, float eta, float3& out)
{
    const float cosI  = devDot(-v, n);                        // positive: n opposes v
    const float sinT2 = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    if (sinT2 >= 1.0f)                                         // total internal reflection
    {
        out = devReflect(v, n);
        return false;
    }
    const float cosT = sqrtf(1.0f - sinT2);
    out = devNormalize(v * eta + n * (eta * cosI - cosT));    // Snell's law, vector form
    return true;
}

#endif // OPTIX_RAYTRACER_PBR_H

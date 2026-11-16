#ifndef OPTIX_RAYTRACER_RNG_SAMPLING_H
#define OPTIX_RAYTRACER_RNG_SAMPLING_H

#include "dev_common.h"

// ─── PCG pseudo-random number generator ──────────────────────────────────────
// O'Neill 2014 — one 32-bit state, good statistical quality, cheap on GPU.

static __forceinline__ __device__
uint32_t pcgNext(uint32_t& state)
{
    uint32_t prev = state * 747796405u + 2891336453u;
    state = prev;
    uint32_t w = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (w >> 22u) ^ w;
}

static __forceinline__ __device__
float rnd(uint32_t& seed) { return (float)pcgNext(seed) * (1.0f / 4294967296.0f); }

// ─── Sampling helpers ─────────────────────────────────────────────────────────

// Cosine-weighted hemisphere direction in a Z-up local frame.
// BRDF = albedo/π, PDF = cosθ/π → BRDF/PDF = albedo (weight is just albedo).
static __forceinline__ __device__
float3 cosineSampleHemisphere(float r1, float r2)
{
    const float phi     = 2.0f * 3.14159265358979f * r1;
    const float sqrtR2  = sqrtf(r2);
    return make_float3(cosf(phi)*sqrtR2, sinf(phi)*sqrtR2, sqrtf(1.0f - r2));
}

// Henyey-Greenstein phase function.  Returns a direction in a local frame
// where the forward (incident) direction is +z.  g in [-1, 1].
static __forceinline__ __device__
float3 sampleHenyeyGreenstein(float g, float xi1, float xi2)
{
    float cosTheta;
    if (fabsf(g) < 1e-4f)
    {
        cosTheta = 1.0f - 2.0f * xi1;
    }
    else
    {
        const float denom = 1.0f - g + 2.0f * g * xi1;
        const float sq    = (1.0f - g * g) / denom;
        cosTheta = (1.0f + g * g - sq * sq) / (2.0f * g);
    }
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi      = 2.0f * 3.14159265358979f * xi2;
    return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

// Orthonormal tangent frame around N (Duff et al. 2017).
static __forceinline__ __device__
void buildONB(float3 N, float3& T, float3& B)
{
    const float s = copysignf(1.0f, N.z);
    const float a = -1.0f / (s + N.z);
    const float b =  N.x * N.y * a;
    T = make_float3(1.0f + s*N.x*N.x*a, s*b, -s*N.x);
    B = make_float3(b, s + N.y*N.y*a, -N.y);
}

#endif // OPTIX_RAYTRACER_RNG_SAMPLING_H

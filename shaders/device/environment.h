#ifndef OPTIX_RAYTRACER_ENVIRONMENT_H
#define OPTIX_RAYTRACER_ENVIRONMENT_H

#include "dev_common.h"

// ─── Background / environment map ────────────────────────────────────────────

static __forceinline__ __device__
float3 sampleBackground(float3 dir)
{
    const float  exposure = exp2f(optixLaunchParams.envExposure);  // EV stops → linear scale
    const float3 tint     = srgbToLinear3(optixLaunchParams.skyColor);

    float3 base;
    if (optixLaunchParams.skyMode == SKY_MODE_HDRI && optixLaunchParams.envMap != 0)
    {
        const float kInvPi  = 0.31830988618f;
        const float kInv2Pi = 0.15915494309f;
        const float theta   = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
        const float phi     = atan2f(dir.x, -dir.z) + optixLaunchParams.envMapRotation;
        const float4 s = tex2D<float4>(optixLaunchParams.envMap,
                                       phi * kInv2Pi + 0.5f,
                                       theta * kInvPi);
        base = make_float3(s.x, s.y, s.z);
    }
    else if (optixLaunchParams.skyMode == SKY_MODE_CONSTANT)
    {
        base = make_float3(1.0f, 1.0f, 1.0f);
    }
    else
    {
        // Procedural sky (also the HDRI-mode fallback when no env map is loaded):
        base = devMix(make_float3(0.25f, 0.3f, 0.6f), make_float3(0.8f, 1.1f, 3.0f),
                      devClamp01(dir.y));
    }

    return base * tint * exposure;
}

// ─── HDRI importance-sampling helpers ────────────────────────────────────────

// Binary search: first index in cdf[0..n-1] where cdf[idx] >= u.
static __forceinline__ __device__
int upperBound(const float* cdf, int n, float u)
{
    int lo = 0, hi = n - 1;
    while (lo < hi)
    {
        const int mid = (lo + hi) >> 1;
        if (cdf[mid] < u)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

// Sample the env map by importance.
// r1, r2: uniform [0,1).  Writes the sampled world-space direction to outDir
// and its solid-angle PDF to outPdf.  Returns the HDR radiance at that direction.
static __forceinline__ __device__
float3 sampleEnvMapIS(float r1, float r2, float3& outDir, float& outPdf)
{
    const int    W           = optixLaunchParams.envCdfW;
    const int    H           = optixLaunchParams.envCdfH;
    const float* marginal    = optixLaunchParams.envMarginalCdf;
    const float* conditional = optixLaunchParams.envConditionalCdf;

    // Sample row j, then column i
    const int j = upperBound(marginal,          H, r1);
    const int i = upperBound(conditional + j*W, W, r2);

    // Continuous UV at texel centre
    const float u = (i + 0.5f) / static_cast<float>(W);
    const float v = (j + 0.5f) / static_cast<float>(H);

    // UV → spherical: undo the envMapRotation sampleBackground would apply
    const float theta    = v * 3.14159265358979f;
    const float phiWorld = (u - 0.5f) * 6.28318530717959f - optixLaunchParams.envMapRotation;
    const float sinT     = sinf(theta);
    outDir = make_float3(sinf(phiWorld) * sinT, cosf(theta), -cosf(phiWorld) * sinT);

    // PDF in solid-angle measure
    const float mPrev = (j > 0) ? marginal[j - 1]             : 0.0f;
    const float cPrev = (i > 0) ? conditional[j * W + i - 1] : 0.0f;
    const float pUV   = (marginal[j] - mPrev) * static_cast<float>(H)
                      * (conditional[j * W + i] - cPrev) * static_cast<float>(W);
    outPdf = pUV / fmaxf(1e-5f, 2.0f * 3.14159265358979f * 3.14159265358979f * sinT);

    return sampleBackground(outDir);  // applies exposure + rotation + fallback
}

// Evaluate the solid-angle PDF for a world-space direction under env map IS.
// Returns 0 when no CDF is loaded.
static __forceinline__ __device__
float evalEnvMapPdf(float3 dir)
{
    if (!optixLaunchParams.envMarginalCdf)
    {
        return 0.0f;
    }

    const int    W           = optixLaunchParams.envCdfW;
    const int    H           = optixLaunchParams.envCdfH;
    const float* marginal    = optixLaunchParams.envMarginalCdf;
    const float* conditional = optixLaunchParams.envConditionalCdf;

    // Direction → (u, v) — mirror exactly what sampleBackground computes
    const float kInvPi  = 0.31830988618f;
    const float kInv2Pi = 0.15915494309f;
    const float theta   = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
    const float phi     = atan2f(dir.x, -dir.z) + optixLaunchParams.envMapRotation;
    float u = phi * kInv2Pi + 0.5f;
    u = u - floorf(u);  // wrap to [0,1]
    const float v = theta * kInvPi;

    const int i = min(static_cast<int>(u * static_cast<float>(W)), W - 1);
    const int j = min(static_cast<int>(v * static_cast<float>(H)), H - 1);

    const float mPrev = (j > 0) ? marginal[j - 1]             : 0.0f;
    const float cPrev = (i > 0) ? conditional[j * W + i - 1] : 0.0f;
    const float pUV   = (marginal[j] - mPrev) * static_cast<float>(H)
                      * (conditional[j * W + i] - cPrev) * static_cast<float>(W);
    const float sinT  = sinf(theta);
    return (sinT > 1e-5f)
        ? pUV / (2.0f * 3.14159265358979f * 3.14159265358979f * sinT)
        : 0.0f;
}

#endif // OPTIX_RAYTRACER_ENVIRONMENT_H

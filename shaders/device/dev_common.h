#ifndef OPTIX_RAYTRACER_DEV_COMMON_H
#define OPTIX_RAYTRACER_DEV_COMMON_H

#include <optix.h>
#include "device_math.h"   // float3 operators (+, -, *, /, unary -)
#include "color_math.h"    // srgbToLinear / srgbToLinear3 (shared with the host)
#include "launch_params.h"
#include "scene_data.h"

extern "C" { __constant__ LaunchParams optixLaunchParams; }

// ─── Tuning ───────────────────────────────────────────────────────────────────

static constexpr int MAX_BOUNCES = 16;

// ─── Scalar / float3 utilities ───────────────────────────────────────────────

static __forceinline__ __device__ float devClamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static __forceinline__ __device__ float devDot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static __forceinline__ __device__ float devLuminance(float3 c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

static __forceinline__ __device__
float3 devNormalize(float3 v)
{
    return v * rsqrtf(devDot(v, v));
}

// Linear interpolation: a + (b-a)*t
static __forceinline__ __device__
float3 devMix(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}

#endif // OPTIX_RAYTRACER_DEV_COMMON_H

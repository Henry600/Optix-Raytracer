#ifndef OPTIX_RAYTRACER_CAMERA_RAY_H
#define OPTIX_RAYTRACER_CAMERA_RAY_H

#include "dev_common.h"
#include "rng_sampling.h"

// Generates the primary ray for pixel `idx`: sub-pixel jitter plus thin-lens
// depth of field.  Consumes RNG samples from `seed` in a fixed order.
static __forceinline__ __device__
void generateCameraRay(const uint3 idx, const uint3 dim, uint32_t& seed,
                       float3& outOrig, float3& outDir)
{
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

    outOrig = rayOrig;
    outDir  = rayDir;
}

#endif // OPTIX_RAYTRACER_CAMERA_RAY_H

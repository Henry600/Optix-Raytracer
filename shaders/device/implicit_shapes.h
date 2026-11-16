#ifndef OPTIX_RAYTRACER_IMPLICIT_SHAPES_H
#define OPTIX_RAYTRACER_IMPLICIT_SHAPES_H

#include "dev_common.h"

// ─── Implicit shape intersection helpers ─────────────────────────────────────
// All shapes are canonical unit forms in object space ([-1,1]^3 AABB).
// OptiX transforms the world-space ray into object space automatically before
// calling the intersection program.

// Unit sphere x²+y²+z²=1.  Returns smallest valid t, or -1 on miss.
static __forceinline__ __device__
float implicitSphereT(const float3& ro, const float3& rd)
{
    const float a    = devDot(rd, rd);
    const float b    = 2.0f * devDot(ro, rd);
    const float c    = devDot(ro, ro) - 1.0f;
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) { return -1.0f; }
    const float sq   = sqrtf(disc);
    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();
    const float t1   = (-b - sq) / (2.0f * a);
    if (t1 >= tMin && t1 <= tMax) { return t1; }
    const float t2   = (-b + sq) / (2.0f * a);
    if (t2 >= tMin && t2 <= tMax) { return t2; }
    return -1.0f;
}

// Unit cube [-1,1]^3.  Returns smallest valid t and writes outward face normal.
// Returns -1 on miss.
static __forceinline__ __device__
float implicitBoxT(const float3& ro, const float3& rd, float3& outN)
{
    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();

    const float invDx = 1.0f / rd.x;
    const float invDy = 1.0f / rd.y;
    const float invDz = 1.0f / rd.z;

    const float tx1 = (-1.0f - ro.x) * invDx;
    const float tx2 = ( 1.0f - ro.x) * invDx;
    const float ty1 = (-1.0f - ro.y) * invDy;
    const float ty2 = ( 1.0f - ro.y) * invDy;
    const float tz1 = (-1.0f - ro.z) * invDz;
    const float tz2 = ( 1.0f - ro.z) * invDz;

    const float txN = fminf(tx1, tx2);  const float txF = fmaxf(tx1, tx2);
    const float tyN = fminf(ty1, ty2);  const float tyF = fmaxf(ty1, ty2);
    const float tzN = fminf(tz1, tz2);  const float tzF = fmaxf(tz1, tz2);

    const float tNear = fmaxf(fmaxf(txN, tyN), tzN);
    const float tFar  = fminf(fminf(txF, tyF), tzF);

    if (tNear > tFar) { return -1.0f; }

    float t = tNear;
    if (t < tMin) { t = tFar; }
    if (t < tMin || t > tMax) { return -1.0f; }

    const float3 p  = ro + rd * t;
    const float  ax = fabsf(p.x);
    const float  ay = fabsf(p.y);
    const float  az = fabsf(p.z);
    if (ax >= ay && ax >= az)
    {
        outN = make_float3(p.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
    }
    else if (ay >= ax && ay >= az)
    {
        outN = make_float3(0.0f, p.y > 0.0f ? 1.0f : -1.0f, 0.0f);
    }
    else
    {
        outN = make_float3(0.0f, 0.0f, p.z > 0.0f ? 1.0f : -1.0f);
    }
    return t;
}

// Y-axis cylinder, radius=1, y∈[-1,1].  Barrel + two caps.
// Returns smallest valid t and writes outward normal.  Returns -1 on miss.
static __forceinline__ __device__
float implicitCylinderT(const float3& ro, const float3& rd, float3& outN)
{
    const float tMin  = optixGetRayTmin();
    const float tMax  = optixGetRayTmax();
    float bestT       = -1.0f;
    float3 bestN      = make_float3(0.0f, 1.0f, 0.0f);

    // ── Barrel: x²+z² = 1 ────────────────────────────────────────────────────
    const float a = rd.x * rd.x + rd.z * rd.z;
    if (fabsf(a) > 1e-8f)
    {
        const float b    = 2.0f * (ro.x * rd.x + ro.z * rd.z);
        const float c    = ro.x * ro.x + ro.z * ro.z - 1.0f;
        const float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f)
        {
            const float sq = sqrtf(disc);
            const float t1 = (-b - sq) / (2.0f * a);
            const float t2 = (-b + sq) / (2.0f * a);
            for (int k = 0; k < 2; ++k)
            {
                const float t = (k == 0) ? t1 : t2;
                if (t < tMin || t > tMax) { continue; }
                const float y = ro.y + t * rd.y;
                if (y < -1.0f || y > 1.0f) { continue; }
                if (bestT < 0.0f || t < bestT)
                {
                    bestT = t;
                    const float3 p = ro + rd * t;
                    bestN = make_float3(p.x, 0.0f, p.z);
                }
            }
        }
    }

    // ── Caps: y = ±1 ─────────────────────────────────────────────────────────
    if (fabsf(rd.y) > 1e-8f)
    {
        for (int k = 0; k < 2; ++k)
        {
            const float capY = (k == 0) ? -1.0f : 1.0f;
            const float t    = (capY - ro.y) / rd.y;
            if (t < tMin || t > tMax) { continue; }
            const float3 p = ro + rd * t;
            if (p.x * p.x + p.z * p.z > 1.0f) { continue; }
            if (bestT < 0.0f || t < bestT)
            {
                bestT = t;
                bestN = make_float3(0.0f, capY, 0.0f);
            }
        }
    }

    if (bestT >= 0.0f) { outN = devNormalize(bestN); }
    return bestT;
}

#endif // OPTIX_RAYTRACER_IMPLICIT_SHAPES_H

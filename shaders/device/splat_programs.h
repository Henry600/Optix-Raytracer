#ifndef OPTIX_RAYTRACER_SPLAT_PROGRAMS_H
#define OPTIX_RAYTRACER_SPLAT_PROGRAMS_H

#include "dev_common.h"
#include "splat_gather.h"

// ─── Gaussian splats ─────────────────────────────────────────────────────────
// Two intersection programs share the ellipsoid math:
//   __intersection__splat       — volume gather for the radiance path: inserts
//                                 (t, α, colour) at the gaussian's maximum
//                                 response into the SplatGather payload and
//                                 never reports, so traversal spans the segment.
//   __intersection__splat_solid — reporting 3σ ellipsoid hit, used by the pick
//                                 SBT so splats remain clickable in the viewport.

extern "C" __global__ void __intersection__splat()
{
    const SplatSetData& sd =
        *reinterpret_cast<const SplatSetData*>(optixGetSbtDataPointer());

    const unsigned int prim = optixGetPrimitiveIndex();
    const float4 mo = sd.meanOpacity[prim];

    if (mo.w < SPLAT_MIN_ALPHA)
    {
        return;  // cannot contribute above the compositing threshold
    }

    const float4 q = sd.quat[prim];   // (x, y, z, w)
    const float4 s = sd.scale[prim];

    // Rotation matrix rows from the unit quaternion
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    const float3 r0 = make_float3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),        2.0f * (xz + wy));
    const float3 r1 = make_float3(2.0f * (xy + wz),        1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx));
    const float3 r2 = make_float3(2.0f * (xz - wy),        2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy));

    const float3 invS = make_float3(1.0f / fmaxf(s.x, 1e-8f),
                                    1.0f / fmaxf(s.y, 1e-8f),
                                    1.0f / fmaxf(s.z, 1e-8f));

    const float3 ro = optixGetObjectRayOrigin();
    const float3 rd = optixGetObjectRayDirection();
    const float3 oc = make_float3(ro.x - mo.x, ro.y - mo.y, ro.z - mo.z);

    // Into the gaussian's σ-normalized frame: u = diag(1/s) · Rᵀ · v,
    // so |u|² is the squared Mahalanobis distance.
    const float3 ou = make_float3(
        (r0.x * oc.x + r1.x * oc.y + r2.x * oc.z) * invS.x,
        (r0.y * oc.x + r1.y * oc.y + r2.y * oc.z) * invS.y,
        (r0.z * oc.x + r1.z * oc.y + r2.z * oc.z) * invS.z);
    const float3 du = make_float3(
        (r0.x * rd.x + r1.x * rd.y + r2.x * rd.z) * invS.x,
        (r0.y * rd.x + r1.y * rd.y + r2.y * rd.z) * invS.y,
        (r0.z * rd.x + r1.z * rd.y + r2.z * rd.z) * invS.z);

    const float a2 = devDot(du, du);
    if (a2 < 1e-12f)
    {
        return;
    }

    // Depth of maximum gaussian response along the ray
    const float tPeak = -devDot(ou, du) / a2;
    if (tPeak < optixGetRayTmin() || tPeak > optixGetRayTmax())
    {
        return;  // peak outside this segment/batch — contributes elsewhere
    }

    // Squared Mahalanobis distance at the peak → gaussian falloff
    const float3 up = make_float3(ou.x + tPeak * du.x,
                                  ou.y + tPeak * du.y,
                                  ou.z + tPeak * du.z);
    const float r2m = devDot(up, up);
    if (r2m > 9.0f)
    {
        return;  // beyond 3σ
    }

    const float alpha = mo.w * expf(-0.5f * r2m);
    if (alpha < SPLAT_MIN_ALPHA)
    {
        return;
    }

    SplatGather* g = reinterpret_cast<SplatGather*>(
        (static_cast<uint64_t>(optixGetPayload_0()) << 32)
        | static_cast<uint64_t>(optixGetPayload_1()));

    // Sorted insert (ascending t); when full, the farthest entry is dropped
    // and `dropped` triggers a re-trace batch from the last kept depth.
    if (g->count == SPLAT_KBUF && tPeak >= g->t[SPLAT_KBUF - 1])
    {
        g->dropped = 1;
        return;
    }

    int i;
    if (g->count == SPLAT_KBUF)
    {
        g->dropped = 1;          // overwriting the farthest kept entry
        i = SPLAT_KBUF - 1;
    }
    else
    {
        i = g->count;
        ++g->count;
    }
    while (i > 0 && g->t[i - 1] > tPeak)
    {
        g->t[i]    = g->t[i - 1];
        g->a[i]    = g->a[i - 1];
        g->inst[i] = g->inst[i - 1];
        g->prim[i] = g->prim[i - 1];
        --i;
    }
    g->t[i]    = tPeak;
    g->a[i]    = alpha;
    g->inst[i] = optixGetInstanceIndex();
    g->prim[i] = prim;

    // Once the buffer is full, report a hit at the farthest kept depth: OptiX
    // shrinks tmax, so traversal culls BVH nodes beyond it — dropped hits out
    // there are re-gathered by the next batch anyway.
    if (g->count == SPLAT_KBUF)
    {
        optixReportIntersection(g->t[SPLAT_KBUF - 1], 0);
    }
}

extern "C" __global__ void __intersection__splat_solid()
{
    const SplatSetData& sd =
        *reinterpret_cast<const SplatSetData*>(optixGetSbtDataPointer());

    const unsigned int prim = optixGetPrimitiveIndex();
    const float4 mo = sd.meanOpacity[prim];

    if (mo.w < 0.05f)
    {
        return;  // don't let near-invisible gaussians eat pick clicks
    }

    const float4 q = sd.quat[prim];   // (x, y, z, w)
    const float4 s = sd.scale[prim];

    // Rotation matrix rows from the unit quaternion
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    const float3 r0 = make_float3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),        2.0f * (xz + wy));
    const float3 r1 = make_float3(2.0f * (xy + wz),        1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx));
    const float3 r2 = make_float3(2.0f * (xz - wy),        2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy));

    const float3 invS = make_float3(1.0f / fmaxf(s.x, 1e-8f),
                                    1.0f / fmaxf(s.y, 1e-8f),
                                    1.0f / fmaxf(s.z, 1e-8f));

    const float3 ro = optixGetObjectRayOrigin();
    const float3 rd = optixGetObjectRayDirection();
    const float3 oc = make_float3(ro.x - mo.x, ro.y - mo.y, ro.z - mo.z);

    const float3 ou = make_float3(
        (r0.x * oc.x + r1.x * oc.y + r2.x * oc.z) * invS.x,
        (r0.y * oc.x + r1.y * oc.y + r2.y * oc.z) * invS.y,
        (r0.z * oc.x + r1.z * oc.y + r2.z * oc.z) * invS.z);
    const float3 du = make_float3(
        (r0.x * rd.x + r1.x * rd.y + r2.x * rd.z) * invS.x,
        (r0.y * rd.x + r1.y * rd.y + r2.y * rd.z) * invS.y,
        (r0.z * rd.x + r1.z * rd.y + r2.z * rd.z) * invS.z);

    // |ou + t·du|² = 9  (3σ boundary), solved in half-b form
    const float a = devDot(du, du);
    if (a < 1e-12f)
    {
        return;
    }
    const float b    = devDot(ou, du);
    const float c    = devDot(ou, ou) - 9.0f;
    const float disc = b * b - a * c;
    if (disc < 0.0f)
    {
        return;
    }

    const float sq   = sqrtf(disc);
    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();

    float t = (-b - sq) / a;
    if (t < tMin || t > tMax)
    {
        t = (-b + sq) / a;
        if (t < tMin || t > tMax)
        {
            return;
        }
    }

    optixReportIntersection(t, 0);
}

#endif // OPTIX_RAYTRACER_SPLAT_PROGRAMS_H

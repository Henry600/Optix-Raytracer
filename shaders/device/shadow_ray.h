#ifndef OPTIX_RAYTRACER_SHADOW_RAY_H
#define OPTIX_RAYTRACER_SHADOW_RAY_H

#include "dev_common.h"

// ─── Shadow / visibility ray ─────────────────────────────────────────────────
// Traces a visibility ray against surface geometry (VIS_MASK_GEOMETRY — splats
// never occlude).  SBT miss index 1 (__miss__shadow) marks the ray as having
// reached the light; __anyhit__radiance terminates it on opaque surfaces and
// multiplies a colour filter through thin-walled glass (payload p1..p3), so
// the filter accumulates in ray order.  No TERMINATE_ON_FIRST_HIT for that
// reason.
//
// Returns true when the light is reachable; `filter` then holds the
// accumulated thin-walled tint (white when nothing was crossed).

static __forceinline__ __device__
bool traceShadow(const float3 origin, const float3 dir, const float tMax, float3& filter)
{
    uint32_t vis = 0u;
    uint32_t fr  = __float_as_uint(1.0f);
    uint32_t fg  = __float_as_uint(1.0f);
    uint32_t fb  = __float_as_uint(1.0f);
    optixTrace(
        optixLaunchParams.traversable,
        origin, dir,
        1e-3f, tMax, 0.0f,
        OptixVisibilityMask(VIS_MASK_GEOMETRY),
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0, 1, 1,   // offset=0, stride=1, missIndex=1
        vis, fr, fg, fb);

    filter = make_float3(__uint_as_float(fr), __uint_as_float(fg), __uint_as_float(fb));
    return vis != 0u;
}

#endif // OPTIX_RAYTRACER_SHADOW_RAY_H

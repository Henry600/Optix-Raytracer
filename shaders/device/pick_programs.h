#ifndef OPTIX_RAYTRACER_PICK_PROGRAMS_H
#define OPTIX_RAYTRACER_PICK_PROGRAMS_H

#include "dev_common.h"

// ─── 1px picking ─────────────────────────────────────────────────────────────
// Fires a single ray from normalised screen-space (pickU, pickV) and writes
// the TLAS instance index of the closest hit into pickResult.  0xFFFFFFFF = miss.

extern "C" __global__ void __raygen__pick()
{
    const float ndcX  = 2.0f * optixLaunchParams.pickU - 1.0f;
    const float ndcY  = 1.0f - 2.0f * optixLaunchParams.pickV;
    const float3 dir  = devNormalize(
        optixLaunchParams.W
        + ndcX * optixLaunchParams.U
        + ndcY * optixLaunchParams.V);

    uint32_t instanceIdx = 0xFFFFFFFFu;
    uint32_t hitT_bits   = __float_as_uint(-1.0f);  // -1 = miss sentinel
    optixTrace(
        optixLaunchParams.traversable,
        optixLaunchParams.eye, dir,
        1e-3f, 1e16f, 0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        instanceIdx, hitT_bits);

    *optixLaunchParams.pickResult = instanceIdx;
    if (optixLaunchParams.pickDistance)
    {
        *optixLaunchParams.pickDistance = __uint_as_float(hitT_bits);
    }
}

extern "C" __global__ void __closesthit__pick()
{
    optixSetPayload_0(optixGetInstanceIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __miss__pick()
{
    optixSetPayload_0(0xFFFFFFFFu);
    optixSetPayload_1(__float_as_uint(-1.0f));
}

#endif // OPTIX_RAYTRACER_PICK_PROGRAMS_H

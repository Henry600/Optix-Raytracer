#ifndef OPTIX_RAYTRACER_MISS_SHADOW_PROGRAMS_H
#define OPTIX_RAYTRACER_MISS_SHADOW_PROGRAMS_H

#include "dev_common.h"

// ─── Miss ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void __miss__radiance()
{
    // intentionally empty — hit==0 default in raygen is sufficient
}

// ─── Miss — NEE shadow ray ────────────────────────────────────────────────────
// Fires when a shadow ray reaches the background unoccluded.
// Sets p0 = 1 (visible); the raygen reads this to confirm the light path is clear.

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(1u);
}

// ─── Any-hit — shadow ray thin-walled colour filter ──────────────────────────
// Only reached by NEE shadow rays; radiance rays set OPTIX_RAY_FLAG_DISABLE_ANYHIT
// so this program is never invoked for primary or bounce rays.
//
// Shadow rays carry 4 payload registers:
//   p0        : vis  — 0 while in flight; set to 1 by __miss__shadow on escape
//   p1/p2/p3  : accumulated colour filter (float bits, caller-initialised to 1.0)
//
// Thin-walled surfaces multiply the running filter by their linearised albedo and
// call optixIgnoreIntersection() so BVH traversal continues toward the light.
// Non-thin-walled surfaces call optixTerminateRay() — this is the correct way to
// signal occlusion when TERMINATE_ON_FIRST_HIT is not set, and it guarantees
// that the first opaque hit (nearest-to-farthest in BVH order) wins.

extern "C" __global__ void __anyhit__radiance()
{
    const MeshData& mesh =
        *reinterpret_cast<const MeshData*>(optixGetSbtDataPointer());
    if (   mesh.materialIndex >= 0
        && optixLaunchParams.materials
        && optixLaunchParams.materials[mesh.materialIndex].thinWalled)
    {
        // Accumulate this surface's (linear) transmission colour into the colour filter.
        const float3 alb = optixLaunchParams.materials[mesh.materialIndex].transmissionColor;
        optixSetPayload_1(__float_as_uint(__uint_as_float(optixGetPayload_1()) * alb.x));
        optixSetPayload_2(__float_as_uint(__uint_as_float(optixGetPayload_2()) * alb.y));
        optixSetPayload_3(__float_as_uint(__uint_as_float(optixGetPayload_3()) * alb.z));
        optixIgnoreIntersection();          // keep traversal going past the glass
    }
    else
    {
        optixTerminateRay();                // opaque surface — shadow ray is blocked
    }
}

#endif // OPTIX_RAYTRACER_MISS_SHADOW_PROGRAMS_H

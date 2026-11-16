#ifndef OPTIX_RAYTRACER_PATH_STATE_H
#define OPTIX_RAYTRACER_PATH_STATE_H

#include "dev_common.h"

// ─── Per-path state ──────────────────────────────────────────────────────────
// Everything the raygen's bounce loop carries between iterations.  Lobe and
// event functions take it by reference; being forced-inline, the members live
// in registers exactly as the old loop-local variables did.

struct PathState
{
    float3   rayOrig;
    float3   rayDir;
    float3   throughput;
    float3   radiance;

    // PDF of the last BSDF-sampled direction (diffuse / specular / coat);
    // 0 = refraction, pass-through or first ray.  Carried into the next
    // iteration so the miss / emission blocks can MIS-weight against NEE.
    float    bsdfPdfForMis;

    // Beer-Lambert absorption coefficient: -log(transmissionColor) per unit
    // distance.  Kept in sync with medAbsorb[medTop].
    float3   absorb;

    uint32_t seed;
};

// Nested-dielectric medium stack.  Index 0 = air (never popped).  Kept apart
// from PathState on purpose: the stack is dynamically indexed, and living in
// the same aggregate keeps the whole path state (ray, throughput, ...) out of
// registers.
struct MediumStack
{
    static constexpr int kMaxDepth = 8;

    float    medIor   [kMaxDepth];
    float3   medAbsorb[kMaxDepth];
    int      medTop;
};

// Initialises everything except the primary ray (see generateCameraRay).
static __forceinline__ __device__
void initPathState(PathState& ps, const uint32_t seed)
{
    ps.throughput    = make_float3(1.0f, 1.0f, 1.0f);
    ps.radiance      = make_float3(0.0f, 0.0f, 0.0f);
    ps.bsdfPdfForMis = 0.0f;
    ps.absorb        = make_float3(0.0f, 0.0f, 0.0f);
    ps.seed          = seed;
}

static __forceinline__ __device__
void initMediumStack(MediumStack& ms)
{
    ms.medTop       = 0;
    ms.medIor[0]    = 1.0f;
    ms.medAbsorb[0] = make_float3(0.0f, 0.0f, 0.0f);
}

#endif // OPTIX_RAYTRACER_PATH_STATE_H

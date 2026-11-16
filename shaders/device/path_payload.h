#ifndef OPTIX_RAYTRACER_PATH_PAYLOAD_H
#define OPTIX_RAYTRACER_PATH_PAYLOAD_H

#include "dev_common.h"

// ─── Path payload ─────────────────────────────────────────────────────────────
// The raygen stack-allocates one of these and passes a packed pointer to each
// optixTrace call.  The closest-hit program fills in the fields and sets hit=1.

struct PathVertex
{
    float3 pos;          // world-space surface point
    float3 N;            // world-space shading normal (unit)
    float3 albedo;       // base colour
    float3 emission;     // pre-scaled emissive radiance (emission * emissionScale)
    float  roughness;    // perceptual roughness [0, 1]
    float  metallic;     // metallic factor [0, 1]
    float  diffuseRoughness; // Oren-Nayar sigma [0, 1]; 0 = Lambertian
    float  transmission;         // 0 = opaque, 1 = fully transmissive
    float  ior;                  // index of refraction
    float  absorptionDistance;   // world-space units for full albedo absorption
    float3 scatteringCoeff;      // per-channel σ_s (1/MFP); (0,0,0) = disabled
    float3 transmissionColor;     // transmission tint / absorption colour (linear)
    float  scatteringAnisotropy; // Henyey-Greenstein g [-1, +1]
    float  clearcoat;            // clearcoat layer intensity [0, 1]
    float  clearcoatRoughness;   // clearcoat layer roughness [0, 1]
    float  clearcoatDarkening;   // OpenPBR coat_darkening [0, 1]
    float3 clearcoatColor;       // coat absorption tint (linear)
    float  clearcoatIor;         // coat IOR
    float  clearcoatKr;          // coat internal diffuse reflectance (host-derived)
    int    thinWalled;           // 1 = zero-thickness surface; pass through without refraction
    int    shadowCatcher;        // 1 = invisible to camera/GI rays; shows background + shadow only
    float        t;          // ray travel distance to this hit (Beer-Lambert)
    unsigned int instanceId; // TLAS instance index; ~0u when not set (miss / mesh)
    int          hit;        // 1 = geometry hit, 0 = ray escaped to background
};

static __forceinline__ __device__
void packPointer(void* ptr, uint32_t& p0, uint32_t& p1)
{
    const uint64_t u = reinterpret_cast<uint64_t>(ptr);
    p0 = static_cast<uint32_t>(u >> 32);
    p1 = static_cast<uint32_t>(u & 0xFFFFFFFFu);
}

static __forceinline__ __device__
PathVertex* unpackVertex(uint32_t p0, uint32_t p1)
{
    return reinterpret_cast<PathVertex*>(
        (static_cast<uint64_t>(p0) << 32) | static_cast<uint64_t>(p1));
}

#endif // OPTIX_RAYTRACER_PATH_PAYLOAD_H

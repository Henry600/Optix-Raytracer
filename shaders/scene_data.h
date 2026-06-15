#ifndef OPTIX_RAYTRACER_SCENE_DATA_H
#define OPTIX_RAYTRACER_SCENE_DATA_H

#include <cuda_runtime.h>  // float3, float2, uint3, cudaTextureObject_t

// GPU view of one mesh — placed in the SBT hit group record per mesh instance.
// In a hit shader: const MeshData* mesh = (const MeshData*)optixGetSbtDataPointer();
struct MeshData
{
    const float3* positions;
    const float3* normals;
    const float2* uvs;
    const uint3*  indices;
    int           materialIndex;  // into the per-launch MaterialData array
};

// Flat POD material — used on the host side and copied directly into SBT records
// or a device-side MaterialData array. No std::string; names live in Scene on the host.
struct MaterialData
{
    // UV transform applied to all texture lookups on this material.
    // xy = tiling (scale), zw = offset.  Default: scale=(1,1), offset=(0,0).
    float4 uvTransform   = { 1.0f, 1.0f, 0.0f, 0.0f };

    float3 albedo        = { 1.0f, 1.0f, 1.0f };
    int    albedoTexture = -1;    // index into device texture array; -1 = no texture

    float  roughness        = 0.5f;
    int    roughnessTexture = -1;  // index into device texture array; red channel; -1 = no texture
    float  metallic         = 0.0f;

    float3 emission         = { 0.0f, 0.0f, 0.0f };
    float  emissionScale    = 1.0f;
    int    emissionTexture  = -1;   // index into device texture array; -1 = no texture

    float  transmission          = 0.0f;  // 0 = opaque, 1 = fully transmissive
    float  ior                   = 1.5f;  // index of refraction (glass ≈ 1.5)
    float  absorptionDistance    = 1.0f;  // world-space distance for full albedo absorption
    float3 scatteringCoeff       = { 0.0f, 0.0f, 0.0f };  // per-channel σ_s (1/MFP); (0,0,0) = disabled
    float  scatteringAnisotropy  = 0.0f;  // Henyey-Greenstein g [-1 back, 0 iso, +1 forward]

    float  clearcoat          = 0.0f;  // KHR_materials_clearcoat intensity [0, 1]
    float  clearcoatRoughness = 0.0f;  // clearcoat layer roughness [0, 1]

    int    thinWalled = 0;  // 1 = invisible to NEE shadow rays (pass-through)
};

// Device-side mirror of ImplicitType from implicit_node.h.
// Plain C-style enum for CUDA device code compatibility.
enum ImplicitTypeGPU : unsigned int
{
    IMPLICIT_SPHERE   = 0,
    IMPLICIT_BOX      = 1,
    IMPLICIT_CYLINDER = 2,
};

// SBT data for one implicit shape instance.
// Read by __intersection__implicit, __anyhit__implicit, __closesthit__implicit.
struct ImplicitShapeData
{
    unsigned int type;          // ImplicitTypeGPU value
    int          materialIndex; // into the per-launch MaterialData array
};

#endif // OPTIX_RAYTRACER_SCENE_DATA_H

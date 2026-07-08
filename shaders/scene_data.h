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

// GPU view of one gaussian splat dataset — all pointers are device pointers
// into buffers owned by the host-side SplatDeviceBuffers.  Arrays are indexed
// by gaussian; float4 packing keeps loads coalesced.
struct SplatSetData
{
    const float4* meanOpacity;  // xyz = mean (splat-local space), w = opacity [0,1]
    const float4* quat;         // unit quaternion (x, y, z, w)
    const float4* scale;        // xyz = linear per-axis scale, w unused
    const float4* sh0;          // xyz = DC SH coefficients (f_dc), w unused
    const float*  shN;          // higher-order SH (see gaussian_splat.h layout); null when absent
    unsigned int  count;
    int           shBands;      // 0 = DC only
    int           antialias;    // 1 = dataset was trained with mip-splatting antialias
};

// One emissive implicit light — uploaded per-frame for direct light sampling (NEE).
// Stores the local-to-world transform and its inverse so the device can sample a
// point on the canonical unit shape and compute a correct solid-angle PDF via the
// surface Jacobian:  dA_world = |det(L)| * |W^T * n_local| * dA_local
// where L = l2w_3x3, W = w2l_3x3 = L^{-1}.
struct EmissiveLightData
{
    float        l2w[12];      // local-to-world 3×4, row-major (last row [0,0,0,1] implicit)
    float        w2l[12];      // world-to-local 3×4, row-major
    float3       emission;     // linear emission (sRGB^2.2 * emissionScale)
    float        invDetW2l;    // |det(w2l_3x3)| = 1/|det(l2w_3x3)|
    float        localArea;    // canonical surface area: 4π (sphere), 24 (box), 6π (cyl)
    unsigned int type;         // ImplicitTypeGPU
    unsigned int instanceId;   // TLAS instance index, for MIS at direct emissive hits
};

#endif // OPTIX_RAYTRACER_SCENE_DATA_H

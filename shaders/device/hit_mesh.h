#ifndef OPTIX_RAYTRACER_HIT_MESH_H
#define OPTIX_RAYTRACER_HIT_MESH_H

#include "dev_common.h"
#include "path_payload.h"
#include "material.h"

// ─── Closest-hit ─────────────────────────────────────────────────────────────

extern "C" __global__ void __closesthit__radiance()
{
    PathVertex* vtx = unpackVertex(optixGetPayload_0(), optixGetPayload_1());

    const MeshData& mesh = *reinterpret_cast<const MeshData*>(optixGetSbtDataPointer());

    // ── Interpolated shading normal ───────────────────────────────────────────
    const uint3  tri = mesh.indices[optixGetPrimitiveIndex()];
    const float2 bc  = optixGetTriangleBarycentrics();
    const float  w0  = 1.0f - bc.x - bc.y;

    const float3 n_obj = devNormalize(
        mesh.normals[tri.x] * w0  +
        mesh.normals[tri.y] * bc.x +
        mesh.normals[tri.z] * bc.y);

    vtx->N = devNormalize(optixTransformNormalFromObjectToWorldSpace(n_obj));

    // ── World-space hit point and travel distance ─────────────────────────────
    vtx->t   = optixGetRayTmax();
    vtx->pos = optixGetWorldRayOrigin()
             + optixGetWorldRayDirection() * vtx->t;

    // ── Material ──────────────────────────────────────────────────────────────
    if (optixLaunchParams.materials && mesh.materialIndex >= 0)
    {
        const MaterialData& mat = optixLaunchParams.materials[mesh.materialIndex];
        applyMaterial(vtx, mat);

        // Texture layers — each sample is sRGB-encoded (except roughness), so it
        // is linearised independently and multiplied onto the already-linear
        // factor that applyMaterial wrote (glTF: factor × texture).
        float4 texSample;
        if (sampleSceneTex(mat.albedoTexture, mesh.uvs, tri, w0, bc, mat.uvTransform, texSample))
        {
            const float3 texLinear = srgbToLinear3(make_float3(texSample.x, texSample.y, texSample.z));
            vtx->albedo = make_float3(
                vtx->albedo.x * texLinear.x,
                vtx->albedo.y * texLinear.y,
                vtx->albedo.z * texLinear.z);
        }

        // Roughness — red channel × roughness factor
        if (sampleSceneTex(mat.roughnessTexture, mesh.uvs, tri, w0, bc, mat.uvTransform, texSample))
        {
            vtx->roughness *= texSample.x;
        }

        // Emission — linear colour × scale × texture (multiplication commutes)
        if (sampleSceneTex(mat.emissionTexture, mesh.uvs, tri, w0, bc, mat.uvTransform, texSample))
        {
            const float3 emTexLinear = srgbToLinear3(
                make_float3(texSample.x, texSample.y, texSample.z));
            vtx->emission = make_float3(vtx->emission.x * emTexLinear.x,
                                        vtx->emission.y * emTexLinear.y,
                                        vtx->emission.z * emTexLinear.z);
        }
    }
    else
    {
        applyDefaultMaterial(vtx);
    }

    vtx->hit        = 1;
    vtx->instanceId = ~0u;  // mesh hits are never in the emissive-lights array
}

#endif // OPTIX_RAYTRACER_HIT_MESH_H

#ifndef OPTIX_RAYTRACER_MATERIAL_H
#define OPTIX_RAYTRACER_MATERIAL_H

#include "dev_common.h"
#include "path_payload.h"

// ─── Scene texture sampling helper ───────────────────────────────────────────

// Sample a scene texture at the barycentric-interpolated UV for the current hit.
// Returns true and writes the sampled float4 to `out` when all of the following hold:
//   • index >= 0 (a texture is assigned)
//   • optixLaunchParams.sceneTextures is non-null
//   • uvs is non-null (the mesh has UV data)
// Returns false and leaves `out` unchanged otherwise.
// uvTransform: xy = tiling (scale), zw = offset.
static __forceinline__ __device__
bool sampleSceneTex(int index, const float2* uvs, const uint3& tri,
                    float w0, float2 bc, float4 uvTransform, float4& out)
{
    if (index < 0 || !optixLaunchParams.sceneTextures || !uvs)
    {
        return false;
    }

    // Interpolate raw mesh UV, then apply tiling and offset.
    const float2 uvRaw = make_float2(
        uvs[tri.x].x * w0 + uvs[tri.y].x * bc.x + uvs[tri.z].x * bc.y,
        uvs[tri.x].y * w0 + uvs[tri.y].y * bc.x + uvs[tri.z].y * bc.y);
    const float2 uv = make_float2(
        uvRaw.x * uvTransform.x + uvTransform.z,
        uvRaw.y * uvTransform.y + uvTransform.w);

    out = tex2D<float4>(optixLaunchParams.sceneTextures[index], uv.x, uv.y);
    return true;
}

// ─── Material → path vertex ──────────────────────────────────────────────────
// Single place where MaterialData is copied into the PathVertex payload, shared
// by every closest-hit program.  A new material field needs adding here only.
// The device MaterialData array already holds linear colours (linearised once at
// upload, see Application::uploadMaterials), so no pow() runs per hit.  Textured
// lookups (albedo / roughness / emission) are layered on top by the caller
// (see __closesthit__radiance).

static __forceinline__ __device__
void applyMaterial(PathVertex* vtx, const MaterialData& mat)
{
    vtx->albedo    = mat.albedo;
    vtx->roughness = mat.roughness;
    vtx->metallic  = mat.metallic;
    vtx->diffuseRoughness = mat.diffuseRoughness;
    vtx->emission  = mat.emission * mat.emissionScale;

    vtx->transmission         = mat.transmission;
    vtx->transmissionColor    = mat.transmissionColor;
    vtx->ior                  = mat.ior;
    vtx->absorptionDistance   = fmaxf(mat.absorptionDistance, 1e-4f);
    vtx->scatteringCoeff      = make_float3(fmaxf(mat.scatteringCoeff.x, 0.0f),
                                            fmaxf(mat.scatteringCoeff.y, 0.0f),
                                            fmaxf(mat.scatteringCoeff.z, 0.0f));
    vtx->scatteringAnisotropy = mat.scatteringAnisotropy;

    vtx->clearcoat          = mat.clearcoat;
    vtx->clearcoatRoughness = mat.clearcoatRoughness;
    vtx->clearcoatDarkening = mat.clearcoatDarkening;
    vtx->clearcoatColor     = mat.clearcoatColor;
    vtx->clearcoatIor       = mat.clearcoatIor;
    vtx->clearcoatKr        = mat.clearcoatKr;

    vtx->thinWalled    = mat.thinWalled;
    vtx->shadowCatcher = mat.shadowCatcher;
}

// Fallback for geometry without a valid material: the MaterialData member
// initialisers (scene_data.h) are the single source of defaults, with a neutral
// grey base colour.
static __forceinline__ __device__
void applyDefaultMaterial(PathVertex* vtx)
{
    const MaterialData def = MaterialData();
    applyMaterial(vtx, def);
    vtx->albedo = make_float3(0.8f, 0.8f, 0.8f);  // linear
}

#endif // OPTIX_RAYTRACER_MATERIAL_H

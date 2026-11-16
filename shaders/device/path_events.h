#ifndef OPTIX_RAYTRACER_PATH_EVENTS_H
#define OPTIX_RAYTRACER_PATH_EVENTS_H

#include "dev_common.h"
#include "path_payload.h"
#include "path_state.h"
#include "environment.h"
#include "emissive_lights.h"

// Denoiser guide layers - written on the first bounce only.
static __forceinline__ __device__
void writeGuideLayers(const unsigned int fbIdx, const PathVertex& vtx)
{
    if (optixLaunchParams.normalBuffer)
    {
        optixLaunchParams.normalBuffer[fbIdx] = vtx.hit
            ? make_float4(vtx.N.x, vtx.N.y, vtx.N.z, 0.0f)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (optixLaunchParams.albedoBuffer)
    {
        optixLaunchParams.albedoBuffer[fbIdx] = vtx.hit
            ? make_float4(vtx.albedo.x, vtx.albedo.y, vtx.albedo.z, 1.0f)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

// Ray escaped: add the env-map / sky radiance with MIS against env NEE.
static __forceinline__ __device__
void shadeBackground(PathState& ps)
{
    float3& rayDir = ps.rayDir;
    float3& throughput = ps.throughput;
    float3& radiance = ps.radiance;
    float& bsdfPdfForMis = ps.bsdfPdfForMis;

    float3 bg = sampleBackground(rayDir);
    if (optixLaunchParams.envMarginalCdf && bsdfPdfForMis > 0.0f)
    {
        // MIS power heuristic: down-weight the BSDF path when the env map
        // IS would have sampled this direction with a higher density.
        const float pEnv  = evalEnvMapPdf(rayDir);
        const float wBsdf = (pEnv > 0.0f)
            ? (bsdfPdfForMis * bsdfPdfForMis)
              / (bsdfPdfForMis * bsdfPdfForMis + pEnv * pEnv)
            : 1.0f;
        radiance += throughput * bg * wBsdf;
    }
    else
    {
        radiance += throughput * bg;
    }
}

// Beer-Lambert attenuation over the segment just travelled, then add the
// hit surface emission (MIS-weighted against emissive-light NEE).
static __forceinline__ __device__
void applyAbsorptionAndEmission(const PathVertex& vtx, PathState& ps)
{
    float3& rayOrig = ps.rayOrig;
    float3& throughput = ps.throughput;
    float3& radiance = ps.radiance;
    float& bsdfPdfForMis = ps.bsdfPdfForMis;
    float3& absorb = ps.absorb;

    throughput.x *= expf(-absorb.x * vtx.t);
    throughput.y *= expf(-absorb.y * vtx.t);
    throughput.z *= expf(-absorb.z * vtx.t);

    // ── Emission ──────────────────────────────────────────────────────────
    // When emissive NEE is active and this hit was an implicit light, apply
    // the MIS power heuristic to avoid double-counting direct contributions:
    // the NEE strategy already sampled this light in the previous bounce.
    if (optixLaunchParams.emissiveLightCount > 0
        && vtx.instanceId != ~0u
        && bsdfPdfForMis > 0.0f)
    {
        const float pLight = evalEmissiveLightPdf(
            vtx.instanceId, rayOrig, vtx.pos, vtx.N, vtx.t);
        const float wBsdf  = (pLight > 0.0f)
            ? (bsdfPdfForMis * bsdfPdfForMis)
              / (bsdfPdfForMis * bsdfPdfForMis + pLight * pLight)
            : 1.0f;
        radiance += throughput * vtx.emission * wBsdf;
    }
    else
    {
        radiance += throughput * vtx.emission;
    }
}

#endif // OPTIX_RAYTRACER_PATH_EVENTS_H

// launch_params.h — shared between host (C++) and device (CUDA) code.
// Included by both application.cpp and device_programs.cu.
// Must use only types valid in both C++17 and CUDA device code:
// no STL, no windows.h, no host-only headers.
#ifndef OPTIX_RAYTRACER_LAUNCH_PARAMS_H
#define OPTIX_RAYTRACER_LAUNCH_PARAMS_H

#include <stdint.h>        // uint32_t and friends (safe in both host and device code)
#include <optix.h>         // OptixTraversableHandle
#include <cuda_runtime.h>  // uchar4, uint2
#include "scene_data.h"     // MaterialData

struct LaunchParams
{
    // Display output — linear float4 for the scRGB FP16 swapchain.
    // hdrDisplay selects the encode written by the raygen:
    //   0 = Reinhard only (linear [0,1])   — scRGB swapchain, HDR Output off.
    //   1 = unclamped linear scRGB         — scRGB swapchain, HDR Output on.
    //   2 = Reinhard + gamma 1/2.2         — SRGB_NONLINEAR swapchain (no HDR display).
    float4* colorBuffer;  // device pointer — output of writeDisplay()
    int     hdrDisplay;   // display encode mode (0/1/2 — see above)
    uint2                  fbSize;       // { width, height } in pixels
    OptixTraversableHandle traversable;  // top-level IAS; 0 = no scene loaded

    // Pinhole camera basis vectors computed on the host each frame.
    // dir = normalize(ndc.x * U + ndc.y * V + W)  where ndc in [-1,1]
    float3 eye;  // camera position in world space
    float3 U;    // right direction * tan(halfFovH)
    float3 V;    // up direction   * tan(halfFovV)
    float3 W;    // forward direction (unit length, pointing into scene)

    // Equirectangular (lat-long) environment map. 0 = not loaded; raygen falls
    // back to the procedural sky gradient when this is 0.
    cudaTextureObject_t envMap;
    float               envMapRotation;  // azimuth offset in radians (Shift+RMB drag)
    float               envExposure;     // exposure in EV stops; applied as 2^n to env radiance

    // HDRI importance-sampling CDFs — null when no env map is loaded.
    // marginalCdf[j]      = P(row ≤ j), normalised prefix-sum of row energies.
    // conditionalCdf[j*W+i] = P(col ≤ i | row j), normalised within each row.
    const float* envMarginalCdf;    // device float[envCdfH]
    const float* envConditionalCdf; // device float[envCdfH × envCdfW]
    int          envCdfW;           // env map width  (= CDF column count)
    int          envCdfH;           // env map height (= CDF row count)

    // Thin-lens depth of field. lensRadius = 0 → pinhole (no DoF).
    // lensRadius is in world units (1 unit = 1 m); focusDistance is in world units.
    float lensRadius;     // half aperture = focalLength_mm / (2 × fStop × 1000)
    float focusDistance;  // distance from eye to focal plane along W
    float bokehEdgeBias;  // [0, 1]: 0 = uniform disk, 1 = pure rim ring

    // Per-pixel HDR accumulation buffer (float4, w unused).  The raygen adds
    // one sample per launch; the display value is accumBuffer[i] / sampleIndex.
    float4*  accumBuffer;
    uint32_t sampleIndex;  // number of samples already accumulated

    // Denoiser guide layers — null when denoiser is disabled.
    float4* normalBuffer;  // first-hit world-space shading normal (written every frame)
    float4* albedoBuffer;  // first-hit material albedo          (written every frame)
    float4* hdrBuffer;     // running HDR average = accumBuffer / sampleIndex

    // Scene materials — device pointer to a MaterialData array, indexed by
    // MeshData::materialIndex.  Null when no scene is loaded.
    const MaterialData* materials;

    // Scene textures — flat device array of cudaTextureObject_t, indexed by
    // MaterialData::albedoTexture.  Null when no textures have been uploaded.
    const cudaTextureObject_t* sceneTextures;

    // Emissive implicit lights for direct sampling (NEE).
    // Null / 0 when no emissive implicit shapes exist in the scene.
    const EmissiveLightData* emissiveLights;
    int                      emissiveLightCount;

    // Picking — only valid during a 1×1 pick launch; ignored by the normal raygen.
    float     pickU;      // normalised mouse X [0, 1]
    float     pickV;      // normalised mouse Y [0, 1]
    uint32_t* pickResult;   // device pointer to 1-element output; null during normal render
    float*    pickDistance; // device pointer to 1-element hit-T output; null during normal render
};

#endif // OPTIX_RAYTRACER_LAUNCH_PARAMS_H

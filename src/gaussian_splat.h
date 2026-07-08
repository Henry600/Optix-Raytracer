#ifndef OPTIX_RAYTRACER_GAUSSIAN_SPLAT_H
#define OPTIX_RAYTRACER_GAUSSIAN_SPLAT_H

#include "gpu_buffer.h"
#include "scene_data.h"    // SplatSetData — shared with device code

#include <cuda_runtime.h>  // float3, float4
#include <cstdint>
#include <string>
#include <vector>

// Host-side storage for a 3D Gaussian Splat dataset decoded from a SOG file.
// All quantization/codebook indirection is resolved at load time so this is a
// flat, render-agnostic representation the upload step can consume directly.
struct GaussianSplatData
{
    uint32_t count     = 0;
    int      shBands   = 0;      // 0 = DC color only, 1..3 = shN present
    bool     antialias = false;  // meta.json antialias hint (mip-splatting style)

    std::vector<float3> positions;   // local-space means
    std::vector<float4> rotations;   // unit quaternion (x, y, z, w)
    std::vector<float3> scales;      // linear per-axis scale (exp applied)
    std::vector<float3> sh0;         // DC spherical-harmonic coefficients (f_dc);
                                     // base color = 0.5 + SH_C0 * f_dc
    std::vector<float>  opacities;   // [0, 1]

    // Higher-order SH coefficients, empty when shBands == 0.
    // Layout: count * coeffsPerChannel(shBands) * 3 floats, coefficient-major
    // with RGB interleaved per coefficient:
    //   [c0.r, c0.g, c0.b, c1.r, c1.g, c1.b, ...]
    // coeffsPerChannel = (shBands + 1)^2 - 1  →  3 / 8 / 15 for bands 1 / 2 / 3.
    std::vector<float>  shN;

    std::string name;  // source file stem, shown in the UI
};

// Device-side buffers for one splat dataset.  Owned by Scene, parallel to its
// GaussianSplatData vector.  upload() packs the host arrays into coalesced
// float4 layouts; view() yields the pointer struct passed to device code.
struct SplatDeviceBuffers
{
    GPUBuffer meanOpacity;  // float4: xyz = mean, w = opacity
    GPUBuffer quat;         // float4: quaternion (x,y,z,w)
    GPUBuffer scale;        // float4: xyz = linear scale, w = 0
    GPUBuffer sh0;          // float4: xyz = f_dc, w = 0
    GPUBuffer shN;          // float[], empty when the dataset has no higher-order SH

    unsigned int count     = 0;
    int          shBands   = 0;
    bool         antialias = false;

    bool valid() const { return meanOpacity.valid(); }

    // Throws std::runtime_error on CUDA allocation/copy failure.
    void upload(const GaussianSplatData& src);

    SplatSetData view() const
    {
        SplatSetData v = {};
        v.meanOpacity  = meanOpacity.typedPtr<const float4>();
        v.quat         = quat.typedPtr<const float4>();
        v.scale        = scale.typedPtr<const float4>();
        v.sh0          = sh0.typedPtr<const float4>();
        v.shN          = shN.valid() ? shN.typedPtr<const float>() : nullptr;
        v.count        = count;
        v.shBands      = shBands;
        v.antialias    = antialias ? 1 : 0;
        return v;
    }
};

#endif // OPTIX_RAYTRACER_GAUSSIAN_SPLAT_H

#ifndef OPTIX_RAYTRACER_DISPLAY_H
#define OPTIX_RAYTRACER_DISPLAY_H

#include "dev_common.h"

// ─── Display write-out ────────────────────────────────────────────────────────
// Writes a linear-radiance colour to the float4 display buffer.
// hdrDisplay=0: Reinhard only — linear [0,1] for scRGB/EXTENDED_SRGB_LINEAR.
// hdrDisplay=1: unclamped linear scRGB — HDR highlights exceed 1.0.
// hdrDisplay=2: Reinhard + gamma 1/2.2 — sRGB-encoded for SRGB_NONLINEAR swapchain.
static __forceinline__ __device__
void writeDisplay(unsigned int fbIdx, float3 linear)
{
    float3 c = linear;
    if (optixLaunchParams.hdrDisplay == 0)
    {
        c.x = c.x / (c.x + 1.0f);
        c.y = c.y / (c.y + 1.0f);
        c.z = c.z / (c.z + 1.0f);
    }
    else if (optixLaunchParams.hdrDisplay == 2)
    {
        c.x = powf(c.x / (c.x + 1.0f), 1.0f / 2.2f);
        c.y = powf(c.y / (c.y + 1.0f), 1.0f / 2.2f);
        c.z = powf(c.z / (c.z + 1.0f), 1.0f / 2.2f);
    }
    optixLaunchParams.colorBuffer[fbIdx] = make_float4(c.x, c.y, c.z, 1.0f);
}

#endif // OPTIX_RAYTRACER_DISPLAY_H

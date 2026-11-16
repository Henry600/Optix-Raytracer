#ifndef OPTIX_RAYTRACER_COLOR_MATH_H
#define OPTIX_RAYTRACER_COLOR_MATH_H

// Colour-space conversion shared by host (C++) and device (CUDA) code, so there
// is exactly one definition of the transfer function.  Same macro scheme as
// device_math.h: __host__ __device__ under nvcc, plain inline otherwise.

#include <cuda_runtime.h>
#include <math.h>

#ifdef __CUDACC__
#  define COLOR_MATH_FN static __forceinline__ __host__ __device__
#else
#  define COLOR_MATH_FN static inline
#endif

// sRGB -> linear: simple gamma-2.2 approximation, matching the linear -> sRGB
// encode path in the display write-out.  Input is clamped to [0, 1].
COLOR_MATH_FN float srgbToLinear(float c)
{
    const float cc = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
    return powf(cc, 2.2f);
}

COLOR_MATH_FN float3 srgbToLinear3(float3 c)
{
    return make_float3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z));
}

// linear -> sRGB: exact inverse of srgbToLinear for values in [0, 1].
COLOR_MATH_FN float linearToSrgb(float c)
{
    const float cc = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
    return powf(cc, 1.0f / 2.2f);
}

#undef COLOR_MATH_FN

#endif // OPTIX_RAYTRACER_COLOR_MATH_H

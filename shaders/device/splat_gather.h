#ifndef OPTIX_RAYTRACER_SPLAT_GATHER_H
#define OPTIX_RAYTRACER_SPLAT_GATHER_H

#include "dev_common.h"
#include "path_payload.h"

// ─── Gaussian splat volume gather ────────────────────────────────────────────
// 3DGRT-style compositing: a dedicated trace against splat instances only
// (VIS_MASK_SPLAT) gathers per-gaussian responses along a ray segment into a
// k-buffer held in the payload.  __intersection__splat evaluates each
// candidate's maximum response along the ray and inserts (t, α, colour) sorted
// by depth — it never reports an intersection, so traversal always covers the
// full segment.  The raygen then composites front-to-back and, if the buffer
// overflowed, re-traces from the last kept depth until transmittance is spent.

#define SPLAT_KBUF 16

// Minimum contributing alpha — gaussians below this are culled from the BLAS
// (see Accel::buildSplatBlas adaptive bounds) and skipped by the IS.
#define SPLAT_MIN_ALPHA 0.01f

struct SplatGather
{
    int      count;               // valid entries in the arrays below
    int      dropped;             // 1 = at least one hit beyond t[count-1] was discarded
    float    t[SPLAT_KBUF];       // max-response depth, ascending
    float    a[SPLAT_KBUF];       // gaussian alpha at max response
    unsigned int inst[SPLAT_KBUF]; // TLAS instance index → optixLaunchParams.splatInstances
    unsigned int prim[SPLAT_KBUF]; // gaussian index within the dataset
};

// Evaluate the view-dependent splat colour: DC term plus optional SH bands
// 1..3, using the 3DGS basis convention.  `d` is the unit view direction in
// splat-local space (object space), pointing from the camera toward the point.
static __forceinline__ __device__
float3 evalSplatColor(const SplatSetData& sd, unsigned int prim, float3 d)
{
    const float4 fdc = sd.sh0[prim];
    const float  C0  = 0.28209479177387814f;
    float3 col = make_float3(0.5f + C0 * fdc.x,
                             0.5f + C0 * fdc.y,
                             0.5f + C0 * fdc.z);

    if (sd.shBands > 0 && sd.shN)
    {
        const int coeffs = (sd.shBands + 1) * (sd.shBands + 1) - 1;  // 3 / 8 / 15
        const float* sh  = sd.shN + static_cast<size_t>(prim) * coeffs * 3;

        const float x = d.x, y = d.y, z = d.z;

        // Band 1
        const float C1 = 0.4886025119029199f;
        col += make_float3(sh[0], sh[1], sh[2]) * (-C1 * y);
        col += make_float3(sh[3], sh[4], sh[5]) * ( C1 * z);
        col += make_float3(sh[6], sh[7], sh[8]) * (-C1 * x);

        if (sd.shBands > 1)
        {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;

            // Band 2
            col += make_float3(sh[ 9], sh[10], sh[11]) * (1.0925484305920792f  * xy);
            col += make_float3(sh[12], sh[13], sh[14]) * (-1.0925484305920792f * yz);
            col += make_float3(sh[15], sh[16], sh[17]) * (0.31539156525252005f * (2.0f * zz - xx - yy));
            col += make_float3(sh[18], sh[19], sh[20]) * (-1.0925484305920792f * xz);
            col += make_float3(sh[21], sh[22], sh[23]) * (0.5462742152960396f  * (xx - yy));

            if (sd.shBands > 2)
            {
                // Band 3
                col += make_float3(sh[24], sh[25], sh[26]) * (-0.5900435899266435f * y * (3.0f * xx - yy));
                col += make_float3(sh[27], sh[28], sh[29]) * ( 2.890611442640554f  * xy * z);
                col += make_float3(sh[30], sh[31], sh[32]) * (-0.4570457994644658f * y * (4.0f * zz - xx - yy));
                col += make_float3(sh[33], sh[34], sh[35]) * ( 0.3731763325901154f * z * (2.0f * zz - 3.0f * xx - 3.0f * yy));
                col += make_float3(sh[36], sh[37], sh[38]) * (-0.4570457994644658f * x * (4.0f * zz - xx - yy));
                col += make_float3(sh[39], sh[40], sh[41]) * ( 1.445305721320277f  * z * (xx - yy));
                col += make_float3(sh[42], sh[43], sh[44]) * (-0.5900435899266435f * x * (xx - 3.0f * yy));
            }
        }
    }

    col.x = fmaxf(col.x, 0.0f);
    col.y = fmaxf(col.y, 0.0f);
    col.z = fmaxf(col.z, 0.0f);

    // 3DGS colours are display-referred; linearize into the working space.
    return srgbToLinear3(col);
}

// Composite all gaussians along [tMin, tMax] front-to-back.  Returns the
// in-scattered radiance and the remaining transmittance for the segment.
static __forceinline__ __device__
void traceSplatSegment(float3 rayOrig, float3 rayDir, float tMin, float tMax,
                       float3& outRadiance, float& outTransmittance)
{
    float3 C = make_float3(0.0f, 0.0f, 0.0f);
    float  T = 1.0f;

    float tStart = tMin;
    for (int batch = 0; batch < 32; ++batch)
    {
        SplatGather g;
        g.count   = 0;
        g.dropped = 0;

        uint32_t p0, p1;
        packPointer(&g, p0, p1);

        optixTrace(
            optixLaunchParams.traversable,
            rayOrig, rayDir,
            tStart, tMax, 0.0f,
            OptixVisibilityMask(VIS_MASK_SPLAT),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0, 1, 0,   // miss 0 = __miss__radiance (empty — payload untouched)
            p0, p1);

        for (int i = 0; i < g.count; ++i)
        {
            // Colour is evaluated here — per composited entry — rather than in
            // the intersection shader, where it would run for every BVH
            // candidate including ones that never make it into the k-buffer.
            const SplatInstanceData& sid = optixLaunchParams.splatInstances[g.inst[i]];
            const float3 dObj = devNormalize(make_float3(
                sid.w2o[0] * rayDir.x + sid.w2o[1] * rayDir.y + sid.w2o[2] * rayDir.z,
                sid.w2o[3] * rayDir.x + sid.w2o[4] * rayDir.y + sid.w2o[5] * rayDir.z,
                sid.w2o[6] * rayDir.x + sid.w2o[7] * rayDir.y + sid.w2o[8] * rayDir.z));

            const float a = fminf(g.a[i], 0.99f);
            C += (evalSplatColor(sid.set, g.prim[i], dObj) * (T * a));
            T *= 1.0f - a;
            if (T < 0.003f)
            {
                break;
            }
        }

        // Continue batching while the buffer filled up (a full buffer reports
        // a hit that shrinks tmax, so farther candidates never reach the IS
        // and `dropped` alone can't detect them) or an overflow was recorded.
        const bool truncated = (g.count == SPLAT_KBUF) || g.dropped;
        if (!truncated || g.count == 0 || T < 0.003f)
        {
            break;  // segment fully gathered or effectively opaque
        }
        tStart = g.t[g.count - 1] * (1.0f + 1e-4f) + 1e-6f;
    }

    outRadiance      = C;
    outTransmittance = fmaxf(T, 0.0f);
}

#endif // OPTIX_RAYTRACER_SPLAT_GATHER_H

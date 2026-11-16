#ifndef OPTIX_RAYTRACER_HIT_IMPLICIT_H
#define OPTIX_RAYTRACER_HIT_IMPLICIT_H

#include "dev_common.h"
#include "path_payload.h"
#include "material.h"
#include "implicit_shapes.h"

// ─── Implicit: intersection program ──────────────────────────────────────────

extern "C" __global__ void __intersection__implicit()
{
    const ImplicitShapeData& shape =
        *reinterpret_cast<const ImplicitShapeData*>(optixGetSbtDataPointer());

    const float3 ro = optixGetObjectRayOrigin();
    const float3 rd = optixGetObjectRayDirection();

    float  t      = -1.0f;
    float3 normal = make_float3(0.0f, 1.0f, 0.0f);

    if (shape.type == IMPLICIT_SPHERE)
    {
        t = implicitSphereT(ro, rd);
        if (t >= 0.0f)
        {
            normal = devNormalize(ro + rd * t);
        }
    }
    else if (shape.type == IMPLICIT_BOX)
    {
        t = implicitBoxT(ro, rd, normal);
    }
    else  // IMPLICIT_CYLINDER
    {
        t = implicitCylinderT(ro, rd, normal);
    }

    if (t >= 0.0f)
    {
        // Pass the object-space normal via 3 attribute registers.
        // numAttributeValues must be >= 3 in the pipeline compile options.
        optixReportIntersection(t, 0,
            __float_as_uint(normal.x),
            __float_as_uint(normal.y),
            __float_as_uint(normal.z));
    }
}

// ─── Implicit: any-hit program ────────────────────────────────────────────────

extern "C" __global__ void __anyhit__implicit()
{
    const ImplicitShapeData& shape =
        *reinterpret_cast<const ImplicitShapeData*>(optixGetSbtDataPointer());
    if (   shape.materialIndex >= 0
        && optixLaunchParams.materials
        && optixLaunchParams.materials[shape.materialIndex].thinWalled)
    {
        const float3 alb = srgbToLinear3(
            optixLaunchParams.materials[shape.materialIndex].transmissionColor);
        optixSetPayload_1(__float_as_uint(__uint_as_float(optixGetPayload_1()) * alb.x));
        optixSetPayload_2(__float_as_uint(__uint_as_float(optixGetPayload_2()) * alb.y));
        optixSetPayload_3(__float_as_uint(__uint_as_float(optixGetPayload_3()) * alb.z));
        optixIgnoreIntersection();
    }
    else
    {
        optixTerminateRay();
    }
}

// ─── Implicit: closest-hit program ───────────────────────────────────────────

extern "C" __global__ void __closesthit__implicit()
{
    PathVertex* vtx = unpackVertex(optixGetPayload_0(), optixGetPayload_1());

    const ImplicitShapeData& shape =
        *reinterpret_cast<const ImplicitShapeData*>(optixGetSbtDataPointer());

    // Object-space normal written by __intersection__implicit via attributes.
    const float3 n_obj = make_float3(
        __uint_as_float(optixGetAttribute_0()),
        __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2()));

    vtx->N   = devNormalize(optixTransformNormalFromObjectToWorldSpace(n_obj));
    vtx->t   = optixGetRayTmax();
    vtx->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * vtx->t;

    if (optixLaunchParams.materials && shape.materialIndex >= 0)
    {
        applyMaterial(vtx, optixLaunchParams.materials[shape.materialIndex]);
    }
    else
    {
        applyDefaultMaterial(vtx);
    }

    vtx->hit        = 1;
    vtx->instanceId = optixGetInstanceId();  // used by emissive NEE MIS
}

#endif // OPTIX_RAYTRACER_HIT_IMPLICIT_H

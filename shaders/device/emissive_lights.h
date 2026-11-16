#ifndef OPTIX_RAYTRACER_EMISSIVE_LIGHTS_H
#define OPTIX_RAYTRACER_EMISSIVE_LIGHTS_H

#include "dev_common.h"

// ─── Emissive-light direct sampling (NEE) ────────────────────────────────────
//
// Each EmissiveLightData stores the local-to-world (l2w) and world-to-local (w2l)
// 3×4 transforms for one implicit shape that carries an emissive material.
// Sampling is done in the canonical local space (unit sphere/cube/cylinder), then
// the result is transformed to world space.  The solid-angle PDF is obtained by
// converting the local uniform area PDF through the surface Jacobian:
//
//   dA_world = |det(L)| · |W^T · n_local| · dA_local
//
// where L = l2w_3x3, W = w2l_3x3 = L^{-1}.  Both matrices are stored per light
// so the Jacobian factor |W^T · n_local| is cheap to evaluate on device.

// Apply the 3×4 l2w transform to a local-space point.
static __forceinline__ __device__
float3 lightL2WPoint(const EmissiveLightData& L, float3 p)
{
    return make_float3(
        L.l2w[0]*p.x + L.l2w[1]*p.y + L.l2w[ 2]*p.z + L.l2w[ 3],
        L.l2w[4]*p.x + L.l2w[5]*p.y + L.l2w[ 6]*p.z + L.l2w[ 7],
        L.l2w[8]*p.x + L.l2w[9]*p.y + L.l2w[10]*p.z + L.l2w[11]);
}

// Transform a local normal to world space: n_world = normalize((w2l_3x3)^T · n_local).
// This is the inverse-transpose normal transform, correct for non-uniform scale.
static __forceinline__ __device__
float3 lightL2WNormal(const EmissiveLightData& L, float3 n)
{
    return devNormalize(make_float3(
        L.w2l[0]*n.x + L.w2l[4]*n.y + L.w2l[ 8]*n.z,
        L.w2l[1]*n.x + L.w2l[5]*n.y + L.w2l[ 9]*n.z,
        L.w2l[2]*n.x + L.w2l[6]*n.y + L.w2l[10]*n.z));
}

// |( w2l_3x3 )^T · n_local| — Jacobian magnitude for the local→world area mapping.
static __forceinline__ __device__
float lightJacobianLen(const EmissiveLightData& L, float3 n)
{
    const float3 wn = make_float3(
        L.w2l[0]*n.x + L.w2l[4]*n.y + L.w2l[ 8]*n.z,
        L.w2l[1]*n.x + L.w2l[5]*n.y + L.w2l[ 9]*n.z,
        L.w2l[2]*n.x + L.w2l[6]*n.y + L.w2l[10]*n.z);
    return sqrtf(devDot(wn, wn));
}

// Sample uniformly on the unit sphere surface; returns the point (= outward normal).
static __forceinline__ __device__
float3 sampleUnitSphere(float r1, float r2)
{
    const float cosTheta = 1.0f - 2.0f * r1;
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi      = 6.28318530718f * r2;
    return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

// Sample uniformly on the unit cube (6 faces, area 4 each, total 24).
// r3 selects the face; r1/r2 pick the point within it.  Writes n_local.
static __forceinline__ __device__
float3 sampleUnitBox(float r1, float r2, float r3, float3& n_local)
{
    const int   face = min((int)(r3 * 6.0f), 5);
    const float sign = (face & 1) ? -1.0f : 1.0f;
    const float u    = r1 * 2.0f - 1.0f;
    const float v    = r2 * 2.0f - 1.0f;
    n_local = make_float3(0.0f, 0.0f, 0.0f);
    switch (face >> 1)
    {
        case 0:  n_local.x = sign; return make_float3(sign, u, v);
        case 1:  n_local.y = sign; return make_float3(u, sign, v);
        default: n_local.z = sign; return make_float3(u, v, sign);
    }
}

// Sample uniformly on the unit cylinder (lateral 4π, each cap π, total 6π).
// r3 selects the region; r1/r2 pick the point.  Writes n_local.
static __forceinline__ __device__
float3 sampleUnitCylinder(float r1, float r2, float r3, float3& n_local)
{
    const float kTwoPi = 6.28318530718f;
    if (r3 < 2.0f / 3.0f)
    {
        const float phi = r1 * kTwoPi;
        const float y   = r2 * 2.0f - 1.0f;
        const float cx  = cosf(phi), cz = sinf(phi);
        n_local = make_float3(cx, 0.0f, cz);
        return make_float3(cx, y, cz);
    }
    else
    {
        const float sign = (r3 < 5.0f / 6.0f) ? -1.0f : 1.0f;
        const float rad  = sqrtf(r1);
        const float phi  = r2 * kTwoPi;
        n_local = make_float3(0.0f, sign, 0.0f);
        return make_float3(rad * cosf(phi), sign, rad * sinf(phi));
    }
}

// Sample one randomly-chosen emissive implicit light.
// r0 selects the light; r1/r2/r3 sample the surface.
// Writes world-space sample position, world-space light normal, the selected
// light record pointer, and the solid-angle PDF (selection × area converted).
// Returns false when no lights exist or the sampled face is back-facing.
static __forceinline__ __device__
bool sampleEmissiveLight(
    float3 shadingPos,
    float r0, float r1, float r2, float r3,
    float3& outPos, float3& outN,
    const EmissiveLightData*& outLight, float& outPdf)
{
    const int N = optixLaunchParams.emissiveLightCount;
    if (N <= 0) { return false; }

    const int idx = min((int)(r0 * (float)N), N - 1);
    const EmissiveLightData& light = optixLaunchParams.emissiveLights[idx];

    float3 p_local, n_local;
    if (light.type == IMPLICIT_SPHERE)
    {
        n_local = sampleUnitSphere(r1, r2);
        p_local = n_local;
    }
    else if (light.type == IMPLICIT_BOX)
    {
        p_local = sampleUnitBox(r1, r2, r3, n_local);
    }
    else
    {
        p_local = sampleUnitCylinder(r1, r2, r3, n_local);
    }

    outPos = lightL2WPoint(light, p_local);
    outN   = lightL2WNormal(light, n_local);

    const float3 toLight = outPos - shadingPos;
    const float  dist2   = devDot(toLight, toLight);
    if (dist2 < 1e-8f) { return false; }
    const float dist = sqrtf(dist2);
    const float3 dir = toLight * (1.0f / dist);

    const float cosLight = devDot(outN, -dir);
    if (cosLight <= 1e-4f) { return false; }

    const float jacobLen = lightJacobianLen(light, n_local);
    if (jacobLen < 1e-8f) { return false; }

    // p_omega = (1/N) * (1/localArea) * (invDetW2l/jacobLen) * dist²/cosLight
    outPdf  = (1.0f / (float)N) * light.invDetW2l * dist2
            / (light.localArea * jacobLen * cosLight);
    outLight = &light;
    return true;
}

// Evaluate the solid-angle PDF that sampleEmissiveLight would assign to a
// path that arrived at hitPos (world normal hitN, distance hitT) from prevPos.
// Returns 0 when instanceId doesn't match any emissive light.
static __forceinline__ __device__
float evalEmissiveLightPdf(
    unsigned int instanceId,
    float3 prevPos, float3 hitPos, float3 hitN, float hitT)
{
    const int N = optixLaunchParams.emissiveLightCount;
    for (int i = 0; i < N; ++i)
    {
        const EmissiveLightData& light = optixLaunchParams.emissiveLights[i];
        if (light.instanceId != instanceId) { continue; }

        // n_local = normalize(l2w_3x3 * hitN): inverse of the normal transform
        // Derivation: hitN = normalize(W^T * n_local) → n_local ∝ L * hitN
        const float3 n_local = devNormalize(make_float3(
            light.l2w[0]*hitN.x + light.l2w[1]*hitN.y + light.l2w[ 2]*hitN.z,
            light.l2w[4]*hitN.x + light.l2w[5]*hitN.y + light.l2w[ 6]*hitN.z,
            light.l2w[8]*hitN.x + light.l2w[9]*hitN.y + light.l2w[10]*hitN.z));

        const float cosLight = devDot(hitN, -(hitPos - prevPos) * (1.0f / hitT));
        if (cosLight <= 1e-4f) { return 0.0f; }

        const float jacobLen = lightJacobianLen(light, n_local);
        if (jacobLen < 1e-8f) { return 0.0f; }

        return (1.0f / (float)N) * light.invDetW2l * (hitT * hitT)
             / (light.localArea * jacobLen * cosLight);
    }
    return 0.0f;
}

#endif // OPTIX_RAYTRACER_EMISSIVE_LIGHTS_H

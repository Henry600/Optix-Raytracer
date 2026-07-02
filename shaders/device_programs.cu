// device_programs.cu — OptiX Monte Carlo path tracer.
//
// Architecture:
//   • The raygen shader drives the entire bounce loop iteratively via a for-loop.
//     No closest-hit program ever calls optixTrace — trace depth is always 1.
//   • Hit data (position, normal, material) is returned to the raygen by packing a
//     pointer into the two uint32 payload values.  The closest-hit program writes
//     through that pointer; the miss program leaves it untouched (hit==0 default).
//   • Each launch adds one sample to a per-pixel float4 accumulation buffer.
//     The raygen reads the running average and writes a tone-mapped uchar4 to the
//     display buffer every frame.

#include <optix.h>
#include "device_math.h"   // float3 operators (+, -, *, /, unary -)
#include "launch_params.h"
#include "scene_data.h"

extern "C" { __constant__ LaunchParams optixLaunchParams; }

// ─── Tuning ───────────────────────────────────────────────────────────────────

static constexpr int MAX_BOUNCES = 16;

// ─── Scalar / float3 utilities ───────────────────────────────────────────────

static __forceinline__ __device__ float devClamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static __forceinline__ __device__ float devDot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static __forceinline__ __device__ float devLuminance(float3 c) { return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

// sRGB → linear: simple γ-2.2 approximation, matching the linear→sRGB encode path.
static __forceinline__ __device__
float srgbToLinear(float c)
{
    return powf(devClamp01(c), 2.2f);
}

static __forceinline__ __device__
float3 srgbToLinear3(float3 c)
{
    return make_float3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z));
}

static __forceinline__ __device__
float3 devNormalize(float3 v)
{
    return v * rsqrtf(devDot(v, v));
}

// Linear interpolation: a + (b-a)*t
static __forceinline__ __device__
float3 devMix(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}

// ─── Path payload ─────────────────────────────────────────────────────────────
// The raygen stack-allocates one of these and passes a packed pointer to each
// optixTrace call.  The closest-hit program fills in the fields and sets hit=1.

struct PathVertex
{
    float3 pos;          // world-space surface point
    float3 N;            // world-space shading normal (unit)
    float3 albedo;       // base colour
    float3 emission;     // pre-scaled emissive radiance (emission * emissionScale)
    float  roughness;    // perceptual roughness [0, 1]
    float  metallic;     // metallic factor [0, 1]
    float  transmission;         // 0 = opaque, 1 = fully transmissive
    float  ior;                  // index of refraction
    float  absorptionDistance;   // world-space units for full albedo absorption
    float3 scatteringCoeff;      // per-channel σ_s (1/MFP); (0,0,0) = disabled
    float  scatteringAnisotropy; // Henyey-Greenstein g [-1, +1]
    float  clearcoat;            // clearcoat layer intensity [0, 1]
    float  clearcoatRoughness;   // clearcoat layer roughness [0, 1]
    int    thinWalled;           // 1 = zero-thickness surface; pass through without refraction
    float        t;          // ray travel distance to this hit (Beer-Lambert)
    unsigned int instanceId; // TLAS instance index; ~0u when not set (miss / mesh)
    int          hit;        // 1 = geometry hit, 0 = ray escaped to background
};

static __forceinline__ __device__
void packPointer(void* ptr, uint32_t& p0, uint32_t& p1)
{
    const uint64_t u = reinterpret_cast<uint64_t>(ptr);
    p0 = static_cast<uint32_t>(u >> 32);
    p1 = static_cast<uint32_t>(u & 0xFFFFFFFFu);
}

static __forceinline__ __device__
PathVertex* unpackVertex(uint32_t p0, uint32_t p1)
{
    return reinterpret_cast<PathVertex*>(
        (static_cast<uint64_t>(p0) << 32) | static_cast<uint64_t>(p1));
}

// ─── PCG pseudo-random number generator ──────────────────────────────────────
// O'Neill 2014 — one 32-bit state, good statistical quality, cheap on GPU.

static __forceinline__ __device__
uint32_t pcgNext(uint32_t& state)
{
    uint32_t prev = state * 747796405u + 2891336453u;
    state = prev;
    uint32_t w = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (w >> 22u) ^ w;
}

static __forceinline__ __device__
float rnd(uint32_t& seed) { return (float)pcgNext(seed) * (1.0f / 4294967296.0f); }

// ─── Sampling helpers ─────────────────────────────────────────────────────────

// Cosine-weighted hemisphere direction in a Z-up local frame.
// BRDF = albedo/π, PDF = cosθ/π → BRDF/PDF = albedo (weight is just albedo).
static __forceinline__ __device__
float3 cosineSampleHemisphere(float r1, float r2)
{
    const float phi     = 2.0f * 3.14159265358979f * r1;
    const float sqrtR2  = sqrtf(r2);
    return make_float3(cosf(phi)*sqrtR2, sinf(phi)*sqrtR2, sqrtf(1.0f - r2));
}

// Henyey-Greenstein phase function.  Returns a direction in a local frame
// where the forward (incident) direction is +z.  g in [-1, 1].
static __forceinline__ __device__
float3 sampleHenyeyGreenstein(float g, float xi1, float xi2)
{
    float cosTheta;
    if (fabsf(g) < 1e-4f)
    {
        cosTheta = 1.0f - 2.0f * xi1;
    }
    else
    {
        const float denom = 1.0f - g + 2.0f * g * xi1;
        const float sq    = (1.0f - g * g) / denom;
        cosTheta = (1.0f + g * g - sq * sq) / (2.0f * g);
    }
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi      = 2.0f * 3.14159265358979f * xi2;
    return make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

// Orthonormal tangent frame around N (Duff et al. 2017).
static __forceinline__ __device__
void buildONB(float3 N, float3& T, float3& B)
{
    const float s = copysignf(1.0f, N.z);
    const float a = -1.0f / (s + N.z);
    const float b =  N.x * N.y * a;
    T = make_float3(1.0f + s*N.x*N.x*a, s*b, -s*N.x);
    B = make_float3(b, s + N.y*N.y*a, -N.y);
}

// ─── PBR helpers ─────────────────────────────────────────────────────────────

// Schlick Fresnel approximation (for metals: F0 = albedo).
static __forceinline__ __device__
float3 devFresnel(float cosTheta, float3 F0)
{
    const float t  = 1.0f - devClamp01(cosTheta);
    const float t5 = t * t * t * t * t;
    return F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * t5;
}

// Exact unpolarised Fresnel for a dielectric interface.
// eta = n_incident / n_transmitted (same convention as devRefract).
// Correctly returns 0.0 at all angles when eta = 1.0 (IOR = 1).
static __forceinline__ __device__
float devFresnelDielectric(float cosI, float eta)
{
    const float sinT2 = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    if (sinT2 >= 1.0f) { return 1.0f; }  // total internal reflection
    const float cosT = sqrtf(1.0f - sinT2);
    const float rs   = (eta * cosI - cosT) / (eta * cosI + cosT);
    const float rp   = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rs * rs + rp * rp);
}

// Sample GGX visible normals (Heitz 2018).
// Standard hemisphere sampling produces many below-horizon reflections on rough
// surfaces (causing energy loss).  VNDF conditions on the view direction so the
// sampled half-vector is always visible, making sub-horizon outcomes nearly
// impossible and simplifying the MC weight to just F·G1(L).
//
// V_local: view direction in tangent space (z = N component, must be > 0).
// Returns: half-vector in tangent space.
static __forceinline__ __device__
float3 devSampleGGX_VNDF(float3 V_local, float alpha, float u1, float u2)
{
    // Stretch view vector into the hemisphere configuration
    const float3 Vh = devNormalize(
        make_float3(alpha * V_local.x, alpha * V_local.y, V_local.z));

    // Orthonormal basis around Vh (T1 ⊥ Vh in the XY plane)
    const float  len2 = Vh.x*Vh.x + Vh.y*Vh.y;
    const float3 T1   = (len2 > 1e-7f)
        ? make_float3(-Vh.y, Vh.x, 0.0f) * rsqrtf(len2)
        : make_float3(1.0f, 0.0f, 0.0f);
    // T2 = cross(Vh, T1)
    const float3 T2 = make_float3(Vh.y*T1.z - Vh.z*T1.y,
                                   Vh.z*T1.x - Vh.x*T1.z,
                                   Vh.x*T1.y - Vh.y*T1.x);

    // Sample a point on the projected area disk
    const float r   = sqrtf(u1);
    const float phi = 2.0f * 3.14159265358979f * u2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    // Blend between uniform disk (s=0) and projected hemisphere (s=1)
    const float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;

    // Lift onto hemisphere, stretch back to GGX ellipsoid
    const float3 Nh = T1 * t1 + T2 * t2
                    + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    return devNormalize(
        make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(1e-6f, Nh.z)));
}

// Smith G1 shadowing-masking term for GGX.
static __forceinline__ __device__
float devSmithG1(float cosTheta, float alpha)
{
    const float a2 = alpha * alpha;
    const float c2 = cosTheta * cosTheta;
    return 2.0f * cosTheta / (cosTheta + sqrtf(a2 + (1.0f - a2) * c2));
}

// GGX normal distribution function: D(H) = α² / (π·(cosH²·(α²−1)+1)²)
static __forceinline__ __device__
float devGGX_D(float cosH, float alpha)
{
    const float a2 = alpha * alpha;
    const float d  = cosH * cosH * (a2 - 1.0f) + 1.0f;
    return a2 / (3.14159265358979f * d * d);
}

// PDF of a direction sampled by GGX VNDF:  p(ωo) = G1(cosV,α)·D(H) / (4·cosV)
static __forceinline__ __device__
float devGGX_PDF(float cosV, float cosH, float alpha)
{
    return devSmithG1(cosV, alpha) * devGGX_D(cosH, alpha)
           / (4.0f * fmaxf(cosV, 1e-5f));
}

// Smith height-correlated G2 for arbitrary light + view directions.
// G2 = 2·cosI·cosO / (cosO·√(α²+(1−α²)·cosI²) + cosI·√(α²+(1−α²)·cosO²))
static __forceinline__ __device__
float devGGX_G2(float cosI, float cosO, float alpha)
{
    const float a2 = alpha * alpha;
    const float lv = cosO * sqrtf(a2 + (1.0f - a2) * cosI * cosI);
    const float ll = cosI * sqrtf(a2 + (1.0f - a2) * cosO * cosO);
    return 2.0f * cosI * cosO / fmaxf(lv + ll, 1e-8f);
}

// GGX BRDF × cosI without Fresnel, for NEE evaluation.
// Returns D(H)·G2(cosI,cosO) / (4·cosO).  Caller multiplies by Fresnel(V·H, F0).
static __forceinline__ __device__
float devGGX_BRDFCosI(float cosI, float cosO, float cosH, float alpha)
{
    return devGGX_D(cosH, alpha) * devGGX_G2(cosI, cosO, alpha)
           / (4.0f * fmaxf(cosO, 1e-5f));
}

// Reflect incident direction v around normal n.
static __forceinline__ __device__
float3 devReflect(float3 v, float3 n)
{
    return v - n * (2.0f * devDot(v, n));
}

// Refract incident direction v through a surface with normal n (opposing v) and
// relative IOR eta = n1/n2.  Returns true on successful refraction and writes the
// transmitted direction to `out`.  Returns false on total internal reflection and
// writes the mirrored direction to `out` so the caller can continue the path.
static __forceinline__ __device__
bool devRefract(float3 v, float3 n, float eta, float3& out)
{
    const float cosI  = devDot(-v, n);                        // positive: n opposes v
    const float sinT2 = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    if (sinT2 >= 1.0f)                                         // total internal reflection
    {
        out = devReflect(v, n);
        return false;
    }
    const float cosT = sqrtf(1.0f - sinT2);
    out = devNormalize(v * eta + n * (eta * cosI - cosT));    // Snell's law, vector form
    return true;
}

// ─── Background / environment map ────────────────────────────────────────────

static __forceinline__ __device__
float3 sampleBackground(float3 dir)
{
    const float exposure = exp2f(optixLaunchParams.envExposure);  // EV stops → linear scale

    if (optixLaunchParams.envMap != 0)
    {
        const float kInvPi  = 0.31830988618f;
        const float kInv2Pi = 0.15915494309f;
        const float theta   = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
        const float phi     = atan2f(dir.x, -dir.z) + optixLaunchParams.envMapRotation;
        const float4 s = tex2D<float4>(optixLaunchParams.envMap,
                                       phi * kInv2Pi + 0.5f,
                                       theta * kInvPi);
        return make_float3(s.x, s.y, s.z) * exposure;
    }
    // Procedural sky:
    return devMix(make_float3(0.25f, 0.3f, 0.6f), make_float3(0.8f, 1.1f, 3.0f),
                  devClamp01(dir.y)) * exposure;
}

// ─── HDRI importance-sampling helpers ────────────────────────────────────────

// Binary search: first index in cdf[0..n-1] where cdf[idx] >= u.
static __forceinline__ __device__
int upperBound(const float* cdf, int n, float u)
{
    int lo = 0, hi = n - 1;
    while (lo < hi)
    {
        const int mid = (lo + hi) >> 1;
        if (cdf[mid] < u)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

// Sample the env map by importance.
// r1, r2: uniform [0,1).  Writes the sampled world-space direction to outDir
// and its solid-angle PDF to outPdf.  Returns the HDR radiance at that direction.
static __forceinline__ __device__
float3 sampleEnvMapIS(float r1, float r2, float3& outDir, float& outPdf)
{
    const int    W           = optixLaunchParams.envCdfW;
    const int    H           = optixLaunchParams.envCdfH;
    const float* marginal    = optixLaunchParams.envMarginalCdf;
    const float* conditional = optixLaunchParams.envConditionalCdf;

    // Sample row j, then column i
    const int j = upperBound(marginal,          H, r1);
    const int i = upperBound(conditional + j*W, W, r2);

    // Continuous UV at texel centre
    const float u = (i + 0.5f) / static_cast<float>(W);
    const float v = (j + 0.5f) / static_cast<float>(H);

    // UV → spherical: undo the envMapRotation sampleBackground would apply
    const float theta    = v * 3.14159265358979f;
    const float phiWorld = (u - 0.5f) * 6.28318530717959f - optixLaunchParams.envMapRotation;
    const float sinT     = sinf(theta);
    outDir = make_float3(sinf(phiWorld) * sinT, cosf(theta), -cosf(phiWorld) * sinT);

    // PDF in solid-angle measure
    const float mPrev = (j > 0) ? marginal[j - 1]             : 0.0f;
    const float cPrev = (i > 0) ? conditional[j * W + i - 1] : 0.0f;
    const float pUV   = (marginal[j] - mPrev) * static_cast<float>(H)
                      * (conditional[j * W + i] - cPrev) * static_cast<float>(W);
    outPdf = pUV / fmaxf(1e-5f, 2.0f * 3.14159265358979f * 3.14159265358979f * sinT);

    return sampleBackground(outDir);  // applies exposure + rotation + fallback
}

// Evaluate the solid-angle PDF for a world-space direction under env map IS.
// Returns 0 when no CDF is loaded.
static __forceinline__ __device__
float evalEnvMapPdf(float3 dir)
{
    if (!optixLaunchParams.envMarginalCdf)
    {
        return 0.0f;
    }

    const int    W           = optixLaunchParams.envCdfW;
    const int    H           = optixLaunchParams.envCdfH;
    const float* marginal    = optixLaunchParams.envMarginalCdf;
    const float* conditional = optixLaunchParams.envConditionalCdf;

    // Direction → (u, v) — mirror exactly what sampleBackground computes
    const float kInvPi  = 0.31830988618f;
    const float kInv2Pi = 0.15915494309f;
    const float theta   = acosf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
    const float phi     = atan2f(dir.x, -dir.z) + optixLaunchParams.envMapRotation;
    float u = phi * kInv2Pi + 0.5f;
    u = u - floorf(u);  // wrap to [0,1]
    const float v = theta * kInvPi;

    const int i = min(static_cast<int>(u * static_cast<float>(W)), W - 1);
    const int j = min(static_cast<int>(v * static_cast<float>(H)), H - 1);

    const float mPrev = (j > 0) ? marginal[j - 1]             : 0.0f;
    const float cPrev = (i > 0) ? conditional[j * W + i - 1] : 0.0f;
    const float pUV   = (marginal[j] - mPrev) * static_cast<float>(H)
                      * (conditional[j * W + i] - cPrev) * static_cast<float>(W);
    const float sinT  = sinf(theta);
    return (sinT > 1e-5f)
        ? pUV / (2.0f * 3.14159265358979f * 3.14159265358979f * sinT)
        : 0.0f;
}

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

// ─── Raygen — iterative path loop ────────────────────────────────────────────

extern "C" __global__ void __raygen__renderFrame()
{
    const uint3        idx   = optixGetLaunchIndex();
    const uint3        dim   = optixGetLaunchDimensions();
    const unsigned int fbIdx = idx.y * dim.x + idx.x;

    // ── No scene: show env map preview or UV placeholder ──────────────────────
    if (optixLaunchParams.traversable == 0)
    {
        if (optixLaunchParams.envMap != 0)
        {
            const float nx  = ((float)idx.x + 0.5f) / (float)dim.x * 2.0f - 1.0f;
            const float ny  = ((float)idx.y + 0.5f) / (float)dim.y * 2.0f - 1.0f;
            const float3 dir = devNormalize(
                optixLaunchParams.U * nx +
                optixLaunchParams.V * ny +
                optixLaunchParams.W);
            writeDisplay(fbIdx, sampleBackground(dir));
        }
        else
        {
            // UV debug gradient — linearise for the scRGB buffer; the UI
            // pipeline applies the paper-white scale when rendering the image.
            const float u = (float)idx.x / (float)dim.x;
            const float v = (float)idx.y / (float)dim.y;
            optixLaunchParams.colorBuffer[fbIdx] = make_float4(
                powf(u,    2.2f),
                powf(v,    2.2f),
                powf(0.5f, 2.2f),
                1.0f);
        }
        return;
    }

    // ── Path tracing ──────────────────────────────────────────────────────────

    // Seed: mix pixel index and sample index for uncorrelated sequences
    uint32_t seed = fbIdx * 1973u + optixLaunchParams.sampleIndex * 9277u + 127u;
    pcgNext(seed);  // warm up

    // Sub-pixel jitter for anti-aliasing
    const float nx = ((float)idx.x + rnd(seed)) / (float)dim.x * 2.0f - 1.0f;
    const float ny = ((float)idx.y + rnd(seed)) / (float)dim.y * 2.0f - 1.0f;

    float3 rayOrig = optixLaunchParams.eye;
    float3 rayDir  = devNormalize(
        optixLaunchParams.U * nx +
        optixLaunchParams.V * ny +
        optixLaunchParams.W);

    // ── Thin-lens depth of field ──────────────────────────────────────────────
    // When lensRadius > 0, offset the ray origin over the aperture disk and
    // redirect the ray to pass through the focal point on the focal plane.
    if (optixLaunchParams.lensRadius > 0.0f)
    {
        // Focal point: intersection of the pinhole ray with the focal plane
        // (plane perpendicular to W at distance focusDistance along W)
        const float  t       = optixLaunchParams.focusDistance
                              / devDot(rayDir, optixLaunchParams.W);
        const float3 focalPt = optixLaunchParams.eye + rayDir * t;

        // Sample the lens disk with optional edge bias.
        // alpha = 0.5 * (1 - bias): alpha=0.5 → sqrt (uniform area), alpha→0 → rim ring.
        // r = R * u^alpha concentrates samples toward the edge as bias → 1.
        const float alpha = 0.5f * (1.0f - devClamp01(optixLaunchParams.bokehEdgeBias));
        const float r     = powf(rnd(seed), fmaxf(alpha, 1e-4f)) * optixLaunchParams.lensRadius;
        const float phi = 2.0f * 3.14159265358979f * rnd(seed);

        // Lens plane axes = normalised camera right (U) and up (V)
        const float3 lensU = devNormalize(optixLaunchParams.U);
        const float3 lensV = devNormalize(optixLaunchParams.V);

        rayOrig = optixLaunchParams.eye
                + lensU * (r * cosf(phi))
                + lensV * (r * sinf(phi));
        rayDir  = devNormalize(focalPt - rayOrig);
    }

    float3 throughput    = make_float3(1.0f, 1.0f, 1.0f);
    float3 radiance      = make_float3(0.0f, 0.0f, 0.0f);
    // PDF of the last BSDF-sampled direction (diffuse only); 0 = specular/glass/first ray.
    // Carried into the next iteration so the miss block can MIS-weight env map samples.
    float  bsdfPdfForMis = 0.0f;

    // Beer-Lambert absorption coefficient: -log(albedo) per unit distance.
    // Kept in sync with medAbsorb[medTop]; updated whenever the stack changes.
    float3 absorb = make_float3(0.0f, 0.0f, 0.0f);

    // Nested-dielectric medium stack.
    // Index 0 = air (always present, never popped).  Push on entry, pop on exit.
    constexpr int kMaxMediumDepth = 8;
    float  medIor   [kMaxMediumDepth];
    float3 medAbsorb[kMaxMediumDepth];
    int    medTop = 0;
    medIor[0]    = 1.0f;
    medAbsorb[0] = make_float3(0.0f, 0.0f, 0.0f);

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        PathVertex vtx;
        vtx.hit        = 0;
        vtx.instanceId = ~0u;

        uint32_t p0, p1;
        packPointer(&vtx, p0, p1);

        optixTrace(
            optixLaunchParams.traversable,
            rayOrig, rayDir,
            1e-3f, 1e30f, 0.0f,
            OptixVisibilityMask(0xFF),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // anyhit only needed for shadow rays
            0, 1, 0,
            p0, p1);

        // ── Denoiser guide layers: write on first bounce only ────────────────
        if (bounce == 0)
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

        if (!vtx.hit)
        {
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
            break;
        }

        // ── Beer-Lambert absorption ───────────────────────────────────────────
        // Apply exp(-σ·t) for each channel; σ = -log(albedo) set when entering glass.
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

        // ── Lobe selection: clearcoat → specular → diffuse / refraction ─────────
        //
        // Clearcoat is sampled first: it's a dielectric layer (IOR 1.5, F0 = 0.04)
        // on top of the base material.  If not chosen, the base material receives
        // attenuated energy: throughput *= (1 − clearcoat · F_cc).
        //
        // Within the base material:
        //   1. specular   — p_spec (metallic / Fresnel)
        //   2. diffuse    — (1 − p_spec) · (1 − transmission)
        //   3. refraction — (1 − p_spec) ·      transmission

        const float alpha = fmaxf(vtx.roughness * vtx.roughness, 1e-3f);

        // Front-facing normal: always faces the incoming ray.
        const float3 Nf  = (devDot(vtx.N, -rayDir) >= 0.0f) ? vtx.N : -vtx.N;
        const float  cosV = devDot(Nf, -rayDir);  // shared by clearcoat + base Fresnel

        // ── Clearcoat Fresnel (fixed IOR 1.5 → F0 = 0.04) ────────────────────
        const float  cc_alpha = fmaxf(vtx.clearcoatRoughness * vtx.clearcoatRoughness, 1e-3f);
        const float3 cc_F     = devFresnel(cosV, make_float3(0.04f, 0.04f, 0.04f));
        const float  p_coat   = vtx.clearcoat * devLuminance(cc_F);

        if (p_coat > 0.0f && rnd(seed) < p_coat)
        {
            // ── Clearcoat specular (GGX-VNDF) ────────────────────────────────
            float3 Tc, Bc;
            buildONB(Nf, Tc, Bc);
            const float3 Vc       = -rayDir;
            const float3 Vc_local = make_float3(
                devDot(Vc, Tc), devDot(Vc, Bc), fmaxf(1e-4f, devDot(Vc, Nf)));
            const float3 Hc_local = devSampleGGX_VNDF(Vc_local, cc_alpha, rnd(seed), rnd(seed));
            const float3 Hc       = devNormalize(Tc * Hc_local.x + Bc * Hc_local.y + Nf * Hc_local.z);
            const float3 Lc       = devReflect(rayDir, Hc);

            if (devDot(Lc, Nf) <= 0.0f)
            {
                break;
            }

            // cc_F is achromatic (F0 = 0.04 is grey → Schlick stays grey), so
            // cc_F * clearcoat / p_coat = cc_F * clearcoat / (clearcoat * luminance(cc_F)) = 1.
            // The probability selection already carries the energy split; only G1 remains.
            const float cosNLc  = fmaxf(1e-4f, devDot(Nf, Lc));
            const float cosO_cc = Vc_local.z;   // dot(Vc, Nf), already clamped ≥ 1e-4
            throughput *= devSmithG1(cosNLc, cc_alpha);
            rayDir        = Lc;
            rayOrig       = vtx.pos + Nf * 1e-3f;
            bsdfPdfForMis = devGGX_PDF(cosO_cc, Hc_local.z, cc_alpha);
        }
        else
        {
            // ── Base material ─────────────────────────────────────────────────
            // (1 - cc_F * clearcoat) / (1 - p_coat) = 1 for achromatic cc_F,
            // so no throughput change is needed here either.

            // Base Fresnel — exact dielectric (handles IOR=1 correctly) blended with metal Schlick.
            // Schlick with F0=0 gives spurious (1−cosV)^5 at grazing for IOR=1; exact gives 0.
            const float  fDiel = devFresnelDielectric(cosV, medIor[medTop] / vtx.ior);
            const float3 F     = devMix(make_float3(fDiel, fDiel, fDiel),
                                        devFresnel(cosV, vtx.albedo), vtx.metallic);

            // Specular probability — metals always specular
            const float p_spec = devClamp01((1.0f - vtx.metallic) * devLuminance(F) + vtx.metallic);

            if (rnd(seed) < p_spec)
            {
                // ── 1. Specular reflection (GGX-VNDF) ─────────────────────────
                float3 T, B;
                buildONB(Nf, T, B);

                const float3 V       = -rayDir;
                const float3 V_local = make_float3(
                    devDot(V, T), devDot(V, B), fmaxf(1e-4f, devDot(V, Nf)));

                const float3 H_local = devSampleGGX_VNDF(V_local, alpha, rnd(seed), rnd(seed));
                const float3 H       = devNormalize(T * H_local.x + B * H_local.y + Nf * H_local.z);
                const float3 L       = devReflect(rayDir, H);

                if (devDot(L, Nf) <= 0.0f)
                {
                    break;
                }

                const float cosNL = fmaxf(1e-4f, devDot(Nf, L));
                const float cosO  = V_local.z;   // dot(V, Nf), already clamped ≥ 1e-4

                // Branch probability compensation — applied before NEE so both
                // the direct and indirect contributions are correctly scaled.
                throughput = throughput * (1.0f / fmaxf(1e-4f, p_spec));

                // ── NEE: specular direct env-map lighting ──────────────────────
                if (optixLaunchParams.envMarginalCdf)
                {
                    float3 neeDir; float neePdf;
                    float3 neeL = sampleEnvMapIS(rnd(seed), rnd(seed), neeDir, neePdf);

                    const float cosI_nee = devDot(neeDir, Nf);
                    if (cosI_nee > 0.0f && neePdf > 0.0f)
                    {
                        const float3 H_nee     = devNormalize(V + neeDir);
                        const float  cosH_nee  = fmaxf(0.0f, devDot(H_nee, Nf));
                        const float  cosVH_nee = fmaxf(0.0f, devDot(V, H_nee));
                        const float  fNEE_diel = devFresnelDielectric(cosVH_nee, medIor[medTop] / vtx.ior);
                        const float3 F_nee     = devMix(make_float3(fNEE_diel, fNEE_diel, fNEE_diel),
                                                        devFresnel(cosVH_nee, vtx.albedo), vtx.metallic);
                        const float  brdfCosI  = devGGX_BRDFCosI(cosI_nee, cosO, cosH_nee, alpha);

                        if (brdfCosI > 0.0f)
                        {
                            // Shadow ray: p0=vis, p1/p2/p3=RGB colour filter (init 1.0).
                            // No TERMINATE_ON_FIRST_HIT — anyhit handles termination
                            // explicitly so thin-walled filter accumulation is in ray order.
                            uint32_t shadowVis = 0u;
                            uint32_t shadowFR  = __float_as_uint(1.0f);
                            uint32_t shadowFG  = __float_as_uint(1.0f);
                            uint32_t shadowFB  = __float_as_uint(1.0f);
                            optixTrace(
                                optixLaunchParams.traversable,
                                vtx.pos + Nf * 1e-3f, neeDir,
                                1e-3f, 1e30f, 0.0f,
                                OptixVisibilityMask(0xFF),
                                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                0, 1, 1,
                                shadowVis, shadowFR, shadowFG, shadowFB);

                            if (shadowVis)
                            {
                                const float3 shadowFilter = make_float3(
                                    __uint_as_float(shadowFR),
                                    __uint_as_float(shadowFG),
                                    __uint_as_float(shadowFB));
                                // MIS power heuristic: pBsdf = (1-p_coat)*p_spec * GGX_PDF(neeDir)
                                const float pGGX_nee = devGGX_PDF(cosO, cosH_nee, alpha);
                                const float pBsdf    = (1.0f - p_coat) * p_spec * pGGX_nee;
                                const float wNee     = neePdf * neePdf
                                                     / fmaxf(neePdf * neePdf + pBsdf * pBsdf, 1e-12f);
                                radiance += throughput * shadowFilter
                                          * F_nee * brdfCosI / neePdf * neeL * wNee;
                            }
                        }
                    }
                }

                // ── NEE: specular direct emissive-light sampling ──────────────
                if (optixLaunchParams.emissiveLightCount > 0)
                {
                    float3 lightPos, lightN;
                    const EmissiveLightData* selectedLight = nullptr;
                    float lightPdf = 0.0f;
                    if (sampleEmissiveLight(vtx.pos,
                            rnd(seed), rnd(seed), rnd(seed), rnd(seed),
                            lightPos, lightN, selectedLight, lightPdf))
                    {
                        const float3 toLight  = lightPos - vtx.pos;
                        const float  dist2    = devDot(toLight, toLight);
                        const float  dist     = sqrtf(dist2);
                        const float3 neeDir   = toLight * (1.0f / dist);
                        const float  cosI_nee = devDot(neeDir, Nf);

                        if (cosI_nee > 0.0f)
                        {
                            const float3 H_nee     = devNormalize(V + neeDir);
                            const float  cosH_nee  = fmaxf(0.0f, devDot(H_nee, Nf));
                            const float  cosVH_nee = fmaxf(0.0f, devDot(V, H_nee));
                            const float  fNEE_diel = devFresnelDielectric(
                                cosVH_nee, medIor[medTop] / vtx.ior);
                            const float3 F_nee     = devMix(
                                make_float3(fNEE_diel, fNEE_diel, fNEE_diel),
                                devFresnel(cosVH_nee, vtx.albedo), vtx.metallic);
                            const float  brdfCosI  = devGGX_BRDFCosI(
                                cosI_nee, cosO, cosH_nee, alpha);

                            if (brdfCosI > 0.0f)
                            {
                                uint32_t shadowVis = 0u;
                                uint32_t shadowFR  = __float_as_uint(1.0f);
                                uint32_t shadowFG  = __float_as_uint(1.0f);
                                uint32_t shadowFB  = __float_as_uint(1.0f);
                                optixTrace(
                                    optixLaunchParams.traversable,
                                    vtx.pos + Nf * 1e-3f, neeDir,
                                    1e-3f, dist - 2e-3f, 0.0f,
                                    OptixVisibilityMask(0xFF),
                                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                    0, 1, 1,
                                    shadowVis, shadowFR, shadowFG, shadowFB);

                                if (shadowVis)
                                {
                                    const float3 shadowFilter = make_float3(
                                        __uint_as_float(shadowFR),
                                        __uint_as_float(shadowFG),
                                        __uint_as_float(shadowFB));
                                    const float pGGX_nee = devGGX_PDF(
                                        cosO, cosH_nee, alpha);
                                    const float pBsdf    = (1.0f - p_coat)
                                                         * p_spec * pGGX_nee;
                                    const float wNee     = lightPdf * lightPdf
                                        / fmaxf(lightPdf * lightPdf + pBsdf * pBsdf,
                                                1e-12f);
                                    radiance += throughput * shadowFilter
                                              * F_nee * brdfCosI / lightPdf
                                              * selectedLight->emission * wNee;
                                }
                            }
                        }
                    }
                }

                // ── Continue path with BSDF-sampled direction ──────────────────
                // p_spec already divided above; only the BRDF weight remains.
                throughput *= F * devSmithG1(cosNL, alpha);
                rayDir        = L;
                rayOrig       = vtx.pos + Nf * 1e-3f;
                // Store GGX PDF so miss block can MIS-weight env-map hits
                bsdfPdfForMis = devGGX_PDF(cosO, H_local.z, alpha);
            }
            else
            {
                // Non-specular weight — Fresnel transmission factor only.
                // albedo is applied inside the diffuse branch; the refraction branch
                // gets its color from Beer-Lambert absorption, not a surface multiply.
                const float3 kNS = (make_float3(1.0f, 1.0f, 1.0f) - F)
                                   * (1.0f / fmaxf(1e-4f, 1.0f - p_spec));
                throughput *= kNS;

                const float p_trans = devClamp01(vtx.transmission);

                if (rnd(seed) > p_trans)
                {
                    // ── 2. Diffuse (Lambertian, cosine-weighted) ───────────────
                    throughput *= vtx.albedo;

                    float3 T, B;
                    buildONB(Nf, T, B);

                    // ── NEE: direct env-map lighting ──────────────────────────
                    // Sample the env map IS distribution, fire a shadow ray to
                    // check visibility, and add a MIS-weighted direct contribution.
                    if (optixLaunchParams.envMarginalCdf)
                    {
                        float3 neeDir; float neePdf;
                        float3 neeL = sampleEnvMapIS(rnd(seed), rnd(seed), neeDir, neePdf);

                        const float cosL = devDot(neeDir, Nf);
                        if (cosL > 0.0f && neePdf > 0.0f)
                        {
                            // Shadow ray: SBT miss index 1 → __miss__shadow sets p0=1.
                            // p1/p2/p3 carry the accumulated thin-walled colour filter.
                            // No TERMINATE_ON_FIRST_HIT — anyhit calls optixTerminateRay()
                            // for opaque surfaces so thin-walled filter order is near-to-far.
                            uint32_t shadowVis = 0u;
                            uint32_t shadowFR  = __float_as_uint(1.0f);
                            uint32_t shadowFG  = __float_as_uint(1.0f);
                            uint32_t shadowFB  = __float_as_uint(1.0f);
                            optixTrace(
                                optixLaunchParams.traversable,
                                vtx.pos + Nf * 1e-3f, neeDir,
                                1e-3f, 1e30f, 0.0f,
                                OptixVisibilityMask(0xFF),
                                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                0, 1, 1,   // offset=0, stride=1, missIndex=1
                                shadowVis, shadowFR, shadowFG, shadowFB);

                            if (shadowVis)
                            {
                                const float3 shadowFilter = make_float3(
                                    __uint_as_float(shadowFR),
                                    __uint_as_float(shadowFG),
                                    __uint_as_float(shadowFB));
                                // throughput already carries kNS = (1-F)*albedo/(1-p_spec)
                                // Lambertian: f×cosθ = kNS * (cosθ/π) evaluated at neeDir
                                const float kInvPi = 0.31830988618f;
                                const float pBsdf  = (1.0f - p_coat) * (1.0f - p_spec)
                                                   * (1.0f - p_trans) * cosL * kInvPi;
                                const float wNee   = (neePdf * neePdf)
                                                   / (neePdf * neePdf + pBsdf * pBsdf);
                                radiance += throughput * shadowFilter
                                          * (cosL * kInvPi / neePdf) * neeL * wNee;
                            }
                        }
                    }

                    // ── NEE: diffuse direct emissive-light sampling ───────────
                    if (optixLaunchParams.emissiveLightCount > 0)
                    {
                        float3 lightPos, lightN;
                        const EmissiveLightData* selectedLight = nullptr;
                        float lightPdf = 0.0f;
                        if (sampleEmissiveLight(vtx.pos,
                                rnd(seed), rnd(seed), rnd(seed), rnd(seed),
                                lightPos, lightN, selectedLight, lightPdf))
                        {
                            const float3 toLight = lightPos - vtx.pos;
                            const float  dist2   = devDot(toLight, toLight);
                            const float  dist    = sqrtf(dist2);
                            const float3 neeDir  = toLight * (1.0f / dist);
                            const float  cosL    = devDot(neeDir, Nf);

                            if (cosL > 0.0f)
                            {
                                uint32_t shadowVis = 0u;
                                uint32_t shadowFR  = __float_as_uint(1.0f);
                                uint32_t shadowFG  = __float_as_uint(1.0f);
                                uint32_t shadowFB  = __float_as_uint(1.0f);
                                optixTrace(
                                    optixLaunchParams.traversable,
                                    vtx.pos + Nf * 1e-3f, neeDir,
                                    1e-3f, dist - 2e-3f, 0.0f,
                                    OptixVisibilityMask(0xFF),
                                    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                    0, 1, 1,
                                    shadowVis, shadowFR, shadowFG, shadowFB);

                                if (shadowVis)
                                {
                                    const float3 shadowFilter = make_float3(
                                        __uint_as_float(shadowFR),
                                        __uint_as_float(shadowFG),
                                        __uint_as_float(shadowFB));
                                    const float kInvPi = 0.31830988618f;
                                    const float pBsdf  = (1.0f - p_coat)
                                        * (1.0f - p_spec) * (1.0f - p_trans)
                                        * cosL * kInvPi;
                                    const float wNee   = (lightPdf * lightPdf)
                                        / (lightPdf * lightPdf + pBsdf * pBsdf);
                                    radiance += throughput * shadowFilter
                                              * (cosL * kInvPi / lightPdf)
                                              * selectedLight->emission * wNee;
                                }
                            }
                        }
                    }

                    // ── Continue: cosine-sample the hemisphere for the path ────
                    const float3 d = cosineSampleHemisphere(rnd(seed), rnd(seed));
                    rayDir  = devNormalize(T * d.x + B * d.y + Nf * d.z);
                    rayOrig = vtx.pos + Nf * 1e-3f;

                    // Store BSDF PDF for MIS in next-iteration miss block
                    const float kInvPi = 0.31830988618f;
                    bsdfPdfForMis = devDot(rayDir, Nf) * kInvPi;
                }
                else
                {
                    bsdfPdfForMis = 0.0f;  // refraction/pass-through — no diffuse NEE on next hit

                if (vtx.thinWalled)
                {
                    // Zero-thickness surface: front and back refractions cancel.
                    // Pass the ray through with no direction change and no Beer-Lambert
                    // absorption (there is no medium interior to accumulate through).
                    // Nf already faces the incoming ray (computed above), so -Nf always
                    // points to the far side regardless of which face was hit.
                    rayOrig = vtx.pos - Nf * 1e-3f;
                    // rayDir and absorb are intentionally left unchanged.
                }
                else
                {
                // ── 3. Refraction (rough Snell's law via GGX microfacet) ─────────
                // vtx.N is the true outward geometry normal — used only for the
                // entering/exiting test, not for the microfacet direction.
                const bool   entering = (devDot(rayDir, vtx.N) < 0.0f);
                const float3 faceN    = entering ? vtx.N : -vtx.N;
                const float  eta      = entering
                                      ? (medIor[medTop] / vtx.ior)
                                      : (vtx.ior / medIor[medTop > 0 ? medTop - 1 : 0]);

                // ── Volumetric scattering ─────────────────────────────────────────
                // Per-channel delta tracking: sample the collision event at σ_max
                // (the highest per-channel rate, typically blue for Rayleigh media).
                // When a scatter fires, weight throughput by σ_c/σ_max per channel.
                // Channels with lower σ carry reduced throughput along the scattered
                // direction, making the transmitted beam yellow when blue scatters more.
                // Beer-Lambert was already applied for vtx.t; undo the excess portion.
                const float3& sc       = vtx.scatteringCoeff;
                const float   sigmaMax = fmaxf(fmaxf(sc.x, sc.y), sc.z);
                bool scattered = false;
                if (!entering && sigmaMax > 0.0f)
                {
                    const float t_scatter = -logf(fmaxf(1e-6f, rnd(seed))) / sigmaMax;
                    if (t_scatter < vtx.t)
                    {
                        const float excess = vtx.t - t_scatter;
                        throughput.x *= expf(absorb.x * excess);
                        throughput.y *= expf(absorb.y * excess);
                        throughput.z *= expf(absorb.z * excess);

                        // Chromatic ratio weight — unbiased delta-tracking estimator.
                        throughput.x *= sc.x / sigmaMax;
                        throughput.y *= sc.y / sigmaMax;
                        throughput.z *= sc.z / sigmaMax;

                        const float3 scatterPos = rayOrig + rayDir * t_scatter;
                        float3 Ts, Bs;
                        buildONB(rayDir, Ts, Bs);
                        const float3 localDir = sampleHenyeyGreenstein(
                            vtx.scatteringAnisotropy, rnd(seed), rnd(seed));
                        rayDir  = devNormalize(Ts * localDir.x + Bs * localDir.y + rayDir * localDir.z);
                        rayOrig = scatterPos;
                        scattered = true;
                    }
                }

                if (!scattered)
                {
                // Sample a GGX microfacet normal (same VNDF as the specular lobe).
                // At alpha → 0 this converges to the macro normal → smooth glass.
                // At higher alpha the half-vector is scattered → frosted glass.
                float3 Tt, Bt;
                buildONB(Nf, Tt, Bt);
                const float3 Vt       = -rayDir;
                const float3 Vt_local = make_float3(
                    devDot(Vt, Tt), devDot(Vt, Bt), fmaxf(1e-4f, devDot(Vt, Nf)));
                const float3 Ht_local = devSampleGGX_VNDF(Vt_local, alpha, rnd(seed), rnd(seed));
                const float3 Ht       = devNormalize(Tt * Ht_local.x + Bt * Ht_local.y + Nf * Ht_local.z);

                // Refract through the microfacet normal; TIR falls back to reflection.
                const bool refracted = devRefract(rayDir, Ht, eta, rayDir);

                // Smith G1 on the transmitted direction — mirrors the specular weight.
                // Skip at eta≈1: the refracted direction equals the incident direction, so
                // G1(cosT,α) would incorrectly darken grazing angles on an IOR=1 surface.
                if (refracted && fabsf(eta - 1.0f) > 1e-4f)
                {
                    // cosine against the far-side normal (-faceN)
                    const float cosT = fmaxf(1e-4f, devDot(rayDir, -faceN));
                    throughput *= devSmithG1(cosT, alpha);
                }

                // Update medium stack and Beer-Lambert absorption.
                // Push on entry, pop on exit; absorb is always medAbsorb[medTop] after.
                // On TIR the stack is unchanged — ray stays in the same medium.
                if (refracted)
                {
                    if (entering)
                    {
                        if (medTop + 1 < kMaxMediumDepth)
                        {
                            ++medTop;
                            medIor[medTop] = vtx.ior;
                            // Total extinction σ_t = σ_absorption + σ_scattering per channel.
                            // Scattering removes photons from the direct beam just like absorption;
                            // including it here makes the transmitted beam yellow when blue
                            // scatters more (Rayleigh).  The scatter event code undoes the excess
                            // σ_t beyond the actual scatter point, keeping the estimator correct.
                            const float invDist = 1.0f / vtx.absorptionDistance;
                            medAbsorb[medTop].x = -logf(fmaxf(vtx.albedo.x, 1e-6f)) * invDist + vtx.scatteringCoeff.x;
                            medAbsorb[medTop].y = -logf(fmaxf(vtx.albedo.y, 1e-6f)) * invDist + vtx.scatteringCoeff.y;
                            medAbsorb[medTop].z = -logf(fmaxf(vtx.albedo.z, 1e-6f)) * invDist + vtx.scatteringCoeff.z;
                        }
                    }
                    else
                    {
                        if (medTop > 0) { --medTop; }
                    }
                    absorb = medAbsorb[medTop];
                }

                // On TIR devRefract writes the reflected direction; offset stays on
                // the same side.  On true refraction offset to the far side.
                rayOrig = vtx.pos + faceN * (refracted ? -1e-3f : 1e-3f);
                }   // end if (!scattered)
                }   // end thick refraction else (not thin-walled)
                }   // end refraction else
            }   // end non-specular else (diffuse / refraction)
        }   // end base material else (specular / diffuse / refraction) = clearcoat else

        // Russian roulette: stochastically terminate dim paths (all branches)
        if (bounce >= 3)
        {
            const float maxThr = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rnd(seed) > maxThr)
            {
                break;
            }
            throughput *= 1.0f / fmaxf(maxThr, 1e-6f);
        }
    }

    // ── Accumulate ────────────────────────────────────────────────────────────
    float4& acc = optixLaunchParams.accumBuffer[fbIdx];
    acc.x += radiance.x;
    acc.y += radiance.y;
    acc.z += radiance.z;

    // ── HDR average for denoiser ──────────────────────────────────────────────
    const float  inv = 1.0f / (float)(optixLaunchParams.sampleIndex + 1);
    if (optixLaunchParams.hdrBuffer)
    {
        optixLaunchParams.hdrBuffer[fbIdx] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 1.0f);
    }

    // ── Display encode ────────────────────────────────────────────────────────
    // Reinhard (hdrDisplay=0) or unclamped (hdrDisplay=1) → linear float4 scRGB.
    writeDisplay(fbIdx, make_float3(acc.x * inv, acc.y * inv, acc.z * inv));
}

// ─── Miss ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void __miss__radiance()
{
    // intentionally empty — hit==0 default in raygen is sufficient
}

// ─── Miss — NEE shadow ray ────────────────────────────────────────────────────
// Fires when a shadow ray reaches the background unoccluded.
// Sets p0 = 1 (visible); the raygen reads this to confirm the light path is clear.

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(1u);
}

// ─── Any-hit — shadow ray thin-walled colour filter ──────────────────────────
// Only reached by NEE shadow rays; radiance rays set OPTIX_RAY_FLAG_DISABLE_ANYHIT
// so this program is never invoked for primary or bounce rays.
//
// Shadow rays carry 4 payload registers:
//   p0        : vis  — 0 while in flight; set to 1 by __miss__shadow on escape
//   p1/p2/p3  : accumulated colour filter (float bits, caller-initialised to 1.0)
//
// Thin-walled surfaces multiply the running filter by their linearised albedo and
// call optixIgnoreIntersection() so BVH traversal continues toward the light.
// Non-thin-walled surfaces call optixTerminateRay() — this is the correct way to
// signal occlusion when TERMINATE_ON_FIRST_HIT is not set, and it guarantees
// that the first opaque hit (nearest-to-farthest in BVH order) wins.

extern "C" __global__ void __anyhit__radiance()
{
    const MeshData& mesh =
        *reinterpret_cast<const MeshData*>(optixGetSbtDataPointer());
    if (   mesh.materialIndex >= 0
        && optixLaunchParams.materials
        && optixLaunchParams.materials[mesh.materialIndex].thinWalled)
    {
        // Accumulate this surface's linearised albedo into the colour filter.
        const float3 alb = srgbToLinear3(optixLaunchParams.materials[mesh.materialIndex].albedo);
        optixSetPayload_1(__float_as_uint(__uint_as_float(optixGetPayload_1()) * alb.x));
        optixSetPayload_2(__float_as_uint(__uint_as_float(optixGetPayload_2()) * alb.y));
        optixSetPayload_3(__float_as_uint(__uint_as_float(optixGetPayload_3()) * alb.z));
        optixIgnoreIntersection();          // keep traversal going past the glass
    }
    else
    {
        optixTerminateRay();                // opaque surface — shadow ray is blocked
    }
}

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
        // Albedo — base colour × texture tint (glTF: baseColorFactor × baseColorTexture).
        // Both the colour factor and the texture sample are sRGB-encoded, so each is
        // linearised independently before the multiplication.
        vtx->albedo = srgbToLinear3(mat.albedo);
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
        vtx->roughness = mat.roughness;
        if (sampleSceneTex(mat.roughnessTexture, mesh.uvs, tri, w0, bc, mat.uvTransform, texSample))
        {
            vtx->roughness *= texSample.x;
        }

        vtx->metallic = mat.metallic;

        // Emission — colour factor × texture (both sRGB-encoded, linearised before
        // multiply, matching the albedo path).
        {
            float3 emLinear = srgbToLinear3(mat.emission);
            float4 emSample;
            if (sampleSceneTex(mat.emissionTexture, mesh.uvs, tri, w0, bc,
                               mat.uvTransform, emSample))
            {
                const float3 emTexLinear = srgbToLinear3(
                    make_float3(emSample.x, emSample.y, emSample.z));
                emLinear = make_float3(emLinear.x * emTexLinear.x,
                                      emLinear.y * emTexLinear.y,
                                      emLinear.z * emTexLinear.z);
            }
            vtx->emission = emLinear * mat.emissionScale;
        }
        vtx->transmission       = mat.transmission;
        vtx->ior                = mat.ior;
        vtx->absorptionDistance   = fmaxf(mat.absorptionDistance, 1e-4f);
        vtx->scatteringCoeff      = make_float3(fmaxf(mat.scatteringCoeff.x, 0.0f),
                                               fmaxf(mat.scatteringCoeff.y, 0.0f),
                                               fmaxf(mat.scatteringCoeff.z, 0.0f));
        vtx->scatteringAnisotropy = mat.scatteringAnisotropy;
        vtx->clearcoat            = mat.clearcoat;
        vtx->clearcoatRoughness = mat.clearcoatRoughness;
        vtx->thinWalled         = mat.thinWalled;
    }
    else
    {
        vtx->albedo             = make_float3(0.8f, 0.8f, 0.8f);
        vtx->roughness          = 0.5f;
        vtx->metallic           = 0.0f;
        vtx->emission           = make_float3(0.0f, 0.0f, 0.0f);
        vtx->transmission       = 0.0f;
        vtx->ior                = 1.5f;
        vtx->absorptionDistance = 1.0f;
        vtx->clearcoat          = 0.0f;
        vtx->clearcoatRoughness = 0.0f;
        vtx->thinWalled         = 0;
    }

    vtx->hit        = 1;
    vtx->instanceId = ~0u;  // mesh hits are never in the emissive-lights array
}

// ─── Implicit shape intersection helpers ─────────────────────────────────────
// All shapes are canonical unit forms in object space ([-1,1]^3 AABB).
// OptiX transforms the world-space ray into object space automatically before
// calling the intersection program.

// Unit sphere x²+y²+z²=1.  Returns smallest valid t, or -1 on miss.
static __forceinline__ __device__
float implicitSphereT(const float3& ro, const float3& rd)
{
    const float a    = devDot(rd, rd);
    const float b    = 2.0f * devDot(ro, rd);
    const float c    = devDot(ro, ro) - 1.0f;
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) { return -1.0f; }
    const float sq   = sqrtf(disc);
    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();
    const float t1   = (-b - sq) / (2.0f * a);
    if (t1 >= tMin && t1 <= tMax) { return t1; }
    const float t2   = (-b + sq) / (2.0f * a);
    if (t2 >= tMin && t2 <= tMax) { return t2; }
    return -1.0f;
}

// Unit cube [-1,1]^3.  Returns smallest valid t and writes outward face normal.
// Returns -1 on miss.
static __forceinline__ __device__
float implicitBoxT(const float3& ro, const float3& rd, float3& outN)
{
    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();

    const float invDx = 1.0f / rd.x;
    const float invDy = 1.0f / rd.y;
    const float invDz = 1.0f / rd.z;

    const float tx1 = (-1.0f - ro.x) * invDx;
    const float tx2 = ( 1.0f - ro.x) * invDx;
    const float ty1 = (-1.0f - ro.y) * invDy;
    const float ty2 = ( 1.0f - ro.y) * invDy;
    const float tz1 = (-1.0f - ro.z) * invDz;
    const float tz2 = ( 1.0f - ro.z) * invDz;

    const float txN = fminf(tx1, tx2);  const float txF = fmaxf(tx1, tx2);
    const float tyN = fminf(ty1, ty2);  const float tyF = fmaxf(ty1, ty2);
    const float tzN = fminf(tz1, tz2);  const float tzF = fmaxf(tz1, tz2);

    const float tNear = fmaxf(fmaxf(txN, tyN), tzN);
    const float tFar  = fminf(fminf(txF, tyF), tzF);

    if (tNear > tFar) { return -1.0f; }

    float t = tNear;
    if (t < tMin) { t = tFar; }
    if (t < tMin || t > tMax) { return -1.0f; }

    const float3 p  = ro + rd * t;
    const float  ax = fabsf(p.x);
    const float  ay = fabsf(p.y);
    const float  az = fabsf(p.z);
    if (ax >= ay && ax >= az)
    {
        outN = make_float3(p.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
    }
    else if (ay >= ax && ay >= az)
    {
        outN = make_float3(0.0f, p.y > 0.0f ? 1.0f : -1.0f, 0.0f);
    }
    else
    {
        outN = make_float3(0.0f, 0.0f, p.z > 0.0f ? 1.0f : -1.0f);
    }
    return t;
}

// Y-axis cylinder, radius=1, y∈[-1,1].  Barrel + two caps.
// Returns smallest valid t and writes outward normal.  Returns -1 on miss.
static __forceinline__ __device__
float implicitCylinderT(const float3& ro, const float3& rd, float3& outN)
{
    const float tMin  = optixGetRayTmin();
    const float tMax  = optixGetRayTmax();
    float bestT       = -1.0f;
    float3 bestN      = make_float3(0.0f, 1.0f, 0.0f);

    // ── Barrel: x²+z² = 1 ────────────────────────────────────────────────────
    const float a = rd.x * rd.x + rd.z * rd.z;
    if (fabsf(a) > 1e-8f)
    {
        const float b    = 2.0f * (ro.x * rd.x + ro.z * rd.z);
        const float c    = ro.x * ro.x + ro.z * ro.z - 1.0f;
        const float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f)
        {
            const float sq = sqrtf(disc);
            const float t1 = (-b - sq) / (2.0f * a);
            const float t2 = (-b + sq) / (2.0f * a);
            for (int k = 0; k < 2; ++k)
            {
                const float t = (k == 0) ? t1 : t2;
                if (t < tMin || t > tMax) { continue; }
                const float y = ro.y + t * rd.y;
                if (y < -1.0f || y > 1.0f) { continue; }
                if (bestT < 0.0f || t < bestT)
                {
                    bestT = t;
                    const float3 p = ro + rd * t;
                    bestN = make_float3(p.x, 0.0f, p.z);
                }
            }
        }
    }

    // ── Caps: y = ±1 ─────────────────────────────────────────────────────────
    if (fabsf(rd.y) > 1e-8f)
    {
        for (int k = 0; k < 2; ++k)
        {
            const float capY = (k == 0) ? -1.0f : 1.0f;
            const float t    = (capY - ro.y) / rd.y;
            if (t < tMin || t > tMax) { continue; }
            const float3 p = ro + rd * t;
            if (p.x * p.x + p.z * p.z > 1.0f) { continue; }
            if (bestT < 0.0f || t < bestT)
            {
                bestT = t;
                bestN = make_float3(0.0f, capY, 0.0f);
            }
        }
    }

    if (bestT >= 0.0f) { outN = devNormalize(bestN); }
    return bestT;
}

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
            optixLaunchParams.materials[shape.materialIndex].albedo);
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
        const MaterialData& mat = optixLaunchParams.materials[shape.materialIndex];

        vtx->albedo    = srgbToLinear3(mat.albedo);
        vtx->roughness = mat.roughness;
        vtx->metallic  = mat.metallic;

        vtx->emission = srgbToLinear3(mat.emission) * mat.emissionScale;

        vtx->transmission         = mat.transmission;
        vtx->ior                  = mat.ior;
        vtx->absorptionDistance   = fmaxf(mat.absorptionDistance, 1e-4f);
        vtx->scatteringCoeff      = make_float3(fmaxf(mat.scatteringCoeff.x, 0.0f),
                                                fmaxf(mat.scatteringCoeff.y, 0.0f),
                                                fmaxf(mat.scatteringCoeff.z, 0.0f));
        vtx->scatteringAnisotropy = mat.scatteringAnisotropy;
        vtx->clearcoat            = mat.clearcoat;
        vtx->clearcoatRoughness   = mat.clearcoatRoughness;
        vtx->thinWalled           = mat.thinWalled;
    }
    else
    {
        vtx->albedo             = make_float3(0.8f, 0.8f, 0.8f);
        vtx->roughness          = 0.5f;
        vtx->metallic           = 0.0f;
        vtx->emission           = make_float3(0.0f, 0.0f, 0.0f);
        vtx->transmission       = 0.0f;
        vtx->ior                = 1.5f;
        vtx->absorptionDistance = 1.0f;
        vtx->clearcoat          = 0.0f;
        vtx->clearcoatRoughness = 0.0f;
        vtx->thinWalled         = 0;
    }

    vtx->hit        = 1;
    vtx->instanceId = optixGetInstanceId();  // used by emissive NEE MIS
}

// ─── 1px picking ─────────────────────────────────────────────────────────────
// Fires a single ray from normalised screen-space (pickU, pickV) and writes
// the TLAS instance index of the closest hit into pickResult.  0xFFFFFFFF = miss.

extern "C" __global__ void __raygen__pick()
{
    const float ndcX  = 2.0f * optixLaunchParams.pickU - 1.0f;
    const float ndcY  = 1.0f - 2.0f * optixLaunchParams.pickV;
    const float3 dir  = devNormalize(
        optixLaunchParams.W
        + ndcX * optixLaunchParams.U
        + ndcY * optixLaunchParams.V);

    uint32_t instanceIdx = 0xFFFFFFFFu;
    uint32_t hitT_bits   = __float_as_uint(-1.0f);  // -1 = miss sentinel
    optixTrace(
        optixLaunchParams.traversable,
        optixLaunchParams.eye, dir,
        1e-3f, 1e16f, 0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        instanceIdx, hitT_bits);

    *optixLaunchParams.pickResult = instanceIdx;
    if (optixLaunchParams.pickDistance)
    {
        *optixLaunchParams.pickDistance = __uint_as_float(hitT_bits);
    }
}

extern "C" __global__ void __closesthit__pick()
{
    optixSetPayload_0(optixGetInstanceIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __miss__pick()
{
    optixSetPayload_0(0xFFFFFFFFu);
    optixSetPayload_1(__float_as_uint(-1.0f));
}

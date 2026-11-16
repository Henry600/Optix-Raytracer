#ifndef OPTIX_RAYTRACER_BASE_LOBE_H
#define OPTIX_RAYTRACER_BASE_LOBE_H

#include "dev_common.h"
#include "path_payload.h"
#include "path_state.h"
#include "surface_context.h"
#include "rng_sampling.h"
#include "pbr.h"
#include "coat_lobe.h"
#include "specular_lobe.h"
#include "diffuse_lobe.h"
#include "transmission_lobe.h"

// Base material under the (optional) coat: specular, diffuse or transmission.
// Returns false when the path terminates.
static __forceinline__ __device__
bool sampleBase(const PathVertex& vtx, const SurfaceCtx& sc, PathState& ps, MediumStack& ms)
{
    applyCoatDarkening(vtx, sc, ps, ms);

    // Base Fresnel - exact dielectric (handles IOR=1 correctly) blended with metal Schlick.
    // Schlick with F0=0 gives spurious (1-cosV)^5 at grazing for IOR=1; exact gives 0.
    const float  fDiel = devFresnelDielectric(sc.cosV, ms.medIor[ms.medTop] / vtx.ior);
    const float3 F     = devMix(make_float3(fDiel, fDiel, fDiel),
                                devFresnel(sc.cosV, vtx.albedo), vtx.metallic);

    // Specular probability - metals always specular
    const float p_spec = devClamp01((1.0f - vtx.metallic) * devLuminance(F) + vtx.metallic);

    if (rnd(ps.seed) < p_spec)
    {
        return sampleSpecular(vtx, sc, ps, ms, F, p_spec);
    }

    // Non-specular weight - Fresnel transmission factor only.
    // albedo is applied inside the diffuse branch; the refraction branch
    // gets its color from Beer-Lambert absorption, not a surface multiply.
    const float3 kNS = (make_float3(1.0f, 1.0f, 1.0f) - F)
                       * (1.0f / fmaxf(1e-4f, 1.0f - p_spec));
    ps.throughput *= kNS;

    const float p_trans = devClamp01(vtx.transmission);

    if (rnd(ps.seed) > p_trans)
    {
        sampleDiffuse(vtx, sc, ps, p_spec, p_trans);
    }
    else
    {
        sampleTransmission(vtx, sc, ps, ms);
    }
    return true;
}

#endif // OPTIX_RAYTRACER_BASE_LOBE_H

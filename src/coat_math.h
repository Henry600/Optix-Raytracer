#ifndef OPTIX_RAYTRACER_COAT_MATH_H
#define OPTIX_RAYTRACER_COAT_MATH_H

#include <cmath>

// Host-side helpers for the OpenPBR clear-coat model.

// Unpolarised dielectric Fresnel reflectance for light arriving from outside
// (air) at cosine `mu` onto an interface with relative IOR `eta` (> 1).
inline float coatFresnel(float mu, float eta)
{
    const float s2 = (1.0f - mu * mu) / (eta * eta);
    const float ct = std::sqrt(std::fmax(0.0f, 1.0f - s2));
    const float rs = (mu - eta * ct) / (mu + eta * ct);
    const float rp = (eta * mu - ct) / (eta * mu + ct);
    return 0.5f * (rs * rs + rp * rp);
}

// Internal diffuse reflectance K_r = 1 - (1 - E_F(eta)) / eta^2, where
// E_F = 2 * integral_0^1 F(mu, eta) * mu dmu is the hemispherical average
// Fresnel reflectance (OpenPBR coat darkening, arXiv 2512.23696 Eq. 77).
// Midpoint-rule integration; computed on the host whenever the coat IOR
// changes so the shader needs no per-vertex integral.
inline float coatInternalReflectance(float eta)
{
    eta = std::fmax(eta, 1.0f);
    constexpr int kSteps = 256;
    float sum = 0.0f;
    for (int i = 0; i < kSteps; ++i)
    {
        const float mu = (static_cast<float>(i) + 0.5f) / static_cast<float>(kSteps);
        sum += coatFresnel(mu, eta) * mu;
    }
    const float eF = 2.0f * sum / static_cast<float>(kSteps);
    return 1.0f - (1.0f - eF) / (eta * eta);
}

#endif // OPTIX_RAYTRACER_COAT_MATH_H

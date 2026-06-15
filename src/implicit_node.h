#ifndef OPTIX_RAYTRACER_IMPLICIT_NODE_H
#define OPTIX_RAYTRACER_IMPLICIT_NODE_H

#include "node_3d.h"
#include <cstdint>

// Analytic implicit shapes rendered via OptiX custom primitives.
// All shapes are canonical unit forms in local (object) space; the node's
// localTransform positions, orients, and scales them in world space.
//
//   Sphere   — unit sphere  x²+y²+z² = 1       (AABB [-1,1]³)
//   Box      — unit cube    each axis ∈ [-1,1]  (AABB [-1,1]³)
//   Cylinder — Y-axis, radius=1, y ∈ [-1,1]    (AABB [-1,1]³)
//
// Values must stay in sync with ImplicitTypeGPU in shaders/scene_data.h.
enum class ImplicitType : uint32_t
{
    Sphere   = 0,
    Box      = 1,
    Cylinder = 2,
};

class ImplicitNode : public Node3D
{
public:
    ImplicitType type          = ImplicitType::Sphere;
    int          materialIndex = 0;

    const char* typeName() const override { return "Implicit"; }
};

#endif // OPTIX_RAYTRACER_IMPLICIT_NODE_H

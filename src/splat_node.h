#ifndef OPTIX_RAYTRACER_SPLAT_NODE_H
#define OPTIX_RAYTRACER_SPLAT_NODE_H

#include "node_3d.h"

// A scene-graph node referencing a gaussian splat dataset owned by Scene.
// The node's transform positions the whole splat cloud in world space; the
// dataset itself is shared, so duplicating the node does not copy the data.
// Rendering is not implemented yet — the node currently only participates in
// the scene graph (selection, transforms, visibility).
class SplatNode : public Node3D
{
public:
    int splatIndex = -1;  // index into Scene::splats()

    const char* typeName() const override { return "Splat"; }
};

#endif // OPTIX_RAYTRACER_SPLAT_NODE_H

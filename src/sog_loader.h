#ifndef OPTIX_RAYTRACER_SOG_LOADER_H
#define OPTIX_RAYTRACER_SOG_LOADER_H

#include "gaussian_splat.h"

#include <string>

// Loads a SOG v2 (Spatially Ordered Gaussians) dataset.
//
// `path` may be either:
//   - a bundled archive (scene.sog) — a ZIP containing meta.json and the
//     WebP property images at the archive root, or
//   - a meta.json path — the multi-file layout, with the WebP images
//     resolved relative to the meta.json directory.
//
// On success fills `out` and returns true.  On failure returns false and
// writes a human-readable reason into `error`.
bool loadSog(const std::string& path, GaussianSplatData& out, std::string& error);

#endif // OPTIX_RAYTRACER_SOG_LOADER_H

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

// The source is split into headers under device/, included in dependency order into
// this single translation unit (one PTX module).  Entry-point programs live in
// the *_programs.h / hit_*.h / raygen_render.h headers.

#include "device/dev_common.h"
#include "device/path_payload.h"
#include "device/rng_sampling.h"
#include "device/pbr.h"
#include "device/environment.h"
#include "device/emissive_lights.h"
#include "device/display.h"
#include "device/eon_diffuse.h"
#include "device/splat_gather.h"
#include "device/raygen_render.h"
#include "device/miss_shadow_programs.h"
#include "device/material.h"
#include "device/hit_mesh.h"
#include "device/implicit_shapes.h"
#include "device/hit_implicit.h"
#include "device/splat_programs.h"
#include "device/pick_programs.h"

#ifndef OPTIX_RAYTRACER_ACCEL_H
#define OPTIX_RAYTRACER_ACCEL_H

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "gpu_buffer.h"

#include <vector>

// Forward declaration — Accel.cpp includes Scene.h
class Scene;

// Manages the OptiX acceleration structure for a loaded scene:
//   - one BLAS (bottom-level AS) per mesh, built with compaction
//   - one shared custom-primitive BLAS for all implicit shape nodes
//   - one TLAS (top-level AS) instancing all BLASes with node world-space transforms
//
// The traversable handle exposed here is stored in LaunchParams and passed to
// optixLaunch so device programs can call optixTrace() against the geometry.
class Accel
{
public:
    Accel()  = default;
    ~Accel() { destroy(); }

    Accel(const Accel&)            = delete;
    Accel& operator=(const Accel&) = delete;
    Accel(Accel&&)                 = delete;
    Accel& operator=(Accel&&)      = delete;

    // Build or rebuild from scene geometry. Destroys any previous state first.
    // Throws std::runtime_error on CUDA / OptiX failure.
    void build(OptixDeviceContext ctx, const Scene& scene);

    // Rebuild only the TLAS (instance transforms + IAS) from the current scene
    // node hierarchy. BLASes and device geometry buffers are reused unchanged.
    // Much cheaper than full build() — use after live node transform edits.
    void rebuildTlas(OptixDeviceContext ctx, const Scene& scene);

    // Free all device memory and reset to empty state. Safe to call multiple times.
    void destroy();

    OptixTraversableHandle traversable()  const { return m_tlas; }
    bool                   valid()        const { return m_tlas != 0; }
    size_t                 meshCount()    const { return m_meshBuffers.size(); }

    // Shared BLAS used by all implicit shape TLAS instances.
    OptixTraversableHandle implicitBlas() const { return m_implicitBlas; }

    // Per-mesh device pointers needed to fill SBT hit group records.
    struct MeshDevicePtrs
    {
        CUdeviceptr positions;  // device float3 array
        CUdeviceptr normals;    // device float3 array
        CUdeviceptr uvs;        // device float2 array (0 when mesh has no UVs)
        CUdeviceptr indices;    // device uint3  array
    };
    MeshDevicePtrs meshDevicePtrs(size_t idx) const
    {
        return { m_meshBuffers[idx].positions.ptr(),
                 m_meshBuffers[idx].normals.ptr(),
                 m_meshBuffers[idx].uvs.ptr(),
                 m_meshBuffers[idx].indices.ptr() };
    }

private:
    // Per-mesh device buffers kept alive for the lifetime of the AS.
    struct MeshBuffers
    {
        GPUBuffer              positions;  // device copy of Mesh::positions
        GPUBuffer              normals;    // device copy of Mesh::normals
        GPUBuffer              uvs;        // device copy of Mesh::uvs (absent = not valid())
        GPUBuffer              indices;    // device copy of Mesh::indices
        GPUBuffer              outputAS;   // compacted BLAS output buffer
        OptixTraversableHandle blas = 0;
    };

    std::vector<MeshBuffers> m_meshBuffers;
    GPUBuffer                m_tlasOutputBuffer;
    GPUBuffer                m_instanceBuffer;   // device OptixInstance array
    OptixTraversableHandle   m_tlas             = 0;

    // Shared custom-primitive BLAS for all implicit shape instances.
    // All canonical shapes occupy [-1,1]^3; the intersection program reads
    // the shape type from the per-instance SBT record.
    GPUBuffer              m_implicitAabbBuf;
    GPUBuffer              m_implicitOutputAS;
    OptixTraversableHandle m_implicitBlas = 0;

    // Uploads vertex/index data and builds one BLAS with compaction.
    // Writes to buffers.positions, buffers.indices, buffers.outputAS, buffers.blas.
    static void buildBlas(
        OptixDeviceContext ctx,
        MeshBuffers&       buffers,
        unsigned int       vertexCount,
        unsigned int       triangleCount);

    // Builds the shared custom-primitive BLAS for implicit shapes.
    void buildImplicitBlas(OptixDeviceContext ctx);

    // Frees old TLAS resources, recomputes world transforms from the scene node
    // hierarchy, then rebuilds the TLAS. Called by both build() and rebuildTlas().
    void buildTlasPhase(OptixDeviceContext ctx, const Scene& scene);
};

#endif // OPTIX_RAYTRACER_ACCEL_H

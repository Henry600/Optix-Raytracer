// Accel.cpp — OptiX acceleration structure builder.
//
// Builds one BLAS per mesh (with compaction) and one TLAS instancing all
// BLASes with world-space transforms derived from the scene node hierarchy. The compacted BLAS output buffers and TLAS
// output buffer are kept alive in MeshBuffers / m_tlasOutputBuffer so OptiX
// can continue to traverse them during rendering.
#include "accel.h"
#include "matrix4x4.h"
#include "node_3d.h"
#include "scene.h"

#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ─── Error macros ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t rc = (call);                                                \
        if (rc != cudaSuccess)                                                  \
        {                                                                       \
            throw std::runtime_error(std::string("CUDA error in " __FILE__     \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + cudaGetErrorString(rc));                                      \
        }                                                                       \
    } while (0)

#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult rc = (call);                                                \
        if (rc != OPTIX_SUCCESS)                                                \
        {                                                                       \
            throw std::runtime_error(std::string("OptiX error in " __FILE__    \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + optixGetErrorString(rc));                                     \
        }                                                                       \
    } while (0)

// ─── Accel::destroy ───────────────────────────────────────────────────────────

void Accel::destroy()
{
    m_meshBuffers.clear();  // GPUBuffer members free device memory via their destructors
    m_tlasOutputBuffer.free();
    m_instanceBuffer.free();
    m_tlas = 0;
}

// ─── Accel::buildBlas ─────────────────────────────────────────────────────────

void Accel::buildBlas(
    OptixDeviceContext ctx,
    MeshBuffers&       buffers,
    unsigned int       vertexCount,
    unsigned int       triangleCount)
{
    // One geometry flag entry (one SBT record)
    const uint32_t buildFlags[] = { OPTIX_GEOMETRY_FLAG_NONE };

    OptixBuildInput buildInput       = {};
    buildInput.type                  = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // OptiX expects a CUdeviceptr* for vertexBuffers — take address of a local copy.
    const CUdeviceptr posPtr         = buffers.positions.ptr();
    auto& tri                        = buildInput.triangleArray;
    tri.vertexFormat                 = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri.vertexStrideInBytes          = sizeof(float3);
    tri.numVertices                  = vertexCount;
    tri.vertexBuffers                = &posPtr;

    tri.indexFormat                  = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri.indexStrideInBytes           = sizeof(uint3);
    tri.numIndexTriplets             = triangleCount;
    tri.indexBuffer                  = buffers.indices.ptr();

    tri.flags                        = buildFlags;
    tri.numSbtRecords                = 1;
    tri.sbtIndexOffsetBuffer         = 0;

    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                    | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // Query memory requirements
    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &opts, &buildInput, 1, &sizes));

    GPUBuffer tempBuffer;
    tempBuffer.alloc(sizes.tempSizeInBytes);

    // Device slot to receive the compacted AS size
    GPUBuffer compactedSizeSlot;
    compactedSizeSlot.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeSlot.ptr();

    // Uncompacted build
    buffers.outputAS.alloc(sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        ctx, nullptr,
        &opts, &buildInput, 1,
        tempBuffer.ptr(),       tempBuffer.size(),
        buffers.outputAS.ptr(), buffers.outputAS.size(),
        &buffers.blas,
        &emitDesc, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back compacted size and compact if it is smaller
    uint64_t compactedSize = 0;
    compactedSizeSlot.download(&compactedSize, sizeof(uint64_t));

    if (compactedSize < sizes.outputSizeInBytes)
    {
        GPUBuffer compactedAS;
        compactedAS.alloc(compactedSize);

        OPTIX_CHECK(optixAccelCompact(ctx, nullptr,
                                      buffers.blas,
                                      compactedAS.ptr(), compactedSize,
                                      &buffers.blas));

        CUDA_CHECK(cudaDeviceSynchronize());

        buffers.outputAS = std::move(compactedAS);  // old uncompacted buffer freed here
    }
    // tempBuffer and compactedSizeSlot freed automatically at end of scope
}

// ─── Accel::build ─────────────────────────────────────────────────────────────

void Accel::build(OptixDeviceContext ctx, const Scene& scene)
{
    destroy();

    const auto& meshes = scene.meshes();
    if (meshes.empty())
    {
        return;
    }

    m_meshBuffers.resize(meshes.size());

    // ── BLAS per mesh ─────────────────────────────────────────────────────────
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        const Mesh& mesh = meshes[i];
        MeshBuffers& mb  = m_meshBuffers[i];

        // Upload vertex attributes to device
        mb.positions.allocAndUpload(mesh.positions.data(), mesh.positions.size() * sizeof(float3));

        if (!mesh.normals.empty())
        {
            mb.normals.allocAndUpload(mesh.normals.data(), mesh.normals.size() * sizeof(float3));
        }

        if (!mesh.uvs.empty())
        {
            mb.uvs.allocAndUpload(mesh.uvs.data(), mesh.uvs.size() * sizeof(float2));
        }

        mb.indices.allocAndUpload(mesh.indices.data(), mesh.indices.size() * sizeof(uint3));

        buildBlas(ctx, mb,
                  static_cast<unsigned int>(mesh.positions.size()),
                  static_cast<unsigned int>(mesh.indices.size()));
    }

    buildTlasPhase(ctx, scene);
}

// ─── Accel::buildTlasPhase ────────────────────────────────────────────────────

void Accel::buildTlasPhase(OptixDeviceContext ctx, const Scene& scene)
{
    // Free any existing TLAS resources — GPUBuffer::alloc() calls free() internally,
    // but we zero m_tlas here so the traversable is not used while we rebuild.
    m_tlas = 0;

    const auto& meshes = scene.meshes();
    if (meshes.empty())
    {
        return;
    }

    // ── World-space transforms from node hierarchy ────────────────────────────
    // Walk the Node3D tree and collect one record per MeshNode mesh reference.
    // Multiple nodes referencing the same mesh each produce their own TLAS
    // instance, so duplicated nodes render independently with independent materials.
    struct MeshInst
    {
        int       meshIdx;
        int       materialIdx;
        Matrix4x4 world;
    };
    std::vector<MeshInst> meshInstances;

    if (!scene.rootNodes().empty())
    {
        std::function<void(int, const Matrix4x4&)> walkNode =
            [&](int nodeIdx, const Matrix4x4& parentWorld)
        {
            const Node3D& node    = *scene.nodes()[nodeIdx];
            const Matrix4x4 world = mat4Multiply(parentWorld, node.localTransform);

            if (const MeshNode* mn = dynamic_cast<const MeshNode*>(&node))
            {
                for (int j = 0; j < static_cast<int>(mn->meshIndices.size()); ++j)
                {
                    const int mi = mn->meshIndices[j];
                    if (mi >= 0 && mi < static_cast<int>(meshes.size()))
                    {
                        const int matIdx = (j < static_cast<int>(mn->materialIndices.size()))
                            ? mn->materialIndices[j]
                            : meshes[mi].materialIndex;
                        meshInstances.push_back({mi, matIdx, world});
                    }
                }
            }

            for (int childIdx : node.children)
            {
                walkNode(childIdx, world);
            }
        };

        const Matrix4x4 identity = mat4Identity();
        for (int rootIdx : scene.rootNodes())
        {
            walkNode(rootIdx, identity);
        }
    }

    if (meshInstances.empty())
    {
        return;
    }

    // ── TLAS — one instance per MeshNode mesh reference ───────────────────────
    // sbtOffset = flat instance index i, matching the per-instance SBT records
    // built by buildSbt() using the same node-tree walk order.
    std::vector<OptixInstance> instances(meshInstances.size());

    for (size_t i = 0; i < meshInstances.size(); ++i)
    {
        const MeshInst&  inst_data = meshInstances[i];
        const Matrix4x4& w         = inst_data.world;
        OptixInstance&   inst      = instances[i];
        std::memset(&inst, 0, sizeof(inst));

        // OptiX instance transform = row-major 3×4 (last row [0,0,0,1] implicit).
        // Our Matrix4x4 is also row-major, so rows 0–2 copy directly.
        inst.transform[0]  = w.m[0][0];  inst.transform[1]  = w.m[0][1];
        inst.transform[2]  = w.m[0][2];  inst.transform[3]  = w.m[0][3];
        inst.transform[4]  = w.m[1][0];  inst.transform[5]  = w.m[1][1];
        inst.transform[6]  = w.m[1][2];  inst.transform[7]  = w.m[1][3];
        inst.transform[8]  = w.m[2][0];  inst.transform[9]  = w.m[2][1];
        inst.transform[10] = w.m[2][2];  inst.transform[11] = w.m[2][3];

        inst.instanceId        = static_cast<unsigned int>(i);
        inst.sbtOffset         = static_cast<unsigned int>(i);
        inst.visibilityMask    = 0xFF;
        inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
        inst.traversableHandle = m_meshBuffers[inst_data.meshIdx].blas;
    }

    m_instanceBuffer.allocAndUpload(instances.data(), instances.size() * sizeof(OptixInstance));

    OptixBuildInput tlasInput                          = {};
    tlasInput.type                                     = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    tlasInput.instanceArray.instances                  = m_instanceBuffer.ptr();
    tlasInput.instanceArray.numInstances               =
        static_cast<unsigned int>(instances.size());

    OptixAccelBuildOptions tlasOpts = {};
    tlasOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    tlasOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes tlasSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        ctx, &tlasOpts, &tlasInput, 1, &tlasSizes));

    GPUBuffer tlasTempBuffer;
    tlasTempBuffer.alloc(tlasSizes.tempSizeInBytes);
    m_tlasOutputBuffer.alloc(tlasSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        ctx, nullptr,
        &tlasOpts, &tlasInput, 1,
        tlasTempBuffer.ptr(),      tlasTempBuffer.size(),
        m_tlasOutputBuffer.ptr(),  m_tlasOutputBuffer.size(),
        &m_tlas, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    // tlasTempBuffer freed automatically at end of scope
}

// ─── Accel::rebuildTlas ───────────────────────────────────────────────────────

void Accel::rebuildTlas(OptixDeviceContext ctx, const Scene& scene)
{
    if (m_meshBuffers.empty())
    {
        return;  // no BLASes built yet — nothing to instance
    }
    buildTlasPhase(ctx, scene);
}

// Accel.cpp — OptiX acceleration structure builder.
//
// Builds one BLAS per mesh (with compaction) and one TLAS instancing all
// BLASes with world-space transforms derived from the scene node hierarchy. The compacted BLAS output buffers and TLAS
// output buffer are kept alive in MeshBuffers / m_tlasOutputBuffer so OptiX
// can continue to traverse them during rendering.
#include "accel.h"
#include "gaussian_splat.h"
#include "implicit_node.h"
#include "matrix4x4.h"
#include "node_3d.h"
#include "scene.h"
#include "splat_node.h"

#include <cmath>
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
    m_implicitAabbBuf.free();
    m_implicitOutputAS.free();
    m_implicitBlas = 0;
    m_splatBlas.clear();  // GPUBuffer members free device memory via their destructors
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

// ─── Accel::buildImplicitBlas ─────────────────────────────────────────────────

void Accel::buildImplicitBlas(OptixDeviceContext ctx)
{
    // All canonical implicit shapes (Sphere, Box, Cylinder) fit within [-1,1]^3
    // in local space.  One shared BLAS covers every shape type; the intersection
    // program reads the actual type from the per-instance SBT record.
    const OptixAabb aabb = { -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };
    m_implicitAabbBuf.allocAndUpload(&aabb, sizeof(OptixAabb));

    const uint32_t    buildFlags[] = { OPTIX_GEOMETRY_FLAG_NONE };
    const CUdeviceptr aabbPtr      = m_implicitAabbBuf.ptr();

    OptixBuildInput buildInput = {};
    buildInput.type            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    auto& cp                   = buildInput.customPrimitiveArray;
    cp.aabbBuffers             = &aabbPtr;
    cp.numPrimitives           = 1;
    cp.strideInBytes           = sizeof(OptixAabb);
    cp.flags                   = buildFlags;
    cp.numSbtRecords           = 1;
    cp.sbtIndexOffsetBuffer    = 0;

    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &opts, &buildInput, 1, &sizes));

    GPUBuffer tempBuf;
    tempBuf.alloc(sizes.tempSizeInBytes);
    m_implicitOutputAS.alloc(sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        ctx, nullptr,
        &opts, &buildInput, 1,
        tempBuf.ptr(),            tempBuf.size(),
        m_implicitOutputAS.ptr(), m_implicitOutputAS.size(),
        &m_implicitBlas,
        nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── Accel::buildSplatBlas ────────────────────────────────────────────────────

void Accel::buildSplatBlas(
    OptixDeviceContext       ctx,
    SplatBlasBuffers&        buffers,
    const GaussianSplatData& splat)
{
    // ── Per-gaussian AABBs in splat-local space ───────────────────────────────
    // The gaussian's 3σ ellipsoid is x = μ + R · diag(s) · u with |u| = 3.
    // Its AABB half-extent along axis j is 3·‖row_j(R · diag(s))‖.
    std::vector<OptixAabb> aabbs(splat.count);

    for (uint32_t i = 0; i < splat.count; ++i)
    {
        const float3& p = splat.positions[i];
        const float4& q = splat.rotations[i];  // (x, y, z, w)
        const float3& s = splat.scales[i];

        // Rotation matrix rows from the unit quaternion
        const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
        const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

        const float r[3][3] = {
            { 1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),        2.0f * (xz + wy)        },
            { 2.0f * (xy + wz),        1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx)        },
            { 2.0f * (xz - wy),        2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy) },
        };

        float half[3];
        for (int j = 0; j < 3; ++j)
        {
            const float mx = r[j][0] * s.x;
            const float my = r[j][1] * s.y;
            const float mz = r[j][2] * s.z;
            half[j] = 3.0f * std::sqrt(mx * mx + my * my + mz * mz);
        }

        aabbs[i] = { p.x - half[0], p.y - half[1], p.z - half[2],
                     p.x + half[0], p.y + half[1], p.z + half[2] };
    }

    buffers.aabbs.allocAndUpload(aabbs.data(), aabbs.size() * sizeof(OptixAabb));

    // ── Custom-primitive BLAS with compaction ─────────────────────────────────
    const uint32_t    buildFlags[] = { OPTIX_GEOMETRY_FLAG_NONE };
    const CUdeviceptr aabbPtr      = buffers.aabbs.ptr();

    OptixBuildInput buildInput = {};
    buildInput.type            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    auto& cp                   = buildInput.customPrimitiveArray;
    cp.aabbBuffers             = &aabbPtr;
    cp.numPrimitives           = splat.count;
    cp.strideInBytes           = sizeof(OptixAabb);
    cp.flags                   = buildFlags;
    cp.numSbtRecords           = 1;  // all gaussians share one record; primitive index selects
    cp.sbtIndexOffsetBuffer    = 0;

    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                    | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &opts, &buildInput, 1, &sizes));

    GPUBuffer tempBuffer;
    tempBuffer.alloc(sizes.tempSizeInBytes);

    GPUBuffer compactedSizeSlot;
    compactedSizeSlot.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeSlot.ptr();

    buffers.outputAS.alloc(sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        ctx, nullptr,
        &opts, &buildInput, 1,
        tempBuffer.ptr(),       tempBuffer.size(),
        buffers.outputAS.ptr(), buffers.outputAS.size(),
        &buffers.blas,
        &emitDesc, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

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

        buffers.outputAS = std::move(compactedAS);
    }
}

// ─── Accel::build ─────────────────────────────────────────────────────────────

void Accel::build(OptixDeviceContext ctx, const Scene& scene)
{
    destroy();

    const auto& meshes = scene.meshes();

    // ── BLAS per mesh ─────────────────────────────────────────────────────────
    if (!meshes.empty())
    {
        m_meshBuffers.resize(meshes.size());

        for (size_t i = 0; i < meshes.size(); ++i)
        {
            const Mesh& mesh = meshes[i];
            MeshBuffers& mb  = m_meshBuffers[i];

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
    }

    // ── Shared implicit BLAS (built if any ImplicitNode exists in scene) ──────
    {
        bool hasImplicits = false;
        std::function<void(int)> scan = [&](int idx)
        {
            const Node3D& n = *scene.nodes()[idx];
            if (dynamic_cast<const ImplicitNode*>(&n)) { hasImplicits = true; }
            for (int c : n.children) { scan(c); }
        };
        for (int r : scene.rootNodes()) { scan(r); }

        if (hasImplicits) { buildImplicitBlas(ctx); }
    }

    // ── Splat BLASes (one per dataset with GPU-resident property buffers) ─────
    // The SBT walk instances a dataset only when splatGpu(i).valid(), so skip
    // datasets whose upload failed to keep the two walks in agreement.
    {
        const auto& splats = scene.splats();
        m_splatBlas.resize(splats.size());
        for (size_t i = 0; i < splats.size(); ++i)
        {
            if (splats[i].count > 0 && scene.splatGpu(static_cast<int>(i)).valid())
            {
                buildSplatBlas(ctx, m_splatBlas[i], splats[i]);
            }
        }
    }

    buildTlasPhase(ctx, scene);
}

// ─── Accel::buildTlasPhase ────────────────────────────────────────────────────

void Accel::buildTlasPhase(OptixDeviceContext ctx, const Scene& scene)
{
    // Zero m_tlas so the old handle is not used while we rebuild.
    m_tlas = 0;

    const auto& meshes   = scene.meshes();
    const auto& allNodes = scene.nodes();

    // ── DFS walk: collect one OptixInstance per mesh-primitive and per implicit ─
    // The walk order must match buildSbt() exactly so that TLAS instance i and
    // SBT record i always correspond.
    std::vector<OptixInstance> instances;

    auto setTransform = [](OptixInstance& inst, const Matrix4x4& w)
    {
        // OptiX stores a row-major 3×4 transform (last row [0 0 0 1] implicit).
        inst.transform[0]  = w.m[0][0];  inst.transform[1]  = w.m[0][1];
        inst.transform[2]  = w.m[0][2];  inst.transform[3]  = w.m[0][3];
        inst.transform[4]  = w.m[1][0];  inst.transform[5]  = w.m[1][1];
        inst.transform[6]  = w.m[1][2];  inst.transform[7]  = w.m[1][3];
        inst.transform[8]  = w.m[2][0];  inst.transform[9]  = w.m[2][1];
        inst.transform[10] = w.m[2][2];  inst.transform[11] = w.m[2][3];
    };

    std::function<void(int)> walkNode = [&](int nodeIdx)
    {
        const Node3D&    node  = *allNodes[nodeIdx];
        const Matrix4x4& world = node.worldTransform;

        if (!node.visible) { return; }  // skip entire subtree

        if (const MeshNode* mn = dynamic_cast<const MeshNode*>(&node))
        {
            for (int j = 0; j < static_cast<int>(mn->meshIndices.size()); ++j)
            {
                const int mi = mn->meshIndices[j];
                if (mi < 0 || mi >= static_cast<int>(meshes.size())) { continue; }

                OptixInstance inst = {};
                setTransform(inst, world);
                inst.instanceId        = static_cast<unsigned int>(instances.size());
                inst.sbtOffset         = static_cast<unsigned int>(instances.size());
                inst.visibilityMask    = VIS_MASK_GEOMETRY;
                inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
                inst.traversableHandle = m_meshBuffers[mi].blas;
                instances.push_back(inst);
            }
        }
        else if (dynamic_cast<const ImplicitNode*>(&node))
        {
            if (m_implicitBlas != 0)
            {
                OptixInstance inst = {};
                setTransform(inst, world);
                inst.instanceId        = static_cast<unsigned int>(instances.size());
                inst.sbtOffset         = static_cast<unsigned int>(instances.size());
                inst.visibilityMask    = VIS_MASK_GEOMETRY;
                inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
                inst.traversableHandle = m_implicitBlas;
                instances.push_back(inst);
            }
        }
        else if (const SplatNode* sn = dynamic_cast<const SplatNode*>(&node))
        {
            const OptixTraversableHandle blas =
                (sn->splatIndex >= 0) ? splatBlas(static_cast<size_t>(sn->splatIndex)) : 0;
            if (blas != 0)
            {
                OptixInstance inst = {};
                setTransform(inst, world);
                inst.instanceId        = static_cast<unsigned int>(instances.size());
                inst.sbtOffset         = static_cast<unsigned int>(instances.size());
                inst.visibilityMask    = VIS_MASK_SPLAT;
                inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
                inst.traversableHandle = blas;
                instances.push_back(inst);
            }
        }

        for (int childIdx : node.children)
        {
            walkNode(childIdx);
        }
    };

    for (int rootIdx : scene.rootNodes())
    {
        walkNode(rootIdx);
    }

    if (instances.empty())
    {
        return;
    }

    m_instanceBuffer.allocAndUpload(instances.data(), instances.size() * sizeof(OptixInstance));

    OptixBuildInput tlasInput                = {};
    tlasInput.type                           = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    tlasInput.instanceArray.instances        = m_instanceBuffer.ptr();
    tlasInput.instanceArray.numInstances     = static_cast<unsigned int>(instances.size());

    OptixAccelBuildOptions tlasOpts = {};
    tlasOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    tlasOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes tlasSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &tlasOpts, &tlasInput, 1, &tlasSizes));

    GPUBuffer tlasTempBuffer;
    tlasTempBuffer.alloc(tlasSizes.tempSizeInBytes);
    m_tlasOutputBuffer.alloc(tlasSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        ctx, nullptr,
        &tlasOpts, &tlasInput, 1,
        tlasTempBuffer.ptr(),     tlasTempBuffer.size(),
        m_tlasOutputBuffer.ptr(), m_tlasOutputBuffer.size(),
        &m_tlas, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─── Accel::rebuildTlas ───────────────────────────────────────────────────────

void Accel::rebuildTlas(OptixDeviceContext ctx, const Scene& scene)
{
    bool anySplatBlas = false;
    for (const SplatBlasBuffers& sb : m_splatBlas)
    {
        if (sb.blas != 0)
        {
            anySplatBlas = true;
            break;
        }
    }
    if (m_meshBuffers.empty() && m_implicitBlas == 0 && !anySplatBlas)
    {
        return;  // no BLASes built yet — nothing to instance
    }
    buildTlasPhase(ctx, scene);
}

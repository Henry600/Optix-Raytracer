// application_pipeline.cpp — OptiX pipeline, program groups, SBT, and pick launch.

#include "application.h"
#include "cuda_optix_check.h"
#include "implicit_node.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// ─── SBT record types ────────────────────────────────────────────────────────
// Each record = 32-byte opaque header (packed by optixSbtRecordPackHeader) + user data.
// alignas(OPTIX_SBT_RECORD_ALIGNMENT) ensures the compiler pads sizeof() to a multiple
// of the required 16-byte SBT stride.

namespace
{

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // no extra raygen data
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // no extra miss data
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    union {
        MeshData          mesh;      // used when packed with m_pgHitgroup
        ImplicitShapeData implicit;  // used when packed with m_pgHitgroupImplicit
    } data;
};

// Per-instance info collected during a scene graph DFS.
struct InstanceInfo
{
    int          nodeIdx;
    bool         isImplicit;
    int          meshIdx;      // valid when !isImplicit
    int          materialIdx;
    ImplicitType implicitType; // valid when isImplicit
};

// Walk the scene graph in the same DFS order as Accel::buildTlasPhase so that
// TLAS instance i and SBT record i always correspond.  Emits one InstanceInfo
// per MeshNode sub-mesh and one per ImplicitNode.
static void walkSceneInstances(const Scene& scene, std::vector<InstanceInfo>& out)
{
    const auto& allNodes = scene.nodes();
    const auto& meshes   = scene.meshes();

    std::function<void(int)> walk = [&](int nodeIdx)
    {
        const Node3D& node = *allNodes[nodeIdx];
        if (!node.visible) { return; }  // must match Accel::buildTlasPhase walk exactly
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
                    out.push_back({nodeIdx, false, mi, matIdx, ImplicitType::Sphere});
                }
            }
        }
        else if (const ImplicitNode* in = dynamic_cast<const ImplicitNode*>(&node))
        {
            out.push_back({nodeIdx, true, 0, in->materialIndex, in->type});
        }
        for (int childIdx : node.children)
        {
            walk(childIdx);
        }
    };

    for (int rootIdx : scene.rootNodes())
    {
        walk(rootIdx);
    }
}

} // anonymous namespace

// ─── Pipeline ─────────────────────────────────────────────────────────────────

void Application::buildPipeline(const std::string& ptxDir)
{
    // Load PTX source from the directory next to the executable
    const std::string ptxPath = (std::filesystem::path(ptxDir) / "device_programs.ptx").string();

    std::ifstream ptxFile(ptxPath, std::ios::binary | std::ios::ate);
    if (!ptxFile)
    {
        throw std::runtime_error("Cannot open PTX file: " + ptxPath);
    }

    const std::streamsize ptxSize = ptxFile.tellg();
    ptxFile.seekg(0, std::ios::beg);
    std::string ptxSource(static_cast<size_t>(ptxSize), '\0');
    ptxFile.read(ptxSource.data(), ptxSize);

    // ── Module ────────────────────────────────────────────────────────────────
    OptixModuleCompileOptions moduleOpts = {};
    moduleOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineOpts = {};
    pipelineOpts.usesMotionBlur                   = 0;
    pipelineOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOpts.numPayloadValues                 = 4;  // radiance: p0/p1 = packed PathVertex ptr
                                                        // shadow: p0=vis, p1/p2/p3=RGB filter
    pipelineOpts.numAttributeValues               = 3;  // barycentrics for triangles; float3 normal for custom primitives
    pipelineOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOpts.pipelineLaunchParamsVariableName = "optixLaunchParams";
    pipelineOpts.usesPrimitiveTypeFlags           = static_cast<unsigned int>(
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

    OPTIX_CHECK(optixModuleCreate(
        m_optixContext,
        &moduleOpts,
        &pipelineOpts,
        ptxSource.c_str(), ptxSource.size(),
        nullptr, nullptr,
        &m_module));

    // ── Program groups ────────────────────────────────────────────────────────
    OptixProgramGroupOptions pgOpts = {};
    OptixProgramGroupDesc    pgDesc = {};

    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = m_module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";
    OPTIX_CHECK(optixProgramGroupCreate(
        m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgRaygen));

    pgDesc                          = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module              = m_module;
    pgDesc.miss.entryFunctionName   = "__miss__radiance";
    OPTIX_CHECK(optixProgramGroupCreate(
        m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgMiss));

    pgDesc                          = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module              = m_module;
    pgDesc.miss.entryFunctionName   = "__miss__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(
        m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgMissShadow));

    pgDesc                              = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = m_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH            = m_module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    pgDesc.hitgroup.moduleIS            = nullptr;  // built-in triangle IS
    pgDesc.hitgroup.entryFunctionNameIS = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgHitgroup));

    pgDesc                              = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = m_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__implicit";
    pgDesc.hitgroup.moduleAH            = m_module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__implicit";
    pgDesc.hitgroup.moduleIS            = m_module;
    pgDesc.hitgroup.entryFunctionNameIS = "__intersection__implicit";
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgHitgroupImplicit));

    // ── Pick program groups ───────────────────────────────────────────────────
    pgDesc                              = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module                = m_module;
    pgDesc.raygen.entryFunctionName     = "__raygen__pick";
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgPickRaygen));

    pgDesc                              = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module                  = m_module;
    pgDesc.miss.entryFunctionName       = "__miss__pick";
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgPickMiss));

    pgDesc                              = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = m_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__pick";
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgPickHitgroup));

    pgDesc                              = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = m_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__pick";
    pgDesc.hitgroup.moduleIS            = m_module;
    pgDesc.hitgroup.entryFunctionNameIS = "__intersection__implicit";
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext, &pgDesc, 1, &pgOpts, nullptr, nullptr, &m_pgPickHitgroupImplicit));

    // ── Pipeline ──────────────────────────────────────────────────────────────
    const OptixProgramGroup pgs[] = {
        m_pgRaygen, m_pgMiss, m_pgMissShadow, m_pgHitgroup, m_pgHitgroupImplicit,
        m_pgPickRaygen, m_pgPickMiss, m_pgPickHitgroup, m_pgPickHitgroupImplicit
    };

    OptixPipelineLinkOptions linkOpts = {};
    // Depth 1: path rays and NEE shadow rays are both called from raygen —
    // no CH/miss ever calls optixTrace, so the chain never exceeds depth 1.
    linkOpts.maxTraceDepth = 1;

    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &pipelineOpts, &linkOpts,
        pgs, 9,
        nullptr, nullptr,
        &m_pipeline));

    // Stack size — 2 KB continuation stack, max traversal depth 2 (TLAS → BLAS)
    OPTIX_CHECK(optixPipelineSetStackSize(m_pipeline, 0, 0, 2048, 2));
}

// ─── Hot reload ───────────────────────────────────────────────────────────────

void Application::reloadPipeline()
{
    // Drain the GPU before touching any pipeline objects
    CUDA_CHECK(cudaDeviceSynchronize());

    // Save the current handles — we restore them if the new PTX fails to compile,
    // which keeps the last working shader running instead of going black.
    const OptixModule       oldModule                    = m_module;
    const OptixProgramGroup oldPgRaygen                  = m_pgRaygen;
    const OptixProgramGroup oldPgMiss                    = m_pgMiss;
    const OptixProgramGroup oldPgMissShadow              = m_pgMissShadow;
    const OptixProgramGroup oldPgHitgroup                = m_pgHitgroup;
    const OptixProgramGroup oldPgHitgroupImplicit        = m_pgHitgroupImplicit;
    const OptixProgramGroup oldPgPickRaygen              = m_pgPickRaygen;
    const OptixProgramGroup oldPgPickMiss                = m_pgPickMiss;
    const OptixProgramGroup oldPgPickHitgroup            = m_pgPickHitgroup;
    const OptixProgramGroup oldPgPickHitgroupImplicit    = m_pgPickHitgroupImplicit;
    const OptixPipeline     oldPipeline                  = m_pipeline;

    m_module                    = nullptr;
    m_pgRaygen                  = nullptr;
    m_pgMiss                    = nullptr;
    m_pgMissShadow              = nullptr;
    m_pgHitgroup                = nullptr;
    m_pgHitgroupImplicit        = nullptr;
    m_pgPickRaygen              = nullptr;
    m_pgPickMiss                = nullptr;
    m_pgPickHitgroup            = nullptr;
    m_pgPickHitgroupImplicit    = nullptr;
    m_pipeline                  = nullptr;

    try
    {
        buildPipeline(m_ptxDir);
    }
    catch (...)
    {
        if (m_pipeline)                  { optixPipelineDestroy(m_pipeline);                      m_pipeline                  = nullptr; }
        if (m_pgPickHitgroupImplicit)    { optixProgramGroupDestroy(m_pgPickHitgroupImplicit);    m_pgPickHitgroupImplicit    = nullptr; }
        if (m_pgPickHitgroup)            { optixProgramGroupDestroy(m_pgPickHitgroup);            m_pgPickHitgroup            = nullptr; }
        if (m_pgPickMiss)                { optixProgramGroupDestroy(m_pgPickMiss);                m_pgPickMiss                = nullptr; }
        if (m_pgPickRaygen)              { optixProgramGroupDestroy(m_pgPickRaygen);              m_pgPickRaygen              = nullptr; }
        if (m_pgHitgroupImplicit)        { optixProgramGroupDestroy(m_pgHitgroupImplicit);        m_pgHitgroupImplicit        = nullptr; }
        if (m_pgHitgroup)                { optixProgramGroupDestroy(m_pgHitgroup);                m_pgHitgroup                = nullptr; }
        if (m_pgMissShadow)              { optixProgramGroupDestroy(m_pgMissShadow);              m_pgMissShadow              = nullptr; }
        if (m_pgMiss)                    { optixProgramGroupDestroy(m_pgMiss);                    m_pgMiss                    = nullptr; }
        if (m_pgRaygen)                  { optixProgramGroupDestroy(m_pgRaygen);                  m_pgRaygen                  = nullptr; }
        if (m_module)                    { optixModuleDestroy(m_module);                          m_module                    = nullptr; }

        m_module                    = oldModule;
        m_pgRaygen                  = oldPgRaygen;
        m_pgMiss                    = oldPgMiss;
        m_pgMissShadow              = oldPgMissShadow;
        m_pgHitgroup                = oldPgHitgroup;
        m_pgHitgroupImplicit        = oldPgHitgroupImplicit;
        m_pgPickRaygen              = oldPgPickRaygen;
        m_pgPickMiss                = oldPgPickMiss;
        m_pgPickHitgroup            = oldPgPickHitgroup;
        m_pgPickHitgroupImplicit    = oldPgPickHitgroupImplicit;
        m_pipeline                  = oldPipeline;
        throw;
    }

    buildSbt();
    m_accumDirty = true;  // new shader = new result; clear accumulation

    if (oldPipeline)                  { optixPipelineDestroy(oldPipeline);                    }
    if (oldPgPickHitgroupImplicit)    { optixProgramGroupDestroy(oldPgPickHitgroupImplicit);  }
    if (oldPgPickHitgroup)            { optixProgramGroupDestroy(oldPgPickHitgroup);          }
    if (oldPgPickMiss)                { optixProgramGroupDestroy(oldPgPickMiss);              }
    if (oldPgPickRaygen)              { optixProgramGroupDestroy(oldPgPickRaygen);            }
    if (oldPgHitgroupImplicit)        { optixProgramGroupDestroy(oldPgHitgroupImplicit);      }
    if (oldPgHitgroup)                { optixProgramGroupDestroy(oldPgHitgroup);              }
    if (oldPgMissShadow)              { optixProgramGroupDestroy(oldPgMissShadow);            }
    if (oldPgMiss)                    { optixProgramGroupDestroy(oldPgMiss);                  }
    if (oldPgRaygen)                  { optixProgramGroupDestroy(oldPgRaygen);                }
    if (oldModule)                    { optixModuleDestroy(oldModule);                        }
}

void Application::checkShaderHotReload()
{
    const auto ptxPath = std::filesystem::path(m_ptxDir) / "device_programs.ptx";

    std::error_code ec;
    const auto newWriteTime = std::filesystem::last_write_time(ptxPath, ec);
    if (ec || newWriteTime == m_ptxWriteTime)
    {
        return;
    }

    // Stamp first — prevents hammering reloadPipeline every frame if the PTX
    // stays broken (the timestamp will have moved but won't keep changing).
    m_ptxWriteTime = newWriteTime;

    try
    {
        reloadPipeline();
        m_shaderError.clear();
    }
    catch (const std::exception& e)
    {
        m_shaderError = e.what();
    }
}

// ─── Shader binding table ────────────────────────────────────────────────────

void Application::buildSbt()
{
    m_sbtRaygenBuffer.free();
    m_sbtMissBuffer.free();
    m_sbtHitgroupBuffer.free();
    m_sbt = {};

    // ── Raygen record ─────────────────────────────────────────────────────────
    RaygenRecord raygenRec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygen, &raygenRec));
    m_sbtRaygenBuffer.allocAndUpload(&raygenRec, sizeof(RaygenRecord));

    // ── Miss records — index 0 = radiance, index 1 = NEE shadow ─────────────
    MissRecord missRecs[2] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMiss,       &missRecs[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMissShadow, &missRecs[1]));
    m_sbtMissBuffer.allocAndUpload(missRecs, sizeof(missRecs));

    // ── Hit group records — one per TLAS instance ────────────────────────────
    // Walk the node tree in the same DFS order as Accel::buildTlasPhase so that
    // TLAS instance i and SBT record i always correspond.
    std::vector<InstanceInfo> instList;
    walkSceneInstances(*m_scene, instList);

    std::vector<HitGroupRecord> hitRecs(instList.size());
    for (size_t i = 0; i < instList.size(); ++i)
    {
        const InstanceInfo& info = instList[i];
        if (info.isImplicit)
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitgroupImplicit, &hitRecs[i]));
            hitRecs[i].data.implicit.type          = static_cast<unsigned int>(info.implicitType);
            hitRecs[i].data.implicit.materialIndex = info.materialIdx;
        }
        else
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitgroup, &hitRecs[i]));
            if (m_scene->hasAccel())
            {
                const auto ptrs                    = m_scene->meshDevicePtrs(info.meshIdx);
                hitRecs[i].data.mesh.positions     = reinterpret_cast<const float3*>(ptrs.positions);
                hitRecs[i].data.mesh.normals       = reinterpret_cast<const float3*>(ptrs.normals);
                hitRecs[i].data.mesh.indices       = reinterpret_cast<const uint3*>(ptrs.indices);
                hitRecs[i].data.mesh.uvs           = reinterpret_cast<const float2*>(ptrs.uvs);
                hitRecs[i].data.mesh.materialIndex = info.materialIdx;
            }
        }
    }

    if (!hitRecs.empty())
    {
        m_sbtHitgroupBuffer.allocAndUpload(hitRecs.data(), hitRecs.size() * sizeof(HitGroupRecord));
    }

    // ── Fill the SBT descriptor ───────────────────────────────────────────────
    m_sbt.raygenRecord                = m_sbtRaygenBuffer.ptr();
    m_sbt.missRecordBase              = m_sbtMissBuffer.ptr();
    m_sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    m_sbt.missRecordCount             = 2;  // [0]=radiance, [1]=shadow
    m_sbt.hitgroupRecordBase          = m_sbtHitgroupBuffer.ptr();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_sbt.hitgroupRecordCount         = static_cast<unsigned int>(hitRecs.size());

    // ── Instance → node map ───────────────────────────────────────────────────
    m_instanceToNode.resize(instList.size());
    for (size_t i = 0; i < instList.size(); ++i)
    {
        m_instanceToNode[i] = instList[i].nodeIdx;
    }

    buildPickSbt();
}

// ─── Pick SBT ────────────────────────────────────────────────────────────────

void Application::buildPickSbt()
{
    m_pickSbtRaygenBuffer.free();
    m_pickSbtMissBuffer.free();
    m_pickSbtHitgroupBuffer.free();
    m_pickSbt = {};

    if (!m_pgPickRaygen)
    {
        return;
    }

    RaygenRecord raygenRec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgPickRaygen, &raygenRec));
    m_pickSbtRaygenBuffer.allocAndUpload(&raygenRec, sizeof(RaygenRecord));

    MissRecord missRec = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgPickMiss, &missRec));
    m_pickSbtMissBuffer.allocAndUpload(&missRec, sizeof(MissRecord));

    // Hit records — same DFS walk order as buildSbt / Accel::buildTlasPhase.
    std::vector<InstanceInfo> instList;
    walkSceneInstances(*m_scene, instList);

    std::vector<HitGroupRecord> hitRecs(instList.size());
    for (size_t i = 0; i < instList.size(); ++i)
    {
        const InstanceInfo& info = instList[i];
        hitRecs[i] = {};
        if (info.isImplicit)
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(m_pgPickHitgroupImplicit, &hitRecs[i]));
            hitRecs[i].data.implicit.type = static_cast<unsigned int>(info.implicitType);
        }
        else
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(m_pgPickHitgroup, &hitRecs[i]));
        }
    }

    if (!hitRecs.empty())
    {
        m_pickSbtHitgroupBuffer.allocAndUpload(hitRecs.data(), hitRecs.size() * sizeof(HitGroupRecord));
    }

    m_pickSbt.raygenRecord                = m_pickSbtRaygenBuffer.ptr();
    m_pickSbt.missRecordBase              = m_pickSbtMissBuffer.ptr();
    m_pickSbt.missRecordStrideInBytes     = sizeof(MissRecord);
    m_pickSbt.missRecordCount             = 1;
    m_pickSbt.hitgroupRecordBase          = m_pickSbtHitgroupBuffer.ptr();
    m_pickSbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    m_pickSbt.hitgroupRecordCount         = static_cast<unsigned int>(hitRecs.size());
}

// ─── Pick launch ─────────────────────────────────────────────────────────────

int Application::launchPick(float u, float v, float* outDistance)
{
    if (!m_scene->hasAccel() || m_instanceToNode.empty() || !m_pickSbt.raygenRecord)
    {
        return -1;
    }

    if (!m_pickResultBuffer.ptr())
    {
        m_pickResultBuffer.alloc(sizeof(uint32_t));
    }
    if (!m_pickDistanceBuffer.ptr())
    {
        m_pickDistanceBuffer.alloc(sizeof(float));
    }

    m_launchParams.pickU         = u;
    m_launchParams.pickV         = v;
    m_launchParams.pickResult    = m_pickResultBuffer.typedPtr<uint32_t>();
    m_launchParams.pickDistance  = m_pickDistanceBuffer.typedPtr<float>();
    m_launchParamsBuffer.upload(&m_launchParams, sizeof(LaunchParams));

    OPTIX_CHECK(optixLaunch(
        m_pipeline, nullptr,
        m_launchParamsBuffer.ptr(), sizeof(LaunchParams),
        &m_pickSbt,
        1, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    m_launchParams.pickResult   = nullptr;
    m_launchParams.pickDistance = nullptr;

    uint32_t rawIdx = 0xFFFFFFFFu;
    m_pickResultBuffer.download(&rawIdx, sizeof(uint32_t));

    if (outDistance)
    {
        float dist = -1.0f;
        m_pickDistanceBuffer.download(&dist, sizeof(float));
        *outDistance = (dist > 0.0f) ? dist : -1.0f;
    }

    if (rawIdx == 0xFFFFFFFFu || rawIdx >= static_cast<uint32_t>(m_instanceToNode.size()))
    {
        return -1;
    }
    return m_instanceToNode[rawIdx];
}

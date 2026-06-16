// Application.cpp — host-side application: window, CUDA/OptiX init, render loop.
//
// IMPORTANT: optix_function_table_definition.h must appear in exactly ONE
// translation unit. This file is that unit — do not include it elsewhere.

#include "application.h"
#include "implicit_node.h"

#include <optix_function_table_definition.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <nfd.h>
#include "scene_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Error macros ─────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t rc = (call);                                               \
        if (rc != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error in " __FILE__     \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + cudaGetErrorString(rc));                                     \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult rc = (call);                                               \
        if (rc != OPTIX_SUCCESS) {                                             \
            throw std::runtime_error(std::string("OptiX error in " __FILE__    \
                ":" + std::to_string(__LINE__) + " — ")                        \
                + optixGetErrorString(rc));                                    \
        }                                                                      \
    } while (0)

// ─── Construction / Destruction ───────────────────────────────────────────────

Application::Application(int width, int height, const std::string& title,
                         const std::string& ptxDir)
    : m_width(width), m_height(height)
{
    initWindow(title);
    m_vkCtx.init(m_window, m_width, m_height);
    initImGui();
    initCuda();
    initOptix();
    initDenoiser();
    buildPipeline(ptxDir);

    // Hot-reload — remember where the PTX lives and when it was last written
    m_ptxDir = ptxDir;
    {
        std::error_code ec;
        m_ptxWriteTime = std::filesystem::last_write_time(std::filesystem::path(ptxDir) / "device_programs.ptx", ec);
    }

    // Thumbnail cache lives alongside the executable so it survives across runs.
    m_hdriBrowser.setCacheDir(
        (std::filesystem::path(m_ptxDir) / "thumbnails").u8string());

    m_scene = std::make_unique<Scene>();
    buildSbt();  // empty SBT — no meshes yet

    m_launchParamsBuffer.alloc(sizeof(LaunchParams));
    NFD_Init();
}

Application::~Application()
{
    // GPUBuffer members (m_colorBuffer, m_accumBuffer, SBT buffers, denoiser buffers,
    // m_materialsBuffer, m_launchParamsBuffer) free themselves via their destructors.
    // std::vector members (m_colorBufferHost, m_hdrBufferHost) do the same.

    // m_envMap destructor frees GPU resources automatically — no explicit call needed

    if (m_scene)
    {
        m_scene->destroyAccel();  // free AS GPU memory before destroying OptiX context
    }

    if (m_denoiser)
    {
        optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
    }

    if (m_pipeline)              { optixPipelineDestroy(m_pipeline);                    m_pipeline              = nullptr; }
    if (m_pgHitgroupImplicit)    { optixProgramGroupDestroy(m_pgHitgroupImplicit);      m_pgHitgroupImplicit    = nullptr; }
    if (m_pgHitgroup)            { optixProgramGroupDestroy(m_pgHitgroup);              m_pgHitgroup            = nullptr; }
    if (m_pgMissShadow)          { optixProgramGroupDestroy(m_pgMissShadow);            m_pgMissShadow          = nullptr; }
    if (m_pgMiss)                { optixProgramGroupDestroy(m_pgMiss);                  m_pgMiss                = nullptr; }
    if (m_pgRaygen)              { optixProgramGroupDestroy(m_pgRaygen);                m_pgRaygen              = nullptr; }
    if (m_module)                { optixModuleDestroy(m_module);                        m_module                = nullptr; }

    if (m_optixContext)
    {
        optixDeviceContextDestroy(m_optixContext);
        m_optixContext = nullptr;
    }

    NFD_Quit();

    // ImGui must shut down before Vulkan resources are destroyed.
    // destroyDisplayImage() calls ImGui_ImplVulkan_RemoveTexture(), so it must
    // run while the ImGui Vulkan backend is still alive (before Shutdown).
    m_vkCtx.waitIdle();
    m_hdriBrowser.shutdown(m_vkCtx);  // destroyImGuiTexture before ImGui_ImplVulkan_Shutdown
    m_vkCtx.destroyDisplayImage();    // unregisters display texture from ImGui
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    // Explicitly clean up Vulkan before destroying the GLFW window.  The m_vkCtx
    // member destructor would do this automatically, but member destructors run
    // *after* the destructor body — by which point glfwDestroyWindow has torn down
    // the underlying wl_surface, causing the Nvidia driver to crash in destroySwapchain.
    m_vkCtx.cleanup();

    if (m_window) { glfwDestroyWindow(m_window); m_window = nullptr; }
    glfwTerminate();
}

// ─── Initialisation ───────────────────────────────────────────────────────────

void Application::initWindow(const std::string& title)
{
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialise GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // no OpenGL context — Vulkan owns presentation

    // On Wayland HiDPI the default GLFW_SCALE_FRAMEBUFFER=TRUE makes glfwGetFramebufferSize
    // return physical pixels while glfwGetCursorPos stays in logical pixels, causing a growing
    // coordinate mismatch (worse toward bottom-right).  Disabling it puts everything in logical-
    // pixel space; the compositor handles upscaling transparently.
    // On other platforms (Win32, macOS, X11) TRUE is correct: the swapchain should match
    // the monitor's physical resolution, so we only disable it on Wayland.
    if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND)
    {
        glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_FALSE);
    }

    m_window = glfwCreateWindow(m_width, m_height, title.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* win, int /*w*/, int /*h*/)
    {
        static_cast<Application*>(glfwGetWindowUserPointer(win))->m_swapchainResizePending = true;
    });
}

// ─── ImGui initialisation ────────────────────────────────────────────────────

void Application::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // ImGuiConfigFlags_ViewportsEnable disabled — Vulkan multi-viewport requires
    // per-viewport swapchains (significant extra complexity).
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.02f, 0.04f, 0.07f, 0.94f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.088f, 0.172f, 0.275f, 1.000f);
    style.FrameRounding = 4.0f;
    style.WindowRounding = 8.0f;
    style.GrabRounding = 4.0f;
    style.WindowPadding = ImVec2(4.0f, 4.0f);
    style.FramePadding = ImVec2(4.0f, 4.0f);
    style.ItemSpacing = ImVec2(8.0f, 4.0f);
    style.IndentSpacing = 16.0f;
    style.TreeLinesFlags = ImGuiTreeNodeFlags_DrawLinesFull;

    // On an scRGB swapchain the UI pipeline bakes the paper-white scale in as a
    // specialization constant.  The scale is only meaningful when Windows HDR is
    // actually active on the display; when HDR is off DWM maps scRGB 1.0 →
    // display-white, so applying the paper-white multiplier would inflate mid-
    // tones (e.g., sRGB 0.5 → 68% brightness instead of the correct 21.7%).
    // Load fonts before initImGui() so the Vulkan backend uploads the atlas
    // on its first NewFrame call.  AddFontDefault must be called first so that
    // the FA glyphs merge INTO the default ProggyClean font (not replacing it).
    {
        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->AddFontDefault();
        if (std::filesystem::exists("fonts/fa-solid-900.ttf"))
        {
            // Restrict to the exact codepoints used in icons.h — avoids
            // rasterising the full FA glyph set (~1 000 glyphs).
            static const ImWchar icon_ranges[] = {
                0xF030, 0xF030,  // fa-camera
                0xF0C8, 0xF0C8,  // fa-square
                0xF111, 0xF111,  // fa-circle
                0xF1B2, 0xF1B2,  // fa-cube
                0xF1C0, 0xF1C0,  // fa-database
                0xF5FD, 0xF5FD,  // fa-layer-group
                0
            };
            ImFontConfig cfg;
            cfg.MergeMode        = true;
            cfg.PixelSnapH       = true;
            cfg.GlyphMinAdvanceX = 13.0f;
            io.Fonts->AddFontFromFileTTF(
                "fonts/fa-solid-900.ttf", 13.0f, &cfg, icon_ranges);
        }
    }

    m_vkCtx.setUiScale(m_vkCtx.isScRgbSwapchain() ? (m_paperWhiteNits / 80.0f) : 1.0f);

    m_vkCtx.initImGui(m_window, m_vkCtx.swapchainImageCount());
}
void Application::initCuda()
{
    // Force CUDA runtime initialisation on device 0
    CUDA_CHECK(cudaFree(nullptr));

    // Query device properties for the performance stats display
    cudaDeviceProp prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    m_deviceName         = prop.name;
    m_deviceComputeMajor = prop.major;
    m_deviceComputeMinor = prop.minor;
    m_deviceMemoryMB     = static_cast<std::uint64_t>(prop.totalGlobalMem)
                           / (1024ULL * 1024ULL);
}

void Application::optixLogCallback(unsigned int level,
                                   const char*  tag,
                                   const char*  message,
                                   void* /*cbdata*/)
{
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << '\n';
}

void Application::initOptix()
{
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions opts = {};
    opts.logCallbackFunction       = &Application::optixLogCallback;
    opts.logCallbackLevel          = 3; // warnings and errors

    CUcontext cuCtx = 0; // 0 = use the CUDA runtime's current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &opts, &m_optixContext));
}

void Application::initDenoiser()
{
    OptixDenoiserOptions denoiserOpts = {};
    denoiserOpts.guideNormal = 1;
    denoiserOpts.guideAlbedo = 1;
    OPTIX_CHECK(optixDenoiserCreate(
        m_optixContext,
        OPTIX_DENOISER_MODEL_KIND_HDR,
        &denoiserOpts,
        &m_denoiser));
    m_denoiserIntensity.alloc(sizeof(float));
}

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

    // ── Pipeline ──────────────────────────────────────────────────────────────
    const OptixProgramGroup pgs[] = {
        m_pgRaygen, m_pgMiss, m_pgMissShadow, m_pgHitgroup, m_pgHitgroupImplicit
    };

    OptixPipelineLinkOptions linkOpts = {};
    // Depth 1: path rays and NEE shadow rays are both called from raygen —
    // no CH/miss ever calls optixTrace, so the chain never exceeds depth 1.
    linkOpts.maxTraceDepth = 1;

    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &pipelineOpts, &linkOpts,
        pgs, 5,
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
    const OptixModule       oldModule               = m_module;
    const OptixProgramGroup oldPgRaygen             = m_pgRaygen;
    const OptixProgramGroup oldPgMiss               = m_pgMiss;
    const OptixProgramGroup oldPgMissShadow         = m_pgMissShadow;
    const OptixProgramGroup oldPgHitgroup           = m_pgHitgroup;
    const OptixProgramGroup oldPgHitgroupImplicit   = m_pgHitgroupImplicit;
    const OptixPipeline     oldPipeline             = m_pipeline;

    m_module               = nullptr;
    m_pgRaygen             = nullptr;
    m_pgMiss               = nullptr;
    m_pgMissShadow         = nullptr;
    m_pgHitgroup           = nullptr;
    m_pgHitgroupImplicit   = nullptr;
    m_pipeline             = nullptr;

    try
    {
        buildPipeline(m_ptxDir);
    }
    catch (...)
    {
        if (m_pipeline)              { optixPipelineDestroy(m_pipeline);                  m_pipeline              = nullptr; }
        if (m_pgHitgroupImplicit)    { optixProgramGroupDestroy(m_pgHitgroupImplicit);    m_pgHitgroupImplicit    = nullptr; }
        if (m_pgHitgroup)            { optixProgramGroupDestroy(m_pgHitgroup);            m_pgHitgroup            = nullptr; }
        if (m_pgMissShadow)          { optixProgramGroupDestroy(m_pgMissShadow);          m_pgMissShadow          = nullptr; }
        if (m_pgMiss)                { optixProgramGroupDestroy(m_pgMiss);                m_pgMiss                = nullptr; }
        if (m_pgRaygen)              { optixProgramGroupDestroy(m_pgRaygen);              m_pgRaygen              = nullptr; }
        if (m_module)                { optixModuleDestroy(m_module);                      m_module                = nullptr; }

        m_module               = oldModule;
        m_pgRaygen             = oldPgRaygen;
        m_pgMiss               = oldPgMiss;
        m_pgMissShadow         = oldPgMissShadow;
        m_pgHitgroup           = oldPgHitgroup;
        m_pgHitgroupImplicit   = oldPgHitgroupImplicit;
        m_pipeline             = oldPipeline;
        throw;
    }

    buildSbt();
    m_accumDirty = true;  // new shader = new result; clear accumulation

    if (oldPipeline)              { optixPipelineDestroy(oldPipeline);                }
    if (oldPgHitgroupImplicit)    { optixProgramGroupDestroy(oldPgHitgroupImplicit);  }
    if (oldPgHitgroup)            { optixProgramGroupDestroy(oldPgHitgroup);          }
    if (oldPgMissShadow)          { optixProgramGroupDestroy(oldPgMissShadow);        }
    if (oldPgMiss)                { optixProgramGroupDestroy(oldPgMiss);              }
    if (oldPgRaygen)              { optixProgramGroupDestroy(oldPgRaygen);            }
    if (oldModule)                { optixModuleDestroy(oldModule);                    }
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
    // Free any previously allocated SBT device buffers
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
    const auto& meshes   = m_scene->meshes();
    const auto& allNodes = m_scene->nodes();

    struct InstRecord
    {
        bool         isImplicit;
        int          meshIdx;      // valid when !isImplicit
        int          materialIdx;
        ImplicitType implicitType; // valid when isImplicit
    };
    std::vector<InstRecord> instList;

    std::function<void(int)> walk = [&](int nodeIdx)
    {
        const Node3D& node = *allNodes[nodeIdx];
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
                    instList.push_back({false, mi, matIdx, ImplicitType::Sphere});
                }
            }
        }
        else if (const ImplicitNode* in = dynamic_cast<const ImplicitNode*>(&node))
        {
            instList.push_back({true, 0, in->materialIndex, in->type});
        }
        for (int childIdx : node.children)
        {
            walk(childIdx);
        }
    };

    for (int rootIdx : m_scene->rootNodes())
    {
        walk(rootIdx);
    }

    std::vector<HitGroupRecord> hitRecs(instList.size());

    for (size_t i = 0; i < instList.size(); ++i)
    {
        const InstRecord& rec = instList[i];
        if (rec.isImplicit)
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitgroupImplicit, &hitRecs[i]));
            hitRecs[i].data.implicit.type          = static_cast<unsigned int>(rec.implicitType);
            hitRecs[i].data.implicit.materialIndex = rec.materialIdx;
        }
        else
        {
            OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitgroup, &hitRecs[i]));
            if (m_scene->hasAccel())
            {
                const auto ptrs                    = m_scene->meshDevicePtrs(rec.meshIdx);
                hitRecs[i].data.mesh.positions     = reinterpret_cast<const float3*>(ptrs.positions);
                hitRecs[i].data.mesh.normals       = reinterpret_cast<const float3*>(ptrs.normals);
                hitRecs[i].data.mesh.indices       = reinterpret_cast<const uint3*>(ptrs.indices);
                hitRecs[i].data.mesh.uvs           = reinterpret_cast<const float2*>(ptrs.uvs);
                hitRecs[i].data.mesh.materialIndex = rec.materialIdx;
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
}

// ─── Framebuffer ─────────────────────────────────────────────────────────────

void Application::resizeFramebuffer(int w, int h)
{
    m_vkCtx.waitIdle();
    m_vkCtx.destroyDisplayImage();

    m_viewportWidth  = w;
    m_viewportHeight = h;

    const size_t pixelCount  = static_cast<size_t>(w) * h;
    const size_t float4Bytes = pixelCount * sizeof(float4);

    m_colorBuffer.alloc(float4Bytes);
    m_colorBufferHost.resize(pixelCount);

    m_accumBuffer.alloc(float4Bytes);
    m_accumDirty = true;

    // ── Denoiser guide + working buffers ──────────────────────────────────────
    m_normalBuffer.alloc(float4Bytes);
    m_albedoBuffer.alloc(float4Bytes);
    m_hdrBuffer.alloc(float4Bytes);
    m_denoisedBuffer.alloc(float4Bytes);
    m_hdrBufferHost.resize(pixelCount);

    if (m_denoiser)
    {
        OptixDenoiserSizes sizes = {};
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(
            m_denoiser,
            static_cast<unsigned int>(w),
            static_cast<unsigned int>(h),
            &sizes));

        m_denoiserState.alloc(sizes.stateSizeInBytes);
        m_denoiserScratch.alloc(sizes.withoutOverlapScratchSizeInBytes);

        OPTIX_CHECK(optixDenoiserSetup(
            m_denoiser, nullptr,
            static_cast<unsigned int>(w),
            static_cast<unsigned int>(h),
            m_denoiserState.ptr(),   m_denoiserState.size(),
            m_denoiserScratch.ptr(), m_denoiserScratch.size()));
    }

    m_vkCtx.createDisplayImage(w, h);
}

void Application::syncFlyCameraFromNode(int nodeIdx)
{
    if (nodeIdx < 0 || nodeIdx >= static_cast<int>(m_scene->nodes().size()))
    {
        return;
    }
    if (std::string(m_scene->nodeAt(nodeIdx).typeName()) != "Camera")
    {
        return;
    }

    // Extract position, yaw, and pitch from the node's world-space transform.
    // This mirrors the one-time extraction in loadScene() so that editing a
    // CameraNode via the gizmo or TRS sliders immediately moves the camera.
    const Matrix4x4 world = m_scene->computeWorldTransform(nodeIdx);
    m_camPos = { world.m[0][3], world.m[1][3], world.m[2][3] };

    // Camera looks down its local -Z axis; column 2 is that -Z in world space.
    const float fx = -world.m[0][2];
    const float fy = -world.m[1][2];
    const float fz = -world.m[2][2];
    const float fLen = std::max(1e-6f, sqrtf(fx*fx + fy*fy + fz*fz));
    m_camPitch = asinf(std::max(-1.0f, std::min(1.0f, fy / fLen)));
    m_camYaw   = atan2f(fx / fLen, -(fz / fLen));
}

// ─── Material upload ─────────────────────────────────────────────────────────

void Application::uploadMaterials()
{
    const auto& mats = m_scene->materials();
    if (mats.empty())
    {
        return;
    }
    const size_t matBytes = mats.size() * sizeof(MaterialData);

    // Reallocate if the buffer is absent or too small; otherwise reuse.
    if (m_materialsBuffer.size() < matBytes)
    {
        m_materialsBuffer.alloc(matBytes);
    }
    m_materialsBuffer.upload(mats.data(), matBytes);
}

// ─── Scene loading ────────────────────────────────────────────────────────────

void Application::loadScene(const std::string& path)
{
    m_scene->clear();  // also resets the accel via Scene::clear()
    m_loadError.clear();
    m_selectedNodeIdx = -1;
    m_sceneFilePath.clear();

    if (loadGltfFile(path, *m_scene, m_loadError))
    {
        m_sceneFilePath = path;

        if (!m_scene->empty())
        {
            m_scene->updateAllWorldTransforms();
            try
            {
                m_scene->buildAccel(m_optixContext);
            }
            catch (const std::exception& e)
            {
                m_loadError = std::string("AS build failed: ") + e.what();
                m_scene->destroyAccel();
            }
        }
    }
    else
    {
        m_scene->clear();  // discard any partial data from a failed load
    }

    // Upload materials to device so the closest-hit shader can look up properties.
    m_materialsBuffer.free();  // force realloc to match new material count
    uploadMaterials();
    m_scene->uploadTextures();  // upload glTF textures to GPU and build device array

    m_accumDirty = true;  // scene changed — clear accumulated samples
    buildSbt();  // rebuild with new mesh count (0 if load failed or no geometry)

    // Sync fly-camera state from the newly loaded (or default) scene camera so
    // movement immediately continues from the correct position and orientation.
    {
        const Camera& cam = m_scene->camera();
        m_camPos.x = cam.transform.m[0][3];
        m_camPos.y = cam.transform.m[1][3];
        m_camPos.z = cam.transform.m[2][3];

        // Forward = -column2 of the camera-to-world matrix
        const float fx = -cam.transform.m[0][2];
        const float fy = -cam.transform.m[1][2];
        const float fz = -cam.transform.m[2][2];
        const float fLen = std::max(1e-6f, sqrtf(fx*fx + fy*fy + fz*fz));
        m_camPitch = asinf(std::max(-1.0f, std::min(1.0f, fy / fLen)));
        m_camYaw   = atan2f(fx / fLen, -(fz / fLen));
    }
}

// ─── Texture loading ─────────────────────────────────────────────────────────

void Application::loadTexture(const std::string& path)
{
    Texture tex;
    tex.name = std::filesystem::path(path).filename().string();
    std::string err;

    // Dispatch on file extension.
    const std::string ext = std::filesystem::path(path).extension().string();
    const bool ok = (ext == ".exr" || ext == ".EXR")
        ? tex.loadEXR(path, err)
        : (ext == ".hdr" || ext == ".HDR")
            ? tex.loadHDR(path, err)
            : tex.loadImage(path, err);

    if (!ok)
    {
        m_loadError = "Texture load failed: " + err;
        return;
    }

    tex.uploadToGpu();
    m_scene->addTexture(std::move(tex));
    m_scene->uploadTextures();  // rebuild the device texture-object array
    m_accumDirty = true;
}

// ─── Environment map ─────────────────────────────────────────────────────────

void Application::loadEnvMap(const std::string& path)
{
    m_envMapError.clear();
    m_envMap.free();  // release old GPU resources before loading new map
    m_envMapPath.clear();

    const std::string ext = std::filesystem::path(path).extension().string();
    const bool isHdr = (ext == ".hdr" || ext == ".HDR");

    const bool ok = isHdr
        ? m_envMap.loadHDR(path, m_envMapError)
        : m_envMap.loadEXR(path, m_envMapError);

    if (ok)
    {
        m_envMap.uploadToGpu();
        m_envMap.buildCdf();
        m_envMapPath = std::filesystem::path(path).filename().string();
        m_accumDirty = true;  // new env map = new lighting; clear accumulated samples
    }
}


// ─── Camera controller ───────────────────────────────────────────────────────

void Application::updateCamera()
{
    ImGuiIO& io = ImGui::GetIO();

    // ── Mouse delta ───────────────────────────────────────────────────────────
    double mouseX, mouseY;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);
    const float dx = static_cast<float>(mouseX - m_prevMouseX);
    const float dy = static_cast<float>(mouseY - m_prevMouseY);
    m_prevMouseX = mouseX;
    m_prevMouseY = mouseY;

    const bool rmb          = glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool rmbFirstFrame = rmb && !m_prevRmb;  // true only on the press event
    m_prevRmb = rmb;

    // ── Rotation / Orbit — right-drag while the Viewport panel is under the cursor
    // Ctrl+RMB: orbit the camera position around the world origin at a fixed radius.
    //           Camera orientation (yaw/pitch) is left unchanged — no snap.
    // Plain RMB: free-look (rotate orientation in place, position fixed).
    // rmbFirstFrame is skipped to avoid a position-jump on the first drag frame.
    if (rmb && !rmbFirstFrame && m_viewportHovered)
    {
        const bool shiftHeld = glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS
                            || glfwGetKey(m_window, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS;
        const bool ctrlHeld  = glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS
                            || glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;

        if (shiftHeld)
        {
            // Rotate environment map azimuthally — same sensitivity as camera rotation
            m_envMapRotation += dx * m_rotSpeed;
            m_accumDirty = true;
        }
        else if (ctrlHeld)
        {
            // Orbit: move the position along a sphere of the same radius and
            // derive yaw/pitch from the new position so the camera always looks
            // at the origin.  Rotation is driven by the mouse delta on the
            // position — not imposed directly on angles — so there is no snap.
            const float r = sqrtf(m_camPos.x*m_camPos.x
                                + m_camPos.y*m_camPos.y
                                + m_camPos.z*m_camPos.z);
            if (r > 1e-4f)
            {
                float azimuth   = atan2f(m_camPos.x, m_camPos.z);
                float elevation = asinf(std::max(-1.0f, std::min(1.0f, m_camPos.y / r)));

                azimuth   -= dx * m_rotSpeed;
                elevation += dy * m_rotSpeed;

                const float kPoleLimit = 1.5533430f;  // 89°
                elevation = std::max(-kPoleLimit, std::min(kPoleLimit, elevation));

                m_camPos.x = r * cosf(elevation) * sinf(azimuth);
                m_camPos.y = r * sinf(elevation);
                m_camPos.z = r * cosf(elevation) * cosf(azimuth);

                // Apply the equivalent delta to yaw/pitch (yaw = -azimuth,
                // pitch = -elevation) so they track the orbit continuously
                // without snapping to an absolute value.
                m_camYaw   += dx * m_rotSpeed;
                m_camPitch -= dy * m_rotSpeed;
                m_camPitch  = std::max(-kPoleLimit, std::min(kPoleLimit, m_camPitch));
            }
        }
        else
        {
            // Free-look: update view direction, position stays fixed.
            m_camYaw   += dx * m_rotSpeed;
            m_camPitch -= dy * m_rotSpeed;

            // Clamp pitch to just under ±90° to avoid gimbal singularity
            const float kPitchLimit = 1.5533430f;  // 89 degrees in radians
            m_camPitch = std::max(-kPitchLimit, std::min(kPitchLimit, m_camPitch));
        }
    }

    // ── Translation — WASD in camera space ───────────────────────────────────
    if (!io.WantCaptureKeyboard)
    {
        // dt clamped so an initial stall frame doesn't teleport the camera
        const float dt   = std::max(0.001f, std::min(m_frameTimeMs * 0.001f, 0.1f));
        const float dist = m_moveSpeed * dt;

        const float sy = sinf(m_camYaw),   cy = cosf(m_camYaw);
        const float sp = sinf(m_camPitch),  cp = cosf(m_camPitch);

        // forward = direction the camera looks, right = camera's local +X, up = camera's local +Y
        const float3 forward = {  sy*cp,  sp, -cy*cp };
        const float3 right   = {  cy,    0.0f,  sy    };
        const float3 up      = { -sy*sp,  cp,  cy*sp };

        const bool wKey = glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS;
        const bool sKey = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
        const bool aKey = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS;
        const bool dKey = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS;
        const bool eKey = glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS;
        const bool qKey = glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS;

        if (wKey || sKey)
        {
            const float fwd = wKey ? dist : -dist;
            m_camPos.x += forward.x * fwd;
            m_camPos.y += forward.y * fwd;
            m_camPos.z += forward.z * fwd;
        }
        if (aKey || dKey)
        {
            const float strafe = dKey ? dist : -dist;
            m_camPos.x += right.x * strafe;
            m_camPos.z += right.z * strafe;
        }
        if (eKey || qKey)
        {
            const float lift = eKey ? dist : -dist;
            m_camPos.x += up.x * lift;
            m_camPos.y += up.y * lift;
            m_camPos.z += up.z * lift;
        }
    }

    // ── Rebuild camera-to-world matrix ────────────────────────────────────────
    // Row-major Matrix4x4, columns are world-space camera axes:
    //   col 0 = right   = {cy,      0,     sy    }
    //   col 1 = up      = {-sy*sp,  cp,    cy*sp }
    //   col 2 = +Z cam  = {-sy*cp, -sp,    cy*cp } (camera looks down -Z)
    //   col 3 = pos
    const float sy = sinf(m_camYaw),   cy = cosf(m_camYaw);
    const float sp = sinf(m_camPitch),  cp = cosf(m_camPitch);

    Camera cam = m_scene->camera();
    cam.transform.m[0][0] =  cy;  cam.transform.m[0][1] = -sy*sp; cam.transform.m[0][2] = -sy*cp; cam.transform.m[0][3] = m_camPos.x;
    cam.transform.m[1][0] = 0.0f; cam.transform.m[1][1] =  cp;    cam.transform.m[1][2] = -sp;    cam.transform.m[1][3] = m_camPos.y;
    cam.transform.m[2][0] =  sy;  cam.transform.m[2][1] =  cy*sp; cam.transform.m[2][2] =  cy*cp; cam.transform.m[2][3] = m_camPos.z;
    cam.transform.m[3][0] = 0.0f; cam.transform.m[3][1] =  0.0f;  cam.transform.m[3][2] =  0.0f;  cam.transform.m[3][3] = 1.0f;
    m_scene->setCamera(std::move(cam));
}


// ─── Per-frame ────────────────────────────────────────────────────────────────

bool Application::tick()
{
    if (glfwWindowShouldClose(m_window))
    {
        return false;
    }

    // ── Shader hot-reload ─────────────────────────────────────────────────────
    checkShaderHotReload();

    // ── Camera-change detection ────────────────────────────────────────────────
    // Save camera state before updateCamera() so we can detect any change.
    const float3 prevPos   = m_camPos;
    const float  prevYaw   = m_camYaw;
    const float  prevPitch = m_camPitch;

    // ── Frame timing ──────────────────────────────────────────────────────────
    {
        const auto now = std::chrono::steady_clock::now();
        if (m_frameTimeMs > 0.0f)
        {
            const float deltaMs = std::chrono::duration<float, std::milli>(
                now - m_frameStart).count();
            // Exponential moving average — α=0.02 ≈ 50-frame window for stable stats
            m_frameTimeMs = 0.02f * deltaMs + 0.98f * m_frameTimeMs;
        }
        else if (m_frameStart != std::chrono::steady_clock::time_point{})
        {
            // Second frame: initialise with the first real measurement
            m_frameTimeMs = std::chrono::duration<float, std::milli>(
                now - m_frameStart).count();
        }
        m_frameStart = now;
    }

    glfwPollEvents();

    if (m_swapchainResizePending)
    {
        int fw, fh;
        glfwGetFramebufferSize(m_window, &fw, &fh);
        if (fw > 0 && fh > 0)
        {
            m_vkCtx.waitIdle();
            m_vkCtx.recreateSwapchain(fw, fh);
        }
        m_swapchainResizePending = false;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();  // must be called once per frame, right after ImGui::NewFrame()
    //ImGui::ShowDemoWindow();
    // Camera input is processed here — after NewFrame() so WantCaptureMouse /
    // WantCaptureKeyboard are current, but before the GPU launch so this frame
    // renders with the updated camera.
    updateCamera();

    if (m_camPos.x != prevPos.x || m_camPos.y != prevPos.y || m_camPos.z != prevPos.z ||
        m_camYaw   != prevYaw   || m_camPitch  != prevPitch)
    {
        m_accumDirty = true;
    }

    // Enable a full-window dockspace so panels can be docked to it
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);

    // ── Viewport panel ────────────────────────────────────────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();

    // Record hover state for use in updateCamera() next frame
    m_viewportHovered = ImGui::IsWindowHovered();

    {
        const ImVec2 regionSize = ImGui::GetContentRegionAvail();
        const int vpW = std::max(1, static_cast<int>(regionSize.x));
        const int vpH = std::max(1, static_cast<int>(regionSize.y));

        if (vpW != m_viewportWidth || vpH != m_viewportHeight)
        {
            resizeFramebuffer(vpW, vpH);
        }

        // ── Update launch parameters ──────────────────────────────────────────
        // ── Reset accumulation buffer if anything changed ─────────────────────
        if (m_accumDirty && m_accumBuffer.valid())
        {
            m_accumBuffer.clear();
            m_sampleCount             = 0;
            m_accumDirty              = false;
            m_hasValidDenoisedFrame   = false;  // stale denoised frame is now invalid
        }

        m_launchParams.colorBuffer = m_colorBuffer.typedPtr<float4>();
        m_launchParams.hdrDisplay  = m_vkCtx.isScRgbSwapchain()
                                         ? (m_hdrOutput ? 1 : 0)
                                         : 2;
        m_launchParams.fbSize            = make_uint2(static_cast<unsigned int>(m_viewportWidth), static_cast<unsigned int>(m_viewportHeight));
        m_launchParams.traversable       = m_scene->traversable();
        m_launchParams.envMap            = m_envMap.gpuTex;
        m_launchParams.envMapRotation    = m_envMapRotation;
        m_launchParams.envExposure       = m_envExposure;
        m_launchParams.envMarginalCdf    = m_envMap.cdfMarginal    ? reinterpret_cast<const float*>(m_envMap.cdfMarginal)    : nullptr;
        m_launchParams.envConditionalCdf = m_envMap.cdfConditional ? reinterpret_cast<const float*>(m_envMap.cdfConditional) : nullptr;
        m_launchParams.envCdfW           = m_envMap.width;
        m_launchParams.envCdfH           = m_envMap.height;
        m_launchParams.accumBuffer       = m_accumBuffer.typedPtr<float4>();
        m_launchParams.sampleIndex       = m_sampleCount;
        m_launchParams.materials         = m_materialsBuffer.typedPtr<const MaterialData>();
        m_launchParams.sceneTextures     = m_scene->textureObjects();
        m_launchParams.normalBuffer      = m_normalBuffer.typedPtr<float4>();
        m_launchParams.albedoBuffer      = m_albedoBuffer.typedPtr<float4>();
        m_launchParams.hdrBuffer         = m_hdrBuffer.typedPtr<float4>();

        // Camera basis vectors derived from the scene camera each frame
        {
            const Camera&    cam = m_scene->camera();
            const Matrix4x4& t   = cam.transform;

            // Camera-to-world columns: right=col0, up=col1, +Z=col2, eye=col3
            m_launchParams.eye   = make_float3(t.m[0][3], t.m[1][3], t.m[2][3]);
            const float3 right   = make_float3(t.m[0][0], t.m[1][0], t.m[2][0]);
            const float3 up      = make_float3(t.m[0][1], t.m[1][1], t.m[2][1]);

            // Camera looks down -Z, so forward = -column2
            const float3 forward = make_float3(-t.m[0][2], -t.m[1][2], -t.m[2][2]);

            // FOV from physical lens + sensor parameters
            const float aspect       = static_cast<float>(m_viewportWidth) / static_cast<float>(m_viewportHeight);
            const float sensorHeight = cam.sensorSize / aspect;  // mm
            const float tanHalfFovV  = sensorHeight / (2.0f * cam.focalLength);
            const float tanHalfFovH  = tanHalfFovV * aspect;

            m_launchParams.U =
                make_float3(right.x * tanHalfFovH,
                            right.y * tanHalfFovH,
                            right.z * tanHalfFovH);
            m_launchParams.V =
                make_float3(up.x * tanHalfFovV,
                            up.y * tanHalfFovV,
                            up.z * tanHalfFovV);
            m_launchParams.W = forward;

            // Thin-lens DoF parameters.
            // lensRadius = 0 → pinhole (no DoF); overridden to 0 when DoF is disabled.
            m_launchParams.lensRadius    = cam.dofEnabled
                ? (cam.focalLength / (2.0f * cam.fStop)) / 1000.0f
                : 0.0f;
            m_launchParams.focusDistance = cam.focusDistance;
            m_launchParams.bokehEdgeBias = cam.bokehEdgeBias;
        }

        // ── GPU launch ────────────────────────────────────────────────────────
        m_launchParamsBuffer.upload(&m_launchParams, sizeof(LaunchParams));

        OPTIX_CHECK(optixLaunch(
            m_pipeline, nullptr,
            m_launchParamsBuffer.ptr(), sizeof(LaunchParams),
            &m_sbt,
            static_cast<unsigned int>(m_viewportWidth),
            static_cast<unsigned int>(m_viewportHeight),
            1));

        CUDA_CHECK(cudaDeviceSynchronize());
        ++m_sampleCount;

        // ── Denoiser post-process ─────────────────────────────────────────────
        const bool runDenoiser = m_denoiserEnabled
                              && m_denoiser
                              && m_hdrBuffer.valid()
                              && m_denoiserState.valid()
                              && m_denoiserInterval > 0
                              && (m_sampleCount % m_denoiserInterval == 0 || m_sampleCount == 25);
        if (runDenoiser)
        {
            const auto makeImage = [&](CUdeviceptr ptr) -> OptixImage2D
            {
                OptixImage2D img          = {};
                img.data                  = ptr;
                img.width                 = static_cast<unsigned int>(m_viewportWidth);
                img.height                = static_cast<unsigned int>(m_viewportHeight);
                img.rowStrideInBytes      = img.width * static_cast<unsigned int>(sizeof(float4));
                img.pixelStrideInBytes    = sizeof(float4);
                img.format                = OPTIX_PIXEL_FORMAT_FLOAT4;
                return img;
            };

            OptixDenoiserLayer layer = {};
            layer.input              = makeImage(m_hdrBuffer.ptr());
            layer.output             = makeImage(m_denoisedBuffer.ptr());

            OptixDenoiserGuideLayer guide = {};
            guide.normal                  = makeImage(m_normalBuffer.ptr());
            guide.albedo                  = makeImage(m_albedoBuffer.ptr());

            OPTIX_CHECK(optixDenoiserComputeIntensity(
                m_denoiser, nullptr,
                &layer.input,
                m_denoiserIntensity.ptr(),
                m_denoiserScratch.ptr(), m_denoiserScratch.size()));

            OptixDenoiserParams denoiserParams = {};
            denoiserParams.hdrIntensity        = m_denoiserIntensity.ptr();
            denoiserParams.blendFactor         = 0.0f;

            OPTIX_CHECK(optixDenoiserInvoke(
                m_denoiser, nullptr,
                &denoiserParams,
                m_denoiserState.ptr(),   m_denoiserState.size(),
                &guide,
                &layer, 1,
                0, 0,
                m_denoiserScratch.ptr(), m_denoiserScratch.size()));

            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy denoised float4 to host and CPU-tone-map into m_colorBufferHost
            const size_t pixelCount = static_cast<size_t>(m_viewportWidth) * m_viewportHeight;
            m_denoisedBuffer.download(m_hdrBufferHost.data(), pixelCount * sizeof(float4));

            // Mirror the raygen encode (hdrDisplay modes 0/1/2).
            const bool scRgb      = m_vkCtx.isScRgbSwapchain();
            const bool applyGamma = !scRgb;
            for (size_t i = 0; i < pixelCount; ++i)
            {
                float r = m_hdrBufferHost[i].x;
                float g = m_hdrBufferHost[i].y;
                float b = m_hdrBufferHost[i].z;
                if (!m_hdrOutput || !scRgb)
                {
                    r = r / (r + 1.0f);
                    g = g / (g + 1.0f);
                    b = b / (b + 1.0f);
                    if (applyGamma)
                    {
                        r = std::pow(r, 1.0f / 2.2f);
                        g = std::pow(g, 1.0f / 2.2f);
                        b = std::pow(b, 1.0f / 2.2f);
                    }
                }
                m_colorBufferHost[i] = make_float4(r, g, b, 1.0f);
            }
            m_hasValidDenoisedFrame = true;
        }
        else if (!m_denoiserEnabled || !m_hasValidDenoisedFrame)
        {
            // Show the live raygen output when:
            //  • denoiser is off, OR
            //  • denoiser is on but hasn't fired yet since the last accum reset
            //    (e.g. camera just moved) — keeps the viewport responsive.
            const size_t pixelCount = static_cast<size_t>(m_viewportWidth) * m_viewportHeight;
            m_colorBuffer.download(m_colorBufferHost.data(), pixelCount * sizeof(float4));
        }
        // else: denoiser enabled, valid denoised frame exists, but interval not reached —
        //        the host colour buffer already holds the last denoised result; leave it untouched.

        // Copy rendered pixels into the persistently-mapped staging buffer.
        // The GPU reads from this in the transfer pass recorded later this frame.
        if (m_vkCtx.displayStagingPtr() && !m_colorBufferHost.empty())
        {
            const size_t pixelCount = static_cast<size_t>(m_viewportWidth) * m_viewportHeight;
            std::memcpy(m_vkCtx.displayStagingPtr(), m_colorBufferHost.data(), pixelCount * sizeof(float4));
        }

        const ImVec2 imageScreenPos = ImGui::GetCursorScreenPos();
        if (m_vkCtx.displayDescSet() != VK_NULL_HANDLE)
        {
            // UV0=(0,1) UV1=(1,0): flip vertically — CUDA/OptiX row-0 is at the
            // top of the buffer; Vulkan/ImGui expect row-0 at the bottom of V.
            ImGui::Image(
                (ImTextureID)m_vkCtx.displayDescSet(),
                ImVec2(static_cast<float>(m_viewportWidth),
                       static_cast<float>(m_viewportHeight)),
                ImVec2(0.0f, 1.0f),
                ImVec2(1.0f, 0.0f));
        }
        else
        {
            ImGui::Dummy(ImVec2(static_cast<float>(m_viewportWidth),
                                static_cast<float>(m_viewportHeight)));
        }

        // ── 3D gizmo overlay ──────────────────────────────────────────────────
        // Render the ImGuizmo gizmo on top of the viewport image for the
        // currently selected scene-graph node.
        // Gizmo operation keyboard shortcuts — active whenever a node is selected
        // and ImGui is not consuming the keyboard for text input.
        // 1 = Scale  |  2 = Rotate  |  3 = Translate
        if (!ImGui::GetIO().WantTextInput)
        {
            if (ImGui::IsKeyPressed(ImGuiKey_1))
            {
                m_gizmoOp = ImGuizmo::SCALE;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_2))
            {
                m_gizmoOp = ImGuizmo::ROTATE;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_3))
            {
                m_gizmoOp = ImGuizmo::TRANSLATE;
            }
        }

        if (m_selectedNodeIdx >= 0
            && m_selectedNodeIdx < static_cast<int>(m_scene->nodes().size()))
        {
            const float vpW = static_cast<float>(m_viewportWidth);
            const float vpH = static_cast<float>(m_viewportHeight);

            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist();
            ImGuizmo::SetRect(imageScreenPos.x, imageScreenPos.y, vpW, vpH);

            // ── View matrix (column-major) ────────────────────────────────────
            const float3 eye = m_launchParams.eye;
            const float3 U   = m_launchParams.U;
            const float3 V   = m_launchParams.V;
            const float3 W   = m_launchParams.W;

            // Normalise U and V to recover unit right/up directions
            auto len3  = [](float3 v){ return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); };
            auto dot3  = [](float3 a, float3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; };
            const float3 R  = { U.x / len3(U), U.y / len3(U), U.z / len3(U) };
            const float3 Up = { V.x / len3(V), V.y / len3(V), V.z / len3(V) };

            const float view[16] = {
                R.x,         Up.x,        -W.x,       0.f,
                R.y,         Up.y,        -W.y,       0.f,
                R.z,         Up.z,        -W.z,       0.f,
                -dot3(R, eye), -dot3(Up, eye), dot3(W, eye), 1.f
            };

            // ── Projection matrix (column-major) ──────────────────────────────
            const float tanHalfFovV = len3(V);
            const float f           = 1.0f / tanHalfFovV;
            const float aspect      = vpW / vpH;
            const float zNear = 0.01f, zFar = 10000.0f;
            const float proj[16] = {
                f / aspect, 0.f, 0.f,  0.f,
                0.f,        f,   0.f,  0.f,
                0.f, 0.f, (zFar + zNear) / (zNear - zFar), -1.f,
                0.f, 0.f, 2.0f * zFar * zNear / (zNear - zFar), 0.f
            };

            // ── World transform → column-major float[16] ─────────────────────
            Matrix4x4 worldTx = m_scene->computeWorldTransform(m_selectedNodeIdx);
            float gizmoMatrix[16];
            mat4ToColMajor(worldTx, gizmoMatrix);

            ImGuizmo::Manipulate(view, proj, m_gizmoOp, m_gizmoMode, gizmoMatrix);

            if (ImGuizmo::IsUsing())
            {
                // Convert result back to row-major world transform
                Matrix4x4 newWorldTx = mat4FromColMajor(gizmoMatrix);

                // Compute the new local transform relative to the parent
                Node3D& node = m_scene->nodeAt(m_selectedNodeIdx);
                if (node.parent >= 0)
                {
                    Matrix4x4 parentWorld = m_scene->computeWorldTransform(node.parent);
                    node.localTransform   = mat4Multiply(mat4Inverse(parentWorld), newWorldTx);
                }
                else
                {
                    node.localTransform = newWorldTx;
                }

                m_scene->updateWorldTransforms(m_selectedNodeIdx);
                m_scene->rebuildTlas(m_optixContext);
                m_accumDirty = true;
                syncFlyCameraFromNode(m_selectedNodeIdx);
            }
        }
    }

    ImGui::End();

    drawRaytracerPanel();
    drawResourcesPanel();
    drawSceneGraphPanel();
    drawNodePropertiesPanel();

    // ── HDRI Browser window (always visible, no close button) ────────────────
    {
        std::string selected;
        if (m_hdriBrowser.draw(nullptr, selected) && !selected.empty())
        {
            loadEnvMap(selected);
            m_hdriBrowser.setActivePath(selected);
        }
    }

    ImGui::Render();

    // Upload any newly-ready thumbnails to the GPU.  Must happen after
    // ImGui::Render() (so ImGui won't reference new textures this frame) and
    // before beginFrame() (so no frame is in-flight during vkQueueWaitIdle).
    m_hdriBrowser.uploadPending(m_vkCtx);

    // ── Vulkan present ─────────────────────────────────────────────────────────

    VulkanFrameContext frame = m_vkCtx.beginFrame();
    if (!frame.valid)
    {
        return true;  // swapchain was out of date; rebuilt, skip this frame
    }

    m_vkCtx.uploadDisplayImage(frame.cmd, m_colorBufferHost.data(),
                               m_viewportWidth, m_viewportHeight);
    m_vkCtx.beginRenderPass(frame.cmd);
    // On an scRGB swapchain uiPipeline() overrides the stock ImGui pipeline
    // with one that converts UI colours sRGB → linear × paper white;
    // VK_NULL_HANDLE (BGRA8 fallback) selects the stock pipeline.
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.cmd, m_vkCtx.uiPipeline());
    m_vkCtx.endFrameAndPresent(frame, 0, 0);

    return true;
}

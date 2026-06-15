#ifndef OPTIX_RAYTRACER_APPLICATION_H
#define OPTIX_RAYTRACER_APPLICATION_H

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "accel.h"
#include "gpu_buffer.h"
#include "hdri_browser.h"
#include "launch_params.h"
#include "scene.h"
#include "texture.h"
#include "vulkan_context.h"

#include <imgui.h>       // must precede ImGuizmo.h — it relies on ImGui types
#include <ImGuizmo.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class Application
{
public:
    // ptxDir: directory containing the compiled .ptx shader files (usually the
    // same directory as the executable — pass std::filesystem::path(argv[0]).parent_path()).
    Application(int width, int height, const std::string& title,
                const std::string& ptxDir = ".");
    ~Application();

    // Returns false when the window should close.
    bool tick();

private:
    GLFWwindow*        m_window        = nullptr;
    int                m_width         = 0;
    int                m_height        = 0;

    OptixDeviceContext m_optixContext   = nullptr;

    std::unique_ptr<Scene> m_scene;  // owns geometry, materials, nodes, and the Accel

    // Framebuffer — CUDA device/host buffers for the rendered image (linear float4, scRGB).
    GPUBuffer           m_colorBuffer;                  // CUDA device buffer
    std::vector<float4> m_colorBufferHost;              // host staging buffer
    int                 m_viewportWidth  = 0;           // current framebuffer dimensions
    int                 m_viewportHeight = 0;           // driven by the Viewport panel size

    // HDR display output — pure shader switch: when off the raygen tone-maps
    // to the SDR range (Reinhard), when on it passes radiance through unclamped.
    bool  m_hdrOutput      = false;
    float m_paperWhiteNits = 250.0f;  // UI + image brightness (scRGB: 1.0 = 80 nits)

    // Vulkan presentation context (owns swapchain, render pass, display image, etc.)
    VulkanContext m_vkCtx;

    // HDRI browser panel — async thumbnail grid for quick environment switching
    HdriBrowser m_hdriBrowser;

    void initWindow(const std::string& title);
    void initImGui();
    void initCuda();
    void initOptix();
    void buildPipeline(const std::string& ptxDir);
    void buildSbt();
    void resizeFramebuffer(int w, int h);

    // Hot-reload: rebuild the pipeline from the PTX file whenever it changes on disk.
    void reloadPipeline();
    void checkShaderHotReload();

    void loadScene(const std::string& path);
    void loadEnvMap(const std::string& path);
    void loadTexture(const std::string& path);
    void uploadMaterials();
    // If nodeIdx is a CameraNode, extract its world-space transform into the
    // fly-camera state so the next updateCamera() renders from the new position.
    void syncFlyCameraFromNode(int nodeIdx);
    void initDenoiser();

    static void optixLogCallback(unsigned int level,
                                 const char*  tag,
                                 const char*  message,
                                 void*        cbdata);

    // OptiX pipeline
    OptixModule       m_module        = nullptr;
    OptixProgramGroup m_pgRaygen          = nullptr;
    OptixProgramGroup m_pgMiss            = nullptr;
    OptixProgramGroup m_pgMissShadow      = nullptr;
    OptixProgramGroup m_pgHitgroup        = nullptr;  // triangle meshes
    OptixProgramGroup m_pgHitgroupImplicit = nullptr; // analytic implicit shapes
    OptixPipeline     m_pipeline          = nullptr;

    // Shader binding table
    GPUBuffer               m_sbtRaygenBuffer;
    GPUBuffer               m_sbtMissBuffer;
    GPUBuffer               m_sbtHitgroupBuffer;
    OptixShaderBindingTable m_sbt = {};

    // Scene materials on device
    GPUBuffer m_materialsBuffer;

    // Sample accumulation
    GPUBuffer m_accumBuffer;
    uint32_t  m_sampleCount = 0;
    bool      m_accumDirty  = true;

    // OptiX AI denoiser
    OptixDenoiser       m_denoiser          = nullptr;
    GPUBuffer           m_denoiserState;
    GPUBuffer           m_denoiserScratch;
    GPUBuffer           m_denoiserIntensity;
    GPUBuffer           m_normalBuffer;
    GPUBuffer           m_albedoBuffer;
    GPUBuffer           m_hdrBuffer;
    GPUBuffer           m_denoisedBuffer;
    std::vector<float4> m_hdrBufferHost;
    bool                m_denoiserEnabled       = false;
    int                 m_denoiserInterval      = 50;
    bool                m_hasValidDenoisedFrame = false;

    // Launch parameters
    LaunchParams m_launchParams       = {};
    GPUBuffer    m_launchParamsBuffer;

    std::string m_sceneFilePath;
    std::string m_loadError;

    // Environment map
    Texture     m_envMap;
    std::string m_envMapPath;
    std::string m_envMapError;
    float       m_envMapRotation = 0.0f;
    float       m_envExposure   = 0.0f;

    // Hot-reload state
    std::string                      m_ptxDir;
    std::filesystem::file_time_type  m_ptxWriteTime = {};
    std::string                      m_shaderError;

    // GPU device info
    std::string   m_deviceName;
    int           m_deviceComputeMajor = 0;
    int           m_deviceComputeMinor = 0;
    std::uint64_t m_deviceMemoryMB     = 0;

    // Frame timing
    std::chrono::steady_clock::time_point m_frameStart;
    float m_frameTimeMs = 0.0f;

    // Free-fly camera
    float3 m_camPos    = { 0.0f, 0.0f, 3.0f };
    float  m_camYaw    = 0.0f;
    float  m_camPitch  = 0.0f;
    float  m_moveSpeed = 5.0f;
    float  m_rotSpeed  = 0.003f;

    // Set by the GLFW framebuffer-size callback; consumed at the top of tick().
    bool m_swapchainResizePending = false;

    // Input state
    double m_prevMouseX      = 0.0;
    double m_prevMouseY      = 0.0;
    bool   m_prevRmb         = false;
    bool   m_viewportHovered  = false;
    int    m_selectedNodeIdx  = -1;

    // 3D gizmo
    ImGuizmo::OPERATION m_gizmoOp   = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE      m_gizmoMode = ImGuizmo::LOCAL;

    void updateCamera();
};

#endif // OPTIX_RAYTRACER_APPLICATION_H

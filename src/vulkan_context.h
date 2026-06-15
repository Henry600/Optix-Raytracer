#ifndef OPTIX_RAYTRACER_VULKAN_CONTEXT_H
#define OPTIX_RAYTRACER_VULKAN_CONTEXT_H

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdint>
#include <vector>

// RAII handle for a GPU texture registered with the ImGui Vulkan backend.
// Created by VulkanContext::createImGuiTexture(); destroyed by destroyImGuiTexture().
struct ImGuiTexture
{
    VkImage         image   = VK_NULL_HANDLE;
    VkDeviceMemory  memory  = VK_NULL_HANDLE;
    VkImageView     view    = VK_NULL_HANDLE;
    VkSampler       sampler = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    bool valid() const { return descSet != VK_NULL_HANDLE; }
};

// Returned by beginFrame() and consumed by endFrameAndPresent().
struct VulkanFrameContext
{
    uint32_t        imageIndex = UINT32_MAX;
    VkCommandBuffer cmd        = VK_NULL_HANDLE;
    bool            valid      = false;  // false = swapchain out of date; caller skips render
};

// Owns all Vulkan resources: instance, device, swapchain, render pass,
// command buffers, sync objects, and the display image used to show the
// CUDA-rendered result through ImGui.
class VulkanContext
{
public:
    VulkanContext()  = default;
    ~VulkanContext() { cleanup(); }

    VulkanContext(const VulkanContext&)            = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void init(GLFWwindow* window, int width, int height);
    void initImGui(GLFWwindow* window, int imageCount);
    void cleanup();
    void waitIdle() const;

    // ── Swapchain ─────────────────────────────────────────────────────────────
    void createSwapchain(int w, int h);
    void destroySwapchain();
    void recreateSwapchain(int w, int h);

    // ── scRGB presentation ────────────────────────────────────────────────────
    // The swapchain uses R16G16B16A16_SFLOAT in all cases.  When an HDR
    // display is active, the color space is EXTENDED_SRGB_LINEAR (true scRGB:
    // values > 1.0 appear above paper white).  Without an HDR display the
    // driver does not enumerate that color space, so the fallback is
    // R16G16B16A16_SFLOAT + SRGB_NONLINEAR (FP16 precision, but values above
    // 1.0 clip).  isScRgbSwapchain() distinguishes the two cases.

    // True when the swapchain uses EXTENDED_SRGB_LINEAR (HDR presentation
    // active).  False = FP16 swapchain present but HDR not available on the
    // current display.
    bool isScRgbSwapchain() const { return m_scRgbSwapchain; }

    // Pipeline to pass to ImGui_ImplVulkan_RenderDrawData.  Returns the custom
    // scRGB-linearising pipeline when scRGB is active; VK_NULL_HANDLE when the
    // fallback FP16+SRGB_NONLINEAR path is used (ImGui uses its default then).
    VkPipeline uiPipeline() const { return m_scRgbSwapchain ? m_uiPipeline : VK_NULL_HANDLE; }

    // Paper-white scale (paperWhiteNits / 80) for the scRGB UI pipeline and
    // the render-pass clear colour.  Recreates the UI pipeline (the scale is
    // baked in as a specialization constant) — call between frames.
    void setUiScale(float scale);

    // ── Display image (CUDA → Vulkan pixel upload) ────────────────────────────
    void createDisplayImage(int w, int h);
    void destroyDisplayImage();

    // ── Per-frame rendering ───────────────────────────────────────────────────

    // Wait for the previous frame's fence, acquire the next swapchain image, and
    // begin recording the command buffer.  Returns valid=false if the swapchain
    // needs to be rebuilt; the caller should skip rendering for that frame.
    VulkanFrameContext beginFrame();

    // Copy pixel data from CPU memory into the display Vulkan image via a
    // buffer-to-image transfer recorded into cmd (before the render pass).
    void uploadDisplayImage(VkCommandBuffer cmd, const void* pixels, int w, int h);

    // Begin the main render pass (clears to the background colour).
    void beginRenderPass(VkCommandBuffer cmd);

    // End the render pass, end the command buffer, submit, and present.
    // Handles VK_SUBOPTIMAL and VK_ERROR_OUT_OF_DATE automatically.
    void endFrameAndPresent(VulkanFrameContext& frame, int windowW, int windowH);

    // ── Generic ImGui textures (thumbnails, previews, …) ─────────────────────
    // Upload an RGBA16F linear image as an ImGui-registered sampled texture.
    // Uses a synchronous one-shot command submit — call outside an active frame
    // (i.e. before beginFrame() or after endFrameAndPresent()).
    ImGuiTexture createImGuiTexture(int w, int h, const uint16_t* rgbaF16);

    // Free all Vulkan resources owned by tex and un-register it from ImGui.
    // Safe to call on a default-constructed (invalid) texture; zeroes tex on return.
    void destroyImGuiTexture(ImGuiTexture& tex);

    // ── Utilities ─────────────────────────────────────────────────────────────
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const;

    // ── Accessors ─────────────────────────────────────────────────────────────
    bool             isValid()           const { return m_device != VK_NULL_HANDLE; }
    VkDevice         device()            const { return m_device; }
    VkInstance       instance()          const { return m_instance; }
    VkPhysicalDevice physDevice()        const { return m_physDevice; }
    uint32_t         queueFamily()       const { return m_queueFamily; }
    VkQueue          queue()             const { return m_queue; }
    VkRenderPass     renderPass()        const { return m_renderPass; }
    VkDescriptorPool imguiDescPool()     const { return m_imguiDescPool; }
    int              swapchainImageCount() const { return static_cast<int>(m_scImages.size()); }
    VkDescriptorSet  displayDescSet()    const { return m_displayDescSet; }
    void*            displayStagingPtr() const { return m_displayStagingPtr; }
    bool             hasDisplayImage()   const { return m_displayImage != VK_NULL_HANDLE; }

private:
    // ── One-shot command buffer helpers ───────────────────────────────────────
    VkCommandBuffer beginOneShot();
    void            endOneShot(VkCommandBuffer cmd);

    // Create/destroy the scRGB UI pipeline (see uiPipeline()).
    void createUiPipeline();
    void destroyUiPipeline(bool keepLayouts);

    GLFWwindow*              m_window         = nullptr;  // stored for resize-on-error

    // ── Core ──────────────────────────────────────────────────────────────────
    VkInstance               m_instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physDevice     = VK_NULL_HANDLE;
    VkDevice                 m_device         = VK_NULL_HANDLE;
    VkQueue                  m_queue          = VK_NULL_HANDLE;
    uint32_t                 m_queueFamily    = 0;
    VkSurfaceKHR             m_surface        = VK_NULL_HANDLE;

    // ── Swapchain ─────────────────────────────────────────────────────────────
    VkSwapchainKHR            m_swapchain    = VK_NULL_HANDLE;
    std::vector<VkImage>      m_scImages;
    std::vector<VkImageView>  m_scImageViews;
    VkFormat                  m_scFormat     = VK_FORMAT_UNDEFINED;
    VkExtent2D                m_scExtent     = {};

    // ── Render pass + framebuffers ────────────────────────────────────────────
    VkRenderPass              m_renderPass       = VK_NULL_HANDLE;
    VkFormat                  m_renderPassFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkFramebuffer> m_framebuffers;

    // ── scRGB presentation state ──────────────────────────────────────────────
    bool  m_hasColorspaceExt = false;  // VK_EXT_swapchain_colorspace enabled on the instance
    bool  m_scRgbSwapchain   = false;  // true = EXTENDED_SRGB_LINEAR; false = SRGB_NONLINEAR fallback
    float m_uiScale          = 1.0f;   // paperWhiteNits / 80 — UI pipeline + clear colour

    // Custom ImGui pipeline for scRGB swapchains.  Layout-compatible clone of
    // the backend's pipeline (same set layouts + push constants) with a
    // fragment shader that converts vertex colours sRGB → linear × m_uiScale.
    VkDescriptorSetLayout m_uiSetLayoutTexture = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_uiSetLayoutSampler = VK_NULL_HANDLE;
    VkPipelineLayout      m_uiPipelineLayout   = VK_NULL_HANDLE;
    VkShaderModule        m_uiVertModule       = VK_NULL_HANDLE;
    VkShaderModule        m_uiFragModule       = VK_NULL_HANDLE;
    VkPipeline            m_uiPipeline         = VK_NULL_HANDLE;

    // ── Commands + sync ───────────────────────────────────────────────────────
    VkCommandPool             m_cmdPool      = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_cmdBuffers;
    VkSemaphore               m_imageReady     = VK_NULL_HANDLE;
    std::vector<VkSemaphore>  m_renderDoneSems;  // one per swapchain image
    VkFence                   m_fence          = VK_NULL_HANDLE;

    // ── ImGui descriptor pool ─────────────────────────────────────────────────
    VkDescriptorPool          m_imguiDescPool = VK_NULL_HANDLE;

    // ── Display image (CUDA output → sampled GPU texture) ─────────────────────
    VkImage          m_displayImage      = VK_NULL_HANDLE;
    VkDeviceMemory   m_displayImageMem   = VK_NULL_HANDLE;
    VkImageView      m_displayImageView  = VK_NULL_HANDLE;
    VkSampler        m_displaySampler    = VK_NULL_HANDLE;
    VkDescriptorSet  m_displayDescSet    = VK_NULL_HANDLE;
    VkBuffer         m_displayStaging    = VK_NULL_HANDLE;
    VkDeviceMemory   m_displayStagingMem = VK_NULL_HANDLE;
    void*            m_displayStagingPtr = nullptr;
};

#endif // OPTIX_RAYTRACER_VULKAN_CONTEXT_H

// VulkanContext.cpp — Vulkan device, swapchain, render loop, and display image.
// Extracted from Application.cpp so that Application focuses on OptiX/CUDA/UI logic.

#include "vulkan_context.h"
#include "ui_scrgb_spv.h"  // SPIR-V for the scRGB ImGui pipeline (generated)

#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


// ─── Lifecycle ────────────────────────────────────────────────────────────────

void VulkanContext::init(GLFWwindow* window, int width, int height)
{
    m_window = window;

    // 1. Instance extensions from GLFW
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

    // VK_EXT_swapchain_colorspace extends vkGetPhysicalDeviceSurfaceFormatsKHR
    // with HDR colour spaces (e.g. EXTENDED_SRGB_LINEAR for scRGB output).
    // Optional — HDR output is simply unavailable without it.
    {
        uint32_t extCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> avail(extCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, avail.data());
        for (const auto& e : avail)
        {
            if (std::strcmp(e.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) == 0)
            {
                extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
                m_hasColorspaceExt = true;
                break;
            }
        }
    }

    std::vector<const char*> layers;
#ifdef _DEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    // 2. VkInstance
    VkApplicationInfo appInfo  = {};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "OptiX Raytracer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instCI    = {};
    instCI.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo        = &appInfo;
    instCI.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    instCI.ppEnabledExtensionNames = extensions.data();
    instCI.enabledLayerCount       = static_cast<uint32_t>(layers.size());
    instCI.ppEnabledLayerNames     = layers.data();

    if (vkCreateInstance(&instCI, nullptr, &m_instance) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan instance");
    }

#ifdef _DEBUG
    // 3. Debug messenger
    VkDebugUtilsMessengerCreateInfoEXT dbgCI = {};
    dbgCI.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    dbgCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbgCI.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                          | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                          | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    dbgCI.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT,
                                VkDebugUtilsMessageTypeFlagsEXT,
                                const VkDebugUtilsMessengerCallbackDataEXT* d,
                                void*) -> VkBool32
    {
        std::cerr << "[Vulkan] " << d->pMessage << '\n';
        return VK_FALSE;
    };
    auto vkCreateDbg = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (vkCreateDbg)
    {
        vkCreateDbg(m_instance, &dbgCI, nullptr, &m_debugMessenger);
    }
#endif

    // 4. Surface
    if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan window surface");
    }

    // 5. Physical device — prefer discrete GPU; fall back to any with graphics + present queue
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(m_instance, &devCount, devs.data());

    auto pickQueueFamily = [&](VkPhysicalDevice pd) -> int
    {
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfProps.data());
        for (uint32_t i = 0; i < qfCount; ++i)
        {
            VkBool32 presentOk = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, m_surface, &presentOk);
            if ((qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && presentOk)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    };

    // First pass: discrete GPU only
    for (auto& pd : devs)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            continue;
        }
        int qf = pickQueueFamily(pd);
        if (qf >= 0)
        {
            m_physDevice = pd;
            m_queueFamily = static_cast<uint32_t>(qf);
            break;
        }
    }
    // Second pass: any suitable device (integrated, virtual, etc.)
    if (m_physDevice == VK_NULL_HANDLE)
    {
        for (auto& pd : devs)
        {
            int qf = pickQueueFamily(pd);
            if (qf >= 0)
            {
                m_physDevice = pd;
                m_queueFamily = static_cast<uint32_t>(qf);
                break;
            }
        }
    }
    if (m_physDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("No suitable Vulkan physical device found");
    }

    // 6. Logical device + queue
    const float qPriority = 1.0f;
    VkDeviceQueueCreateInfo queueCI = {};
    queueCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = m_queueFamily;
    queueCI.queueCount       = 1;
    queueCI.pQueuePriorities = &qPriority;

    const char* swapchainExt    = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo deviceCI = {};
    deviceCI.sType                    = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCI.queueCreateInfoCount     = 1;
    deviceCI.pQueueCreateInfos        = &queueCI;
    deviceCI.enabledExtensionCount    = 1;
    deviceCI.ppEnabledExtensionNames  = &swapchainExt;

    if (vkCreateDevice(m_physDevice, &deviceCI, nullptr, &m_device) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan logical device");
    }
    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

    // 7–12. Swapchain, render pass, framebuffers, command pool/buffers, sync objects
    createSwapchain(width, height);
}

void VulkanContext::initImGui(GLFWwindow* window, int imageCount)
{
    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo vkInfo = {};
    vkInfo.ApiVersion     = VK_API_VERSION_1_2;
    vkInfo.Instance       = m_instance;
    vkInfo.PhysicalDevice = m_physDevice;
    vkInfo.Device         = m_device;
    vkInfo.QueueFamily    = m_queueFamily;
    vkInfo.Queue          = m_queue;
    vkInfo.DescriptorPool = m_imguiDescPool;
    vkInfo.MinImageCount  = 2;
    vkInfo.ImageCount     = static_cast<uint32_t>(imageCount);
    vkInfo.PipelineInfoMain.RenderPass  = m_renderPass;
    vkInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&vkInfo);
    // Font texture is uploaded automatically on the first NewFrame call
}

void VulkanContext::cleanup()
{
    if (m_device == VK_NULL_HANDLE)
    {
        return;
    }
    vkDeviceWaitIdle(m_device);

    destroyDisplayImage();
    destroyUiPipeline(false);

    if (m_imguiDescPool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(m_device, m_imguiDescPool, nullptr); m_imguiDescPool = VK_NULL_HANDLE; }
    if (m_fence         != VK_NULL_HANDLE) { vkDestroyFence(m_device,          m_fence,         nullptr); m_fence         = VK_NULL_HANDLE; }
    for (VkSemaphore& sem : m_renderDoneSems)
    {
        if (sem != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(m_device, sem, nullptr);
            sem = VK_NULL_HANDLE;
        }
    }
    m_renderDoneSems.clear();
    if (m_imageReady    != VK_NULL_HANDLE) { vkDestroySemaphore(m_device,   m_imageReady, nullptr); m_imageReady = VK_NULL_HANDLE; }
    if (m_cmdPool       != VK_NULL_HANDLE) { vkDestroyCommandPool(m_device, m_cmdPool,    nullptr); m_cmdPool    = VK_NULL_HANDLE; }
    if (m_renderPass    != VK_NULL_HANDLE) { vkDestroyRenderPass(m_device,  m_renderPass, nullptr); m_renderPass = VK_NULL_HANDLE; }

    destroySwapchain();

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;

    if (m_surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    if (m_debugMessenger != VK_NULL_HANDLE)
    {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn)
        {
            fn(m_instance, m_debugMessenger, nullptr);
        }
        m_debugMessenger = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

void VulkanContext::waitIdle() const
{
    if (m_device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(m_device);
    }
}

// ─── Swapchain ────────────────────────────────────────────────────────────────

void VulkanContext::createSwapchain(int w, int h)
{
    VkSurfaceCapabilitiesKHR caps = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physDevice, m_surface, &caps);

    // Format — prefer R16G16B16A16_SFLOAT + EXTENDED_SRGB_LINEAR (true scRGB HDR).
    // NVIDIA only enumerates EXTENDED_SRGB_LINEAR when Windows HDR is enabled on
    // the active display.  Fall back to R16G16B16A16_SFLOAT + SRGB_NONLINEAR when
    // the HDR color space is absent: FP16 precision is retained but values above
    // 1.0 clip at the compositor.  m_scRgbSwapchain tracks which case is active.
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physDevice, m_surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physDevice, m_surface, &fmtCount, fmts.data());

    VkSurfaceFormatKHR chosen = {};
    for (auto& f : fmts)
    {
        if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT
         && f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
        {
            chosen = f;
            break;
        }
    }
    if (chosen.format == VK_FORMAT_UNDEFINED)
    {
        // No HDR display / HDR off — look for plain FP16 as SDR fallback.
        for (auto& f : fmts)
        {
            if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT)
            {
                chosen = f;
                break;
            }
        }
    }
    if (chosen.format == VK_FORMAT_UNDEFINED)
    {
        // FP16 surface not available (DWM restricts swapchain formats to 8-bit
        // on non-HDR displays regardless of GPU capability) — fall back to BGRA8.
        for (auto& f : fmts)
        {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM
             && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                chosen = f;
                break;
            }
        }
    }
    if (chosen.format == VK_FORMAT_UNDEFINED)
    {
        chosen = fmts[0];  // last resort: whatever the driver offers
    }
    m_scRgbSwapchain = (chosen.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT);
    m_scFormat = chosen.format;

    // Extent
    if (caps.currentExtent.width != UINT32_MAX)
    {
        m_scExtent = caps.currentExtent;
    }
    else
    {
        m_scExtent.width  = std::max(caps.minImageExtent.width,
                             std::min(caps.maxImageExtent.width,  (uint32_t)w));
        m_scExtent.height = std::max(caps.minImageExtent.height,
                             std::min(caps.maxImageExtent.height, (uint32_t)h));
    }

    // Image count (at least 2)
    uint32_t imageCount = std::max(2u, caps.minImageCount);
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
    {
        imageCount = caps.maxImageCount;
    }

    // Present mode — prefer MAILBOX, fall back to FIFO
    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physDevice, m_surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> pms(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physDevice, m_surface, &pmCount, pms.data());
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto& pm : pms)
    {
        if (pm == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            presentMode = pm;
            break;
        }
    }

    // Create swapchain (retaining old handle for retirement)
    VkSwapchainCreateInfoKHR scCI = {};
    scCI.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    scCI.surface          = m_surface;
    scCI.minImageCount    = imageCount;
    scCI.imageFormat      = chosen.format;
    scCI.imageColorSpace  = chosen.colorSpace;
    scCI.imageExtent      = m_scExtent;
    scCI.imageArrayLayers = 1;
    scCI.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    scCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    scCI.preTransform     = caps.currentTransform;
    scCI.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    scCI.presentMode      = presentMode;
    scCI.clipped          = VK_TRUE;
    scCI.oldSwapchain     = m_swapchain;

    VkSwapchainKHR newSC = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(m_device, &scCI, nullptr, &newSC) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan swapchain");
    }
    if (m_swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    }
    m_swapchain = newSC;

    // Images
    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualCount, nullptr);
    m_scImages.resize(actualCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &actualCount, m_scImages.data());

    // Image views
    for (auto& v : m_scImageViews)
    {
        if (v != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_device, v, nullptr);
        }
    }
    m_scImageViews.resize(actualCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < actualCount; ++i)
    {
        VkImageViewCreateInfo viewCI = {};
        viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.image            = m_scImages[i];
        viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format           = m_scFormat;
        viewCI.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCreateImageView(m_device, &viewCI, nullptr, &m_scImageViews[i]);
    }

    // Render pass — recreated only when the attachment format changes (HDR
    // toggle).  Plain resizes keep the same format so ImGui's pipeline, which
    // references this render pass, stays valid across resizes.
    if (m_renderPass != VK_NULL_HANDLE && m_renderPassFormat != m_scFormat)
    {
        vkDestroyRenderPass(m_device, m_renderPass, nullptr);
        m_renderPass = VK_NULL_HANDLE;
    }
    if (m_renderPass == VK_NULL_HANDLE)
    {
        m_renderPassFormat = m_scFormat;

        VkAttachmentDescription colorAtt = {};
        colorAtt.format         = m_scFormat;
        colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments    = &colorRef;

        VkSubpassDependency dep = {};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpCI = {};
        rpCI.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpCI.attachmentCount = 1;
        rpCI.pAttachments    = &colorAtt;
        rpCI.subpassCount    = 1;
        rpCI.pSubpasses      = &subpass;
        rpCI.dependencyCount = 1;
        rpCI.pDependencies   = &dep;

        if (vkCreateRenderPass(m_device, &rpCI, nullptr, &m_renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan render pass");
        }
    }

    // Framebuffers
    for (auto& fb : m_framebuffers)
    {
        if (fb != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(m_device, fb, nullptr);
        }
    }
    m_framebuffers.resize(actualCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < actualCount; ++i)
    {
        VkFramebufferCreateInfo fbCI = {};
        fbCI.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCI.renderPass      = m_renderPass;
        fbCI.attachmentCount = 1;
        fbCI.pAttachments    = &m_scImageViews[i];
        fbCI.width           = m_scExtent.width;
        fbCI.height          = m_scExtent.height;
        fbCI.layers          = 1;
        vkCreateFramebuffer(m_device, &fbCI, nullptr, &m_framebuffers[i]);
    }

    // Command pool (created once)
    if (m_cmdPool == VK_NULL_HANDLE)
    {
        VkCommandPoolCreateInfo poolCI = {};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.queueFamilyIndex = m_queueFamily;
        poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(m_device, &poolCI, nullptr, &m_cmdPool);
    }

    // Command buffers (reallocate if count changed)
    if (!m_cmdBuffers.empty())
    {
        vkFreeCommandBuffers(m_device, m_cmdPool,
            (uint32_t)m_cmdBuffers.size(), m_cmdBuffers.data());
    }
    m_cmdBuffers.resize(actualCount, VK_NULL_HANDLE);
    VkCommandBufferAllocateInfo allocCI = {};
    allocCI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocCI.commandPool        = m_cmdPool;
    allocCI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCI.commandBufferCount = actualCount;
    vkAllocateCommandBuffers(m_device, &allocCI, m_cmdBuffers.data());

    // Sync objects (created once)
    if (m_imageReady == VK_NULL_HANDLE)
    {
        VkSemaphoreCreateInfo semCI = {};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(m_device, &semCI, nullptr, &m_imageReady);

        VkFenceCreateInfo fenceCI = {};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(m_device, &fenceCI, nullptr, &m_fence);
    }

    // Per-image render-done semaphores — recreated whenever the swapchain image
    // count changes (resize may produce a different count).  Each semaphore is
    // only reused when its image is re-acquired, so the swapchain has released
    // any internal reference to the previous present operation by then.
    if (m_renderDoneSems.size() != actualCount)
    {
        for (VkSemaphore& sem : m_renderDoneSems)
        {
            if (sem != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(m_device, sem, nullptr);
                sem = VK_NULL_HANDLE;
            }
        }
        m_renderDoneSems.clear();

        VkSemaphoreCreateInfo semCI = {};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        m_renderDoneSems.resize(actualCount, VK_NULL_HANDLE);
        for (VkSemaphore& sem : m_renderDoneSems)
        {
            vkCreateSemaphore(m_device, &semCI, nullptr, &sem);
        }
    }

    // ImGui descriptor pool (created once).
    // Includes SAMPLED_IMAGE + SAMPLER in addition to COMBINED_IMAGE_SAMPLER because
    // the ImGui Vulkan backend (1.91+) uses separate image/sampler descriptors.
    if (m_imguiDescPool == VK_NULL_HANDLE)
    {
        const VkDescriptorPoolSize poolSizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLER,                1000 },
        };
        VkDescriptorPoolCreateInfo descPoolCI = {};
        descPoolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descPoolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descPoolCI.maxSets       = 3000;
        descPoolCI.poolSizeCount = 3;
        descPoolCI.pPoolSizes    = poolSizes;
        vkCreateDescriptorPool(m_device, &descPoolCI, nullptr, &m_imguiDescPool);
    }

    // scRGB UI pipeline (created once — render pass format never changes)
    if (m_uiPipeline == VK_NULL_HANDLE)
    {
        createUiPipeline();
    }
}

void VulkanContext::destroySwapchain()
{
    for (auto& fb : m_framebuffers)
    {
        if (fb != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(m_device, fb, nullptr);
        }
    }
    m_framebuffers.clear();

    for (auto& v : m_scImageViews)
    {
        if (v != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_device, v, nullptr);
        }
    }
    m_scImageViews.clear();
    m_scImages.clear();

    if (m_swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

void VulkanContext::recreateSwapchain(int w, int h)
{
    if (w == 0 || h == 0)
    {
        return;  // minimised — skip until window has a valid size
    }
    vkDeviceWaitIdle(m_device);
    createSwapchain(w, h);
}

// ─── scRGB presentation ───────────────────────────────────────────────────────

void VulkanContext::setUiScale(float scale)
{
    if (scale == m_uiScale)
    {
        return;
    }
    m_uiScale = scale;

    // The scale is a specialization constant — rebuild the UI pipeline.
    if (m_device != VK_NULL_HANDLE && m_renderPass != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(m_device);
        destroyUiPipeline(true);  // pipeline only; layouts and modules are reused
        createUiPipeline();
    }
}

// Layout-compatible clone of the ImGui Vulkan backend pipeline: identical
// descriptor set layouts (set 0 = sampled image, set 1 = sampler), push
// constant range, vertex input, and blend state — so the backend's per-draw
// descriptor binds (recorded with ITS pipeline layout) stay valid while OUR
// pipeline is bound via the RenderDrawData pipeline override.  Only the
// fragment shader differs: vertex colours are converted sRGB → linear and
// scaled to paper white (see shaders/ui_scrgb.frag).
void VulkanContext::createUiPipeline()
{
    // Descriptor set layouts + pipeline layout (created once)
    if (m_uiSetLayoutTexture == VK_NULL_HANDLE)
    {
        VkDescriptorSetLayoutBinding binding = {};
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo slCI = {};
        slCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slCI.bindingCount = 1;
        slCI.pBindings    = &binding;
        vkCreateDescriptorSetLayout(m_device, &slCI, nullptr, &m_uiSetLayoutTexture);

        binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        vkCreateDescriptorSetLayout(m_device, &slCI, nullptr, &m_uiSetLayoutSampler);

        VkPushConstantRange pcRange = {};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(float) * 4;  // vec2 uScale + vec2 uTranslate

        const VkDescriptorSetLayout setLayouts[2] =
            { m_uiSetLayoutTexture, m_uiSetLayoutSampler };

        VkPipelineLayoutCreateInfo plCI = {};
        plCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plCI.setLayoutCount         = 2;
        plCI.pSetLayouts            = setLayouts;
        plCI.pushConstantRangeCount = 1;
        plCI.pPushConstantRanges    = &pcRange;
        vkCreatePipelineLayout(m_device, &plCI, nullptr, &m_uiPipelineLayout);
    }

    // Shader modules (created once)
    if (m_uiVertModule == VK_NULL_HANDLE)
    {
        VkShaderModuleCreateInfo smCI = {};
        smCI.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smCI.codeSize = sizeof(kUiScrgbVertSpv);
        smCI.pCode    = kUiScrgbVertSpv;
        vkCreateShaderModule(m_device, &smCI, nullptr, &m_uiVertModule);

        smCI.codeSize = sizeof(kUiScrgbFragSpv);
        smCI.pCode    = kUiScrgbFragSpv;
        vkCreateShaderModule(m_device, &smCI, nullptr, &m_uiFragModule);
    }

    // uUiScale (constant_id 0) — baked into the pipeline; setUiScale() rebuilds.
    VkSpecializationMapEntry specEntry = { 0, 0, sizeof(float) };
    VkSpecializationInfo specInfo = {};
    specInfo.mapEntryCount = 1;
    specInfo.pMapEntries   = &specEntry;
    specInfo.dataSize      = sizeof(float);
    specInfo.pData         = &m_uiScale;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_uiVertModule;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_uiFragModule;
    stages[1].pName  = "main";
    stages[1].pSpecializationInfo = &specInfo;

    // Vertex input — ImDrawVert layout (pos2f, uv2f, col RGBA8)
    VkVertexInputBindingDescription bindingDesc = {};
    bindingDesc.stride    = sizeof(ImDrawVert);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[3] = {};
    attrs[0].location = 0;
    attrs[0].format   = VK_FORMAT_R32G32_SFLOAT;
    attrs[0].offset   = offsetof(ImDrawVert, pos);
    attrs[1].location = 1;
    attrs[1].format   = VK_FORMAT_R32G32_SFLOAT;
    attrs[1].offset   = offsetof(ImDrawVert, uv);
    attrs[2].location = 2;
    attrs[2].format   = VK_FORMAT_R8G8B8A8_UNORM;
    attrs[2].offset   = offsetof(ImDrawVert, col);

    VkPipelineVertexInputStateCreateInfo viCI = {};
    viCI.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    viCI.vertexBindingDescriptionCount   = 1;
    viCI.pVertexBindingDescriptions      = &bindingDesc;
    viCI.vertexAttributeDescriptionCount = 3;
    viCI.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo iaCI = {};
    iaCI.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vpCI = {};
    vpCI.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpCI.viewportCount = 1;
    vpCI.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rsCI = {};
    rsCI.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsCI.polygonMode = VK_POLYGON_MODE_FILL;
    rsCI.cullMode    = VK_CULL_MODE_NONE;
    rsCI.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsCI.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msCI = {};
    msCI.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt = {};
    blendAtt.blendEnable         = VK_TRUE;
    blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAtt.colorBlendOp        = VK_BLEND_OP_ADD;
    blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAtt.alphaBlendOp        = VK_BLEND_OP_ADD;
    blendAtt.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                 | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cbCI = {};
    cbCI.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbCI.attachmentCount = 1;
    cbCI.pAttachments    = &blendAtt;

    VkPipelineDepthStencilStateCreateInfo dsCI = {};
    dsCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    const VkDynamicState dynStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynCI = {};
    dynCI.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynCI.dynamicStateCount = 2;
    dynCI.pDynamicStates    = dynStates;

    VkGraphicsPipelineCreateInfo pipeCI = {};
    pipeCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeCI.stageCount          = 2;
    pipeCI.pStages             = stages;
    pipeCI.pVertexInputState   = &viCI;
    pipeCI.pInputAssemblyState = &iaCI;
    pipeCI.pViewportState      = &vpCI;
    pipeCI.pRasterizationState = &rsCI;
    pipeCI.pMultisampleState   = &msCI;
    pipeCI.pDepthStencilState  = &dsCI;
    pipeCI.pColorBlendState    = &cbCI;
    pipeCI.pDynamicState       = &dynCI;
    pipeCI.layout              = m_uiPipelineLayout;
    pipeCI.renderPass          = m_renderPass;
    pipeCI.subpass             = 0;

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeCI,
                                  nullptr, &m_uiPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create scRGB UI pipeline");
    }
}

void VulkanContext::destroyUiPipeline(bool keepLayouts)
{
    if (m_uiPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(m_device, m_uiPipeline, nullptr);
        m_uiPipeline = VK_NULL_HANDLE;
    }
    if (keepLayouts)
    {
        return;
    }
    if (m_uiFragModule       != VK_NULL_HANDLE) { vkDestroyShaderModule(m_device,        m_uiFragModule,       nullptr); m_uiFragModule       = VK_NULL_HANDLE; }
    if (m_uiVertModule       != VK_NULL_HANDLE) { vkDestroyShaderModule(m_device,        m_uiVertModule,       nullptr); m_uiVertModule       = VK_NULL_HANDLE; }
    if (m_uiPipelineLayout   != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_device,      m_uiPipelineLayout,   nullptr); m_uiPipelineLayout   = VK_NULL_HANDLE; }
    if (m_uiSetLayoutSampler != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_uiSetLayoutSampler, nullptr); m_uiSetLayoutSampler = VK_NULL_HANDLE; }
    if (m_uiSetLayoutTexture != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_device, m_uiSetLayoutTexture, nullptr); m_uiSetLayoutTexture = VK_NULL_HANDLE; }
}

// ─── Display image ────────────────────────────────────────────────────────────

void VulkanContext::createDisplayImage(int w, int h)
{
    if (m_device == VK_NULL_HANDLE || m_imguiDescPool == VK_NULL_HANDLE)
    {
        return;
    }
    // float4 display image — matches the CUDA buffer byte-for-byte so the
    // staging upload path is format-agnostic.
    const VkFormat displayFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    const size_t   pixelBytes    = static_cast<size_t>(w) * h * 16u;  // float4 = 4 × float

    // Device-local image (SAMPLED + TRANSFER_DST)
    VkImageCreateInfo imgCI = {};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = displayFormat;
    imgCI.extent        = { (uint32_t)w, (uint32_t)h, 1 };
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(m_device, &imgCI, nullptr, &m_displayImage);

    VkMemoryRequirements imgMemReqs = {};
    vkGetImageMemoryRequirements(m_device, m_displayImage, &imgMemReqs);
    VkMemoryAllocateInfo imgAlloc = {};
    imgAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imgAlloc.allocationSize  = imgMemReqs.size;
    imgAlloc.memoryTypeIndex = findMemoryType(imgMemReqs.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(m_device, &imgAlloc, nullptr, &m_displayImageMem);
    vkBindImageMemory(m_device, m_displayImage, m_displayImageMem, 0);

    // Transition UNDEFINED → SHADER_READ_ONLY as the initial layout
    {
        VkCommandBufferAllocateInfo cbAlloc = {};
        cbAlloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbAlloc.commandPool        = m_cmdPool;
        cbAlloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAlloc.commandBufferCount = 1;
        VkCommandBuffer initCmd    = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(m_device, &cbAlloc, &initCmd);

        VkCommandBufferBeginInfo beginCI = {};
        beginCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginCI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(initCmd, &beginCI);

        VkImageMemoryBarrier barrier       = {};
        barrier.sType                      = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                  = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout                  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex        = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex        = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                      = m_displayImage;
        barrier.subresourceRange           = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask              = 0;
        barrier.dstAccessMask              = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(initCmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        vkEndCommandBuffer(initCmd);

        VkSubmitInfo si = {};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &initCmd;
        vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);
        vkFreeCommandBuffers(m_device, m_cmdPool, 1, &initCmd);
    }

    // Image view
    VkImageViewCreateInfo viewCI = {};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = m_displayImage;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = displayFormat;
    viewCI.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(m_device, &viewCI, nullptr, &m_displayImageView);

    // Sampler (nearest, clamp)
    VkSamplerCreateInfo samplerCI = {};
    samplerCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter    = VK_FILTER_NEAREST;
    samplerCI.minFilter    = VK_FILTER_NEAREST;
    samplerCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(m_device, &samplerCI, nullptr, &m_displaySampler);

    // Staging buffer (host-visible, persistently mapped)
    VkBufferCreateInfo bufCI = {};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size  = pixelBytes;
    bufCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(m_device, &bufCI, nullptr, &m_displayStaging);

    VkMemoryRequirements bufReqs = {};
    vkGetBufferMemoryRequirements(m_device, m_displayStaging, &bufReqs);
    VkMemoryAllocateInfo bufAlloc = {};
    bufAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufAlloc.allocationSize  = bufReqs.size;
    bufAlloc.memoryTypeIndex = findMemoryType(bufReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(m_device, &bufAlloc, nullptr, &m_displayStagingMem);
        vkBindBufferMemory(m_device, m_displayStaging, m_displayStagingMem, 0);
        vkMapMemory(m_device, m_displayStagingMem, 0, pixelBytes, 0, &m_displayStagingPtr);

    // Register with ImGui → VkDescriptorSet used as ImTextureID
    m_displayDescSet = ImGui_ImplVulkan_AddTexture(m_displaySampler, m_displayImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void VulkanContext::destroyDisplayImage()
{
    if (m_device == VK_NULL_HANDLE)
    {
        return;
    }
    if (m_displayDescSet    != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(m_displayDescSet);  m_displayDescSet             = VK_NULL_HANDLE; }
    if (m_displayStagingPtr)                   { vkUnmapMemory(m_device,      m_displayStagingMem); m_displayStagingPtr          = nullptr;        }
    if (m_displayStaging    != VK_NULL_HANDLE) { vkDestroyBuffer(m_device,    m_displayStaging,    nullptr); m_displayStaging    = VK_NULL_HANDLE; }
    if (m_displayStagingMem != VK_NULL_HANDLE) { vkFreeMemory(m_device,       m_displayStagingMem, nullptr); m_displayStagingMem = VK_NULL_HANDLE; }
    if (m_displaySampler    != VK_NULL_HANDLE) { vkDestroySampler(m_device,   m_displaySampler,    nullptr); m_displaySampler    = VK_NULL_HANDLE; }
    if (m_displayImageView  != VK_NULL_HANDLE) { vkDestroyImageView(m_device, m_displayImageView,  nullptr); m_displayImageView  = VK_NULL_HANDLE; }
    if (m_displayImage      != VK_NULL_HANDLE) { vkDestroyImage(m_device,     m_displayImage,      nullptr); m_displayImage      = VK_NULL_HANDLE; }
    if (m_displayImageMem   != VK_NULL_HANDLE) { vkFreeMemory(m_device,       m_displayImageMem,   nullptr); m_displayImageMem   = VK_NULL_HANDLE; }
}

// ─── Per-frame rendering ──────────────────────────────────────────────────────

VulkanFrameContext VulkanContext::beginFrame()
{
    vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(
        m_device, m_swapchain, UINT64_MAX,
        m_imageReady, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        int fw, fh;
        glfwGetFramebufferSize(m_window, &fw, &fh);
        recreateSwapchain(fw, fh);
        return VulkanFrameContext{};  // invalid — caller skips this frame
    }

    vkResetFences(m_device, 1, &m_fence);

    VkCommandBuffer cmd = m_cmdBuffers[imageIndex];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    return VulkanFrameContext{ imageIndex, cmd, true };
}

void VulkanContext::uploadDisplayImage(VkCommandBuffer cmd, const void* pixels, int w, int h)
{
    if (m_displayImage == VK_NULL_HANDLE || m_displayStaging == VK_NULL_HANDLE)
    {
        return;
    }
    // Barrier: SHADER_READ_ONLY → TRANSFER_DST
    VkImageMemoryBarrier toXfer        = {};
    toXfer.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toXfer.oldLayout                   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toXfer.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toXfer.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    toXfer.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    toXfer.image                       = m_displayImage;
    toXfer.subresourceRange            = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toXfer.srcAccessMask               = VK_ACCESS_SHADER_READ_BIT;
    toXfer.dstAccessMask               = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toXfer);

    // Buffer → image copy
    VkBufferImageCopy region       = {};
    region.imageSubresource        = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent             = { (uint32_t)w, (uint32_t)h, 1 };
    vkCmdCopyBufferToImage(cmd, m_displayStaging, m_displayImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Barrier: TRANSFER_DST → SHADER_READ_ONLY
    VkImageMemoryBarrier toShader  = toXfer;
    toShader.oldLayout             = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout             = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.srcAccessMask         = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask         = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toShader);
}

void VulkanContext::beginRenderPass(VkCommandBuffer cmd)
{
    // Background clear — when using the scRGB pipeline, linearise the
    // sRGB-authored value and scale to paper white.  On the FP16+SRGB_NONLINEAR
    // fallback path the compositor expects sRGB-encoded values, so keep as-is.
    VkClearValue clearColor = {};
    {
        const float s  = m_scRgbSwapchain ? m_uiScale : 1.0f;
        const float rg = m_scRgbSwapchain ? std::pow(0.10f, 2.2f) * s : 0.10f;
        const float b  = m_scRgbSwapchain ? std::pow(0.15f, 2.2f) * s : 0.15f;
        clearColor.color = {{ rg, rg, b, 1.0f }};
    }

    // Retrieve the current image index from the command buffer slot
    const uint32_t idx = [&]() -> uint32_t {
        for (uint32_t i = 0; i < (uint32_t)m_cmdBuffers.size(); ++i)
        {
            if (m_cmdBuffers[i] == cmd)
            {
                return i;
            }
        }
        return 0;
    }();

    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass        = m_renderPass;
    rpBegin.framebuffer       = m_framebuffers[idx];
    rpBegin.renderArea.extent = m_scExtent;
    rpBegin.clearValueCount   = 1;
    rpBegin.pClearValues      = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
}

void VulkanContext::endFrameAndPresent(VulkanFrameContext& frame, int /*windowW*/, int /*windowH*/)
{
    vkCmdEndRenderPass(frame.cmd);
    vkEndCommandBuffer(frame.cmd);

    VkSemaphore renderDone = m_renderDoneSems[frame.imageIndex];

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si                = {};
    si.sType                       = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount          = 1;
    si.pWaitSemaphores             = &m_imageReady;
    si.pWaitDstStageMask           = &waitStage;
    si.commandBufferCount          = 1;
    si.pCommandBuffers             = &frame.cmd;
    si.signalSemaphoreCount        = 1;
    si.pSignalSemaphores           = &renderDone;
    vkQueueSubmit(m_queue, 1, &si, m_fence);

    VkPresentInfoKHR pi   = {};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderDone;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &m_swapchain;
    pi.pImageIndices      = &frame.imageIndex;

    VkResult result = vkQueuePresentKHR(m_queue, &pi);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        int fw, fh;
        glfwGetFramebufferSize(m_window, &fw, &fh);
        recreateSwapchain(fw, fh);
    }
}

// ─── One-shot command buffer helpers ─────────────────────────────────────────

VkCommandBuffer VulkanContext::beginOneShot()
{
    VkCommandBufferAllocateInfo allocCI = {};
    allocCI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocCI.commandPool        = m_cmdPool;
    allocCI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCI.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &allocCI, &cmd);

    VkCommandBufferBeginInfo beginCI = {};
    beginCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginCI);
    return cmd;
}

void VulkanContext::endOneShot(VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si       = {};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(m_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &cmd);
}

// ─── Generic ImGui textures ───────────────────────────────────────────────────

ImGuiTexture VulkanContext::createImGuiTexture(int w, int h, const uint8_t* rgba8)
{
    ImGuiTexture tex;
    const size_t pixelBytes = static_cast<size_t>(w) * h * 4u;

    // ── Device-local image ────────────────────────────────────────────────────
    // Use an sRGB view so the sampler auto-linearises texels — the UI
    // pipeline's shader multiplies textures in unconverted.
    const VkFormat texFormat = VK_FORMAT_R8G8B8A8_SRGB;

    VkImageCreateInfo imgCI = {};
    imgCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType     = VK_IMAGE_TYPE_2D;
    imgCI.format        = texFormat;
    imgCI.extent        = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    imgCI.mipLevels     = 1;
    imgCI.arrayLayers   = 1;
    imgCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(m_device, &imgCI, nullptr, &tex.image);

    VkMemoryRequirements imgReqs = {};
    vkGetImageMemoryRequirements(m_device, tex.image, &imgReqs);
    VkMemoryAllocateInfo imgAlloc = {};
    imgAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imgAlloc.allocationSize  = imgReqs.size;
    imgAlloc.memoryTypeIndex = findMemoryType(imgReqs.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(m_device, &imgAlloc, nullptr, &tex.memory);
    vkBindImageMemory(m_device, tex.image, tex.memory, 0);

    // ── Staging buffer (host-visible, temporary) ──────────────────────────────
    VkBuffer       staging    = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;

    VkBufferCreateInfo bufCI = {};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size  = pixelBytes;
    bufCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(m_device, &bufCI, nullptr, &staging);

    VkMemoryRequirements bufReqs = {};
    vkGetBufferMemoryRequirements(m_device, staging, &bufReqs);
    VkMemoryAllocateInfo bufAlloc = {};
    bufAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufAlloc.allocationSize  = bufReqs.size;
    bufAlloc.memoryTypeIndex = findMemoryType(bufReqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(m_device, &bufAlloc, nullptr, &stagingMem);
    vkBindBufferMemory(m_device, staging, stagingMem, 0);

    void* mapped = nullptr;
    vkMapMemory(m_device, stagingMem, 0, pixelBytes, 0, &mapped);
    std::memcpy(mapped, rgba8, pixelBytes);
    vkUnmapMemory(m_device, stagingMem);

    // ── One-shot: UNDEFINED → TRANSFER_DST, copy, TRANSFER_DST → SHADER_READ ─
    VkCommandBuffer cmd = beginOneShot();

    VkImageMemoryBarrier toXfer    = {};
    toXfer.sType                   = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toXfer.oldLayout               = VK_IMAGE_LAYOUT_UNDEFINED;
    toXfer.newLayout               = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toXfer.srcQueueFamilyIndex     = VK_QUEUE_FAMILY_IGNORED;
    toXfer.dstQueueFamilyIndex     = VK_QUEUE_FAMILY_IGNORED;
    toXfer.image                   = tex.image;
    toXfer.subresourceRange        = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toXfer.srcAccessMask           = 0;
    toXfer.dstAccessMask           = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toXfer);

    VkBufferImageCopy region = {};
    region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent       = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 };
    vkCmdCopyBufferToImage(cmd, staging, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier toShader = toXfer;
    toShader.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout            = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.srcAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask        = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &toShader);

    endOneShot(cmd);  // submit + vkQueueWaitIdle + free

    // Free staging resources now that the GPU copy is complete.
    vkDestroyBuffer(m_device, staging,    nullptr);
    vkFreeMemory(m_device,    stagingMem, nullptr);

    // ── Image view ────────────────────────────────────────────────────────────
    VkImageViewCreateInfo viewCI = {};
    viewCI.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image            = tex.image;
    viewCI.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format           = texFormat;
    viewCI.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(m_device, &viewCI, nullptr, &tex.view);

    // ── Sampler (bilinear, clamp — thumbnails benefit from linear filtering) ──
    VkSamplerCreateInfo samplerCI = {};
    samplerCI.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter    = VK_FILTER_LINEAR;
    samplerCI.minFilter    = VK_FILTER_LINEAR;
    samplerCI.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(m_device, &samplerCI, nullptr, &tex.sampler);

    // ── Register with ImGui ───────────────────────────────────────────────────
    tex.descSet = ImGui_ImplVulkan_AddTexture(tex.sampler, tex.view,
                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return tex;
}

void VulkanContext::destroyImGuiTexture(ImGuiTexture& tex)
{
    if (m_device == VK_NULL_HANDLE) { return; }
    if (tex.descSet != VK_NULL_HANDLE) { ImGui_ImplVulkan_RemoveTexture(tex.descSet);        tex.descSet = VK_NULL_HANDLE; }
    if (tex.sampler != VK_NULL_HANDLE) { vkDestroySampler(m_device,   tex.sampler, nullptr); tex.sampler = VK_NULL_HANDLE; }
    if (tex.view    != VK_NULL_HANDLE) { vkDestroyImageView(m_device, tex.view,    nullptr); tex.view    = VK_NULL_HANDLE; }
    if (tex.image   != VK_NULL_HANDLE) { vkDestroyImage(m_device,     tex.image,   nullptr); tex.image   = VK_NULL_HANDLE; }
    if (tex.memory  != VK_NULL_HANDLE) { vkFreeMemory(m_device,       tex.memory,  nullptr); tex.memory  = VK_NULL_HANDLE; }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties memProps = {};
    vkGetPhysicalDeviceMemoryProperties(m_physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1u << i))
         && (memProps.memoryTypes[i].propertyFlags & props) == props)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

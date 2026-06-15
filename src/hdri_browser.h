#ifndef OPTIX_RAYTRACER_HDRI_BROWSER_H
#define OPTIX_RAYTRACER_HDRI_BROWSER_H

#include "vulkan_context.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// ─── Thumbnail dimensions ─────────────────────────────────────────────────────
// 256 × 128 matches the 2:1 equirectangular aspect ratio of most HDRIs and
// keeps GPU memory modest (~256 KB per thumbnail as RGBA16F).
inline constexpr int THUMB_W       = 256;
inline constexpr int THUMB_H       = 128;
inline constexpr int THUMB_WORKERS = 16;   // max concurrent loader threads

// ─── Per-file state ───────────────────────────────────────────────────────────

enum class ThumbState { Idle, Loading, GPUReady, Error };

struct ThumbEntry
{
    std::string  path;
    std::string  name;
    ThumbState   state    = ThumbState::Idle;
    ImGuiTexture texture;   // valid only when state == GPUReady
    std::string  errorMsg;  // non-empty only when state == Error
};

// ─── HdriBrowser ─────────────────────────────────────────────────────────────
//
// Async HDRI/EXR thumbnail browser panel for Dear ImGui.
//
// Typical call pattern (Application::tick):
//   1. m_browser.uploadPending(vkCtx)  — before beginFrame()
//   2. m_browser.draw(&open, path)     — inside ImGui frame
//   3. if (selected) loadEnvMap(path)
//
// Call shutdown(vkCtx) in the Application destructor BEFORE
// ImGui_ImplVulkan_Shutdown() to safely release GPU resources.

class HdriBrowser
{
public:
    HdriBrowser();
    ~HdriBrowser();

    HdriBrowser(const HdriBrowser&)            = delete;
    HdriBrowser& operator=(const HdriBrowser&) = delete;

    // Scan a folder and restart thumbnail loading.  Blocks briefly while the
    // previous worker threads finish their current (non-interruptible) load.
    void setFolder(VulkanContext& vkCtx, const std::string& folderPath);

    // Upload CPU-ready thumbnails to GPU.  Call once per frame BEFORE
    // beginFrame() — uses vkQueueWaitIdle internally so no frame may be
    // in-flight.  Also processes any pending folder change queued by draw().
    void uploadPending(VulkanContext& vkCtx);

    // Draw the browser window.  Returns true when the user clicks a thumbnail;
    // selectedPath is set to the chosen file's absolute path.
    // Pass a bool* open to show a close button (may be nullptr).
    bool draw(bool* open, std::string& selectedPath);

    // Mark a path as the currently active environment map (shows ● highlight).
    void setActivePath(const std::string& path) { m_activePath = path; }

    // Set the directory where generated thumbnails are cached on disk.
    // Call once at startup (before the first setFolder).  An empty or
    // unwritable path silently disables caching.
    void setCacheDir(const std::string& dir);

    // Release all GPU textures and stop worker threads.
    // Must be called before ImGui_ImplVulkan_Shutdown().
    void shutdown(VulkanContext& vkCtx);

private:
    // ── Internal work/ready items ─────────────────────────────────────────────
    struct WorkItem  { int idx; std::string path; };
    struct ReadyItem
    {
        int                    idx;
        std::vector<uint16_t>  pixels;    // RGBA16F linear, THUMB_W × THUMB_H; empty = error
        std::string            errorMsg;
    };

    // ── Thread pool ───────────────────────────────────────────────────────────
    void startWorkers();
    void stopWorkers();
    void workerLoop();

    // ── Helpers ───────────────────────────────────────────────────────────────
    void clearEntries(VulkanContext& vkCtx);

    // Box-filter downsample + log-average auto-exposure → linear RGBA16F.
    // No tone-mapping: highlights above 1.0 are preserved so the scRGB pipeline
    // can display them above paper-white on HDR monitors.  Called on a worker thread.
    static void generateThumbnail(const float* srcRgba32f, int srcW, int srcH,
                                  uint16_t*    dstRgba16f, int dstW, int dstH);

    // ── State ─────────────────────────────────────────────────────────────────
    std::string             m_folder;
    std::string             m_activePath;
    std::string             m_pendingFolder;  // set by draw(); consumed by uploadPending()
    std::string             m_cacheDir;       // UTF-8; empty = cache disabled
    std::vector<ThumbEntry> m_entries;        // accessed from main thread only
    int                     m_thumbSizeIdx = 0; // 0=Large, 1=Medium, 2=Small

    // ── Thread pool ───────────────────────────────────────────────────────────
    std::vector<std::thread>    m_workers;
    std::mutex                  m_workMutex;
    std::condition_variable     m_workCv;
    std::queue<WorkItem>        m_workQueue;
    bool                        m_stopWorkers = false;

    // ── Ready queue (worker → main thread) ───────────────────────────────────
    std::mutex               m_readyMutex;
    std::queue<ReadyItem>    m_readyQueue;
};

#endif // OPTIX_RAYTRACER_HDRI_BROWSER_H

// HdriBrowser.cpp — Async HDRI/EXR thumbnail browser panel for Dear ImGui.
//
// Workers load + downsample images on background threads.  The main thread
// drains the ready queue in uploadPending() (before beginFrame) and does the
// Vulkan upload there.  draw() is a pure ImGui call with no GPU work.

#include "hdri_browser.h"
#include "texture.h"

#include <imgui.h>
#include <nfd.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Portable IEEE 754 float32 → float16 (round toward zero).
// Matches the implementation in vulkan_context.cpp.
static uint16_t f32ToF16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign   = (bits >> 16) & 0x8000u;
    const int32_t  expUnb = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant   = bits & 0x7FFFFFu;
    if (expUnb <= 0)  { return static_cast<uint16_t>(sign); }
    if (expUnb >= 31) { return static_cast<uint16_t>(sign | 0x7C00u); }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(expUnb) << 10) | (mant >> 13));
}

// FNV-1a 64-bit hash — fast, dependency-free, good distribution for file paths.
static uint64_t fnv1a64(const std::string& s)
{
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s)
    {
        h = (h ^ static_cast<uint64_t>(c)) * 1099511628211ULL;
    }
    return h;
}

// ─── Construction / Destruction ───────────────────────────────────────────────

HdriBrowser::HdriBrowser()
{
    startWorkers();
}

HdriBrowser::~HdriBrowser()
{
    // GPU resources must have been freed by shutdown() already.
    stopWorkers();
}

void HdriBrowser::setCacheDir(const std::string& dir)
{
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(dir), ec);
    m_cacheDir = ec ? std::string{} : dir;  // disable silently on failure
}

// ─── Thread pool ─────────────────────────────────────────────────────────────

void HdriBrowser::startWorkers()
{
    m_stopWorkers = false;
    m_workers.reserve(THUMB_WORKERS);
    for (int i = 0; i < THUMB_WORKERS; ++i)
    {
        m_workers.emplace_back([this] { workerLoop(); });
    }
}

void HdriBrowser::stopWorkers()
{
    {
        std::lock_guard<std::mutex> lock(m_workMutex);
        m_stopWorkers = true;
        while (!m_workQueue.empty())
        {
            m_workQueue.pop();  // discard pending
        }
    }
    m_workCv.notify_all();
    for (auto& t : m_workers)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    m_workers.clear();
    m_stopWorkers = false;
}

void HdriBrowser::workerLoop()
{
    while (true)
    {
        WorkItem item;
        {
            std::unique_lock<std::mutex> lock(m_workMutex);
            m_workCv.wait(lock, [this]
            {
                return m_stopWorkers || !m_workQueue.empty();
            });
            if (m_stopWorkers && m_workQueue.empty())
            {
                return;
            }
            item = std::move(m_workQueue.front());
            m_workQueue.pop();
        }

        ReadyItem ready;
        ready.idx = item.idx;

        // ── Cache check BEFORE loading the source file ────────────────────────
        // Only two cheap stat syscalls needed to validate a hit; the full HDR
        // decode (potentially hundreds of MB) is skipped entirely on a cache hit.
        //
        // Cache file layout (binary):
        //   uint32_t version  = THUMB_W<<16|THUMB_H  (invalidates on dim change)
        //   int64_t  mtime    = last_write_time().time_since_epoch().count()
        //   uint64_t filesize = source file size in bytes
        //   float    pixels[THUMB_W * THUMB_H * 4]   (RGBA32F linear)
        bool cacheHit = false;
        std::filesystem::path cachePath;
        std::error_code statEc;
        std::filesystem::file_time_type srcMtime;
        uint64_t srcSize = 0;

        if (!m_cacheDir.empty())
        {
            char name[24];
            std::snprintf(name, sizeof(name), "%016llx.thumb",
                          static_cast<unsigned long long>(fnv1a64(item.path)));
            cachePath = std::filesystem::u8path(m_cacheDir) / name;

            srcMtime = std::filesystem::last_write_time(
                           std::filesystem::u8path(item.path), statEc);
            if (!statEc)
            {
                srcSize = static_cast<uint64_t>(std::filesystem::file_size(
                              std::filesystem::u8path(item.path), statEc));
            }

            if (!statEc)
            {
                std::ifstream ifs(cachePath, std::ios::binary);
                if (ifs)
                {
                    // Top two bits = 0b11 → RGBA16F format (0b10 = RGBA32F, 0b00 = old RGBA8).
                    constexpr uint32_t kVersion =
                        ((static_cast<uint32_t>(THUMB_W) << 16) | THUMB_H) | 0xC0000000u;
                    uint32_t ver = 0;
                    int64_t  mt  = 0;
                    uint64_t sz  = 0;
                    if (ifs.read(reinterpret_cast<char*>(&ver), 4) &&
                        ifs.read(reinterpret_cast<char*>(&mt),  8) &&
                        ifs.read(reinterpret_cast<char*>(&sz),  8) &&
                        ver == kVersion &&
                        mt  == static_cast<int64_t>(
                                   srcMtime.time_since_epoch().count()) &&
                        sz  == srcSize)
                    {
                        const size_t nElems =
                            static_cast<size_t>(THUMB_W) * THUMB_H * 4u;
                        ready.pixels.resize(nElems);
                        if (ifs.read(reinterpret_cast<char*>(ready.pixels.data()),
                                     nElems * sizeof(uint16_t)))
                        {
                            cacheHit = true;
                        }
                        else
                        {
                            ready.pixels.clear();  // partial read — fall through
                        }
                    }
                }
            }
        }

        // ── Cache miss: load source file, generate, write cache ───────────────
        if (!cacheHit)
        {
            // item.path is UTF-8; use u8path() so extension() isn't confused by
            // non-ASCII bytes on Windows.
            Texture tex;
            std::string err;
            std::string ext =
                std::filesystem::u8path(item.path).extension().string();
            for (char& c : ext)
            {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            const bool isHdr = (ext == ".hdr");
            const bool ok    = isHdr ? tex.loadHDR(item.path, err)
                                     : tex.loadEXR(item.path, err);

            if (!ok || !tex.isHdr() || tex.width == 0 || tex.height == 0)
            {
                ready.errorMsg = ok ? "Expected HDR (float) image" : err;
            }
            else
            {
                ready.pixels.resize(static_cast<size_t>(THUMB_W) * THUMB_H * 4u);
                generateThumbnail(tex.floatPixels(), tex.width, tex.height,
                                  ready.pixels.data(), THUMB_W, THUMB_H);

                // ── Write to cache ────────────────────────────────────────────
                // Reuse the mtime/size we already fetched; write via tmp+rename
                // so a crash mid-write can't leave a corrupt cache file.
                if (!m_cacheDir.empty() && !statEc)
                {
                    const auto tmpPath =
                        cachePath.parent_path() /
                        (cachePath.filename().string() + ".tmp");
                    std::ofstream ofs(tmpPath, std::ios::binary);
                    if (ofs)
                    {
                        constexpr uint32_t kVersion =
                            ((static_cast<uint32_t>(THUMB_W) << 16) | THUMB_H) | 0xC0000000u;
                        const int64_t mt = static_cast<int64_t>(
                                               srcMtime.time_since_epoch().count());
                        const size_t byteCount = ready.pixels.size() * sizeof(uint16_t);
                        ofs.write(reinterpret_cast<const char*>(&kVersion), 4);
                        ofs.write(reinterpret_cast<const char*>(&mt),       8);
                        ofs.write(reinterpret_cast<const char*>(&srcSize),  8);
                        ofs.write(reinterpret_cast<const char*>(
                                      ready.pixels.data()), byteCount);
                        ofs.close();
                        std::error_code ec;
                        std::filesystem::rename(tmpPath, cachePath, ec);
                        if (ec)
                        {
                            std::filesystem::remove(tmpPath, ec);
                        }
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_readyMutex);
            m_readyQueue.push(std::move(ready));
        }
    }
}

// ─── Thumbnail generation ─────────────────────────────────────────────────────
// Runs on a worker thread.  Box-filter downsample + log-average auto-exposure
// → linear RGBA16F.  No tone-mapping: values above 1.0 are kept so that the
// scRGB display pipeline can show highlights above paper-white on HDR monitors.
// On SDR, values above 1.0 are clamped by the framebuffer write — acceptable
// for a preview thumbnail.

void HdriBrowser::generateThumbnail(const float* src, int srcW, int srcH,
                                    uint16_t* dst, int dstW, int dstH)
{
    // ── Log-average luminance for auto-exposure ───────────────────────────────
    double logSum = 0.0;
    const int totalPx = srcW * srcH;
    for (int i = 0; i < totalPx; ++i)
    {
        const float lum = 0.2126f * src[i * 4 + 0]
                        + 0.7152f * src[i * 4 + 1]
                        + 0.0722f * src[i * 4 + 2];
        logSum += std::log(static_cast<double>(std::fmax(lum, 1e-4f)));
    }
    const float logAvgLum = static_cast<float>(std::exp(logSum / totalPx));
    const float exposure   = 0.5f / logAvgLum;  // bring scene average to 0.5 linear

    // ── Box-filter downsample ─────────────────────────────────────────────────
    const float scaleX = static_cast<float>(srcW) / dstW;
    const float scaleY = static_cast<float>(srcH) / dstH;

    for (int dy = 0; dy < dstH; ++dy)
    {
        const int sy0 = static_cast<int>(dy * scaleY);
        const int sy1 = std::max(sy0 + 1, std::min(static_cast<int>((dy + 1) * scaleY), srcH));

        for (int dx = 0; dx < dstW; ++dx)
        {
            const int sx0 = static_cast<int>(dx * scaleX);
            const int sx1 = std::max(sx0 + 1, std::min(static_cast<int>((dx + 1) * scaleX), srcW));

            float r = 0.0f, g = 0.0f, b = 0.0f;
            int   n = 0;
            for (int sy = sy0; sy < sy1; ++sy)
            {
                for (int sx = sx0; sx < sx1; ++sx)
                {
                    const float* p = src + (static_cast<size_t>(sy) * srcW + sx) * 4;
                    r += p[0]; g += p[1]; b += p[2]; ++n;
                }
            }
            if (n > 0)
            {
                r /= n; g /= n; b /= n;
            }

            // Apply exposure; clamp negatives (EXR files may contain negative values).
            uint16_t* out = dst + (static_cast<size_t>(dy) * dstW + dx) * 4;
            out[0] = f32ToF16(std::fmax(r * exposure, 0.0f));
            out[1] = f32ToF16(std::fmax(g * exposure, 0.0f));
            out[2] = f32ToF16(std::fmax(b * exposure, 0.0f));
            out[3] = f32ToF16(1.0f);
        }
    }
}

// ─── Folder management ────────────────────────────────────────────────────────

void HdriBrowser::clearEntries(VulkanContext& vkCtx)
{
    stopWorkers();

    // Drain the ready queue before touching entries
    {
        std::lock_guard<std::mutex> lock(m_readyMutex);
        while (!m_readyQueue.empty())
        {
            m_readyQueue.pop();
        }
    }

    for (auto& e : m_entries)
    {
        vkCtx.destroyImGuiTexture(e.texture);
    }
    m_entries.clear();

    startWorkers();
}

void HdriBrowser::setFolder(VulkanContext& vkCtx, const std::string& folderPath)
{
    clearEntries(vkCtx);
    m_folder = folderPath;

    // Interpret folderPath as UTF-8 (NFD always returns UTF-8 on all platforms).
    // On Windows, std::filesystem::path(std::string) uses the ANSI code page, so
    // non-ASCII folder names would produce a broken path.  u8path() decodes the
    // bytes as UTF-8 and stores them as UTF-16 internally.
    const std::filesystem::path fPath = std::filesystem::u8path(folderPath);

    // Collect matching files first so they can be sorted before enqueueing.
    // Root-level files are pushed to the front of the work queue so they
    // appear ready in the browser before the deeper subdirectory files do.
    struct ScannedFile
    {
        std::filesystem::path absPath;
        std::filesystem::path relPath;  // relative to fPath
    };
    std::vector<ScannedFile> scanned;

    try
    {
        // skip_permission_denied: silently skip sub-folders we can't read instead
        // of aborting the whole scan with an exception.
        const auto iterOpts = std::filesystem::directory_options::skip_permission_denied;
        for (const auto& dirEntry :
             std::filesystem::recursive_directory_iterator(fPath, iterOpts))
        {
            // Use the no-ec overload so it reads from the cached directory-entry
            // status (fast, no extra syscall).  The ec overload on MSVC writes back
            // to the error_code after each call; if the same ec was passed to the
            // iterator constructor, a non-zero write here stops iteration early.
            if (!dirEntry.is_regular_file())
            {
                continue;
            }

            // Case-insensitive extension check — tolower handles .EXR / .Exr / etc.
            std::string ext = dirEntry.path().extension().string();
            for (char& c : ext)
            {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            if (ext != ".exr" && ext != ".hdr")
            {
                continue;
            }

            scanned.push_back({
                dirEntry.path(),
                std::filesystem::relative(dirEntry.path(), fPath)
            });
        }
    }
    catch (const std::exception& /*e*/)
    {
        // Root directory not accessible — m_entries stays empty and the UI will
        // show "No .exr or .hdr files found in this folder."
    }

    // Sort: root files (no parent component) first, then subdirectory files.
    // Within each group, keep the natural alphabetical order from the iterator.
    std::stable_sort(scanned.begin(), scanned.end(),
        [](const ScannedFile& a, const ScannedFile& b)
        {
            const bool aRoot = a.relPath.parent_path().empty();
            const bool bRoot = b.relPath.parent_path().empty();
            if (aRoot != bRoot)
            {
                return aRoot;               // root before subdir
            }
            return a.relPath < b.relPath;   // alphabetical within each group
        });

    // Build entries and work queue in sorted order.
    {
        std::lock_guard<std::mutex> lock(m_workMutex);
        for (auto& f : scanned)
        {
            const int idx = static_cast<int>(m_entries.size());
            ThumbEntry te;
            // u8string() gives UTF-8 on every platform.  On Windows, .string()
            // would use the ANSI code page and corrupt paths containing ä/ö/å etc.
            te.path  = f.absPath.u8string();
            // Show the path relative to the chosen root so files in different
            // sub-folders with the same filename are distinguishable in the grid.
            te.name  = f.relPath.u8string();
            te.state = ThumbState::Loading;
            m_entries.push_back(std::move(te));
            m_workQueue.push({ idx, m_entries.back().path });
        }
    }

    m_workCv.notify_all();
}

// ─── Main-thread GPU upload ───────────────────────────────────────────────────

void HdriBrowser::uploadPending(VulkanContext& vkCtx)
{
    // Process a folder change queued by draw() on the previous frame.
    if (!m_pendingFolder.empty())
    {
        setFolder(vkCtx, m_pendingFolder);
        m_pendingFolder.clear();
    }

    // Drain the ready queue and upload all completed thumbnails to GPU.
    std::vector<ReadyItem> batch;
    {
        std::lock_guard<std::mutex> lock(m_readyMutex);
        while (!m_readyQueue.empty())
        {
            batch.push_back(std::move(m_readyQueue.front()));
            m_readyQueue.pop();
        }
    }

    for (auto& item : batch)
    {
        if (item.idx < 0 || item.idx >= static_cast<int>(m_entries.size()))
        {
            continue;
        }

        auto& entry = m_entries[item.idx];
        if (item.errorMsg.empty())
        {
            entry.texture = vkCtx.createImGuiTexture(
                THUMB_W, THUMB_H, item.pixels.data());
            entry.state = ThumbState::GPUReady;
        }
        else
        {
            entry.state    = ThumbState::Error;
            entry.errorMsg = item.errorMsg;
        }
    }
}

// ─── Shutdown ─────────────────────────────────────────────────────────────────

void HdriBrowser::shutdown(VulkanContext& vkCtx)
{
    stopWorkers();

    {
        std::lock_guard<std::mutex> lock(m_readyMutex);
        while (!m_readyQueue.empty())
        {
            m_readyQueue.pop();
        }
    }

    for (auto& e : m_entries)
    {
        vkCtx.destroyImGuiTexture(e.texture);
    }
    m_entries.clear();
}

// ─── ImGui panel ─────────────────────────────────────────────────────────────

bool HdriBrowser::draw(bool* open, std::string& selectedPath)
{
    ImGui::SetNextWindowSize(ImVec2(900.0f, 600.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("HDRI Browser", open))
    {
        ImGui::End();
        return false;
    }

    bool selected = false;

    // ── Toolbar ───────────────────────────────────────────────────────────────
    if (ImGui::Button("Select Folder..."))
    {
        nfdu8char_t* outPath = nullptr;
        if (NFD_PickFolderU8(&outPath, nullptr) == NFD_OKAY)
        {
            // Store the folder; setFolder() (which needs VulkanContext) is
            // called from uploadPending() at the start of the next frame.
            m_pendingFolder = reinterpret_cast<const char*>(outPath);
            NFD_FreePathU8(outPath);
        }
    }

    ImGui::SameLine();
    if (!m_folder.empty())
    {
        ImGui::TextUnformatted(m_folder.c_str());
    }
    else
    {
        ImGui::TextDisabled("No folder selected");
    }

    // Loading progress indicator
    {
        int nLoading = 0;
        for (const auto& e : m_entries)
        {
            if (e.state == ThumbState::Loading)
            {
                ++nLoading;
            }
        }
        if (nLoading > 0)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("  (%d loading...)", nLoading);
        }
    }

    // ── Size selector — right-aligned on the same toolbar row ────────────────
    {
        static const char* const sizeLabels[] = { "Large", "Medium", "Small" };
        const float comboW = 90.0f;
        ImGui::SameLine(ImGui::GetContentRegionMax().x - comboW);
        ImGui::SetNextItemWidth(comboW);
        ImGui::Combo("##thumbsz", &m_thumbSizeIdx, sizeLabels, 3);
    }

    ImGui::Separator();

    // ── Thumbnail grid ────────────────────────────────────────────────────────
    if (m_entries.empty())
    {
        if (!m_folder.empty())
        {
            ImGui::TextDisabled("No .exr or .hdr files found in this folder.");
        }
        ImGui::End();
        return false;
    }

    // Display dimensions per size level (pixel data is always THUMB_W×THUMB_H).
    static const int kDispW[] = { THUMB_W, 192, 128 };
    static const int kDispH[] = { THUMB_H,  96,  64 };
    const int dispW = kDispW[m_thumbSizeIdx];
    const int dispH = kDispH[m_thumbSizeIdx];

    const float padding = 8.0f;
    const float labelH  = ImGui::GetTextLineHeightWithSpacing();

    // ImageButton adds FramePadding on each side of the image; account for it
    // so that cellW, cellH, and the label wrap position are all exact.
    const float fp  = ImGui::GetStyle().FramePadding.x;
    const float fpy = ImGui::GetStyle().FramePadding.y;
    const float cellW = static_cast<float>(dispW) + 2.0f * fp  + padding;
    const float cellH = static_cast<float>(dispH) + 2.0f * fpy + labelH + padding;

    ImGui::BeginChild("##thumbgrid", ImVec2(0.0f, 0.0f), false,
                      ImGuiWindowFlags_HorizontalScrollbar);

    // Query available width INSIDE the child so the column count accounts
    // for the child's own scrollbar reservation and inner padding.
    // Measuring in the parent window (before BeginChild) gives a wider value,
    // causing the rightmost column to overflow the child's clip rect.
    const float avail = ImGui::GetContentRegionAvail().x;

    // n items with (n-1) gaps: row = n*cellW - padding ≤ avail → n ≤ (avail+padding)/cellW.
    const int   cols  = std::max(1, static_cast<int>((avail + padding) / cellW));

    // thumbSz is the *image* size passed to ImageButton; the button's outer size
    // (and therefore the effective cell width) is thumbSz + 2 × FramePadding.
    const ImVec2 thumbSz (static_cast<float>(dispW),              static_cast<float>(dispH));
    // Error / Loading use Button with an explicit outer size that matches the
    // ImageButton outer size so all three states occupy the same cell width.
    const ImVec2 buttonSz(static_cast<float>(dispW) + 2.0f * fp,
                          static_cast<float>(dispH) + 2.0f * fpy);

    // ── Build folder groups ───────────────────────────────────────────────────
    // std::map keeps folder keys alphabetically sorted.
    // Files directly in the root directory map to the empty-string key "".
    std::map<std::string, std::vector<int>> groups;
    for (int i = 0; i < static_cast<int>(m_entries.size()); ++i)
    {
        const std::string folder =
            std::filesystem::u8path(m_entries[i].name)
                .parent_path()
                .generic_u8string();
        groups[folder].push_back(i);
    }

    // Show folder headers only when there is more than one group, or when the
    // single group is a named subfolder (not the root).
    const bool multiGroup = groups.size() > 1
                         || (groups.size() == 1 && !groups.begin()->first.empty());

    // ── Render each folder group ──────────────────────────────────────────────
    for (auto& [folder, indices] : groups)
    {
        if (multiGroup)
        {
            const std::string header = folder.empty() ? "/" : folder;
            ImGui::SeparatorText(header.c_str());
        }

        int col = 0;
        for (int idx : indices)
        {
            auto& e = m_entries[idx];

            // Bare filename for the label; keep e.name (full relative path) for tooltips.
            const std::string filename =
                std::filesystem::u8path(e.name).filename().u8string();

            if (col > 0)
            {
                ImGui::SameLine(0.0f, padding);
            }

            ImGui::PushID(idx);
            ImGui::BeginGroup();

            const bool isActive = !m_activePath.empty() && (e.path == m_activePath);

            if (e.state == ThumbState::GPUReady && e.texture.valid())
            {
                if (isActive)
                {
                    ImGui::PushStyleColor(ImGuiCol_Button,
                                          ImVec4(0.20f, 0.50f, 0.90f, 0.60f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                          ImVec4(0.30f, 0.60f, 1.00f, 0.80f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                          ImVec4(0.10f, 0.40f, 0.80f, 1.00f));
                }

                if (ImGui::ImageButton("##img",
                        reinterpret_cast<ImTextureID>(e.texture.descSet), thumbSz))
                {
                    selectedPath = e.path;
                    m_activePath = e.path;
                    selected     = true;
                }

                if (isActive)
                {
                    ImGui::PopStyleColor(3);
                }

                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("%s", e.name.c_str());
                }
            }
            else if (e.state == ThumbState::Error)
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.40f, 0.10f, 0.10f, 1.0f));
                ImGui::Button("##err", buttonSz);
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Error: %s\n%s", e.name.c_str(), e.errorMsg.c_str());
                }
            }
            else
            {
                // Loading placeholder — dark background + rotating arc spinner
                ImGui::PushStyleColor(ImGuiCol_Button,
                                      ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                      ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
                ImGui::Button("##load", buttonSz);
                ImGui::PopStyleColor(2);

                // Arc spinner drawn directly onto the window draw list so it
                // sits on top of the button without disturbing the ImGui layout.
                const ImVec2 bMin   = ImGui::GetItemRectMin();
                const ImVec2 bMax   = ImGui::GetItemRectMax();
                const ImVec2 center = ImVec2((bMin.x + bMax.x) * 0.5f,
                                             (bMin.y + bMax.y) * 0.5f);
                const float  radius = std::min(bMax.x - bMin.x,
                                               bMax.y - bMin.y) * 0.22f;

                constexpr float kPi    = 3.14159265358979f;
                constexpr float kTwoPi = 2.0f * kPi;
                const float     t      = static_cast<float>(ImGui::GetTime());

                // Rotation: one full spin every ~0.33 s
                const float rotation = t * kTwoPi / 0.33f;

                // Sweep oscillates between ~20° and ~280° on a ~1.5 s sine cycle
                // so the arc visibly stretches and compresses while it spins.
                constexpr float kMinSweep = kPi * 0.11f;   // ~20°
                constexpr float kMaxSweep = kPi * 1.55f;   // ~279°
                const float sweepT     = 0.5f + 0.5f * std::sinf(t * kTwoPi / 1.5f);
                const float sweep      = kMinSweep + (kMaxSweep - kMinSweep) * sweepT;

                const float angleStart = rotation;
                const float angleEnd   = rotation + sweep;

                ImDrawList* dl = ImGui::GetWindowDrawList();

                // Faint full-circle track
                dl->AddCircle(center, radius, IM_COL32(255, 255, 255, 10), 32, 4.0f);
                // Bright animated arc
                dl->PathArcTo(center, radius, angleStart, angleEnd, 24);
                dl->PathStroke(IM_COL32(200, 210, 255, 80), false, 2.5f);

                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Loading: %s", e.name.c_str());
                }
            }

            // ── File name label ───────────────────────────────────────────────
            // Show only the bare filename (no subfolder prefix).
            // Wrap at the button's outer width (image + 2 × FramePadding).
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + buttonSz.x);
            if (isActive)
            {
                ImGui::TextColored(ImVec4(0.40f, 0.75f, 1.00f, 1.0f), "%s",
                                   filename.c_str());
            }
            else
            {
                ImGui::TextUnformatted(filename.c_str());
            }
            ImGui::PopTextWrapPos();

            // Zero-size Dummy after the label.  Its purpose is purely to extend
            // the group's bounding-box max-y to include the trailing ItemSpacing.y
            // that follows the label.  Without it, EndGroup's ItemSize reports a
            // height that is one ItemSpacing.y short, making the row advance by
            // cellH - ItemSpacing.y instead of cellH.  A Dummy placed *outside*
            // EndGroup would overwrite CursorPosPrevLine.y and break SameLine's
            // row-start y for subsequent columns.
            ImGui::Dummy(ImVec2(0.0f, 0.0f));

            ImGui::EndGroup();
            ImGui::PopID();

            ++col;
            if (col >= cols)
            {
                col = 0;
            }
        }
    }

    ImGui::EndChild();
    ImGui::End();
    return selected;
}

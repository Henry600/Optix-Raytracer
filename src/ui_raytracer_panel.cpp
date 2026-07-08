#include "application.h"

#include <nfd.h>
#include <filesystem>
#include <string>

void Application::drawRaytracerPanel()
{
    ImGui::Begin("Raytracer");

    // ── Performance stats ─────────────────────────────────────────────────────
    ImGui::Text("GPU: %s", m_deviceName.c_str());
    ImGui::Text("     SM %d.%d  |  %llu MB VRAM",
        m_deviceComputeMajor, m_deviceComputeMinor,
        static_cast<unsigned long long>(m_deviceMemoryMB));
    ImGui::Text("Resolution: %d x %d", m_viewportWidth, m_viewportHeight);

    if (m_frameTimeMs > 0.0f)
    {
        const float    fps      = 1000.0f / m_frameTimeMs;
        const double   raysPerS = static_cast<double>(m_viewportWidth)
                                * static_cast<double>(m_viewportHeight) * fps;

        ImGui::Text("Frame: %.2f ms  (%.0f fps)", m_frameTimeMs, fps);

        if (raysPerS >= 1.0e9)
        {
            ImGui::Text("Rays: %.2f Grays/s", raysPerS * 1.0e-9);
        }
        else if (raysPerS >= 1.0e6)
        {
            ImGui::Text("Rays: %.2f Mrays/s", raysPerS * 1.0e-6);
        }
        else
        {
            ImGui::Text("Rays: %.2f Krays/s", raysPerS * 1.0e-3);
        }
    }
    else
    {
        ImGui::TextDisabled("Frame: --");
    }

    ImGui::Separator();

    if (ImGui::Button("Open glTF..."))
    {
        nfdu8char_t*    outPath = nullptr;
        nfdfilteritem_t filters[] = { { "glTF Scene", "gltf,glb" } };
        if (NFD_OpenDialogU8(&outPath, filters, 1, nullptr) == NFD_OKAY)
        {
            loadScene(reinterpret_cast<const char*>(outPath));
            NFD_FreePathU8(outPath);
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Open Splat..."))
    {
        nfdu8char_t*    outPath = nullptr;
        nfdfilteritem_t filters[] = { { "Gaussian Splat", "sog,json" } };
        if (NFD_OpenDialogU8(&outPath, filters, 1, nullptr) == NFD_OKAY)
        {
            loadSplat(reinterpret_cast<const char*>(outPath));
            NFD_FreePathU8(outPath);
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Open Env Map..."))
    {
        nfdu8char_t*    outPath = nullptr;
        nfdfilteritem_t filters[] = { { "HDR Image", "exr,hdr" } };
        if (NFD_OpenDialogU8(&outPath, filters, 1, nullptr) == NFD_OKAY)
        {
            const std::string p = reinterpret_cast<const char*>(outPath);
            loadEnvMap(p);
            m_hdriBrowser.setActivePath(p);
            NFD_FreePathU8(outPath);
        }
    }

    if (m_envMap.gpuTex != 0)
    {
        ImGui::SameLine();
        if (ImGui::Button("Clear"))
        {
            m_envMap.free();
            m_envMapPath.clear();
            m_envMapError.clear();
            m_accumDirty = true;
        }
    }

    if (!m_sceneFilePath.empty())
    {
        const std::string filename =
            std::filesystem::path(m_sceneFilePath).filename().string();
        ImGui::Text("Scene: %s", filename.c_str());
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("%s", m_sceneFilePath.c_str());
        }
    }
    else
    {
        ImGui::TextDisabled("No scene loaded");
    }

    if (!m_loadError.empty())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                           "Error: %s", m_loadError.c_str());
    }

    if (!m_envMapPath.empty())
    {
        ImGui::Text("Env: %s  (%d x %d)", m_envMapPath.c_str(),
                    m_envMap.width, m_envMap.height);
    }
    else if (!m_envMapError.empty())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                           "Error: %s", m_envMapError.c_str());
    }
    else
    {
        ImGui::TextDisabled("No environment map");
    }

    if (ImGui::SliderFloat("Env Exposure (EV)", &m_envExposure, -10.0f, 10.0f, "%.1f"))
    {
        m_accumDirty = true;
    }

    ImGui::Separator();
    ImGui::Text("Meshes: %d  Materials: %d  Textures: %d",
        static_cast<int>(m_scene->meshes().size()),
        static_cast<int>(m_scene->materials().size()),
        static_cast<int>(m_scene->textures().size()));

    ImGui::Separator();
    const Camera& cam = m_scene->camera();
    ImGui::Text("Camera: %s", cam.name.c_str());
    ImGui::Text("Position: (%.2f, %.2f, %.2f)",
        cam.transform.m[0][3],
        cam.transform.m[1][3],
        cam.transform.m[2][3]);
    {
        const float aspect      = static_cast<float>(m_viewportWidth)
                                 / static_cast<float>(m_viewportHeight);
        const float sH          = cam.sensorSize / aspect;
        const float derivedFov  = 2.0f * std::atan(sH / (2.0f * cam.focalLength));
        ImGui::Text("FOV: %.1f deg (%.0f mm / %.0f mm sensor)",
            derivedFov * (180.0f / 3.14159265f), cam.focalLength, cam.sensorSize);
    }

    ImGui::Separator();
    if (m_scene->hasAccel())
    {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f), "AS: ready");
    }
    else if (!m_scene->empty())
    {
        ImGui::TextDisabled("AS: build failed");
    }
    else
    {
        ImGui::TextDisabled("AS: no geometry");
    }

    // ── Path tracing progress ─────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Text("Samples: %u", m_sampleCount);
    if (ImGui::Button("Reset Accumulation"))
    {
        m_accumDirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Denoiser", &m_denoiserEnabled))
    {
        m_accumDirty = true;
    }
    if (m_denoiserEnabled)
    {
        ImGui::DragInt("Denoise every N samples", &m_denoiserInterval, 1, 1, 10000);
    }

    // ── HDR output (scRGB swapchain) ──────────────────────────────────────────
    ImGui::Separator();
    {
        const bool scRgb = m_vkCtx.isScRgbSwapchain();
        if (!scRgb)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Checkbox("HDR Output", &m_hdrOutput))
        {
            // The display encode changed — a cached denoised frame is stale;
            // the live raygen output picks up the new encode next launch.
            m_hasValidDenoisedFrame = false;
        }
        if (!scRgb)
        {
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("(no HDR display)");
        }

        // Update the UI pipeline scale whenever the scRGB state changes (e.g.
        // after a swapchain recreation triggered by moving to a different display).
        static bool prevScRgb = false;
        if (scRgb != prevScRgb)
        {
            prevScRgb = scRgb;
            m_vkCtx.setUiScale(scRgb ? (m_paperWhiteNits / 80.0f) : 1.0f);
        }

        // Paper-white slider only applies when the scRGB swapchain is active.
        if (!scRgb)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::SliderFloat("Paper White (nits)", &m_paperWhiteNits, 80.0f, 480.0f, "%.0f"))
        {
            m_vkCtx.setUiScale(m_paperWhiteNits / 80.0f);
        }
        if (!scRgb)
        {
            ImGui::EndDisabled();
        }
    }

    ImGui::Separator();
    if (m_shaderError.empty())
    {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f), "Shader: OK");
        ImGui::TextDisabled("(auto-reloads on PTX change)");
    }
    else
    {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.2f, 1.0f), "Shader error — last good pipeline active");
        ImGui::TextWrapped("%s", m_shaderError.c_str());
    }

    ImGui::End();
}

// ui_environment_panel.cpp — sky mode, environment map loading, and the HDRI browser.

#include "application.h"

#include <nfd.h>
#include <string>

void Application::drawEnvironmentPanel()
{
    ImGui::Begin("Environment");

    // ── Sky mode ──────────────────────────────────────────────────────────────
    {
        int mode = m_skyMode;
        if (ImGui::Combo("Sky Mode", &mode, "Constant\0Procedural\0HDRI\0"))
        {
            m_skyMode    = mode;
            m_accumDirty = true;
        }
    }

    if (ImGui::ColorEdit3("Sky Color", &m_skyColor.x))
    {
        m_accumDirty = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip(
            "Tints every sky mode.\n"
            "In Constant mode this is the sky's exact colour.");
    }

    if (ImGui::SliderAngle("Rotation", &m_envMapRotation, -180.0f, 180.0f))
    {
        m_accumDirty = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Also adjustable by Shift + right-drag in the viewport.");
    }

    if (ImGui::SliderFloat("Env Exposure (EV)", &m_envExposure, -10.0f, 10.0f, "%.1f"))
    {
        m_accumDirty = true;
    }

    ImGui::Separator();

    // ── HDRI source ───────────────────────────────────────────────────────────
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
            m_skyMode    = SKY_MODE_PROCEDURAL;
            m_accumDirty = true;
        }
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

    ImGui::End();

    // ── HDRI Browser window (always visible, no close button) ────────────────
    {
        std::string selected;
        if (m_hdriBrowser.draw(nullptr, selected) && !selected.empty())
        {
            loadEnvMap(selected);
            m_hdriBrowser.setActivePath(selected);
        }
    }
}

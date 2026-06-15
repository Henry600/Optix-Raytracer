#include "application.h"

#include <nfd.h>
#include <string>

void Application::drawResourcesPanel()
{
    ImGui::Begin("Resources");

    // ── Materials ─────────────────────────────────────────────────────────────
    {
        auto& mats = m_scene->materials();
        const std::string matsHeader = "Materials (" + std::to_string(mats.size()) + ")";
        bool anyMatChanged = false;

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 8.0f));
        const bool matsOpen = ImGui::CollapsingHeader(matsHeader.c_str(),
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap);
        {
            const float btnH = ImGui::GetFrameHeight();
            ImGui::SameLine(ImGui::GetContentRegionMax().x - btnH);
            if (ImGui::Button("+##addmat", ImVec2(btnH, btnH)))
            {
                m_scene->addMaterial(MaterialData{}, "New Material");
                uploadMaterials();
                m_accumDirty = true;
            }
        }
        ImGui::PopStyleVar();
        if (matsOpen)
        {
            if (mats.empty())
            {
                ImGui::TextDisabled("No materials loaded");
            }
            else
            {
                for (int i = 0; i < static_cast<int>(mats.size()); ++i)
                {
                    ImGui::PushID(i);

                    const std::string& rawName = m_scene->materialName(i);
                    const std::string  header  = rawName.empty()
                                               ? ("Material " + std::to_string(i))
                                               : rawName;

                    const bool matOpen = ImGui::CollapsingHeader(header.c_str(),
                        ImGuiTreeNodeFlags_AllowOverlap);

                    // When collapsed, show a compact clickable colour swatch
                    // inline with the header — clicking it opens a colour picker.
                    if (!matOpen)
                    {
                        const float swatchSize = ImGui::GetFrameHeight();
                        ImGui::SameLine(ImGui::GetContentRegionMax().x - swatchSize);
                        if (ImGui::ColorEdit3("##albedo_swatch", &mats[i].albedo.x,
                                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float))
                        {
                            anyMatChanged = true;
                        }
                    }

                    if (matOpen)
                    {
                        // Albedo colour + texture selector on the same line
                        if (ImGui::ColorEdit3("Albedo", &mats[i].albedo.x, ImGuiColorEditFlags_Float))
                        {
                            anyMatChanged = true;
                        }

                        ImGui::SameLine();
                        {
                            const auto& textures = m_scene->textures();
                            const int   cur      = mats[i].albedoTexture;
                            const std::string preview = (cur < 0 || cur >= (int)textures.size())
                                ? "None"
                                : (textures[cur].name.empty()
                                    ? "Texture " + std::to_string(cur)
                                    : textures[cur].name);

                            ImGui::PushItemWidth(-1.0f);  // fill remaining line width
                            if (ImGui::BeginCombo("##albedoTex", preview.c_str()))
                            {
                                if (ImGui::Selectable("None", cur < 0))
                                {
                                    mats[i].albedoTexture = -1;
                                    anyMatChanged = true;
                                }

                                for (int t = 0; t < (int)textures.size(); ++t)
                                {
                                    const std::string label = textures[t].name.empty()
                                        ? ("Texture " + std::to_string(t))
                                        : textures[t].name;
                                    if (ImGui::Selectable(label.c_str(), cur == t))
                                    {
                                        mats[i].albedoTexture = t;
                                        anyMatChanged = true;
                                    }
                                }
                                ImGui::EndCombo();
                            }
                            ImGui::PopItemWidth();
                        }
                        if (ImGui::SliderFloat("Roughness", &mats[i].roughness, 0.0f, 1.0f))
                        {
                            anyMatChanged = true;
                        }
                        ImGui::SameLine();
                        {
                            const auto& textures = m_scene->textures();
                            const int   cur      = mats[i].roughnessTexture;
                            const std::string preview = (cur < 0 || cur >= (int)textures.size())
                                ? "None"
                                : (textures[cur].name.empty()
                                    ? "Texture " + std::to_string(cur)
                                    : textures[cur].name);
                            ImGui::PushItemWidth(-1.0f);
                            if (ImGui::BeginCombo("##roughnessTex", preview.c_str()))
                            {
                                if (ImGui::Selectable("None", cur < 0))
                                {
                                    mats[i].roughnessTexture = -1;
                                    anyMatChanged = true;
                                }
                                for (int t = 0; t < (int)textures.size(); ++t)
                                {
                                    const std::string label = textures[t].name.empty()
                                        ? ("Texture " + std::to_string(t))
                                        : textures[t].name;
                                    if (ImGui::Selectable(label.c_str(), cur == t))
                                    {
                                        mats[i].roughnessTexture = t;
                                        anyMatChanged = true;
                                    }
                                }
                                ImGui::EndCombo();
                            }
                            ImGui::PopItemWidth();
                        }
                        if (ImGui::SliderFloat("Metallic",  &mats[i].metallic,  0.0f, 1.0f))
                        {
                            anyMatChanged = true;
                        }
                        if (ImGui::ColorEdit3("Emission", &mats[i].emission.x, ImGuiColorEditFlags_Float))
                        {
                            anyMatChanged = true;
                        }
                        ImGui::SameLine();
                        {
                            const auto& textures = m_scene->textures();
                            const int   cur      = mats[i].emissionTexture;
                            const std::string preview = (cur < 0 || cur >= (int)textures.size())
                                ? "None"
                                : (textures[cur].name.empty()
                                    ? "Texture " + std::to_string(cur)
                                    : textures[cur].name);
                            ImGui::PushItemWidth(-1.0f);
                            if (ImGui::BeginCombo("##emissionTex", preview.c_str()))
                            {
                                if (ImGui::Selectable("None", cur < 0))
                                {
                                    mats[i].emissionTexture = -1;
                                    anyMatChanged = true;
                                }
                                for (int t = 0; t < (int)textures.size(); ++t)
                                {
                                    const std::string label = textures[t].name.empty()
                                        ? ("Texture " + std::to_string(t))
                                        : textures[t].name;
                                    if (ImGui::Selectable(label.c_str(), cur == t))
                                    {
                                        mats[i].emissionTexture = t;
                                        anyMatChanged = true;
                                    }
                                }
                                ImGui::EndCombo();
                            }
                            ImGui::PopItemWidth();
                        }
                        if (ImGui::DragFloat("Emission Scale", &mats[i].emissionScale, 0.1f, 0.0f, 1000.0f, "%.2f"))
                        {
                            anyMatChanged = true;
                        }
                        if (ImGui::SliderFloat("Transmission", &mats[i].transmission, 0.0f, 1.0f, "%.3f"))
                        {
                            anyMatChanged = true;
                        }
                        if (ImGui::SliderFloat("IOR",          &mats[i].ior,          1.0f, 3.0f, "%.3f"))
                        {
                            anyMatChanged = true;
                        }
                        if (ImGui::DragFloat("Absorption Dist.", &mats[i].absorptionDistance, 0.002f, 0.0001f, 1000.0f, "%.4f"))
                        {
                            anyMatChanged = true;
                        }
                        {
                            float3& sc = mats[i].scatteringCoeff;

                            // Scale slider — adjusts all channels proportionally,
                            // preserving the R:G:B ratios set by the Rayleigh button.
                            float scale = fmaxf(fmaxf(sc.x, sc.y), sc.z);
                            if (ImGui::DragFloat("Scatter Scale", &scale, 0.01f, 0.0f, 50.0f, "%.3f"))
                            {
                                scale = fmaxf(scale, 0.0f);
                                const float oldMax = fmaxf(fmaxf(sc.x, sc.y), sc.z);
                                if (oldMax > 1e-6f)
                                {
                                    const float ratio = scale / oldMax;
                                    sc.x *= ratio;
                                    sc.y *= ratio;
                                    sc.z *= ratio;
                                }
                                else
                                {
                                    sc = make_float3(scale, scale, scale);
                                }
                                anyMatChanged = true;
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Rayleigh##scat"))
                            {
                                // λ^-4 ratios: R(680nm):G(550nm):B(440nm) → 0.174 : 0.405 : 1.0
                                const float base = scale > 0.0f ? scale : 1.0f;
                                sc = make_float3(0.174f * base, 0.405f * base, 1.0f * base);
                                anyMatChanged = true;
                            }
                            if (ImGui::DragFloat3("Scatter Coeff.", &sc.x, 0.01f, 0.0f, 50.0f, "%.3f"))
                            {
                                sc.x = fmaxf(sc.x, 0.0f);
                                sc.y = fmaxf(sc.y, 0.0f);
                                sc.z = fmaxf(sc.z, 0.0f);
                                anyMatChanged = true;
                            }
                        }
                        if (ImGui::SliderFloat("Scatter Aniso.", &mats[i].scatteringAnisotropy, -1.0f, 1.0f, "%.3f"))
                        {
                            anyMatChanged = true;
                        }
                        if (ImGui::SliderFloat("Clearcoat",      &mats[i].clearcoat,          0.0f, 1.0f, "%.3f"))
                        {
                            anyMatChanged = true;
                        }
                        if (ImGui::SliderFloat("Coat Roughness", &mats[i].clearcoatRoughness,  0.0f, 1.0f, "%.3f"))
                        {
                            anyMatChanged = true;
                        }

                        {
                            bool tw = mats[i].thinWalled != 0;
                            if (ImGui::Checkbox("Thin Walled", &tw))
                            {
                                mats[i].thinWalled = tw ? 1 : 0;
                                anyMatChanged = true;
                            }
                            if (ImGui::IsItemHovered())
                            {
                                ImGui::SetTooltip(
                                    "Shadow rays pass through this surface.\n"
                                    "Use for window glass to speed up interior lighting convergence.");
                            }
                        }

                        ImGui::Separator();
                        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.6f);
                        if (ImGui::DragFloat2("Tiling",  &mats[i].uvTransform.x, 0.01f, 0.0f, 0.0f, "%.3f"))
                        {
                            anyMatChanged = true;
                        }
                        if (ImGui::DragFloat2("Offset",  &mats[i].uvTransform.z, 0.01f, 0.0f, 0.0f, "%.3f"))
                        {
                            anyMatChanged = true;
                        }
                        ImGui::PopItemWidth();
                    }

                    ImGui::PopID();
                }
            }
        }

        if (anyMatChanged)
        {
            uploadMaterials();
            m_accumDirty = true;
        }
    }

    // ── Textures ──────────────────────────────────────────────────────────────
    ImGui::Separator();
    {
        const auto& textures  = m_scene->textures();
        const std::string texHeader = "Textures (" + std::to_string(textures.size()) + ")";

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 8.0f));
        const bool texOpen = ImGui::CollapsingHeader(texHeader.c_str());
        ImGui::PopStyleVar();
        if (texOpen)
        {
            if (ImGui::Button("Load Image..."))
            {
                nfdu8char_t* outPath = nullptr;
                nfdfilteritem_t filters[] = { { "Image Files", "png,jpg,jpeg,bmp,tga,exr,hdr" } };
                if (NFD_OpenDialogU8(&outPath, filters, 1, nullptr) == NFD_OKAY)
                {
                    loadTexture(reinterpret_cast<const char*>(outPath));
                    NFD_FreePathU8(outPath);
                }
            }
            ImGui::Separator();

            if (textures.empty())
            {
                ImGui::TextDisabled("No textures loaded");
            }
            else
            {
                for (int i = 0; i < static_cast<int>(textures.size()); ++i)
                {
                    const Texture& tex = textures[i];
                    const std::string label = tex.name.empty()
                        ? ("Texture " + std::to_string(i))
                        : tex.name;

                    ImGui::PushID(i);
                    ImGui::Bullet();
                    ImGui::Text("%-32s  %d \xc3\x97 %d  %s",
                        label.c_str(),
                        tex.width, tex.height,
                        tex.format == PixelFormat::RGBA32F ? "RGBA32F" : "RGBA8");
                    ImGui::PopID();
                }
            }
        }
    }

    ImGui::End();
}

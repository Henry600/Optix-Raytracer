#include "application.h"
#include "implicit_node.h"

#include <algorithm>
#include <cmath>
#include <string>

void Application::drawNodePropertiesPanel()
{
    ImGui::Begin("Node Properties");

    if (m_selectedNodeIdx < 0 || m_selectedNodeIdx >= static_cast<int>(m_scene->nodes().size()))
    {
        ImGui::TextDisabled("No node selected");
    }
    else
    {
        Node3D& node = m_scene->nodeAt(m_selectedNodeIdx);

        // ── Identity ──────────────────────────────────────────────────────────
        ImGui::Text("Type: %s", node.typeName());
        ImGui::Text("Name: %s", node.name.empty() ? "(unnamed)" : node.name.c_str());
        ImGui::Separator();

        // ── Gizmo operation selector ──────────────────────────────────────────
        ImGui::Text("Operation:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Translate", m_gizmoOp == ImGuizmo::TRANSLATE))
        {
            m_gizmoOp = ImGuizmo::TRANSLATE;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", m_gizmoOp == ImGuizmo::ROTATE))
        {
            m_gizmoOp = ImGuizmo::ROTATE;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", m_gizmoOp == ImGuizmo::SCALE))
        {
            m_gizmoOp = ImGuizmo::SCALE;
        }

        ImGui::Text("Space:    ");
        ImGui::SameLine();
        if (ImGui::RadioButton("Local", m_gizmoMode == ImGuizmo::LOCAL))
        {
            m_gizmoMode = ImGuizmo::LOCAL;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("World", m_gizmoMode == ImGuizmo::WORLD))
        {
            m_gizmoMode = ImGuizmo::WORLD;
        }

        ImGui::Separator();

        // ── Local transform — TRS editor via ImGuizmo decomposition ──────────
        if (ImGui::CollapsingHeader("Local Transform", ImGuiTreeNodeFlags_DefaultOpen))
        {
            // Convert our row-major Matrix4x4 to the column-major float[16] that
            // ImGuizmo::DecomposeMatrixToComponents / RecomposeMatrixFromComponents expect.
            float colMajor[16];
            mat4ToColMajor(node.localTransform, colMajor);

            float translation[3], rotation[3], scale[3];
            ImGuizmo::DecomposeMatrixToComponents(colMajor, translation, rotation, scale);

            // Reserve enough width on the left for the three drag widgets so the
            // labels ("Translation", "Rotation", "Scale") always remain visible.
            bool changed = false;
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.75f);
            changed |= ImGui::DragFloat3("Translation", translation, 0.01f);
            changed |= ImGui::DragFloat3("Rotation",    rotation,    0.1f, 0.f, 0.f, "%.2f deg");
            changed |= ImGui::DragFloat3("Scale",       scale,       0.01f);
            ImGui::PopItemWidth();

            if (changed)
            {
                ImGuizmo::RecomposeMatrixFromComponents(translation, rotation, scale, colMajor);
                node.localTransform = mat4FromColMajor(colMajor);
                m_scene->updateWorldTransforms(m_selectedNodeIdx);
                m_scene->rebuildTlas(m_optixContext);
                uploadEmissiveLights();
                m_accumDirty = true;
                syncFlyCameraFromNode(m_selectedNodeIdx);
            }
        }

        // ── World transform — read-only display ───────────────────────────────
        if (ImGui::CollapsingHeader("World Transform"))
        {
            float wColMajor[16];
            mat4ToColMajor(node.worldTransform, wColMajor);

            float wTranslation[3], wRotation[3], wScale[3];
            ImGuizmo::DecomposeMatrixToComponents(wColMajor, wTranslation, wRotation, wScale);

            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.75f);
            ImGui::BeginDisabled();
            ImGui::DragFloat3("Translation##world", wTranslation, 0.0f);
            ImGui::DragFloat3("Rotation##world",    wRotation,    0.0f, 0.0f, 0.0f, "%.2f deg");
            ImGui::DragFloat3("Scale##world",       wScale,       0.0f);
            ImGui::EndDisabled();
            ImGui::PopItemWidth();
        }

        // ── Type-specific content ─────────────────────────────────────────────
        if (MeshNode* meshNode = dynamic_cast<MeshNode*>(&node))
        {
            ImGui::Separator();
            ImGui::Text("Meshes: %d primitive(s)", static_cast<int>(meshNode->meshIndices.size()));

            auto& mats      = m_scene->materials();
            bool  anyChanged = false;

            // Ensure materialIndices is sized to match meshIndices.
            // Can be shorter for nodes created before this field existed.
            {
                const size_t needed  = meshNode->meshIndices.size();
                const size_t current = meshNode->materialIndices.size();
                for (size_t k = current; k < needed; ++k)
                {
                    const int mi  = meshNode->meshIndices[k];
                    const int def = (mi >= 0 && mi < static_cast<int>(mats.size()))
                        ? m_scene->meshes()[mi].materialIndex : 0;
                    meshNode->materialIndices.push_back(def);
                }
            }

            const auto matLabel = [&](int idx) -> std::string {
                const std::string& n = m_scene->materialName(idx);
                return n.empty() ? ("Material " + std::to_string(idx)) : n;
            };

            for (int j = 0; j < static_cast<int>(meshNode->meshIndices.size()); ++j)
            {
                int& matIdx = meshNode->materialIndices[j];

                ImGui::PushID(j);

                const std::string preview = (matIdx >= 0 && matIdx < static_cast<int>(mats.size()))
                    ? matLabel(matIdx) : "(none)";

                if (ImGui::BeginCombo("Material##mesh", preview.c_str()))
                {
                    for (int k = 0; k < static_cast<int>(mats.size()); ++k)
                    {
                        ImGui::PushID(k);
                        const bool selected = (k == matIdx);
                        if (ImGui::Selectable(matLabel(k).c_str(), selected))
                        {
                            matIdx     = k;
                            anyChanged = true;
                        }
                        if (selected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                        ImGui::PopID();
                    }
                    ImGui::EndCombo();
                }

                ImGui::PopID();
            }

            if (anyChanged)
            {
                buildSbt();
                m_accumDirty = true;
            }
        }
        else if (ImplicitNode* implNode = dynamic_cast<ImplicitNode*>(&node))
        {
            ImGui::Separator();

            static const char* kTypeNames[] = { "Sphere", "Box", "Cylinder" };
            int typeIdx = static_cast<int>(implNode->type);
            ImGui::Text("Shape:");
            ImGui::SameLine();
            bool typeChanged = false;
            for (int k = 0; k < 3; ++k)
            {
                if (k > 0) { ImGui::SameLine(); }
                if (ImGui::RadioButton(kTypeNames[k], typeIdx == k))
                {
                    typeIdx     = k;
                    typeChanged = true;
                }
            }
            if (typeChanged)
            {
                implNode->type = static_cast<ImplicitType>(typeIdx);
                try
                {
                    m_scene->buildAccel(m_optixContext);
                }
                catch (const std::exception& e)
                {
                    m_loadError = std::string("AS rebuild failed: ") + e.what();
                }
                buildSbt();
                uploadEmissiveLights();
                m_accumDirty = true;
            }

            ImGui::Separator();

            auto& mats = m_scene->materials();
            const auto matLabel = [&](int idx) -> std::string
            {
                const std::string& n = m_scene->materialName(idx);
                return n.empty() ? ("Material " + std::to_string(idx)) : n;
            };

            const std::string preview = (implNode->materialIndex >= 0
                && implNode->materialIndex < static_cast<int>(mats.size()))
                ? matLabel(implNode->materialIndex) : "(none)";

            if (ImGui::BeginCombo("Material##impl", preview.c_str()))
            {
                for (int k = 0; k < static_cast<int>(mats.size()); ++k)
                {
                    ImGui::PushID(k);
                    const bool selected = (k == implNode->materialIndex);
                    if (ImGui::Selectable(matLabel(k).c_str(), selected))
                    {
                        implNode->materialIndex = k;
                        buildSbt();
                        uploadEmissiveLights();
                        m_accumDirty = true;
                    }
                    if (selected) { ImGui::SetItemDefaultFocus(); }
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }
        }
        else if (dynamic_cast<CameraNode*>(&node))
        {
            ImGui::Separator();
            {
                {
                    Camera cam = m_scene->camera();

                    // Focal length — drives FOV together with sensor size
                    float fl = cam.focalLength;
                    if (ImGui::SliderFloat("Focal Length (mm)", &fl, 8.0f, 800.0f, "%.1f mm"))
                    {
                        cam.focalLength = fl;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // Sensor size (horizontal width)
                    float ss = cam.sensorSize;
                    if (ImGui::SliderFloat("Sensor Size (mm)", &ss, 1.0f, 100.0f, "%.1f mm"))
                    {
                        cam.sensorSize = ss;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // Depth of field toggle
                    bool dof = cam.dofEnabled;
                    if (ImGui::Checkbox("Depth of Field", &dof))
                    {
                        cam.dofEnabled = dof;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    // F-stop, focus distance, bokeh — all greyed out when DoF is disabled
                    if (!cam.dofEnabled)
                    {
                        ImGui::BeginDisabled();
                    }

                    float fs = cam.fStop;
                    if (ImGui::SliderFloat("F-Stop", &fs, 0.5f, 64.0f, "f/%.1f"))
                    {
                        cam.fStop = fs;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    float fd = cam.focusDistance;
                    if (ImGui::DragFloat("Focus Distance", &fd, 0.1f, 0.1f, 10000.0f, "%.2f m"))
                    {
                        cam.focusDistance = std::max(0.001f, fd);
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    float eb = cam.bokehEdgeBias;
                    if (ImGui::SliderFloat("Bokeh Edge Bias", &eb, 0.0f, 1.0f, "%.2f"))
                    {
                        cam.bokehEdgeBias = eb;
                        m_scene->setCamera(cam);
                        m_accumDirty = true;
                    }

                    if (!cam.dofEnabled)
                    {
                        ImGui::EndDisabled();
                    }

                    // Derived FOV display
                    const float aspect     = static_cast<float>(m_viewportWidth)
                                            / static_cast<float>(m_viewportHeight);
                    const float sH         = cam.sensorSize / aspect;
                    const float derivedFov = 2.0f * std::atan(sH / (2.0f * cam.focalLength));
                    ImGui::Text("FOV: %.1f deg  |  Aperture: %.1f mm",
                        derivedFov * (180.0f / 3.14159265f),
                        cam.focalLength / cam.fStop);
                    ImGui::TextDisabled("(transform driven by fly-cam controller)");
                }
            }
        }
    }

    ImGui::End();
}

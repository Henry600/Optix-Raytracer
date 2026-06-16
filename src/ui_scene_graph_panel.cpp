#include "application.h"
#include "implicit_node.h"
#include "matrix4x4.h"

#include <filesystem>
#include <memory>
#include <string>

static void drawNode3D(const Scene& scene, int nodeIdx,
                       int& selectedNodeIdx, int& duplicateNodeIdx)
{
    const Node3D& node = *scene.nodes()[nodeIdx];

    std::string label = node.name.empty()
        ? (std::string(node.typeName()) + " " + std::to_string(nodeIdx))
        : node.name;
    label += std::string("  <") + node.typeName() + ">";

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow
                             | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (nodeIdx == selectedNodeIdx)
    {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    if (node.children.empty())
    {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    }

    ImGui::PushID(nodeIdx);
    const bool open = ImGui::TreeNodeEx(label.c_str(), flags);

    if (ImGui::IsItemClicked())
    {
        selectedNodeIdx = nodeIdx;
    }

    if (ImGui::BeginPopupContextItem("##node_ctx"))
    {
        selectedNodeIdx = nodeIdx;   // right-click also selects the node
        if (ImGui::MenuItem("Duplicate"))
        {
            duplicateNodeIdx = nodeIdx;
        }
        ImGui::EndPopup();
    }

    if (open && !node.children.empty())
    {
        for (int childIdx : node.children)
        {
            drawNode3D(scene, childIdx, selectedNodeIdx, duplicateNodeIdx);
        }
        ImGui::TreePop();
    }

    ImGui::PopID();
}

void Application::drawSceneGraphPanel()
{
    ImGui::Begin("Scene Graph");

    // ── Add Implicit Shape button ─────────────────────────────────────────────
    if (ImGui::Button("Add Implicit Shape"))
    {
        ImGui::OpenPopup("add_implicit_popup");
    }
    if (ImGui::BeginPopup("add_implicit_popup"))
    {
        const char* shapeNames[] = { "Sphere", "Box", "Cylinder" };
        for (int k = 0; k < 3; ++k)
        {
            if (ImGui::MenuItem(shapeNames[k]))
            {
                auto implNode            = std::make_unique<ImplicitNode>();
                implNode->name           = shapeNames[k];
                implNode->type           = static_cast<ImplicitType>(k);
                implNode->materialIndex  = 0;
                implNode->localTransform = mat4Identity();
                const int idx = m_scene->addNode(std::move(implNode));
                m_scene->addRootNode(idx);
                m_scene->updateWorldTransforms(idx);
                try
                {
                    m_scene->buildAccel(m_optixContext);
                }
                catch (const std::exception& e)
                {
                    m_loadError = std::string("AS build failed: ") + e.what();
                }
                buildSbt();
                m_selectedNodeIdx = idx;
                m_accumDirty      = true;
            }
        }
        ImGui::EndPopup();
    }
    ImGui::Separator();

    if (m_scene->rootNodes().empty())
    {
        ImGui::TextDisabled("No scene loaded");
    }
    else
    {
        if (!m_sceneFilePath.empty())
        {
            ImGui::TextDisabled("%s",
                std::filesystem::path(m_sceneFilePath).filename().string().c_str());
            ImGui::Separator();
        }
        int duplicateNodeIdx = -1;
        for (int rootIdx : m_scene->rootNodes())
        {
            drawNode3D(*m_scene, rootIdx, m_selectedNodeIdx, duplicateNodeIdx);
        }

        if (duplicateNodeIdx >= 0)
        {
            const int newIdx = m_scene->duplicateSubtree(duplicateNodeIdx);
            try
            {
                m_scene->buildAccel(m_optixContext);
            }
            catch (const std::exception& e)
            {
                m_loadError = std::string("AS build failed: ") + e.what();
            }
            buildSbt();
            m_selectedNodeIdx = newIdx;
            m_accumDirty      = true;
        }
    }

    ImGui::End();
}

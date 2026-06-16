#include "application.h"
#include "icons.h"
#include "implicit_node.h"
#include "matrix4x4.h"

#include <filesystem>
#include <memory>
#include <string>

static const char* nodeIcon(const Node3D& node)
{
    if (dynamic_cast<const MeshNode*>(&node))   { return ICON_FA_CUBE; }
    if (dynamic_cast<const CameraNode*>(&node)) { return ICON_FA_CAMERA; }
    if (const auto* in = dynamic_cast<const ImplicitNode*>(&node))
    {
        switch (in->type)
        {
            case ImplicitType::Sphere:   return ICON_FA_CIRCLE;
            case ImplicitType::Box:      return ICON_FA_SQUARE;
            case ImplicitType::Cylinder: return ICON_FA_DATABASE;
        }
    }
    return ICON_FA_LAYER_GROUP;
}

static ImVec4 nodeIconColor(const Node3D& node)
{
    if (dynamic_cast<const MeshNode*>(&node))   { return { 0.40f, 0.70f, 1.00f, 1.0f }; }  // blue
    if (dynamic_cast<const CameraNode*>(&node)) { return { 1.00f, 0.85f, 0.30f, 1.0f }; }  // gold
    if (dynamic_cast<const ImplicitNode*>(&node)) { return { 0.50f, 1.00f, 0.60f, 1.0f }; } // green
    return { 0.75f, 0.75f, 0.75f, 1.0f };  // gray for groups
}

// Pending actions collected during the tree draw — processed after the walk completes
// so that mutations don't invalidate the node list mid-traversal.
struct NodeAction
{
    int duplicateNodeIdx = -1;
    int deleteNodeIdx    = -1;
};

static void drawNode3D(const Scene& scene, int nodeIdx,
                       int& selectedNodeIdx, NodeAction& action)
{
    const Node3D& node = *scene.nodes()[nodeIdx];

    const char* icon      = nodeIcon(node);
    const ImVec4 iconCol  = nodeIconColor(node);
    const std::string name = node.name.empty()
        ? (std::string(node.typeName()) + " " + std::to_string(nodeIdx))
        : node.name;

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

    // Render tree node with a hidden label so we can draw icon and name
    // separately with different colors.  Click/hover/selection still work
    // because SpanAvailWidth extends the item rect across the full row.
    const bool open = ImGui::TreeNodeEx("##node", flags);

    // Check click and context menu immediately after TreeNodeEx — before
    // SameLine adds other items that would shift the "last item" reference.
    if (ImGui::IsItemClicked())
    {
        selectedNodeIdx = nodeIdx;
    }
    if (ImGui::BeginPopupContextItem("##node_ctx"))
    {
        selectedNodeIdx = nodeIdx;
        if (ImGui::MenuItem("Duplicate"))
        {
            action.duplicateNodeIdx = nodeIdx;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Delete"))
        {
            action.deleteNodeIdx = nodeIdx;
        }
        ImGui::EndPopup();
    }

    // Draw colored icon and plain name on the same row as the tree arrow.
    ImGui::SameLine(0.0f, 2.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, iconCol);
    ImGui::TextUnformatted(icon);
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::TextUnformatted(name.c_str());

    if (open && !node.children.empty())
    {
        for (int childIdx : node.children)
        {
            drawNode3D(scene, childIdx, selectedNodeIdx, action);
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
            ImGui::TextDisabled("%s", std::filesystem::path(m_sceneFilePath).filename().string().c_str());
            ImGui::Separator();
        }
        NodeAction action;
        for (int rootIdx : m_scene->rootNodes())
        {
            drawNode3D(*m_scene, rootIdx, m_selectedNodeIdx, action);
        }

        if (action.duplicateNodeIdx >= 0)
        {
            const int newIdx = m_scene->duplicateSubtree(action.duplicateNodeIdx);
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

        if (action.deleteNodeIdx >= 0)
        {
            m_scene->deleteSubtree(action.deleteNodeIdx);
            if (!m_scene->nodeAlive(m_selectedNodeIdx))
            {
                m_selectedNodeIdx = -1;
            }
            try
            {
                m_scene->buildAccel(m_optixContext);
            }
            catch (const std::exception& e)
            {
                m_loadError = std::string("AS build failed: ") + e.what();
            }
            buildSbt();
            m_accumDirty = true;
        }
    }

    ImGui::End();
}

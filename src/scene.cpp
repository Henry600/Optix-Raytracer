#include "scene.h"
#include "implicit_node.h"
#include "matrix4x4.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

// For CUDA calls in uploadTextures / destroyTextureObjects
#define CUDA_CHECK_SCENE(call)                                                   \
    do {                                                                         \
        cudaError_t rc = (call);                                                 \
        if (rc != cudaSuccess)                                                   \
        {                                                                        \
            throw std::runtime_error(                                            \
                std::string("CUDA error in Scene: ") + cudaGetErrorString(rc)); \
        }                                                                        \
    } while (0)

Scene::Scene()
{
    addDefaultCameraNode();
}

void Scene::addDefaultCameraNode()
{
    auto camNode = std::make_unique<CameraNode>();
    camNode->name           = "Default Camera";
    camNode->localTransform = m_camera.transform;  // root node: local == world
    m_defaultCameraNodeIdx  = addNode(std::move(camNode));
    addRootNode(m_defaultCameraNodeIdx);
    updateWorldTransforms(m_defaultCameraNodeIdx);
}

int Scene::addMesh(Mesh mesh)
{
    m_meshes.push_back(std::move(mesh));
    return static_cast<int>(m_meshes.size()) - 1;
}

int Scene::addMaterial(MaterialData material, std::string name)
{
    m_materials.push_back(material);
    m_materialNames.push_back(std::move(name));
    return static_cast<int>(m_materials.size()) - 1;
}

int Scene::addTexture(Texture texture)
{
    m_textures.push_back(std::move(texture));
    return static_cast<int>(m_textures.size()) - 1;
}

const std::vector<Mesh>& Scene::meshes() const
{
    return m_meshes;
}

const std::vector<MaterialData>& Scene::materials() const
{
    return m_materials;
}

std::vector<MaterialData>& Scene::materials()
{
    return m_materials;
}

const std::vector<Texture>& Scene::textures() const
{
    return m_textures;
}

std::vector<Texture>& Scene::textures()
{
    return m_textures;
}

const std::string& Scene::materialName(int index) const
{
    if (index < 0 || index >= static_cast<int>(m_materialNames.size()))
    {
        throw std::out_of_range("Scene::materialName index out of range");
    }
    return m_materialNames[index];
}

const Camera& Scene::camera() const
{
    return m_camera;
}

void Scene::setCamera(Camera camera)
{
    m_camera = std::move(camera);
}

int Scene::addNode(std::unique_ptr<Node3D> node)
{
    // An imported camera node replaces the default one so the graph always
    // holds exactly one camera; the render camera follows the new node.
    if (m_defaultCameraNodeIdx >= 0 && dynamic_cast<CameraNode*>(node.get()))
    {
        const int idx = m_defaultCameraNodeIdx;
        m_rootNodes.erase(
            std::remove(m_rootNodes.begin(), m_rootNodes.end(), idx),
            m_rootNodes.end());
        m_nodes[idx]           = std::move(node);
        m_defaultCameraNodeIdx = -1;

        // Parent links of ancestors are wired before addNode is called, so the
        // world transform is already resolvable here.
        m_camera.transform = computeWorldTransform(idx);
        m_camera.view      = mat4RigidInverse(m_camera.transform);
        if (!m_nodes[idx]->name.empty())
        {
            m_camera.name = m_nodes[idx]->name;
        }
        return idx;
    }

    m_nodes.push_back(std::move(node));
    return static_cast<int>(m_nodes.size()) - 1;
}

void Scene::addRootNode(int index)
{
    m_rootNodes.push_back(index);
}

// ─── Subtree duplication ─────────────────────────────────────────────────────

static int duplicateSubtreeImpl(Scene& scene, int srcIdx, int newParentIdx)
{
    const Node3D& src = *scene.nodes()[srcIdx];

    std::unique_ptr<Node3D> copy;
    if (const auto* mn = dynamic_cast<const MeshNode*>(&src))
    {
        auto m              = std::make_unique<MeshNode>();
        m->meshIndices      = mn->meshIndices;      // share existing geometry
        m->materialIndices  = mn->materialIndices;  // copy per-instance material assignments
        copy                = std::move(m);
    }
    else if (const auto* in = dynamic_cast<const ImplicitNode*>(&src))
    {
        auto m              = std::make_unique<ImplicitNode>();
        m->type             = in->type;
        m->materialIndex    = in->materialIndex;
        copy                = std::move(m);
    }
    else if (dynamic_cast<const CameraNode*>(&src))
    {
        copy = std::make_unique<CameraNode>();
    }
    else
    {
        copy = std::make_unique<GroupNode>();
    }

    copy->name           = src.name.empty() ? "" : src.name + " (copy)";
    copy->localTransform = src.localTransform;
    copy->parent         = newParentIdx;

    // Snapshot children before addNode — the push_back may reallocate m_nodes,
    // invalidating the `src` reference obtained above.
    const std::vector<int> srcChildren = src.children;

    const int newIdx = scene.addNode(std::move(copy));

    if (newParentIdx >= 0)
    {
        scene.nodeAt(newParentIdx).children.push_back(newIdx);
    }
    else
    {
        scene.addRootNode(newIdx);
    }

    for (int childIdx : srcChildren)
    {
        duplicateSubtreeImpl(scene, childIdx, newIdx);
    }

    return newIdx;
}

int Scene::duplicateSubtree(int nodeIdx)
{
    const int parentIdx = m_nodes[nodeIdx]->parent;
    const int newIdx    = duplicateSubtreeImpl(*this, nodeIdx, parentIdx);
    updateWorldTransforms(newIdx);
    return newIdx;
}

// Recursively resets every node in the subtree rooted at nodeIdx.
// The parent's children list is NOT modified here — the caller handles that.
static void freeSubtree(std::vector<std::unique_ptr<Node3D>>& nodes, int nodeIdx)
{
    for (int childIdx : nodes[nodeIdx]->children)
    {
        freeSubtree(nodes, childIdx);
    }
    nodes[nodeIdx].reset();
}

void Scene::deleteSubtree(int nodeIdx)
{
    Node3D& node = *m_nodes[nodeIdx];

    // Detach from the graph so traversals can no longer reach this subtree.
    if (node.parent >= 0)
    {
        auto& siblings = m_nodes[node.parent]->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), nodeIdx),
                       siblings.end());
    }
    else
    {
        m_rootNodes.erase(std::remove(m_rootNodes.begin(), m_rootNodes.end(), nodeIdx),
                          m_rootNodes.end());
    }

    freeSubtree(m_nodes, nodeIdx);

    // Keep the camera-node sentinel consistent.
    if (!nodeAlive(m_defaultCameraNodeIdx))
    {
        m_defaultCameraNodeIdx = -1;
    }
}

bool Scene::nodeAlive(int nodeIdx) const
{
    return nodeIdx >= 0
        && nodeIdx < static_cast<int>(m_nodes.size())
        && m_nodes[nodeIdx] != nullptr;
}

Node3D& Scene::nodeAt(int index)
{
    return *m_nodes[index];
}

const std::vector<std::unique_ptr<Node3D>>& Scene::nodes() const
{
    return m_nodes;
}

const std::vector<int>& Scene::rootNodes() const
{
    return m_rootNodes;
}

// ─── Scene texture GPU management ────────────────────────────────────────────

void Scene::uploadTextures()
{
    // Upload any texture that is still CPU-only (e.g. loaded from glTF)
    for (Texture& tex : m_textures)
    {
        if (tex.gpuTex == 0 && !tex.pixels.empty())
        {
            tex.uploadToGpu();
        }
    }

    // Build a flat host-side array of texture objects, then copy to device
    std::vector<cudaTextureObject_t> objs;
    objs.reserve(m_textures.size());
    for (const Texture& tex : m_textures)
    {
        objs.push_back(tex.gpuTex);
    }

    destroyTextureObjects();

    if (!objs.empty())
    {
        m_textureObjectsBuffer.allocAndUpload(objs.data(), objs.size() * sizeof(cudaTextureObject_t));
    }
}

void Scene::destroyTextureObjects()
{
    m_textureObjectsBuffer.free();
}

const cudaTextureObject_t* Scene::textureObjects() const
{
    return m_textureObjectsBuffer.typedPtr<const cudaTextureObject_t>();
}

// ─── Emissive implicit lights ─────────────────────────────────────────────────

void Scene::uploadEmissiveLights()
{
    // Walk the scene graph in the same DFS order as Accel::buildTlasPhase so
    // each EmissiveLightData.instanceId matches the TLAS instance index OptiX
    // assigns.  The instance counter increments for every TLAS instance (mesh
    // or implicit); only ImplicitNodes with non-zero emission emit a record.

    std::vector<EmissiveLightData> lights;
    unsigned int instanceCounter = 0;

    const float kFourPi = 12.56637061436f;  // sphere area
    const float kSixPi  = 18.84955592154f;  // cylinder area (lateral 4π + 2 caps π each)

    std::function<void(int)> walk = [&](int nodeIdx)
    {
        const Node3D& node = *m_nodes[nodeIdx];

        if (const MeshNode* mn = dynamic_cast<const MeshNode*>(&node))
        {
            for (int j = 0; j < static_cast<int>(mn->meshIndices.size()); ++j)
            {
                const int mi = mn->meshIndices[j];
                if (mi < 0 || mi >= static_cast<int>(m_meshes.size())) { continue; }
                ++instanceCounter;
            }
        }
        else if (const ImplicitNode* imp = dynamic_cast<const ImplicitNode*>(&node))
        {
            const unsigned int thisId = instanceCounter++;

            const int mi = imp->materialIndex;
            if (mi >= 0 && mi < static_cast<int>(m_materials.size()))
            {
                const MaterialData& mat = m_materials[mi];
                const float3 em = make_float3(
                    mat.emission.x * mat.emissionScale,
                    mat.emission.y * mat.emissionScale,
                    mat.emission.z * mat.emissionScale);

                if (em.x > 0.0f || em.y > 0.0f || em.z > 0.0f)
                {
                    const Matrix4x4& w   = node.worldTransform;
                    const Matrix4x4  inv = mat4Inverse(w);

                    EmissiveLightData ld;

                    ld.l2w[ 0] = w.m[0][0];  ld.l2w[ 1] = w.m[0][1];
                    ld.l2w[ 2] = w.m[0][2];  ld.l2w[ 3] = w.m[0][3];
                    ld.l2w[ 4] = w.m[1][0];  ld.l2w[ 5] = w.m[1][1];
                    ld.l2w[ 6] = w.m[1][2];  ld.l2w[ 7] = w.m[1][3];
                    ld.l2w[ 8] = w.m[2][0];  ld.l2w[ 9] = w.m[2][1];
                    ld.l2w[10] = w.m[2][2];  ld.l2w[11] = w.m[2][3];

                    ld.w2l[ 0] = inv.m[0][0];  ld.w2l[ 1] = inv.m[0][1];
                    ld.w2l[ 2] = inv.m[0][2];  ld.w2l[ 3] = inv.m[0][3];
                    ld.w2l[ 4] = inv.m[1][0];  ld.w2l[ 5] = inv.m[1][1];
                    ld.w2l[ 6] = inv.m[1][2];  ld.w2l[ 7] = inv.m[1][3];
                    ld.w2l[ 8] = inv.m[2][0];  ld.w2l[ 9] = inv.m[2][1];
                    ld.w2l[10] = inv.m[2][2];  ld.w2l[11] = inv.m[2][3];

                    ld.emission = em;

                    // |det(w2l_3x3)| for the Jacobian; equals 1/|det(l2w_3x3)|
                    const float a = inv.m[0][0], b = inv.m[0][1], c = inv.m[0][2];
                    const float d = inv.m[1][0], e = inv.m[1][1], f = inv.m[1][2];
                    const float g = inv.m[2][0], h = inv.m[2][1], i = inv.m[2][2];
                    const float det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
                    ld.invDetW2l = (det != 0.0f) ? std::abs(det) : 1.0f;

                    switch (imp->type)
                    {
                        case ImplicitType::Sphere:
                            ld.localArea = kFourPi;
                            ld.type      = IMPLICIT_SPHERE;
                            break;
                        case ImplicitType::Box:
                            ld.localArea = 24.0f;
                            ld.type      = IMPLICIT_BOX;
                            break;
                        case ImplicitType::Cylinder:
                            ld.localArea = kSixPi;
                            ld.type      = IMPLICIT_CYLINDER;
                            break;
                        default:
                            ld.localArea = kFourPi;
                            ld.type      = IMPLICIT_SPHERE;
                            break;
                    }

                    ld.instanceId = thisId;
                    lights.push_back(ld);
                }
            }
        }

        for (int childIdx : node.children)
        {
            walk(childIdx);
        }
    };

    for (int rootIdx : m_rootNodes)
    {
        walk(rootIdx);
    }

    if (lights.empty())
    {
        m_emissiveLightsBuffer.free();
        m_emissiveLightCount = 0;
        return;
    }

    const size_t bytes = lights.size() * sizeof(EmissiveLightData);
    if (m_emissiveLightsBuffer.size() < bytes)
    {
        m_emissiveLightsBuffer.alloc(bytes);
    }
    m_emissiveLightsBuffer.upload(lights.data(), bytes);
    m_emissiveLightCount = static_cast<int>(lights.size());
}

const EmissiveLightData* Scene::emissiveLights() const
{
    return m_emissiveLightsBuffer.typedPtr<const EmissiveLightData>();
}

int Scene::emissiveLightCount() const
{
    return m_emissiveLightCount;
}

void Scene::clear()
{
    destroyTextureObjects();
    m_emissiveLightsBuffer.free();
    m_emissiveLightCount = 0;
    m_accel.reset();  // free GPU AS memory before geometry is cleared
    m_meshes.clear();
    m_materials.clear();
    m_materialNames.clear();
    m_textures.clear();
    m_nodes.clear();
    m_rootNodes.clear();
    m_camera = Camera::makeDefault();

    // Restore the freshly-constructed invariant: a default camera node exists.
    m_defaultCameraNodeIdx = -1;  // old index is gone with m_nodes
    addDefaultCameraNode();
}

bool Scene::empty() const
{
    return m_meshes.empty();
}

// ─── Acceleration structure ───────────────────────────────────────────────────

void Scene::buildAccel(OptixDeviceContext ctx)
{
    m_accel = std::make_unique<Accel>();
    m_accel->build(ctx, *this);
}

void Scene::rebuildTlas(OptixDeviceContext ctx)
{
    if (m_accel) { m_accel->rebuildTlas(ctx, *this); }
}

void Scene::destroyAccel()
{
    m_accel.reset();
}

bool Scene::hasAccel() const
{
    return m_accel && m_accel->valid();
}

OptixTraversableHandle Scene::traversable() const
{
    return m_accel ? m_accel->traversable() : 0;
}

Accel::MeshDevicePtrs Scene::meshDevicePtrs(size_t meshIdx) const
{
    return m_accel->meshDevicePtrs(meshIdx);
}

// ─── Node graph mutations ─────────────────────────────────────────────────────

bool Scene::isDescendantOf(int nodeIdx, int maybeDescendantIdx) const
{
    if (maybeDescendantIdx == nodeIdx) { return true; }
    for (int childIdx : m_nodes[nodeIdx]->children)
    {
        if (isDescendantOf(childIdx, maybeDescendantIdx)) { return true; }
    }
    return false;
}

void Scene::moveNode(int srcIdx, int newParentIdx, int insertBeforeSiblingIdx)
{
    Node3D& src = *m_nodes[srcIdx];

    // Detach from current location.
    if (src.parent >= 0)
    {
        auto& old = m_nodes[src.parent]->children;
        old.erase(std::remove(old.begin(), old.end(), srcIdx), old.end());
    }
    else
    {
        m_rootNodes.erase(std::remove(m_rootNodes.begin(), m_rootNodes.end(), srcIdx),
                          m_rootNodes.end());
    }

    // Attach to new location.
    src.parent = newParentIdx;
    std::vector<int>& dest = (newParentIdx >= 0)
        ? m_nodes[newParentIdx]->children
        : m_rootNodes;

    if (insertBeforeSiblingIdx < 0)
    {
        dest.push_back(srcIdx);
    }
    else
    {
        auto it = std::find(dest.begin(), dest.end(), insertBeforeSiblingIdx);
        dest.insert(it, srcIdx);
    }
}

// ─── Node transforms ──────────────────────────────────────────────────────────

void Scene::updateWorldTransforms(int nodeIdx)
{
    Node3D& node = *m_nodes[nodeIdx];
    node.worldTransform = (node.parent >= 0)
        ? mat4Multiply(m_nodes[node.parent]->worldTransform, node.localTransform)
        : node.localTransform;
    for (int childIdx : node.children)
    {
        updateWorldTransforms(childIdx);
    }
}

void Scene::updateAllWorldTransforms()
{
    for (int rootIdx : m_rootNodes)
    {
        updateWorldTransforms(rootIdx);
    }
}

Matrix4x4 Scene::computeWorldTransform(int nodeIdx) const
{
    // Build the ancestor chain from nodeIdx up to the root.
    std::vector<int> chain;
    for (int i = nodeIdx; i >= 0; i = m_nodes[i]->parent)
    {
        chain.push_back(i);
    }

    // Multiply from root → node (reverse order).
    Matrix4x4 world = mat4Identity();
    for (int i = static_cast<int>(chain.size()) - 1; i >= 0; --i)
    {
        world = mat4Multiply(world, m_nodes[chain[i]]->localTransform);
    }
    return world;
}

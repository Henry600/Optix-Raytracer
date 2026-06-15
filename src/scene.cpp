#include "scene.h"
#include "matrix4x4.h"

#include <algorithm>
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
    return duplicateSubtreeImpl(*this, nodeIdx, parentIdx);
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

void Scene::clear()
{
    destroyTextureObjects();
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
    if (m_accel && m_accel->valid())
    {
        m_accel->rebuildTlas(ctx, *this);
    }
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

// ─── Node transforms ──────────────────────────────────────────────────────────

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

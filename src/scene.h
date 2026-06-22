#ifndef OPTIX_RAYTRACER_SCENE_H
#define OPTIX_RAYTRACER_SCENE_H

#include "accel.h"
#include "camera.h"
#include "gpu_buffer.h"
#include "mesh.h"
#include "node_3d.h"
#include "texture.h"
#include "scene_data.h"  // MaterialData — shared with device code; on include path via CMake

#include <memory>
#include <vector>
#include <string>

class Scene
{
public:
    Scene();             // starts with a default camera node in the graph
    ~Scene() = default;

    Scene(const Scene&)            = delete;
    Scene& operator=(const Scene&) = delete;
    Scene(Scene&&)                 = default;
    Scene& operator=(Scene&&)      = default;

    // Returns the index of the added item.
    int addMesh(Mesh mesh);
    int addMaterial(MaterialData material, std::string name = {});
    int addTexture(Texture texture);

    const std::vector<Mesh>&         meshes()                   const;
    const std::vector<MaterialData>& materials()                const;
    std::vector<MaterialData>&       materials();               // mutable for in-place editing
    const std::vector<Texture>&      textures()                 const;
    std::vector<Texture>&            textures();               // mutable — for uploadToGpu()
    const std::string&               materialName(int index)    const;

    const Camera& camera()             const;
    void          setCamera(Camera camera);

    // Scene graph — glTF node hierarchy preserved at load time.
    // Adding a CameraNode (e.g. from the glTF importer) replaces the default
    // camera node created at construction/clear() and syncs m_camera to it,
    // so the graph always holds exactly one camera.
    int     addNode(std::unique_ptr<Node3D> node);  // takes ownership; returns index
    void    addRootNode(int index);
    Node3D& nodeAt(int index);                       // mutable access for child-link wiring

    // Deep-copies nodeIdx and its entire child subtree as a sibling of nodeIdx.
    // Mesh and material indices are shared (not duplicated). Returns new root index.
    int duplicateSubtree(int nodeIdx);

    // Detach nodeIdx and its entire subtree from the scene graph and free them.
    // The slot in m_nodes is set to nullptr (tombstone); all other indices stay valid.
    void deleteSubtree(int nodeIdx);

    // Returns true when nodeIdx refers to a live node (not tombstoned or out of range).
    bool nodeAlive(int nodeIdx) const;

    const std::vector<std::unique_ptr<Node3D>>& nodes()     const;
    const std::vector<int>&                     rootNodes() const;

    // Returns true when maybeDescendantIdx is nodeIdx itself or any node in
    // nodeIdx's subtree.  Used to guard against cycles when reparenting.
    bool isDescendantOf(int nodeIdx, int maybeDescendantIdx) const;

    // Detach srcIdx from its current parent (or root list) and reattach it
    // under newParentIdx (pass -1 to make it a root), inserting immediately
    // before insertBeforeSiblingIdx in the new sibling list (pass -1 to append).
    // worldTransform is NOT updated — call updateWorldTransforms(srcIdx) after.
    void moveNode(int srcIdx, int newParentIdx, int insertBeforeSiblingIdx);

    // Recompute worldTransform for nodeIdx and every descendant.
    // Call after any localTransform change on a node in the hierarchy.
    void updateWorldTransforms(int nodeIdx);

    // Rebuild worldTransform for every node in the scene (all roots and their subtrees).
    // Call once after scene load and after any operation that may affect multiple nodes.
    void updateAllWorldTransforms();

    // Walk the ancestor chain and return the accumulated world transform on demand.
    // Prefer node.worldTransform for hot paths; this is kept for internal use during load.
    Matrix4x4 computeWorldTransform(int nodeIdx) const;

    // ── Acceleration structure ────────────────────────────────────────────────
    // The Accel is derived entirely from the scene's geometry and node hierarchy,
    // so Scene owns it.  All methods require an OptiX device context.

    // Build (or rebuild) all BLASes and the TLAS from the current scene data.
    // Destroys any previous acceleration structure first.
    // Throws std::runtime_error on CUDA / OptiX failure.
    void buildAccel(OptixDeviceContext ctx);

    // Rebuild only the TLAS from the current node world-space transforms.
    // BLASes and device geometry buffers are reused unchanged.
    // Much cheaper than buildAccel() — use after live node transform edits.
    void rebuildTlas(OptixDeviceContext ctx);

    // Free all GPU acceleration-structure memory.
    // Called automatically by clear() and in Application's destructor
    // (before the OptiX context is destroyed).
    void destroyAccel();

    bool                   hasAccel()    const;  // true when built and valid
    OptixTraversableHandle traversable() const;  // TLAS handle; 0 when not built

    // Per-mesh device pointers for filling SBT hit group records.
    Accel::MeshDevicePtrs meshDevicePtrs(size_t meshIdx) const;

    // ── Scene textures on GPU ─────────────────────────────────────────────────
    // Upload any CPU-only textures, then build a flat device cudaTextureObject_t[]
    // array for indexing from shaders via MaterialData::albedoTexture.
    void uploadTextures();

    // Device pointer to the flat texture-object array; null until uploadTextures().
    const cudaTextureObject_t* textureObjects() const;

    // Free the device texture array.  Called automatically by clear().
    void destroyTextureObjects();

    // ── Emissive implicit lights for direct sampling (NEE) ────────────────────
    // Walk the scene graph in the same DFS order as the TLAS build and collect
    // one EmissiveLightData per ImplicitNode whose material has non-zero emission.
    // Must be called after any operation that changes node transforms, the scene
    // graph structure, or material emission properties.
    void uploadEmissiveLights();

    // Device pointer to the EmissiveLightData array; null when none exist.
    const EmissiveLightData* emissiveLights()     const;
    int                      emissiveLightCount() const;

    void clear();        // remove all scene data and free the acceleration structure
    bool empty() const;  // true when there are no meshes

private:
    // Add the "Default Camera" root node at the current m_camera transform.
    // Called from the constructor and clear().
    void addDefaultCameraNode();

    std::vector<Mesh>         m_meshes;
    std::vector<MaterialData> m_materials;
    std::vector<std::string>  m_materialNames;  // parallel to m_materials
    std::vector<Texture>      m_textures;
    Camera                    m_camera = Camera::makeDefault();

    // Index of the default camera node in m_nodes; -1 once an imported
    // camera node has replaced it (see addNode()).
    int m_defaultCameraNodeIdx = -1;

    std::vector<std::unique_ptr<Node3D>> m_nodes;
    std::vector<int>                     m_rootNodes;

    std::unique_ptr<Accel> m_accel;  // null until buildAccel() is called
    GPUBuffer m_textureObjectsBuffer;   // device cudaTextureObject_t[]
    GPUBuffer m_emissiveLightsBuffer;   // device EmissiveLightData[]
    int       m_emissiveLightCount = 0;
};

#endif // OPTIX_RAYTRACER_SCENE_H

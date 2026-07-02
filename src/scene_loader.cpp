// TINYGLTF_IMPLEMENTATION in exactly one TU — same pattern as optix_function_table_definition.h
// in Application.cpp. STB_IMAGE_WRITE_IMPLEMENTATION is required even though we only load;
// tinygltf's implementation block unconditionally includes both stb headers.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "scene_loader.h"
#include "node_3d.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

// Matrix4x4 helpers (mat4Identity, mat4Multiply, mat4Inverse, mat4RigidInverse,
// mat4ToColMajor, mat4FromColMajor) are all defined inline in Matrix4x4.h,
// included transitively via SceneLoader.h → Scene.h → Node3D.h → Matrix4x4.h.

// Builds the node's local matrix from either the stored 4x4 or TRS values.
// glTF matrices are column-major; we store row-major so we transpose on import.
static Matrix4x4 gltfNodeLocalMatrix(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        Matrix4x4 m;
        for (int r = 0; r < 4; ++r)
        {
            for (int c = 0; c < 4; ++c)
            {
                m.m[r][c] = static_cast<float>(node.matrix[c * 4 + r]);
            }
        }
        return m;
    }

    float sx = 1.0f, sy = 1.0f, sz = 1.0f;
    if (node.scale.size() == 3)
    {
        sx = static_cast<float>(node.scale[0]);
        sy = static_cast<float>(node.scale[1]);
        sz = static_cast<float>(node.scale[2]);
    }

    float qx = 0.0f, qy = 0.0f, qz = 0.0f, qw = 1.0f;
    if (node.rotation.size() == 4)
    {
        qx = static_cast<float>(node.rotation[0]);
        qy = static_cast<float>(node.rotation[1]);
        qz = static_cast<float>(node.rotation[2]);
        qw = static_cast<float>(node.rotation[3]);
    }

    Matrix4x4 m = mat4Identity();
    // Rotation from unit quaternion, columns scaled by S
    m.m[0][0] = (1.0f - 2.0f*(qy*qy + qz*qz)) * sx;
    m.m[0][1] = (2.0f*(qx*qy - qz*qw))        * sy;
    m.m[0][2] = (2.0f*(qx*qz + qy*qw))        * sz;
    m.m[1][0] = (2.0f*(qx*qy + qz*qw))        * sx;
    m.m[1][1] = (1.0f - 2.0f*(qx*qx + qz*qz)) * sy;
    m.m[1][2] = (2.0f*(qy*qz - qx*qw))        * sz;
    m.m[2][0] = (2.0f*(qx*qz - qy*qw))        * sx;
    m.m[2][1] = (2.0f*(qy*qz + qx*qw))        * sy;
    m.m[2][2] = (1.0f - 2.0f*(qx*qx + qy*qy)) * sz;

    if (node.translation.size() == 3)
    {
        m.m[0][3] = static_cast<float>(node.translation[0]);
        m.m[1][3] = static_cast<float>(node.translation[1]);
        m.m[2][3] = static_cast<float>(node.translation[2]);
    }
    return m;
}

// Recursively searches the node subtree for the first node with a camera.
// Accumulates the parent-to-world transform as it descends.
// outGltfNodeIdx is set to the glTF node index where the camera was found.
static bool findCameraNode(
    const tinygltf::Model& model,
    int                    nodeIdx,
    const Matrix4x4&       parentWorld,
    Camera&                outCamera,
    int&                   outGltfNodeIdx)
{
    const tinygltf::Node& node = model.nodes[nodeIdx];
    const Matrix4x4 world = mat4Multiply(parentWorld, gltfNodeLocalMatrix(node));

    if (node.camera >= 0)
    {
        const tinygltf::Camera& gc = model.cameras[node.camera];
        outCamera.name      = node.name.empty() ? gc.name : node.name;
        outCamera.transform = world;
        outCamera.view      = mat4RigidInverse(world);

        if (gc.type == "perspective")
        {
            outCamera.yFov = static_cast<float>(gc.perspective.yfov);
            if (gc.perspective.aspectRatio > 0.0)
            {
                outCamera.aspectRatio = static_cast<float>(gc.perspective.aspectRatio);
            }
            outCamera.zNear = static_cast<float>(gc.perspective.znear);
            if (gc.perspective.zfar > 0.0)
            {
                outCamera.zFar = static_cast<float>(gc.perspective.zfar);
            }
        }
        // Convert imported yFov to focalLength so the physical parameters are consistent.
        // Use sensor_height = sensorSize / aspectRatio with the camera's aspect ratio.
        {
            const float ar           = (outCamera.aspectRatio > 0.0f)
                                       ? outCamera.aspectRatio
                                       : (16.0f / 9.0f);
            const float sensorHeight = outCamera.sensorSize / ar;
            outCamera.focalLength    = sensorHeight / (2.0f * tanf(outCamera.yFov * 0.5f));
        }
        outGltfNodeIdx = nodeIdx;
        return true;
    }

    for (int child : node.children)
    {
        if (findCameraNode(model, child, world, outCamera, outGltfNodeIdx))
        {
            return true;
        }
    }
    return false;
}

// Recursively builds a Node3D for gltfNodeIdx and all its descendants.
// Returns the index of the newly created node in outScene.
static int buildNode3D(
    const tinygltf::Model&  model,
    int                     gltfNodeIdx,
    int                     parentSceneIdx,
    const std::vector<int>& gltfMeshToSceneIdx,
    int                     cameraGltfNodeIdx,
    Scene&                  outScene)
{
    const tinygltf::Node& gNode = model.nodes[gltfNodeIdx];

    std::unique_ptr<Node3D> node;

    if (gltfNodeIdx == cameraGltfNodeIdx)
    {
        node = std::make_unique<CameraNode>();
    }
    else if (gNode.mesh >= 0 && gNode.mesh < static_cast<int>(gltfMeshToSceneIdx.size()))
    {
        auto meshNode = std::make_unique<MeshNode>();
        const int first     = gltfMeshToSceneIdx[gNode.mesh];
        const int primCount = static_cast<int>(model.meshes[gNode.mesh].primitives.size());
        for (int p = 0; p < primCount; ++p)
        {
            const int sceneIdx = first + p;
            meshNode->meshIndices.push_back(sceneIdx);
            meshNode->materialIndices.push_back(outScene.meshes()[sceneIdx].materialIndex);
        }
        node = std::move(meshNode);
    }
    else
    {
        node = std::make_unique<GroupNode>();
    }

    node->name           = gNode.name;
    node->localTransform = gltfNodeLocalMatrix(gNode);
    node->parent         = parentSceneIdx;

    const int sceneIdx = outScene.addNode(std::move(node));

    for (int childGltfIdx : gNode.children)
    {
        const int childSceneIdx = buildNode3D(
            model, childGltfIdx, sceneIdx,
            gltfMeshToSceneIdx, cameraGltfNodeIdx, outScene);
        outScene.nodeAt(sceneIdx).children.push_back(childSceneIdx);
    }

    return sceneIdx;
}

// ─── float3 host-side math helpers ───────────────────────────────────────────
// cuda_runtime.h defines make_float3/make_float2/make_uint3 but NOT operator
// overloads for float3 in host code — those live in CUDA sample helper headers
// that are not part of the SDK install.

static float3 f3Sub(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static float3 f3Cross(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static float3 f3Normalize(float3 v)
{
    const float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f)
    {
        return make_float3(0.0f, 1.0f, 0.0f);  // degenerate triangle → arbitrary up
    }
    return make_float3(v.x / len, v.y / len, v.z / len);
}

// ─── Accessor helpers ─────────────────────────────────────────────────────────

static const uint8_t* accessorBase(
    const tinygltf::Model& model,
    int                    accessorIndex,
    size_t&                count,
    size_t&                byteStride)
{
    const tinygltf::Accessor&   acc = model.accessors[accessorIndex];
    const tinygltf::BufferView& bv  = model.bufferViews[acc.bufferView];
    count      = acc.count;
    byteStride = bv.byteStride;  // 0 means tightly packed
    return model.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset;
}

static void readVec3Accessor(
    const tinygltf::Model& model,
    int                    accessorIndex,
    std::vector<float3>&   out)
{
    size_t count = 0, stride = 0;
    const uint8_t* base = accessorBase(model, accessorIndex, count, stride);
    if (stride == 0)
    {
        stride = sizeof(float) * 3;
    }

    out.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        const float* src = reinterpret_cast<const float*>(base + i * stride);
        out[i] = make_float3(src[0], src[1], src[2]);
    }
}

static void readVec2Accessor(
    const tinygltf::Model& model,
    int                    accessorIndex,
    std::vector<float2>&   out)
{
    size_t count = 0, stride = 0;
    const uint8_t* base = accessorBase(model, accessorIndex, count, stride);
    if (stride == 0)
    {
        stride = sizeof(float) * 2;
    }

    out.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        const float* src = reinterpret_cast<const float*>(base + i * stride);
        out[i] = make_float2(src[0], src[1]);
    }
}

static void readIndexAccessor(
    const tinygltf::Model& model,
    int                    accessorIndex,
    std::vector<uint3>&    out)
{
    const tinygltf::Accessor&   acc = model.accessors[accessorIndex];
    const tinygltf::BufferView& bv  = model.bufferViews[acc.bufferView];
    const uint8_t* base = model.buffers[bv.buffer].data.data()
                        + bv.byteOffset + acc.byteOffset;
    const size_t triCount = acc.count / 3;
    out.resize(triCount);

    for (size_t tri = 0; tri < triCount; ++tri)
    {
        uint32_t i0, i1, i2;
        if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        {
            i0 = base[tri * 3 + 0];
            i1 = base[tri * 3 + 1];
            i2 = base[tri * 3 + 2];
        }
        else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
        {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(base) + tri * 3;
            i0 = p[0]; i1 = p[1]; i2 = p[2];
        }
        else
        {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(base) + tri * 3;
            i0 = p[0]; i1 = p[1]; i2 = p[2];
        }
        out[tri] = make_uint3(i0, i1, i2);
    }
}

// ─── Normals ──────────────────────────────────────────────────────────────────

static void generateFlatNormals(Mesh& mesh)
{
    mesh.normals.resize(mesh.positions.size(), make_float3(0.0f, 0.0f, 0.0f));
    for (const uint3& tri : mesh.indices)
    {
        const float3 e0 = f3Sub(mesh.positions[tri.y], mesh.positions[tri.x]);
        const float3 e1 = f3Sub(mesh.positions[tri.z], mesh.positions[tri.x]);
        const float3 n  = f3Normalize(f3Cross(e0, e1));
        mesh.normals[tri.x] = n;
        mesh.normals[tri.y] = n;
        mesh.normals[tri.z] = n;
    }
}

// ─── Image loading ────────────────────────────────────────────────────────────

static void loadImage(const tinygltf::Image& gltfImage, Scene& outScene)
{
    Texture tex;
    tex.name   = gltfImage.name.empty() ? gltfImage.uri : gltfImage.name;
    tex.width  = gltfImage.width;
    tex.height = gltfImage.height;
    tex.format = PixelFormat::RGBA8;

    if (gltfImage.component == 3)
    {
        const size_t pixelCount =
            static_cast<size_t>(gltfImage.width) * gltfImage.height;
        tex.pixels.resize(pixelCount * 4);
        for (size_t i = 0; i < pixelCount; ++i)
        {
            tex.pixels[i * 4 + 0] = gltfImage.image[i * 3 + 0];
            tex.pixels[i * 4 + 1] = gltfImage.image[i * 3 + 1];
            tex.pixels[i * 4 + 2] = gltfImage.image[i * 3 + 2];
            tex.pixels[i * 4 + 3] = 255;
        }
    }
    else
    {
        tex.pixels = gltfImage.image;
    }

    outScene.addTexture(std::move(tex));
}

// ─── Material loading ─────────────────────────────────────────────────────────

// glTF stores baseColorFactor and emissiveFactor in linear space (per spec).
// Our MaterialData convention is sRGB-encoded values: the shader always calls
// srgbToLinear3() on mat.albedo/mat.emission before shading, and ImGui's
// ColorEdit3 treats them as sRGB.  Convert at import time so the round-trip
// linearise → shade is correct and the UI shows the right perceived colour.
static float linearToSrgb(float c)
{
    c = std::max(0.0f, std::min(1.0f, c));
    return powf(c, 1.0f / 2.2f);  // matches srgbToLinear(x)=pow(x,2.2) in shader
}

static MaterialData buildMaterial(
    const tinygltf::Material& gltfMat,
    const tinygltf::Model&    model,
    int                       textureOffset)
{
    MaterialData mat;
    const tinygltf::PbrMetallicRoughness& pbr = gltfMat.pbrMetallicRoughness;

    if (pbr.baseColorFactor.size() == 4)
    {
        mat.albedo = make_float3(
            linearToSrgb(static_cast<float>(pbr.baseColorFactor[0])),
            linearToSrgb(static_cast<float>(pbr.baseColorFactor[1])),
            linearToSrgb(static_cast<float>(pbr.baseColorFactor[2])));
    }

    mat.roughness = static_cast<float>(pbr.roughnessFactor);
    mat.metallic  = static_cast<float>(pbr.metallicFactor);

    if (pbr.baseColorTexture.index >= 0)
    {
        const int imageIdx = model.textures[pbr.baseColorTexture.index].source;
        if (imageIdx >= 0)
        {
            mat.albedoTexture = textureOffset + imageIdx;
        }
    }

    if (gltfMat.emissiveFactor.size() == 3)
    {
        mat.emission = make_float3(
            linearToSrgb(static_cast<float>(gltfMat.emissiveFactor[0])),
            linearToSrgb(static_cast<float>(gltfMat.emissiveFactor[1])),
            linearToSrgb(static_cast<float>(gltfMat.emissiveFactor[2])));
    }

    if (gltfMat.emissiveTexture.index >= 0)
    {
        const int imageIdx =
            model.textures[gltfMat.emissiveTexture.index].source;
        if (imageIdx >= 0)
        {
            mat.emissionTexture = textureOffset + imageIdx;
        }
    }

    // KHR_materials_emissive_strength — scalar multiplier on emissiveFactor
    // (core glTF clamps emissiveFactor to [0,1]; this carries HDR intensity)
    auto emsIt = gltfMat.extensions.find("KHR_materials_emissive_strength");
    if (emsIt != gltfMat.extensions.end() && emsIt->second.Has("emissiveStrength"))
    {
        mat.emissionScale = static_cast<float>(
            emsIt->second.Get("emissiveStrength").GetNumberAsDouble());
    }

    // KHR_materials_transmission — transmissionFactor ∈ [0, 1]
    auto transIt = gltfMat.extensions.find("KHR_materials_transmission");
    if (transIt != gltfMat.extensions.end() && transIt->second.Has("transmissionFactor"))
    {
        mat.transmission = static_cast<float>(
            transIt->second.Get("transmissionFactor").GetNumberAsDouble());
    }

    // KHR_materials_ior — index of refraction (default 1.5 per spec)
    auto iorIt = gltfMat.extensions.find("KHR_materials_ior");
    if (iorIt != gltfMat.extensions.end() && iorIt->second.Has("ior"))
    {
        mat.ior = static_cast<float>(
            iorIt->second.Get("ior").GetNumberAsDouble());
    }

    // KHR_materials_clearcoat
    auto ccIt = gltfMat.extensions.find("KHR_materials_clearcoat");
    if (ccIt != gltfMat.extensions.end())
    {
        if (ccIt->second.Has("clearcoatFactor"))
        {
            mat.clearcoat = static_cast<float>(
                ccIt->second.Get("clearcoatFactor").GetNumberAsDouble());
        }
        if (ccIt->second.Has("clearcoatRoughnessFactor"))
        {
            mat.clearcoatRoughness = static_cast<float>(
                ccIt->second.Get("clearcoatRoughnessFactor").GetNumberAsDouble());
        }
    }

    return mat;
}

// ─── Mesh loading ─────────────────────────────────────────────────────────────

static void loadMesh(
    const tinygltf::Mesh&  gltfMesh,
    const tinygltf::Model& model,
    int                    materialOffset,
    Scene&                 outScene)
{
    for (size_t primIdx = 0; primIdx < gltfMesh.primitives.size(); ++primIdx)
    {
        const tinygltf::Primitive& prim = gltfMesh.primitives[primIdx];

        if (prim.mode != TINYGLTF_MODE_TRIANGLES)
        {
            continue;
        }

        auto posIt = prim.attributes.find("POSITION");
        if (posIt == prim.attributes.end())
        {
            continue;
        }

        Mesh mesh;
        mesh.name = (gltfMesh.primitives.size() == 1)
            ? gltfMesh.name
            : gltfMesh.name + "_" + std::to_string(primIdx);

        readVec3Accessor(model, posIt->second, mesh.positions);

        auto normIt = prim.attributes.find("NORMAL");
        if (normIt != prim.attributes.end())
        {
            readVec3Accessor(model, normIt->second, mesh.normals);
        }

        auto uvIt = prim.attributes.find("TEXCOORD_0");
        if (uvIt != prim.attributes.end())
        {
            readVec2Accessor(model, uvIt->second, mesh.uvs);
        }

        if (prim.indices >= 0)
        {
            readIndexAccessor(model, prim.indices, mesh.indices);
        }
        else
        {
            const size_t vertCount = mesh.positions.size();
            mesh.indices.reserve(vertCount / 3);
            for (size_t i = 0; i + 2 < vertCount; i += 3)
            {
                mesh.indices.push_back(make_uint3(
                    static_cast<uint32_t>(i),
                    static_cast<uint32_t>(i + 1),
                    static_cast<uint32_t>(i + 2)));
            }
        }

        if (mesh.normals.empty())
        {
            generateFlatNormals(mesh);
        }

        if (mesh.uvs.empty())
        {
            mesh.uvs.resize(mesh.positions.size(), make_float2(0.0f, 0.0f));
        }

        mesh.materialIndex = (prim.material >= 0)
            ? materialOffset + prim.material
            : materialOffset;

        outScene.addMesh(std::move(mesh));
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

bool loadGltfFile(const std::string& path, Scene& outScene, std::string& outError)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        err;
    std::string        warn;

    const bool isBinary =
        std::filesystem::path(path).extension().string() == ".glb";

    const bool ok = isBinary
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!ok)
    {
        outError = err.empty() ? "tinygltf: unknown error loading " + path : err;
        return false;
    }

    const int materialOffset = static_cast<int>(outScene.materials().size());
    const int textureOffset  = static_cast<int>(outScene.textures().size());
    const int meshOffset     = static_cast<int>(outScene.meshes().size());

    for (const tinygltf::Image& img : model.images)
    {
        loadImage(img, outScene);
    }

    for (const tinygltf::Material& mat : model.materials)
    {
        outScene.addMaterial(buildMaterial(mat, model, textureOffset), mat.name);
    }

    if (model.materials.empty())
    {
        outScene.addMaterial(MaterialData{}, "default");
    }

    for (const tinygltf::Mesh& gltfMesh : model.meshes)
    {
        loadMesh(gltfMesh, model, materialOffset, outScene);
    }

    // Map from glTF mesh index → first Scene mesh index for that mesh.
    // Primitives within each glTF mesh become consecutive Scene mesh entries.
    std::vector<int> gltfMeshToSceneIdx(model.meshes.size(), -1);
    {
        int idx = meshOffset;
        for (int mi = 0; mi < static_cast<int>(model.meshes.size()); ++mi)
        {
            gltfMeshToSceneIdx[mi] = idx;
            idx += static_cast<int>(model.meshes[mi].primitives.size());
        }
    }

    if (!model.scenes.empty())
    {
        const int defaultSceneIdx = (model.defaultScene >= 0) ? model.defaultScene : 0;

        // Find which glTF node carries the camera (for both Camera import and CameraNode tagging).
        Camera importedCamera;
        int    cameraGltfNodeIdx = -1;
        const Matrix4x4 identity = mat4Identity();
        for (int rootNode : model.scenes[defaultSceneIdx].nodes)
        {
            if (findCameraNode(model, rootNode, identity, importedCamera, cameraGltfNodeIdx))
            {
                outScene.setCamera(importedCamera);
                break;
            }
        }

        // Build scene graph node tree from the glTF node hierarchy.
        for (int rootGltfIdx : model.scenes[defaultSceneIdx].nodes)
        {
            const int sceneRootIdx = buildNode3D(
                model, rootGltfIdx, -1,
                gltfMeshToSceneIdx, cameraGltfNodeIdx, outScene);
            outScene.addRootNode(sceneRootIdx);
        }
    }

    return true;
}

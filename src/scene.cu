#include "scene.h"

#include <iostream>

#include "config.h"
#include "logger.h"
#include "utils.h"
#include <filesystem>

// do not define STB_IMAGE_IMPLEMENTATION here, all stuff is included in a single location
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include "tiny_gltf/tiny_gltf.h"

#include "MikkTSpace/mikktspace.h"

namespace tg = tinygltf;
namespace fs = std::filesystem;

static Vec3 get_vec3(const std::vector<double> &v) {
    assert(v.size() >= 3);
    return {(float)v[0], (float)v[1], (float)v[2]};
}

static Vec4 get_vec4(const std::vector<double> &v) {
    assert(v.size() >= 4);
    return {(float)v[0], (float)v[1], (float)v[2], (float)v[3]};
}

static Mat4 get_transform(const tg::Node &node) {
    if (node.matrix.size()) {
        // matrix is in column major order

        float m00 = node.matrix[0];
        float m10 = node.matrix[1];
        float m20 = node.matrix[2];
        float m30 = node.matrix[3];

        float m01 = node.matrix[4];
        float m11 = node.matrix[5];
        float m21 = node.matrix[6];
        float m31 = node.matrix[7];

        float m02 = node.matrix[8];
        float m12 = node.matrix[9];
        float m22 = node.matrix[10];
        float m32 = node.matrix[11];

        float m03 = node.matrix[12];
        float m13 = node.matrix[13];
        float m23 = node.matrix[14];
        float m33 = node.matrix[15];

        return Mat4(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32,
                    m33);
    }

    Mat4 forward = Mat4::Identity();

    if (node.scale.size()) {
        float sx = node.scale[0];
        float sy = node.scale[1];
        float sz = node.scale[2];
        Mat4 scale = Mat4(sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, sz, 0, 0, 0, 0, 1);
        forward = scale * forward;
    }

    if (node.rotation.size()) {
        // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

        float qi = node.rotation[0];
        float qj = node.rotation[1];
        float qk = node.rotation[2];
        float qr = node.rotation[3];

        float s = 1; // s = 1/norm(Q) ASSUME NORMALIZED QUATERNION IDK

        float m00 = 1 - 2 * s * (qj * qj + qk * qk);
        float m01 = 2 * s * (qi * qj - qk * qr);
        float m02 = 2 * s * (qi * qk + qj * qr);

        float m10 = 2 * s * (qi * qj + qk * qr);
        float m11 = 1 - 2 * s * (qi * qi + qk * qk);
        float m12 = 2 * s * (qj * qk - qi * qr);

        float m20 = 2 * s * (qi * qk - qj * qr);
        float m21 = 2 * s * (qj * qk + qi * qr);
        float m22 = 1 - 2 * s * (qi * qi + qj * qj);

        Mat4 rotation = Mat4(m00, m01, m02, 0, m10, m11, m12, 0, m20, m21, m22, 0, 0, 0, 0, 1);
        forward = rotation * forward;
    }

    if (node.translation.size()) {
        float tx = node.translation[0];
        float ty = node.translation[1];
        float tz = node.translation[2];

        Mat4 translation = Mat4(1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1);
        forward = translation * forward;
    }

    return forward;
}

static void parse_camera(temp_scene_t &scene, const tg::Model &model, const tg::Node &node,
                         const Mat4 &modelTransform) {
    const tg::Camera &scene_cam = model.cameras[node.camera];

    if (scene_cam.type != "perspective") {
        log_error("unsupported camera type: %s\n", scene_cam.type.c_str());
        exit(EXIT_FAILURE);
    }

    camera_t cam;

    cam.yfov = scene_cam.perspective.yfov;

    Vec4 position = modelTransform * Vec4{0, 0, 0, 1};
    Vec4 target = modelTransform * Vec4{0, 0, -1, 1};
    Vec4 updir = modelTransform * Vec4{0, 1, 0, 1};

    cam.position = position.dehomogenise();
    cam.target = target.dehomogenise();
    cam.updir = updir.dehomogenise() - cam.position;

    scene.cameras.push_back(cam);
}

static void parse_light(temp_scene_t &scene, const tg::Model &model, const tg::Node &node,
                        const Mat4 &modelTransform) {
    const tg::Light &modelLight = model.lights[node.light];

    light_t light;

    light.color = {1, 1, 1};
    if (modelLight.color.size()) {
        light.color[0] = modelLight.color[0];
        light.color[1] = modelLight.color[1];
        light.color[2] = modelLight.color[2];
    }
    light.intensity = modelLight.intensity;

    light.position = (modelTransform * Vec4(0, 0, 0, 1)).dehomogenise();

    Mat3 linearTransform = modelTransform.getLinearPart();
    light.direction = (linearTransform * Vec3(0, 0, -1)).normalized();
    // printf("Light direction: \n");
    // light.direction.print();

    if (modelLight.type == "directional") {
        light.type = LIGHT_DIRECTIONAL;
        log_trace("parsing directional light: \n");

    } else if (modelLight.type == "point") {
        log_trace("parsing point light: \n");
        light.type = LIGHT_POINT;
    } else {
        log_error("Unsupported light '%s'\n", modelLight.type.c_str());
        exit(EXIT_FAILURE);
    }

    // printf("Light:\n");
    // light.color.print();
    // printf("%f\n", light.intensity);
    // printf("%d\n", light.type);

    scene.lights.push_back(light);
}

static void print_material(const material_t *material) {
    log_trace("Material:\n");
    log_trace("  Name: %s\n", material->name);
    log_trace("  BaseColorFactor: (%.2f, %.2f, %.2f; %.2f)\n", material->baseColorFactor.x,
              material->baseColorFactor.y, material->baseColorFactor.z,
              material->baseColorFactor.w);
    log_trace("  Color texture: (%d)\n", material->baseColorTexture);
    log_trace("  Normal texture: (%d)\n", material->normalTexture);
    log_trace("  Transmission: %.2f\n", material->transmission);
    log_trace("  Emissive: (%.2f, %.2f, %.2f)\n", material->emissive.x, material->emissive.y,
              material->emissive.z);
    log_trace("  Metallic: %.2f\n", material->metallic);
    log_trace("  Roughness: %.2f\n", material->roughness);
    log_trace("  IoR: %.2f\n", material->ior);
}

static void parse_material(temp_scene_t &scene, const tg::Material &sceneMat) {
    material_t mat;

    strncpy(mat.name, sceneMat.name.c_str(), MATERIAL_NAME_SIZE - 1);
    mat.name[MATERIAL_NAME_SIZE - 1] = '\0';

    // PARSE MATERIAL

    float emissiveStrength = 0;
    float transmissionFactor = 0;
    float ior = 1.333;

    for (const auto &extension : sceneMat.extensions) {
        if (extension.first == "KHR_materials_emissive_strength") {
            emissiveStrength =
                (float)extension.second.Get("emissiveStrength").GetNumberAsDouble();
        } else if (extension.first == "KHR_materials_transmission") {
            transmissionFactor =
                (float)extension.second.Get("transmissionFactor").GetNumberAsDouble();
        } else if (extension.first == "KHR_materials_ior") {
            ior = (float)extension.second.Get("ior").GetNumberAsDouble();
        } else {
            log_warning("Unknown extension: %s\n", extension.first.c_str());
        }
    }

    if (sceneMat.alphaMode == "OPAQUE") {
        mat.alphaMode = ALPHA_OPAQUE;
    }
    if (sceneMat.alphaMode == "MASK") {
        mat.alphaMode = ALPHA_MASK;
    }
    if (sceneMat.alphaMode == "BLEND") {
        mat.alphaMode = ALPHA_BLEND;
    }
    mat.alphaCutoff = sceneMat.alphaCutoff;

    // SET MATERIAL VALUES

    mat.baseColorFactor = get_vec4(sceneMat.pbrMetallicRoughness.baseColorFactor);
    mat.emissive = emissiveStrength * get_vec3(sceneMat.emissiveFactor);
    mat.metallic = (float)sceneMat.pbrMetallicRoughness.metallicFactor;
    mat.roughness = (float)sceneMat.pbrMetallicRoughness.roughnessFactor;
    mat.ior = ior;
    mat.transmission = (float)transmissionFactor;

    mat.baseColorTexture = sceneMat.pbrMetallicRoughness.baseColorTexture.index;

    mat.normalTexture = sceneMat.normalTexture.index;

    if (sceneMat.pbrMetallicRoughness.baseColorTexture.texCoord != 0) {
        log_warning("TODO implement other texCoords");
    }
    for (const auto &extension : sceneMat.pbrMetallicRoughness.baseColorTexture.extensions) {
        log_warning("TODO texture extentions");
    }

    print_material(&mat);

    scene.materials.push_back(mat);
}

template <typename T>
static void scene_parse_acc_to_vec(std::vector<T> &out, const tg::Model &model, int accIndex,
                                   int arity) {
    const tg::Accessor &acc = model.accessors[accIndex];
    const tg::BufferView &bufView = model.bufferViews[acc.bufferView];
    const tg::Buffer &buf = model.buffers[bufView.buffer];

    assert(arity == 2 || arity == 3 || arity == 4);

    // AABB actualBounds;
    // AABB givenBounds;
    // bool hasGivenBounds = false;

    // if (acc.maxValues.size() && acc.maxValues.size()) {
    //     hasGivenBounds = true;
    //     givenBounds.min = {(float)acc.minValues[0], (float)acc.minValues[1],
    //                        (float)acc.minValues[2]};
    //     givenBounds.max = {(float)acc.maxValues[0], (float)acc.maxValues[1],
    //                        (float)acc.maxValues[2]};
    //     if (arity == 2) {
    //         givenBounds.min.z = -1e10f;
    //         givenBounds.max.z = 1e10f;
    //     }
    //     log_trace("Bounds given: (%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)\n",
    //     givenBounds.min.x,
    //               givenBounds.min.y, givenBounds.min.z, givenBounds.max.x, givenBounds.max.y,
    //               givenBounds.max.z);
    // }

    assert(!acc.sparse.isSparse && "sparse mesh unsupported"); // TODO maybe

    int byteStride = acc.ByteStride(bufView);

    for (size_t i = 0; i < acc.count; i++) {
        // https://github.com/syoyo/tinygltf/wiki/Accessing-vertex-data

        size_t byteNumber = acc.byteOffset + bufView.byteOffset + byteStride * i;
        const float *item = reinterpret_cast<const float *>(&buf.data[byteNumber]);

        T v = T::Zero();
        for (int j = 0; j < arity; j++) {
            v[j] = item[j];
        }
        // v.print();
        out.push_back(v);

        // if (arity == 2) {
        //     printf("texcoord0: (%f, %f)\n", v.x, v.y);
        // }

        // actualBounds.grow(v);

        // if (hasGivenBounds) {
        //     // sanity check
        //     assert(givenBounds.contains(v));
        // }
    }

    // char s_min[64], s_max[64];
    // actualBounds.min.snprint(s_min, 64);
    // actualBounds.max.snprint(s_max, 64);
    // log_trace("%lu vec3 parsed in given min and max:\n", acc.count);
}

static void scene_parse_acc_indices(std::vector<uint32_t> &list, const tg::Model &model,
                                    int accIndex) {
    const tg::Accessor &acc = model.accessors[accIndex];
    const tg::BufferView &bufView = model.bufferViews[acc.bufferView];
    const tg::Buffer &buf = model.buffers[bufView.buffer];

    assert(!acc.sparse.isSparse && "sparse mesh unsupported"); // TODO maybe

    int byteStride = acc.ByteStride(bufView);

    int min = std::numeric_limits<int>::max();
    int max = -1;

    for (size_t i = 0; i < acc.count; i++) {
        // https://github.com/syoyo/tinygltf/wiki/Accessing-vertex-data

        size_t byteNumber = acc.byteOffset + bufView.byteOffset + byteStride * i;

        // printf("acc.byteOffset %lu, bufView.byteOffset %lu, byteStride %lu, i %ld\n",
        //     acc.byteOffset, bufView.byteOffset, byteStride, i);

        int index;

        switch (acc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            index = (int)*reinterpret_cast<const uint16_t *>(&buf.data[byteNumber]);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            index = (int)*reinterpret_cast<const uint32_t *>(&buf.data[byteNumber]);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            index = (int)*reinterpret_cast<const uint8_t *>(&buf.data[byteNumber]);
            break;

        default:
            log_error("Component type not implemented: %d\n", acc.componentType);
            exit(EXIT_FAILURE);
            break;
        }

        min = std::min(min, index);
        max = std::max(max, index);

        // printf("%d\n", index);
        list.push_back(index);
    }

    log_trace("Indices scanned in range [%d, %d]\n", min, max);
}

static void calculate_tangents(temp_scene_t &scene, int firstFace, int numFaces) {

    typedef struct {
        temp_scene_t *scene;
        int firstFace, numFaces;
    } user_data_t;

    user_data_t user_data = {.scene = &scene, .firstFace = firstFace, .numFaces = numFaces};

    // Returns the number of faces (triangles/quads) on the mesh to be processed.
    auto getNumFaces = [](const SMikkTSpaceContext *context) -> int32_t {
        user_data_t *data = (user_data_t *)context->m_pUserData;
        return data->numFaces;
    };

    // Returns the number of vertices on face number iFace
    // iFace is a number in the range {0, 1, ..., getNumFaces()-1}
    auto getNumVerticesOfFace = [](const SMikkTSpaceContext *context,
                                   const int32_t iFace) -> int32_t {
        user_data_t *data = (user_data_t *)context->m_pUserData;
        int faceIndex = data->firstFace + iFace;
        return data->scene->faces[faceIndex].vertexCount;
    };

    // returns the position/normal/texcoord of the referenced face of vertex number iVert.
    // iVert is in the range {0,1,2} for triangles and {0,1,2,3} for quads.
    auto getPosition = [](const SMikkTSpaceContext *context, float posOut[],
                          const int32_t iFace, const int32_t iVert) -> void {
        user_data_t *data = (user_data_t *)context->m_pUserData;
        int faceIndex = data->firstFace + iFace;
        int vertIndex = data->scene->faces[faceIndex].vertices[iVert];
        const Vec3 &position = data->scene->vertices[vertIndex].position;
        for (int i = 0; i < 3; i++) {
            posOut[i] = position[i];
        }
    };

    auto getNormal = [](const SMikkTSpaceContext *context, float normOut[], const int32_t iFace,
                        const int32_t iVert) -> void {
        user_data_t *data = (user_data_t *)context->m_pUserData;
        int faceIndex = data->firstFace + iFace;
        int vertIndex = data->scene->faces[faceIndex].vertices[iVert];
        const Vec3 &normal = data->scene->vertices[vertIndex].normal;
        for (int i = 0; i < 3; i++) {
            normOut[i] = normal[i];
        }
    };

    auto getTexCoord = [](const SMikkTSpaceContext *context, float uvOut[], const int32_t iFace,
                          const int32_t iVert) -> void {
        user_data_t *data = (user_data_t *)context->m_pUserData;
        int faceIndex = data->firstFace + iFace;
        int vertIndex = data->scene->faces[faceIndex].vertices[iVert];
        const Vec3 &texcoord0 = data->scene->vertices[vertIndex].texcoord0;
        for (int i = 0; i < 2; i++) {
            uvOut[i] = texcoord0[i];
        }
    };

    // This function is used to return the tangent and fSign to the application.
    // fvTangent is a unit length vector.
    // For normal maps it is sufficient to use the following simplified version of the bitangent
    // which is generated at pixel/vertex level. bitangent = fSign * cross(vN, tangent); Note
    // that the results are returned unindexed. It is possible to generate a new index list But
    // averaging/overwriting tangent spaces by using an already existing index list WILL produce
    // INCRORRECT results. DO NOT! use an already existing index list.
    auto setTSpaceBasic = [](const SMikkTSpaceContext *context, const float mikkTangent[],
                             const float sign, const int32_t iFace, const int32_t iVert) {
        user_data_t *data = (user_data_t *)context->m_pUserData;
        int faceIndex = data->firstFace + iFace;
        int vertIndex = data->scene->faces[faceIndex].vertices[iVert];
        Vec4 &tangent = data->scene->vertices[vertIndex].tangent;
        for (int i = 0; i < 3; i++) {
            tangent[i] = mikkTangent[i];
        }
        tangent.w = -sign;
    };

    SMikkTSpaceInterface mikktspace_i = {
        .m_getNumFaces = getNumFaces,
        .m_getNumVerticesOfFace = getNumVerticesOfFace,
        .m_getPosition = getPosition,
        .m_getNormal = getNormal,
        .m_getTexCoord = getTexCoord,
        .m_setTSpaceBasic = setTSpaceBasic,
    };
    SMikkTSpaceContext mikktspace_c;
    mikktspace_c.m_pInterface = &mikktspace_i;
    mikktspace_c.m_pUserData = &user_data;

    if (!genTangSpaceDefault(&mikktspace_c)) {
        log_error("Mikktspace tangent calculation failed!\n");
        exit(EXIT_FAILURE);
    }
}

static void parse_mesh(temp_scene_t &scene, const tg::Model &model, const tg::Node &node,
                       const Mat4 &modelTransform) {
    log_trace("Parsing mesh of node: %s\n", node.name.c_str());

    const tg::Mesh &mesh = model.meshes[node.mesh];
    Mat3 linearModelTransform = modelTransform.getLinearPart();

    for (const tg::Primitive &prim : mesh.primitives) {
        log_trace("Parsing primitive\n");

        if (prim.material < 0 || prim.material >= (int)scene.materials.size()) {
            log_trace("Invalid material %d\n", prim.material);
        }

        std::vector<Vec3> positions;
        std::vector<Vec4> vertexTangents;
        std::vector<Vec3> vertexNormals;
        std::vector<Vec3> texcoord0;
        std::vector<uint32_t> indices;

        for (const auto &attr : prim.attributes) {
            if (attr.first == "POSITION") {
                log_trace("Scanning POSITION\n");
                scene_parse_acc_to_vec(positions, model, attr.second, 3);
            } else if (attr.first == "NORMAL") {
                log_trace("Scanning NORMAL\n");
                scene_parse_acc_to_vec(vertexNormals, model, attr.second, 3);
            } else if (attr.first == "TANGENT") {
                log_trace("Scanning TANGENT\n");
                scene_parse_acc_to_vec(vertexTangents, model, attr.second, 4);
            } else if (attr.first == "TEXCOORD_0") {
                log_trace("Scanning TEXCOORD_0\n");
                scene_parse_acc_to_vec(texcoord0, model, attr.second, 2);
            } else {
                log_warning("skipped unsupported attribute '%s'\n", attr.first.c_str());
            }
        }

        log_trace("Scanning INDICES\n");
        scene_parse_acc_indices(indices, model, prim.indices);

        int vertCount = positions.size();
        assert(vertCount);

        bool hasVertexNormals = vertexNormals.size() > 0;
        assert(!hasVertexNormals || vertexNormals.size() == vertCount);

        bool hasVertexTangents = vertexTangents.size() > 0;
        assert(!hasVertexTangents || vertexTangents.size() == vertCount);

        bool hasTexcoord0 = texcoord0.size() > 0;
        assert(!hasTexcoord0 || texcoord0.size() == vertCount);

        int numIndices = indices.size();
        assert(numIndices && numIndices % 3 == 0);

        log_trace("Adding faces\n");

        int firstVertexIndex = scene.vertices.size();

        for (int i = 0; i < vertCount; i++) {
            vertex_t v;

            Vec4 p = modelTransform * Vec4::toHomogeneous(positions[i]);
            v.position = p.dehomogenise();

            if (hasVertexNormals) {
                v.normal = linearModelTransform * vertexNormals[i];
                v.normal.normalize();
            } else {
                v.normal.set(0); // not used, face normals will be used here
            }

            if (hasVertexNormals && hasVertexTangents) {
                // tangents should be ignored if no normals otherwise would not make much sense
                Vec3 tw = linearModelTransform * vertexTangents[i].xyz();
                tw.normalize();
                v.tangent = Vec4(tw.x, tw.y, tw.z, vertexTangents[i].w); // include handedness
            } else {
                v.tangent.set(0); // will be calculated later if hasVertexNormals
            }

            if (hasTexcoord0) {
                v.texcoord0 = texcoord0[i];
            } else {
                v.texcoord0.set(0);
            }

            scene.vertices.push_back(v);
        }

        int firstFaceOfPrimitive = scene.faces.size();

        for (int i = 0; i < numIndices; i += 3) {
            face_t face;
            face.material = prim.material;

            face.vertexCount = 3;
            face.vertices[0] = firstVertexIndex + indices[i];
            face.vertices[1] = firstVertexIndex + indices[i + 1];
            face.vertices[2] = firstVertexIndex + indices[i + 2];

            if (hasVertexNormals) {
                face.shading = BARY_SHADING;
                face.normal.set(0);
                face.tangent.set(0);
            } else {
                face.shading = FLAT_SHADING;
                // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
                // spec says no normals present -> flat shading
                const Vec3 &A = scene.vertices[face.vertices[0]].position;
                const Vec3 &B = scene.vertices[face.vertices[1]].position;
                const Vec3 &C = scene.vertices[face.vertices[2]].position;
                face.normal = (B - A).cross(C - A).normalized();

                // https://github.com/KhronosGroup/glTF/issues/2480
                // anisotropic materials are rarely seen with flat shading (missing normals)
                // a workaround would be a pain to implement
                face.tangent.set(0);
            }

            scene.faces.push_back(face);
        }

        if (hasVertexNormals && !hasVertexTangents) {
            // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
            // must calculate tangents here using mikktspace after gltf spec
            int numFaces = scene.faces.size() - firstFaceOfPrimitive;
            calculate_tangents(scene, firstFaceOfPrimitive, numFaces);
        }

        log_trace("Done with primitive\n");
    }
    log_trace("Done with mesh\n");
}

static void scene_parse_node(temp_scene_t &scene, const tg::Model &model, const tg::Node &node,
                             Mat4 parentTransform) {
    Mat4 nodeForwardTransform = get_transform(node);
    Mat4 modelTransform = parentTransform * nodeForwardTransform;

    if (node.camera >= 0) {
        parse_camera(scene, model, node, modelTransform);
    }
    if (node.light >= 0) {
        parse_light(scene, model, node, modelTransform);
    }
    if (node.mesh >= 0) {
        parse_mesh(scene, model, node, modelTransform);
    }

    for (const int &childIndex : node.children) {
        const tg::Node &child = model.nodes[childIndex];
        scene_parse_node(scene, model, child, modelTransform);
    }
}

static cudaTextureDesc createCudaTextureDescFromGLTF(int minFilter, int magFilter, int wrapS,
                                                     int wrapT, bool mipmaps) {
    cudaTextureDesc desc = {};

    auto gltfWrapToCuda = [](int wrap) {
        switch (wrap) {
        case TINYGLTF_TEXTURE_WRAP_REPEAT:
            return cudaAddressModeWrap;
        case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
            return cudaAddressModeClamp;
        case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
            return cudaAddressModeMirror;
        default:
            return cudaAddressModeWrap;
        }
    };

    desc.addressMode[0] = gltfWrapToCuda(wrapS);
    desc.addressMode[1] = gltfWrapToCuda(wrapT);
    desc.addressMode[2] = cudaAddressModeWrap; // for 3D textures

    auto gltfFilterToCuda = [](int filter) {
        switch (filter) {
        case TINYGLTF_TEXTURE_FILTER_NEAREST:
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
        case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
            return cudaFilterModePoint;
        case TINYGLTF_TEXTURE_FILTER_LINEAR:
        case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
        case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
            return cudaFilterModeLinear;
        default:
            return cudaFilterModeLinear;
        }
    };

    desc.filterMode = gltfFilterToCuda(magFilter); // prioritize magFilter

    // ---- Mipmaps ----
    if (mipmaps) {
        // Only enable if you have a mipmapped CUDA array
        desc.mipmapFilterMode = gltfFilterToCuda(minFilter);
        desc.maxMipmapLevelClamp = 1000.0f; // allow all mip levels
        desc.minMipmapLevelClamp = 0.0f;
    } else {
        desc.mipmapFilterMode = cudaFilterModePoint; // ignored
        desc.maxMipmapLevelClamp = 0.0f;
        desc.minMipmapLevelClamp = 0.0f;
    }

    desc.normalizedCoords = 1; // use [0,1] texture coordinates
    desc.maxAnisotropy = 1;

    return desc;
}

static cudaChannelFormatDesc getChannelDescription(bool isFloat, int channels) {
    if (isFloat) {
        switch (channels) {
        case 1:
            return cudaCreateChannelDesc<float>();
        case 2:
            return cudaCreateChannelDesc<float2>();
        case 4:
            return cudaCreateChannelDesc<float4>();
        default:
            log_error("Unsupported channels %d\n", channels);
            exit(EXIT_FAILURE);
        }
    } else {
        switch (channels) {
        case 1:
            return cudaCreateChannelDesc<uchar1>();
        case 2:
            return cudaCreateChannelDesc<uchar2>();
        case 4:
            return cudaCreateChannelDesc<uchar4>();
        default:
            log_error("Unsupported channels %d\n", channels);
            exit(EXIT_FAILURE);
        }
    }
    return {}; // should never be reached
}

static image_resource_t load_image(fs::path &path_img) {
    log_trace("Loading image: %s\n", path_img.c_str());

    bool isFloat;

    if (path_img.extension() == ".png" || path_img.extension() == ".jpg") {
        isFloat = false;
    } else if (path_img.extension() == ".hdr") {
        isFloat = true;
    } else {
        log_error("Unknown image extension: %s\n", path_img.extension().c_str());
        exit(EXIT_FAILURE);
    }

    // get initial info of image
    int width, height, actual_components;
    if (!stbi_info(path_img.c_str(), &width, &height, &actual_components)) {
        log_error("STBI error: %s\n", stbi_failure_reason());
        exit(EXIT_FAILURE);
    }

    int loaded_components = 0;

    switch (actual_components) {
    case 3:
    case 4:
        loaded_components = 4;
        break;
    default:
        log_error("image with %d components unsupported\n", actual_components);
        exit(EXIT_FAILURE);
    }

    cudaChannelFormatDesc channelDesc = getChannelDescription(isFloat, loaded_components);

    void *imageData;
    size_t size_of_component;

    if (isFloat) {
        imageData = (void *)stbi_loadf(path_img.c_str(), &width, &height, &actual_components,
                                       loaded_components);
        size_of_component = sizeof(float);
    } else {
        imageData = (void *)stbi_load(path_img.c_str(), &width, &height, &actual_components,
                                      loaded_components);
        size_of_component = sizeof(stbi_uc);
    }
    if (!imageData) {
        log_error("STBI error: %s\n", stbi_failure_reason());
        exit(EXIT_FAILURE);
    }
    assert(loaded_components != 3);

    // setup array
    cudaArray_t cuArray;
    cudaError_t cuerr;
    cuerr = cudaMallocArray(&cuArray, &channelDesc, width, height);
    if (check_cuda_err(cuerr) != cudaSuccess) {
        log_error("cudaMallocArray\n");
        exit(EXIT_FAILURE);
    }

    // copy mem to array
    size_t copy_size = size_of_component * loaded_components * height * width;
    cuerr = cudaMemcpyToArray(cuArray, 0, 0, imageData, copy_size, cudaMemcpyHostToDevice);
    if (check_cuda_err(cuerr) != cudaSuccess) {
        log_error("cudaMemcpyToArray\n");
        exit(EXIT_FAILURE);
    }

    stbi_image_free(imageData);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    log_trace("Loaded image (w=%d, h=%d, actual_components=%d, loaded_components=%d)\n", width,
              height, actual_components, loaded_components);

    image_resource_t resource;
    resource.resourceDesc = resDesc;
    resource.channels = loaded_components;
    resource.isFloat = isFloat;

    return resource;
}

int load_texture(std::vector<texture_t> &textures, const image_resource_t &resource,
                 int minFilter, int magFilter, int wrapS, int wrapT) {
    cudaTextureDesc texDesc =
        createCudaTextureDescFromGLTF(minFilter, magFilter, wrapS, wrapT, false);

    if (resource.isFloat) {
        texDesc.readMode = cudaReadModeElementType;
    } else {
        texDesc.readMode = cudaReadModeNormalizedFloat;
    }

    texture_t tex;
    tex.channels = resource.channels;
    tex.isFloat = resource.isFloat;

    cudaError_t cuerr;

    // raw view of texture
    texDesc.sRGB = 0;
    cuerr =
        cudaCreateTextureObject(&tex.texture_raw, &resource.resourceDesc, &texDesc, nullptr);
    if (check_cuda_err(cuerr) != cudaSuccess) {
        log_error("cudaCreateTextureObject(&texObj, &resource.resource, &texDesc, nullptr);\n");
        exit(EXIT_FAILURE);
    }

    // now linearize same texture
    texDesc.sRGB = 1;
    cuerr = cudaCreateTextureObject(&tex.texture_linearized, &resource.resourceDesc, &texDesc,
                                    nullptr);
    if (check_cuda_err(cuerr) != cudaSuccess) {
        log_error("cudaCreateTextureObject(&texObj, &resource.resource, &texDesc, nullptr);\n");
        exit(EXIT_FAILURE);
    }

    textures.push_back(tex);
    return textures.size() - 1; // index of new texture
}

void Scene::read_gltf(const char *filename, config_t &config) {
    log_info("Parsing .gltf... \n");

    tg::Model model;
    tg::TinyGLTF loader;
    std::string err;
    std::string warn;
    const std::string ext = tg::GetFilePathExtension(filename);

    bool ret = false;
    if (ext.compare("glb") == 0) {
        // assume binary glTF.
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    } else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    }

    if (!warn.empty()) {
        log_warning("glTF parse warning: %s\n", warn);
    }

    if (!err.empty()) {
        log_error("glTF parse error: %s\n", err);
    }
    if (!ret) {
        log_error("Failed to load glTF: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    log_trace("Loaded gltf contains:\n");
    log_trace("  %d accessors\n", model.accessors.size());
    log_trace("  %d animations\n", model.animations.size());
    log_trace("  %d buffers\n", model.buffers.size());
    log_trace("  %d bufferViews\n", model.bufferViews.size());
    log_trace("  %d materials\n", model.materials.size());
    log_trace("  %d meshes\n", model.meshes.size());
    log_trace("  %d nodes\n", model.nodes.size());
    log_trace("  %d textures\n", model.textures.size());
    log_trace("  %d images\n", model.images.size());
    log_trace("  %d skins\n", model.skins.size());
    log_trace("  %d samplers\n", model.samplers.size());
    log_trace("  %d cameras\n", model.cameras.size());
    log_trace("  %d scenes\n", model.scenes.size());
    log_trace("  %d lights\n", model.lights.size());

    temp_scene_t temp_scene;
    std::vector<image_resource_t> image_resources;
    std::vector<texture_t> textures;

    // IMAGES
    for (const auto &img : model.images) {

        fs::path img_name = img.uri.c_str();
        fs::path path_gltf = filename;
        fs::path path_img = path_gltf.parent_path() / img_name;

        image_resource_t image_resource = load_image(path_img);
        image_resources.push_back(image_resource);
    }

    // TEXTURES
    for (const auto &tex : model.textures) {
        log_trace("Creating texture: source=%d, sampler=%d\n", tex.source, tex.sampler);

        image_resource_t &resource = image_resources[tex.source];
        const auto &sampler = model.samplers[tex.sampler];

        load_texture(textures, resource, sampler.minFilter, sampler.magFilter, sampler.wrapS,
                     sampler.wrapT);
    }

    // MATERIALS

    for (const auto &sceneMat : model.materials) {
        parse_material(temp_scene, sceneMat);
    }

    // NODES

    tg::Scene &modelScene = model.scenes[model.defaultScene];
    for (const int &nodeIndex : modelScene.nodes) {
        const tg::Node &node = model.nodes[nodeIndex];
        scene_parse_node(temp_scene, model, node, Mat4::Identity());
    }

    // additional stuff from config
    // clear color and texture
    this->clearColor = config.world_clear_color;
    fs::path path_clear_color_texture = config.world_clear_color_texture;
    image_resource_t clear_color_image_resource = load_image(path_clear_color_texture);

    this->clearTexture =
        load_texture(textures, clear_color_image_resource, TINYGLTF_TEXTURE_FILTER_LINEAR,
                     TINYGLTF_TEXTURE_FILTER_LINEAR, TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT,
                     TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT);

    // FINALIZE CLASS
    this->_location = CudaLocation::Host;

    if (temp_scene.cameras.size() == 0) {
        log_warning("No camera found in scene! Placing default camera.\n");
        camera_t default_camera;
        default_camera.position = config.default_camera_position;
        default_camera.target = config.default_camera_target;
        default_camera.updir = config.default_camera_updir;
        default_camera.yfov = config.default_camera_yfov;

        // log_trace("  position: (%.2f, %.2f, %.2f)\n", default_camera.position.x,
        //           default_camera.position.y, default_camera.position.z);
        // log_trace("  target: (%.2f, %.2f, %.2f)\n", default_camera.target.x,
        //           default_camera.target.y, default_camera.target.z);
        // log_trace("  updir: (%.2f, %.2f, %.2f)\n", default_camera.updir.x,
        //           default_camera.updir.y, default_camera.updir.z);
        // log_trace("  yfov: %.2f", default_camera.yfov);

        temp_scene.cameras.push_back(default_camera);
    } else if (temp_scene.cameras.size() > 1) {
        log_warning("Multiple cameras found in scene, choosing camera 0.\n");
    }

    this->camera = temp_scene.cameras[0];

    // copy into final scene
    fixed_array_from_vector(this->vertices, temp_scene.vertices);
    fixed_array_from_vector(this->faces, temp_scene.faces);
    fixed_array_from_vector(this->materials, temp_scene.materials);
    fixed_array_from_vector(this->lights, temp_scene.lights);
    fixed_array_from_vector(this->textures, textures);

    log_info("Done parsing .gltf\n");
}

void Scene::device_from_host(const Scene &h_scene) {
    log_info("Copying scene to device... \n");
    _location = CudaLocation::Device;

    *this = h_scene;

    size_t totalSize = 0;
    totalSize += copy_device_fixed_array(&this->vertices, &h_scene.vertices);
    totalSize += copy_device_fixed_array(&this->faces, &h_scene.faces);
    totalSize += copy_device_fixed_array(&this->lights, &h_scene.lights);
    totalSize += copy_device_fixed_array(&this->materials, &h_scene.materials);
    totalSize += copy_device_fixed_array(&this->textures, &h_scene.textures);

    char buf[64];
    human_readable_size(buf, totalSize);
    log_info("Done [%s]\n", buf);
}

void Scene::_free() {
    if (_location == CudaLocation::Host) {
        free(vertices.items);
        vertices.items = NULL;
        free(faces.items);
        faces.items = NULL;
        free(materials.items);
        materials.items = NULL;
        free(lights.items);
        lights.items = NULL;
        free(textures.items);
        textures.items = NULL;
    }

    if (_location == CudaLocation::Device) {
        device_free(vertices.items);
        device_free(faces.items);
        device_free(lights.items);
        device_free(materials.items);
        device_free(textures.items);
    }
}

// __device__ float linearize_from_srgb_scalar(float c) {
//     // Standard sRGB -> linear conversion
//     if (c <= 0.04045f) {
//         return c / 12.92f;
//     } else {
//         return powf((c + 0.055f) / 1.055f, 2.4f);
//     }
// }

__device__ Vec4 sample_texture(const texture_t &tex, float u, float v, bool linearize_srgb) {
    if (tex.isFloat && linearize_srgb) {
        // the only case where it should be linearized
        return tex2D<float4>(tex.texture_linearized, u, v);
    } else {
        return tex2D<float4>(tex.texture_raw, u, v);
    }
}

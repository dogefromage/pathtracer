#include "scene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>

#include "logger.h"
#include "tiny_gltf.h"
#include "utils.h"

using namespace tinygltf;

static Vec3 parseVec3(const std::vector<double> &v) {
    assert(v.size() >= 3);
    return {(float)v[0], (float)v[1], (float)v[2]};
}

static Mat4 getForwardTransform(const Node &node) {
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

static void scene_parse_camera(temp_scene_t &scene, const Model &model, const Node &node,
                               const Mat4 &modelTransform) {
    const Camera &cam = model.cameras[node.camera];

    if (cam.type != "perspective") {
        printf("unsupported camera type: %s\n", cam.type.c_str());
        exit(EXIT_FAILURE);
    }

    scene.camera.yfov = cam.perspective.yfov;

    Vec4 position = modelTransform * Vec4{0, 0, 0, 1};
    Vec4 target = modelTransform * Vec4{0, 0, -1, 1};
    Vec4 updir = modelTransform * Vec4{0, 1, 0, 1};

    scene.camera.position = position.dehomogenise();
    scene.camera.target = target.dehomogenise();
    scene.camera.updir = updir.dehomogenise() - scene.camera.position;
}

static void scene_parse_light(temp_scene_t &scene, const Model &model, const Node &node,
                              const Mat4 &modelTransform) {
    const Light &modelLight = model.lights[node.light];

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

    if (modelLight.type == "directional") {
        light.type = LIGHT_DIRECTIONAL;
    } else if (modelLight.type == "point") {
        light.type = LIGHT_POINT;
    } else {
        printf("Unsupported light '%s'\n", modelLight.type.c_str());
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
    log_trace("  Color: (%.2f, %.2f, %.2f)\n", material->color.x, material->color.y,
              material->color.z);
    log_trace("  Transmission: %.2f\n", material->transmission);
    log_trace("  Emissive: (%.2f, %.2f, %.2f)\n", material->emissive.x, material->emissive.y,
              material->emissive.z);
    log_trace("  Metallic: %.2f\n", material->metallic);
    log_trace("  Roughness: %.2f\n", material->roughness);
    log_trace("  IoR: %.2f\n", material->ior);
}

static uint32_t scene_parse_material(temp_scene_t &scene, const Model &model,
                                     int materialIndex) {
    auto previousMatIndex = scene.materialMapping.find(materialIndex);
    if (previousMatIndex != scene.materialMapping.end()) {
        return previousMatIndex->second;
    }

    const Material &sceneMat = model.materials[materialIndex];
    material_t mat;

    log_trace("Material: %s\n", sceneMat.name.c_str());

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
            printf("Unknown extension: %s\n", extension.first.c_str());
        }
    }

    // SET MATERIAL VALUES

    mat.color = parseVec3(sceneMat.pbrMetallicRoughness.baseColorFactor);
    mat.emissive = emissiveStrength * parseVec3(sceneMat.emissiveFactor);
    mat.metallic = (float)sceneMat.pbrMetallicRoughness.metallicFactor;
    mat.roughness = (float)sceneMat.pbrMetallicRoughness.roughnessFactor;
    mat.ior = ior;
    mat.transmission = (float)transmissionFactor;

    // ADD MATERIAL

    uint32_t newIndex = scene.materials.size();
    scene.materials.push_back(mat);
    scene.materialMapping[materialIndex] = newIndex;

    print_material(&mat);

    return newIndex;
}

static void scene_parse_acc_vec3(std::vector<Vec3> &list, const Model &model, int accIndex) {
    const Accessor &acc = model.accessors[accIndex];
    const BufferView &bufView = model.bufferViews[acc.bufferView];
    const Buffer &buf = model.buffers[bufView.buffer];

    AABB actualBounds;
    AABB givenBounds;
    bool hasGivenBounds = false;

    if (acc.maxValues.size() && acc.maxValues.size()) {
        hasGivenBounds = true;
        givenBounds.min = {(float)acc.minValues[0], (float)acc.minValues[1],
                           (float)acc.minValues[2]};
        givenBounds.max = {(float)acc.maxValues[0], (float)acc.maxValues[1],
                           (float)acc.maxValues[2]};
    }

    assert(!acc.sparse.isSparse && "sparse mesh unsupported"); // TODO maybe

    int byteStride = acc.ByteStride(bufView);

    for (size_t i = 0; i < acc.count; i++) {
        // https://github.com/syoyo/tinygltf/wiki/Accessing-vertex-data

        size_t byteNumber = acc.byteOffset + bufView.byteOffset + byteStride * i;
        const float *item = reinterpret_cast<const float *>(&buf.data[byteNumber]);

        Vec3 v = {item[0], item[1], item[2]};
        // v.print();
        list.push_back(v);

        actualBounds.grow(v);

        if (hasGivenBounds) {
            // sanity check
            assert(givenBounds.contains(v));
        }
    }

    // char s_min[64], s_max[64];
    // actualBounds.min.snprint(s_min, 64);
    // actualBounds.max.snprint(s_max, 64);
    // log_trace("%lu vec3 parsed in given min and max:\n", acc.count);
}

static void scene_parse_acc_indices(std::vector<uint32_t> &list, const Model &model,
                                    int accIndex) {
    const Accessor &acc = model.accessors[accIndex];
    const BufferView &bufView = model.bufferViews[acc.bufferView];
    const Buffer &buf = model.buffers[bufView.buffer];

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
        default:
            printf("Component type not implemented: %d\n", acc.componentType);
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

static void scene_parse_mesh(temp_scene_t &scene, const Model &model, const Node &node,
                             const Mat4 &modelTransform) {
    log_trace("Parsing mesh of %s\n", node.name.c_str());

    const Mesh &mesh = model.meshes[node.mesh];
    Mat3 linearModelTransform = modelTransform.getLinearPart();

    for (const Primitive &prim : mesh.primitives) {
        int newMatIndex = scene_parse_material(scene, model, prim.material);

        std::vector<Vec3> positions;
        std::vector<Vec3> normals;
        std::vector<uint32_t> indices;

        for (const auto &attr : prim.attributes) {
            if (attr.first == "POSITION") {
                log_trace("Scanning POSITION\n");
                scene_parse_acc_vec3(positions, model, attr.second);
            } else if (attr.first == "NORMAL") {
                log_trace("Scanning NORMAL\n");
                scene_parse_acc_vec3(normals, model, attr.second);
            } else {
                log_warning("skipped unsupported attribute '%s'\n", attr.first.c_str());
            }
        }

        log_trace("Scanning INDICES\n");
        scene_parse_acc_indices(indices, model, prim.indices);

        int vertCount = positions.size();
        assert(vertCount);

        bool hasNormals = normals.size() > 0;
        assert(!hasNormals || normals.size() == vertCount);

        int numIndices = indices.size();
        assert(numIndices && numIndices % 3 == 0);

        int vertOffset = scene.vertices.size();
        for (int i = 0; i < vertCount; i++) {
            vertex_t v;

            Vec4 p = modelTransform * Vec4::toHomogeneous(positions[i]);
            v.position = p.dehomogenise();

            if (hasNormals) {
                v.normal = linearModelTransform * normals[i];
            } else {
                v.normal.set(0);
            }
            scene.vertices.push_back(v);
        }

        for (int i = 0; i < numIndices; i += 3) {
            face_t face;
            face.material = newMatIndex;

            face.vertexCount = 3;
            face.vertices[0] = vertOffset + indices[i];
            face.vertices[1] = vertOffset + indices[i + 1];
            face.vertices[2] = vertOffset + indices[i + 2];

            if (hasNormals) {
                face.shading = BARY_SHADING;
                face.faceNormal.set(0);
            } else {
                face.shading = FLAT_SHADING;

                // compute right handed flat normals
                // world space -> does not need transform
                const Vec3 &A = scene.vertices[face.vertices[0]].position;
                const Vec3 &B = scene.vertices[face.vertices[1]].position;
                const Vec3 &C = scene.vertices[face.vertices[2]].position;

                face.faceNormal = (B - A).cross(C - A).normalized();
            }

            scene.faces.push_back(face);
        }
    }
}

static void scene_parse_node(temp_scene_t &scene, const Model &model, const Node &node,
                             Mat4 parentTransform) {
    Mat4 nodeForwardTransform = getForwardTransform(node);
    Mat4 modelTransform = parentTransform * nodeForwardTransform;

    if (node.camera >= 0) {
        scene_parse_camera(scene, model, node, modelTransform);
    }
    if (node.light >= 0) {
        scene_parse_light(scene, model, node, modelTransform);
    }
    if (node.mesh >= 0) {
        scene_parse_mesh(scene, model, node, modelTransform);
    }

    for (const int &childIndex : node.children) {
        const Node &child = model.nodes[childIndex];
        scene_parse_node(scene, model, child, modelTransform);
    }
}

void scene_parse_gltf(scene_t &finalScene, const char *filename) {
    log_info("Parsing .gltf... \n");

    Model model;
    TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        log_warning("TinyGLTF WARNING: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        log_error("TinyGLTF ERROR: %s\n", err.c_str());
    }
    if (!ret) {
        log_error("Failed to parse glTF\n");
        exit(EXIT_FAILURE);
    }

    temp_scene_t tempScene;

    Scene &modelScene = model.scenes[model.defaultScene];

    for (const int &nodeIndex : modelScene.nodes) {
        const Node &node = model.nodes[nodeIndex];
        scene_parse_node(tempScene, model, node, Mat4::Identity());
    }

    finalScene.camera = tempScene.camera;

    // copy into final scene
    fixed_array_from_vector(finalScene.vertices, tempScene.vertices);
    fixed_array_from_vector(finalScene.faces, tempScene.faces);
    fixed_array_from_vector(finalScene.materials, tempScene.materials);
    fixed_array_from_vector(finalScene.lights, tempScene.lights);

    log_info("Done parsing .gltf\n");
}

void scene_delete_host(scene_t &scene) {
    free(scene.vertices.items);
    scene.vertices.items = NULL;

    free(scene.faces.items);
    scene.faces.items = NULL;

    free(scene.materials.items);
    scene.materials.items = NULL;

    free(scene.lights.items);
    scene.lights.items = NULL;
}

void scene_copy_to_device(scene_t **dev_scene, scene_t *host_scene) {
    log_info("Copying scene to device... \n");

    scene_t placeholder = *host_scene;
    size_t totalSize = 0;

    totalSize += copy_device_fixed_array(&placeholder.vertices, &host_scene->vertices);
    totalSize += copy_device_fixed_array(&placeholder.faces, &host_scene->faces);
    totalSize += copy_device_fixed_array(&placeholder.lights, &host_scene->lights);
    totalSize += copy_device_fixed_array(&placeholder.materials, &host_scene->materials);

    totalSize += copy_device_struct(dev_scene, &placeholder);

    char buf[64];
    human_readable_size(buf, totalSize);
    log_info("Done [%s]\n", buf);
}

void free_device_scene(scene_t *dev_scene) {
    scene_t placeholder;
    copy_host_struct(&placeholder, dev_scene);

    device_free(placeholder.vertices.items);
    device_free(placeholder.faces.items);
    device_free(placeholder.lights.items);
    device_free(placeholder.materials.items);

    device_free(dev_scene);
}

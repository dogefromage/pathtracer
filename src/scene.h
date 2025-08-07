#pragma once

#include <unordered_map>
#include <vector>

#include "mathops.h"

#define MATERIAL_NAME_SIZE 32
#define MAX_FACE_VERTICES 3

typedef struct {
    Vec3 position;
    /* if no normals passed, normal field set to (0,0,0)
     * and face shading is set to SHADING_FLAT
     */
    Vec3 normal;

} vertex_t;

enum face_shading_t {
    FLAT_SHADING,
    BARY_SHADING,
};

typedef struct {
    uint32_t vertices[MAX_FACE_VERTICES];
    uint32_t vertexCount;
    uint32_t material;
    Vec3 faceNormal;
    face_shading_t shading;
} face_t;

typedef struct {
    char name[MATERIAL_NAME_SIZE];
    Vec3 color, emissive;
    float metallic, roughness, ior, colorAlpha;
} material_t;

enum light_type_t {
    LIGHT_DIRECTIONAL,
    LIGHT_POINT,
    // LIGHT_SPOT, // TODO
};

// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md
typedef struct {
    light_type_t type;
    Vec3 color, position, direction;
    float intensity;
} light_t;

typedef struct {
    Vec3 position, target, updir;
    float yfov;
} camera_t;

typedef struct {
    fixed_array<vertex_t> vertices;
    fixed_array<face_t> faces;
    fixed_array<material_t> materials;
    fixed_array<light_t> lights;

    camera_t camera;
} scene_t;

typedef struct {
    std::vector<vertex_t> vertices;
    std::vector<face_t> faces;
    std::vector<material_t> materials;
    std::vector<light_t> lights;

    std::unordered_map<int, uint32_t> materialMapping;

    camera_t camera;

} temp_scene_t;

void scene_parse_gltf(scene_t &scene, const char *filename);
void scene_delete_host(scene_t &scene);

void scene_copy_to_device(scene_t **dev_scene, scene_t *host_scene);
void free_device_scene(scene_t *dev_scene);

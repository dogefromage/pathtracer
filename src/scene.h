#pragma once

#include <unordered_map>
#include <vector>

#include "config.h"
#include "mathops.h"
#include "utils.h"

#define MATERIAL_NAME_SIZE 32
#define MAX_FACE_VERTICES 3

template <typename T> struct fixed_array {
    uint32_t count;
    T *items;

    MATH_PLATFORM const T &operator[](size_t k) const {
        assert(k < count);
        return items[k];
    }

    MATH_PLATFORM T &operator[](size_t k) {
        assert(k < count);
        return items[k];
    }
};

// host placeholder struct which will be copied over later, must get device pointers
template <typename T>
size_t copy_device_fixed_array(fixed_array<T> *placeholder, const fixed_array<T> *src) {
    cudaError_t err;
    size_t size = sizeof(T) * src->count;

    // log_info("A\n");

    err = cudaMalloc(&placeholder->items, size);
    if (check_cuda_err(err))
        exit(EXIT_FAILURE);

    // log_info("B\n");

    err = cudaMemcpy(placeholder->items, src->items, size, cudaMemcpyHostToDevice);
    if (check_cuda_err(err))
        exit(EXIT_FAILURE);

    return size;
}

template <typename T>
void fixed_array_from_vector(fixed_array<T> &dest, const std::vector<T> &src) {
    dest.count = src.size();
    dest.items = (T *)malloc(sizeof(T) * dest.count);
    std::copy(src.begin(), src.end(), dest.items);
}

typedef struct {
    /* if no normals passed, normal field set to (0,0,0)
     * and face shading is set to SHADING_FLAT
     */
    Vec3 position, normal, texcoord0;
    Vec4 tangent;

} vertex_t;

enum face_shading_t {
    FLAT_SHADING,
    BARY_SHADING,
};

enum alpha_mode_t {
    ALPHA_OPAQUE,
    ALPHA_MASK,
    ALPHA_BLEND,
};

typedef struct {
    uint32_t vertices[MAX_FACE_VERTICES];
    uint32_t vertexCount;
    uint32_t material;
    Vec3 normal;
    Vec4 tangent;
    face_shading_t shading;
} face_t;

typedef struct {
    char name[MATERIAL_NAME_SIZE];
    Vec4 baseColorFactor;
    int32_t baseColorTexture, normalTexture;
    Vec3 emissive;
    float metallic, roughness, ior, transmission;
    alpha_mode_t alphaMode;
    float alphaCutoff;
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
    cudaResourceDesc resourceDesc;
    int channels, isFloat;
} image_resource_t;

typedef struct {
    // texture_linearized: srgb to linear has been applied
    // texture_raw: the original values of the image
    cudaTextureObject_t texture_linearized, texture_raw;
    int channels, isFloat;
} texture_t;

typedef struct {
    std::vector<vertex_t> vertices;
    std::vector<face_t> faces;
    std::vector<material_t> materials;
    std::vector<light_t> lights;

    std::vector<camera_t> cameras;
} temp_scene_t;

// typedef struct {
//     fixed_array<vertex_t> vertices;
//     fixed_array<face_t> faces;
//     fixed_array<material_t> materials;
//     fixed_array<light_t> lights;

//     camera_t camera;
// } scene_t;

// void scene_parse_gltf(scene_t &scene, const char *filename);
// void scene_delete_host(scene_t &scene);
// void scene_copy_to_device(scene_t **dev_scene, scene_t *host_scene);
// void free_device_scene(scene_t *dev_scene);

class Scene {

    CudaLocation _location = CudaLocation::Host;

  public:
    fixed_array<vertex_t> vertices;
    fixed_array<face_t> faces;
    fixed_array<material_t> materials;
    fixed_array<light_t> lights;
    fixed_array<texture_t> textures;

    camera_t camera;
    int clearTexture{-1};
    Vec3 clearColor;

    void read_gltf(const char *filename, config_t &config);
    void device_from_host(const Scene &h_scene);
    void _free();
};

__device__ Vec4 sample_texture(const texture_t &tex, float u, float v, bool linearize_srgb);

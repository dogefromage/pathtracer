#pragma once
#include "intersect.h"
#include "mathc.h"
#include "scene.h"

typedef struct {
    mfloat_t min[VEC3_SIZE], max[VEC3_SIZE];
} AABB;

typedef struct {
    AABB bounds;
    uint32_t leftChild, rightChild;
    uint32_t start, end;
} BVHNode;

typedef struct {
    BVHNode* nodes;
    uint32_t* indices;
    uint32_t nodeCount, primitiveCount, maxNodeCount;
    struct vec3* centroids;
} BVH;

__host__ void bvh_build(BVH* bvh, const obj_scene_data* scene);
__host__ void bvh_free_host(BVH* h_bvh);
__host__ int bvh_copy_device(BVH** d_bvh, const BVH* h_bvh);
__host__ int bvh_free_device(BVH* d_bvh);

__device__ void
bvh_intersect(const __restrict__ BVH* bvh, uint32_t nodeIndex,
              const __restrict__ obj_scene_data* scene, const Ray* ray, Intersection* hit);

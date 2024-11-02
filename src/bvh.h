#pragma once
#include <cstdint>

#include "intersect.h"
#include "scene.h"

typedef struct {
    Vec3 min, max;
} aabb_t;

typedef struct {
    aabb_t bounds;
    uint32_t leftChild, rightChild;
    uint32_t start, end;
} bvh_node_t;

typedef struct {
    bvh_node_t* nodes;
    uint32_t* indices;
    uint32_t nodeCount, maxNodeCount, primitiveCount;
    Vec3* centroids;
} bvh_t;

typedef struct {
    uint32_t totalSkippedFaces, numberLeaves;
    float averageLeafSize;
    time_t lastInfo;
    int maxDepth;
} bvh_stats_t;

#define BVH_ROOT_NODE 0

void bvh_build(bvh_t& bvh, const obj_scene_data& scene);
void bvh_free_host(bvh_t& h_bvh);
int bvh_copy_device(bvh_t** d_bvh, const bvh_t* h_bvh);
int bvh_free_device(bvh_t* d_bvh);

PLATFORM void
bvh_intersect(const __restrict__ bvh_t* bvh, uint32_t nodeIndex,
              const __restrict__ obj_scene_data* scene, const Ray& ray, intersection_t& hit);

PLATFORM void
bvh_intersect_iterative(const __restrict__ bvh_t* bvh,
                        const __restrict__ obj_scene_data* scene, const Ray& ray, intersection_t& hit);

#pragma once
#include <cstdint>

#include "intersect.h"
#include "scene.h"

typedef struct {
    AABB bounds;
    uint32_t leftChild, rightChild;
    uint32_t start, end;
} bvh_node_t;

typedef struct {
    fixed_array<bvh_node_t> nodes;
    fixed_array<uint32_t> indices;
    fixed_array<Vec3> centroids;
    uint32_t nodeCount; //, maxNodeCount, primitiveCount;
} bvh_t;

typedef struct {
    uint32_t totalSkippedFaces, numberLeaves;
    float averageLeafSize;
    time_t lastInfo;
    int maxDepth;
} bvh_stats_t;

void bvh_build(bvh_t& bvh, const scene_t& scene);
void bvh_free_host(bvh_t& h_bvh);
void bvh_copy_device(bvh_t** d_bvh, const bvh_t* h_bvh);
void bvh_free_device(bvh_t* d_bvh);

PLATFORM void
bvh_intersect_iterative(const __restrict__ bvh_t* bvh,
                        const __restrict__ scene_t* scene, const Ray& ray, intersection_t& hit);

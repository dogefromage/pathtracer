#pragma once
#include <cstdint>

#include "intersect.h"
#include "scene.h"

typedef struct {
    AABB bounds;
    uint32_t leftChild, rightChild;
    uint32_t start, end;
} bvh_node_t;

// typedef struct {
//     fixed_array<bvh_node_t> nodes;
//     fixed_array<uint32_t> indices;
//     fixed_array<Vec3> centroids;
//     uint32_t nodeCount; //, maxNodeCount, primitiveCount;
// } bvh_t;

typedef struct {
    uint32_t totalSkippedFaces, numberLeaves;
    float averageLeafSize;
    time_t lastInfo;
    int maxDepth;
} bvh_stats_t;

class BVH {
    CudaLocation _location = CudaLocation::Host;

  public:
    fixed_array<bvh_node_t> nodes;
    fixed_array<uint32_t> indices;
    fixed_array<Vec3> centroids;
    uint32_t nodeCount;

    void build(const Scene &scene);
    void device_from_host(const BVH &h_bvh);
    void _free();
};

__device__ void bvh_intersect_iterative(const BVH &bvh, const Scene &scene, const Ray &ray, intersection_t &hit);

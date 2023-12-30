#pragma once
#include "mathc.h"
#include "obj_parser.h"
#include "intersect.h"

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
    uint32_t nodeCount;
    struct vec3* centroids;
} BVH;

void bvh_build(BVH* bvh, obj_scene_data* scene);
void bvh_intersect(BVH* bvh, uint32_t nodeIndex, obj_scene_data* scene, Ray* ray, Intersection* hit);
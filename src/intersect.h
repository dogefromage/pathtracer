#pragma once
#include "mathc.h"
#include "scene.h"

#define CLEAR_DISTANCE 1e30

typedef struct {
    mfloat_t o[VEC3_SIZE], r[VEC3_SIZE];
} Ray;

typedef struct {
    int has_hit;
    mfloat_t distance;
    mfloat_t position[VEC3_SIZE], normal[VEC3_SIZE], texture_coord[VEC3_SIZE];
    obj_material* mat;
} Intersection;

__host__ __device__ void 
intersect_face(const __restrict__ obj_scene_data* scene, const Ray* ray, Intersection* hit, int faceIndex);

__host__ __device__ void 
intersect_crude(const __restrict__ obj_scene_data* scene, const Ray* ray, Intersection* hit);

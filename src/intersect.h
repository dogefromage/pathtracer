#pragma once
#include "mathc.h"
#include "obj_parser.h"

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

void intersect_face(obj_scene_data* scene, Ray* ray, Intersection* hit, int faceIndex);
void intersect_crude(obj_scene_data* scene, Ray* ray, Intersection* hit);

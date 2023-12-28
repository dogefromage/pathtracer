#pragma once
#include "mathc.h"
#include "obj_parser.h"

#define CLEAR_DISTANCE 1e30

typedef struct {
    mfloat_t o[VEC3_SIZE], r[VEC3_SIZE];
} Ray;

// Describes direction sampled from a hemi or
// full sphere with a probability of occuring
typedef struct {
    struct vec3 dir;
    mfloat_t prob;
} DirectionalSample;

typedef struct {
    int has_hit;
    mfloat_t distance;
    mfloat_t position[VEC3_SIZE], normal[VEC3_SIZE], texture_coord[VEC3_SIZE];
    obj_material* mat;
} Intersection;

void Ray_at(mfloat_t* out, Ray* ray, mfloat_t t);
void hemi_sample_uniform(mfloat_t* out, mfloat_t* normal_unit);
void sphere_sample_uniform(mfloat_t* out);
void barycentric_lincom(mfloat_t* out, mfloat_t* A, mfloat_t* B, mfloat_t* C, mfloat_t t, mfloat_t u, mfloat_t v);
void get_camera_ray(Ray* ray, obj_scene_data* scene, mfloat_t u, mfloat_t v);
void color_correct(mfloat_t* c);
void render(mfloat_t* pixel, obj_scene_data* scene, mfloat_t u, mfloat_t v, int samples);
#pragma once
#include "mathc.h"
#include "obj_parser.h"

#define CLEAR_DISTANCE 1e30

typedef struct {
    struct vec3 o, r;
} ray_t;

void ray_at(mfloat_t* out, ray_t* ray, mfloat_t t);

void barycentric_lincom(mfloat_t* out, mfloat_t* A, mfloat_t* B, mfloat_t* C, mfloat_t t, mfloat_t u, mfloat_t v);

typedef struct {
    mfloat_t distance;
    struct vec3 position, normal, texture_coord;
    int material;
} Intersection;

void get_camera_ray(ray_t* ray, obj_scene_data* scene, mfloat_t u, mfloat_t v);
void render(struct vec3 *pixel, obj_scene_data* scene, mfloat_t u, mfloat_t v);

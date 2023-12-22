#pragma once
#include "math3d.h"
#include "scene.h"
#include "obj_parser.h"

typedef struct {
    vec3_t o, r;
} ray_t;

static inline vec3_t ray_at(ray_t* ray, float t) {
    return v3_add(ray->o, v3_muls(ray->r, t));
}

typedef struct {
    float distance;
    double position[3], normal[3], texture_coord[3];
    int material;
} Intersection;

void render(vec3_t* pixel, obj_scene_data* scene, float u, float v);

#pragma once
#include "mathops.h"
#include "config.h"
#include "scene.h"

#define CLEAR_DISTANCE 1e30

typedef struct {
    Vec3 o, r;
} Ray;

inline std::ostream& operator<<(std::ostream& os, const Ray& ray) {
    os << "[ " << ray.o << ", " << ray.r << " ]";
    return os;
}

typedef struct {
    int has_hit;
    float distance;
    Vec3 position, normal, texture_coord;
    obj_material* mat;
} intersection_t;

PLATFORM void 
intersect_face(const __restrict__ obj_scene_data* scene, const Ray& ray, intersection_t& hit, int faceIndex);

PLATFORM void 
intersect_crude(const __restrict__ obj_scene_data* scene, const Ray& ray, intersection_t& hit);

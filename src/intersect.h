#pragma once
#include "mathops.h"
#include "config.h"
#include "scene.h"

#define CLEAR_DISTANCE 1e30

inline std::ostream& operator<<(std::ostream& os, const Ray& ray) {
    os << "[ " << ray.o << ", " << ray.r << " ]";
    return os;
}

typedef struct {
    int has_hit;
    float distance;
    Vec3 position, lightingNormal, trueNormal, texture_coord;
    int faceIndex;
    const material_t* mat;
} intersection_t;

PLATFORM void 
intersect_face(const __restrict__ scene_t* scene, const Ray& ray, intersection_t& hit, int faceIndex);

PLATFORM void 
intersect_crude(const __restrict__ scene_t* scene, const Ray& ray, intersection_t& hit);

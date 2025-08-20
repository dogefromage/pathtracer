#pragma once
#include "headers.h"
#include "mathops.h"
#include "scene.h"
#include <stdint.h>

#define CLEAR_DISTANCE 1e30

inline std::ostream &operator<<(std::ostream &os, const Ray &ray) {
    os << "[ " << ray.o << ", " << ray.r << " ]";
    return os;
}

struct intersection_t {
    bool has_hit = false;
    float distance = CLEAR_DISTANCE;
    Vec3 position, incident_normal, true_normal;
    int faceIndex;
    const material_t *mat;
};

PLATFORM void intersect_face(const __restrict__ scene_t *scene, const Ray &ray,
                             intersection_t &hit, int faceIndex);

PLATFORM void intersect_crude(const __restrict__ scene_t *scene, const Ray &ray,
                              intersection_t &hit);

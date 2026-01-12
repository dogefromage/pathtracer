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
    Vec3 position, incident_normal, true_normal, texcoord0;
    int faceIndex;
    Vec3 color;
    const material_t *mat;
};

__device__ void intersect_face(const Scene &scene, const Ray &ray, intersection_t &hit, int faceIndex);

__device__ void intersect_crude(const Scene &scene, const Ray &ray, intersection_t &hit);

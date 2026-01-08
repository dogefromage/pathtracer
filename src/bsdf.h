#pragma once
#include "intersect.h"
#include "random.h"

typedef struct {
    float prob_i;
    Vec3 omega_i, bsdf;
} bsdf_sample_t;

__device__ Vec3 sphere_sample_uniform(rand_state_t &rstate);

__device__ void sample_bsdf(bsdf_sample_t &out, const Vec3 &v_inv, const intersection_t &hit, rand_state_t &rstate);

__device__ void evaluate_bsdf(bsdf_sample_t &out, const Vec3 &v_inv, const Vec3 &w, const intersection_t &hit,
                              rand_state_t &rstate);

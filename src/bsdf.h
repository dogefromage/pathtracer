#pragma once
#include "random.h"
#include "intersect.h"

typedef struct {
    float prob_i;
    Vec3 omega_i, bsdf;
} bsdf_sample_t;

PLATFORM void
sample_bsdf(bsdf_sample_t& out, const Vec3& v_inv, const intersection_t& hit, rand_state_t& rstate);

PLATFORM void
evaluate_bsdf(bsdf_sample_t& out, const Vec3& v_inv, const Vec3& w, const intersection_t& hit, rand_state_t& rstate);

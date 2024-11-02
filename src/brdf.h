#pragma once
#include "random.h"
#include "intersect.h"

typedef struct {
    float prob_i;
    Vec3 omega_i, brdf;
} brdf_t;

PLATFORM void
sample_brdf(brdf_t& out, const Vec3& v_inv, const intersection_t& hit, rand_state_t& rstate);
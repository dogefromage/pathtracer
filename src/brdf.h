#pragma once
#include "mathc.h"
#include "random.h"
#include "intersect.h"

typedef struct {
    mfloat_t prob_i, omega_i[3], brdf[3];
} brdf_t;

PLATFORM void
sample_brdf(brdf_t* brdf, mfloat_t* v_inv, Intersection* hit, rand_state_t* rstate);
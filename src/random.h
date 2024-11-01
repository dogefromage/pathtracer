#pragma once
#include <curand_kernel.h>

#include <cstdint>

#include "config.h"
#include "mathc.h"

typedef struct {
    curandState curand;
} rand_state_t;

PLATFORM void
random_init(rand_state_t* rstate, uint64_t seed, uint64_t tid);

PLATFORM mfloat_t
random_uniform(rand_state_t* rstate);

PLATFORM mfloat_t
random_normal(rand_state_t* rstate);

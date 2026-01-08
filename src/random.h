#pragma once
#include <curand_kernel.h>

#include <cstdint>

#include "headers.h"

typedef struct {
    curandState curand;
} rand_state_t;

__device__ void random_init(rand_state_t &rstate, uint64_t seed, uint64_t tid);

__device__ float random_uniform(rand_state_t &rstate);

__device__ float random_normal(rand_state_t &rstate);

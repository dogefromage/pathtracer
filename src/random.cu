#include "random.h"

__device__ void random_init(rand_state_t &rstate, uint64_t seed, uint64_t tid) {
    curand_init(seed, tid, 0, &rstate.curand);
}

__device__ float random_uniform(rand_state_t &rstate) {
    return curand_uniform(&rstate.curand);
}

__device__ float random_normal(rand_state_t &rstate) {
    return curand_normal(&rstate.curand);
}

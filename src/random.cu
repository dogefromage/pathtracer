#include "random.h"

PLATFORM void
random_init(rand_state_t& rstate, uint64_t seed, uint64_t tid) {
#ifdef USE_CPU_RENDER
    srand(seed * 123123 + tid * 19230124);  // idk
#else
    curand_init(seed, tid, 0, &rstate.curand);
#endif
}

PLATFORM float
random_uniform(rand_state_t& rstate) {
#ifdef USE_CPU_RENDER
    return (float)rand() / (float)RAND_MAX;
#else
    return curand_uniform(&rstate.curand);
#endif
}

PLATFORM float
random_normal(rand_state_t& rstate) {
#ifdef USE_CPU_RENDER
    return 0.0; /* TODO */
#else
    return curand_normal(&rstate.curand);
#endif
}

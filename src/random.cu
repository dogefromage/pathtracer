#include "random.h"

PLATFORM void
random_init(rand_state_t* rstate, uint64_t seed, uint64_t tid) {
#ifdef USE_CPU_RENDER
    srand(seed * 123123 + tid * 19230124);  // idk
#else
    curand_init(seed, tid, 0, &rstate->curand);
#endif
}

PLATFORM mfloat_t
random_uniform(rand_state_t* rstate) {
#ifdef USE_CPU_RENDER
    return (mfloat_t)rand() / (mfloat_t)RAND_MAX;
#else
    return curand_uniform(&rstate->curand);
#endif
}

PLATFORM mfloat_t
random_normal(rand_state_t* rstate) {
#ifdef USE_CPU_RENDER
    return 0.0; /* TODO */
#else
    return curand_normal(&rstate->curand);
#endif
}

// // shared across threads
// class CurandGenerator {
//     // curandState* states;
//     curandState state;

//    public:
//     inline __device__ void init(uint64_t seed, uint64_t tid) {
//         // uint32_t tid = get_tid();
//         curand_init(seed, tid, 0, &state);
//     }
//     inline __device__ mfloat_t uniform() {
//         // uint32_t tid = get_tid();
//         return curand_uniform(&state);
//     }
//     inline __device__ mfloat_t normal() {
//         // uint32_t tid = get_tid();
//         return curand_normal(&state);
//     }
// };

// class RandGenerator {
//    public:
//     inline __host__ void init(int seed, int tid) {
//         srand(seed * 123123 + tid * 19230124);  // idk
//     }
//     inline __host__ mfloat_t uniform() {
//         return (mfloat_t)rand() / (mfloat_t)RAND_MAX;
//     }
//     inline __host__ mfloat_t normal() {
//         // uint32_t tid = get_tid();
//         // return curand_normal(&states[tid]);
//         return 0.0;
//     }
// };

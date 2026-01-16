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

__device__ Vec3 sphere_sample_uniform(rand_state_t &rstate) {
    Vec3 r;
    do {
        r.x = 2 * random_uniform(rstate) - 1;
        r.y = 2 * random_uniform(rstate) - 1;
        r.z = 2 * random_uniform(rstate) - 1;
    } while (r.dot(r) > 1);

    return r.normalized();
}

// static __device__ Vec3 world_of_local_dir(const Vec3 &normal, const Vec3 &v) {
//     // let normal be new z coordinate
//     Vec3 z = normal;

//     Vec3 e = {1, 0, 0};
//     if (std::abs(e.dot(normal)) >= 0.9) {
//         // choose other e since very close
//         e = {0, 1, 0};
//     }

//     Vec3 x = z.cross(e);
//     x.normalize();
//     Vec3 y = z.cross(x);

//     return x * v.x + y * v.y + z * v.z;
// }

// static __device__ Vec3 hemi_sample_uniform(const Vec3 &normal_unit, rand_state_t &rstate) {
//     Vec3 r = sphere_sample_uniform(rstate);
//     r.z = std::abs(r.z);
//     return world_of_local_dir(normal_unit, r);
// }

// // static __device__ sample_t hemi_sample_cosine(const Vec3 normal_unit, rand_state_t
// &rstate) {
// //     // weird trick: sample a circle and just "push" upwards into hemisphere
// //     // https://www.youtube.com/watch?v=c6NvZ74LAhE

// //     float x, y, d = 0;
// //     do {
// //         x = 2 * random_uniform(rstate) - 1;
// //         y = 2 * random_uniform(rstate) - 1;
// //         d = x * x + y * y;
// //     } while (d >= 1);

// //     float z = SQRT(1 - d);
// //     Vec3 local = Vec3(x, y, z);

// //     float cos_theta = z;
// //     float p = cos_theta / M_PI;

// //     Vec3 world = world_of_local_dir(normal_unit, local);
// //     return {world, p};
// // }

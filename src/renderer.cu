#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "renderer.h"
#include "scene.h"

#ifdef USE_CPU_RENDER
#define RAND_QUALIFIER __host__
#else
#define RAND_QUALIFIER __device__
#endif

// inline __device__ uint32_t
// get_tid() {
//     uint32_t pixel_x = threadIdx.x + blockDim.x * blockIdx.x;
//     uint32_t pixel_y = threadIdx.y + blockDim.y * blockIdx.y;
//     uint32_t outer_width = gridDim.x * blockDim.x;
//     return pixel_y * outer_width + pixel_x;    
// }

// shared across threads
class CurandGenerator {
    // curandState* states;
    curandState state;
public:
    inline __device__ void init(uint64_t seed, uint64_t tid) {
        // uint32_t tid = get_tid();
        curand_init(seed, tid, 0, &state);
    }
    inline __device__ mfloat_t uniform() {
        // uint32_t tid = get_tid();
        return curand_uniform(&state);
    }
    inline __device__ mfloat_t normal() {
        // uint32_t tid = get_tid();
        return curand_normal(&state);
    }
};

class RandGenerator {
public:
    inline __host__ void init(int seed, int tid) {
        srand(seed * 123123 + tid * 19230124); // idk
    }
    inline __host__ mfloat_t uniform() {
        return (mfloat_t)rand()/(mfloat_t)RAND_MAX;
    }
    inline __host__ mfloat_t normal() {
        // uint32_t tid = get_tid();
        // return curand_normal(&states[tid]);
        return 0.0;
    }
};

static __host__ __device__ void
get_camera_ray(Ray* ray, const __restrict__ obj_scene_data* scene, mfloat_t u, mfloat_t v) {
    // camera settings
    mfloat_t sensor_height = 0.2,
             focal_length = 0.25;

    struct vec3 U, V, W,
        *P = &scene->vertex_list[scene->camera.position],
        *T = &scene->vertex_list[scene->camera.target],
        *Up = &scene->vertex_normal_list[scene->camera.updir];

    vec3_subtract(W.v, T->v, P->v);
    vec3_cross(U.v, W.v, Up->v);
    vec3_cross(V.v, U.v, W.v);

    vec3_normalize(U.v, U.v);
    vec3_normalize(V.v, V.v);
    vec3_normalize(W.v, W.v);

    vec3_multiply_f(U.v, U.v, 0.5 * sensor_height);
    vec3_multiply_f(V.v, V.v, 0.5 * sensor_height);
    vec3_multiply_f(W.v, W.v, focal_length);

    // U, V, W build orthogonal basis for camera ray direction
    // D = u*U + v*V + 1*W

    // build change of basis matrix A
    struct mat3 A;
    vec3_assign(&A.m11, U.v);
    vec3_assign(&A.m12, V.v);
    vec3_assign(&A.m13, W.v);

    // camera coordinate vector for pixel
    struct vec3 x = {u, v, 1};

    vec3_assign(ray->o, P->v);
    vec3_multiply_mat3(ray->r, x.v, A.v);
    vec3_normalize(ray->r, ray->r);
}

// offset new ray slightly from triangle in normal dir
#define SAVE_RAY_EPS 1e-6

static __host__ __device__ void 
initialize_safe_ray(Ray* ray, mfloat_t* origin, mfloat_t* dir, mfloat_t* normal) {
    vec3_multiply_f(ray->o, normal, SAVE_RAY_EPS);
    vec3_add(ray->o, ray->o, origin);
    vec3_assign(ray->r, dir);
}

static __host__ __device__ void
get_brdf(const __restrict__ obj_material* mat, mfloat_t* out) {
    mfloat_t distribution = 1 / MPI;  // basic diffuse
    vec3_assign(out, mat->diff);
    vec3_multiply_f(out, out, distribution);
}

static __host__ __device__ void
intersect(const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
          const Ray* ray, Intersection* hit) {
    // intersectionCalls++;
    hit->has_hit = 0;
    hit->distance = CLEAR_DISTANCE;
#if 0
    intersect_crude(scene, ray, hit);
#else
    bvh_intersect_iterative(bvh, scene, ray, hit);
    // bvh_intersect(bvh, BVH_ROOT_NODE, scene, ray, hit);
#endif
}

#define RR_PROB_MAX 0.99

// static __host__ __device__ void
// integrate_Li(mfloat_t* light,
//              const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
//              const Ray* v_inv, int depth, mfloat_t* throughput, curandState* rand_state) {
//     Intersection hit;
//     intersect(bvh, scene, v_inv, &hit);
//     if (!hit.has_hit) {
//         return;
//     }

//     vec3_add(light, light, hit.mat->emit);

//     mfloat_t rr_prob = MFMIN(vec3_max_component(throughput), RR_PROB_MAX);
//     if (curand_uniform(rand_state) >= rr_prob) {
//         return;  // ray dies
//     }

//     mfloat_t omega_i[VEC3_SIZE];
//     mfloat_t prob_i = 1 / (2 * MPI);  // because even distribution
//     hemi_sample_uniform(omega_i, hit.normal, rand_state);

//     Ray bounce_ray;
//     initialize_safe_ray(&bounce_ray, hit.position, omega_i, hit.normal);

//     mfloat_t brdf[VEC3_SIZE], next_throughput[VEC3_SIZE], bounce_light[VEC3_SIZE];

//     mfloat_t cosTheta = vec3_dot(hit.normal, omega_i);
//     get_brdf(hit.mat, brdf);

//     vec3_multiply(next_throughput, throughput, brdf);
//     vec3_multiply_f(next_throughput, next_throughput, cosTheta / (prob_i * rr_prob));

//     // recurse
//     vec3_zero(bounce_light);
//     // rayBounces++;
//     integrate_Li(bounce_light, bvh, scene, &bounce_ray, depth + 1, next_throughput, rand_state);

//     vec3_multiply(bounce_light, bounce_light, brdf);
//     vec3_multiply_f(bounce_light, bounce_light, cosTheta / (prob_i * rr_prob));

//     vec3_add(light, light, bounce_light);
// }

template<class R> 
RAND_QUALIFIER void 
sphere_sample_uniform(mfloat_t* out, R* rand) {
    mfloat_t x, y, z, d;
    do {
        x = 2 * rand->uniform() - 1;
        y = 2 * rand->uniform() - 1;
        z = 2 * rand->uniform() - 1;
        d = x * x + y * y + z * z;
    } while (d > 1);
    mfloat_t inv_l = 1.0 / MSQRT(d);
    out[0] = x * inv_l;
    out[1] = y * inv_l;
    out[2] = z * inv_l;
}

template<class R> 
RAND_QUALIFIER void 
hemi_sample_uniform(mfloat_t* out, mfloat_t* normal_unit, R* rand) {
    sphere_sample_uniform(out, rand);
    double dotp = vec3_dot(out, normal_unit);
    if (dotp < 0) {
        // mirror out by normal
        mfloat_t corr[VEC3_SIZE];
        vec3_multiply_f(corr, normal_unit, -2 * dotp);
        vec3_add(out, out, corr);
    }
}

#define MAX_BOUNCES 20

typedef struct {
    Intersection hit;
    mfloat_t brdf[3], cos_theta, prob_i, rr_prob;
} bounce_t;

template<class R> 
RAND_QUALIFIER void
integrate_Li_iterative(mfloat_t* light,
                       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
                       Ray ray, R* rand) {
    mfloat_t throughput[3];
    vec3_one(throughput);

    bounce_t bounces[MAX_BOUNCES];
    int depth = 0;

    while (depth < MAX_BOUNCES) {
        bounce_t* bounce = &bounces[depth];

        intersect(bvh, scene, &ray, &bounce->hit);
        if (!bounce->hit.has_hit) {
            break;
        }

        bounce->rr_prob = MFMIN(vec3_max_component(throughput), RR_PROB_MAX);
        if (rand->uniform() >= bounce->rr_prob) {
            break;  // ray dies
        }

        mfloat_t omega_i[VEC3_SIZE];
        bounce->prob_i = 1 / (2 * MPI);  // because even distribution
        hemi_sample_uniform(omega_i, bounce->hit.normal, rand);

        initialize_safe_ray(&ray, bounce->hit.position, omega_i, bounce->hit.normal);

        bounce->cos_theta = vec3_dot(bounce->hit.normal, omega_i);
        get_brdf(bounce->hit.mat, bounce->brdf);

        vec3_multiply(throughput, throughput, bounce->brdf);
        vec3_multiply_f(throughput, throughput, bounce->cos_theta / (bounce->prob_i * bounce->rr_prob));

        depth++;
    }

    while (depth >= MAX_BOUNCES) {
        depth--;
    }

    vec3_zero(light);
    bool is_bounce_ray = false;

    while (depth >= 0) {
        bounce_t* bounce = &bounces[depth];

        if (bounce->hit.has_hit) {
            if (is_bounce_ray) {
                vec3_multiply(light, light, bounce->brdf);
                vec3_multiply_f(light, light, bounce->cos_theta / (bounce->prob_i * bounce->rr_prob));
            }

            vec3_add(light, light, bounce->hit.mat->emit);
        }

        depth--;
        is_bounce_ray = true;
    }
}

#ifdef USE_CPU_RENDER

__host__ void
render_host(struct vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       int pixel_x, int pixel_y,
       render_settings_t settings, int previous_samples) {

    uint64_t tid = pixel_y * settings.width + pixel_x;
    RandGenerator rand;
    rand.init(settings.seed, tid);

    mfloat_t current_light[3], total_light[3];
    vec3_zero(total_light);

    for (int i = 0; i < settings.samples; i++) {
        float sensor_variance = 0.33;
        mfloat_t sensor_x = (mfloat_t)pixel_x + sensor_variance * rand.normal();
        mfloat_t sensor_y = (mfloat_t)pixel_y + sensor_variance * rand.normal();

        mfloat_t u = (2 * sensor_x - settings.width) / (mfloat_t)settings.height;
        mfloat_t v = (2 * sensor_y - settings.height) / (mfloat_t)settings.height;

        Ray camera_ray;
        get_camera_ray(&camera_ray, scene, u, v);

        integrate_Li_iterative(current_light, bvh, scene, camera_ray, &rand);
        vec3_add(total_light, total_light, current_light);
    }

    vec3_divide_f(total_light, total_light, (mfloat_t)settings.samples);

    mfloat_t* pixel = img[pixel_y * settings.width + pixel_x].v;

    mfloat_t total_samples = (mfloat_t)(settings.samples + previous_samples);
    vec3_multiply_f(pixel, pixel, previous_samples / total_samples);
    vec3_multiply_f(total_light, total_light, settings.samples / total_samples);
    vec3_add(pixel, pixel, total_light);
}

#else

__global__ void
render_kernel(struct vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       render_settings_t settings, int previous_samples) {
        
    int pixel_x = threadIdx.x + blockDim.x * blockIdx.x;
    int pixel_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (pixel_x >= settings.width || pixel_y >= settings.height) {
        return;  // out of image
    }

    uint64_t curand_tid = pixel_y * settings.width + pixel_x;

    CurandGenerator rand;
    rand.init(settings.seed, curand_tid);

    mfloat_t current_light[3], total_light[3];
    vec3_zero(total_light);

    for (int i = 0; i < settings.samples; i++) {
        float sensor_variance = 0.33;
        mfloat_t sensor_x = (mfloat_t)pixel_x + sensor_variance * rand.normal();
        mfloat_t sensor_y = (mfloat_t)pixel_y + sensor_variance * rand.normal();

        mfloat_t u = (2 * sensor_x - settings.width) / (mfloat_t)settings.height;
        mfloat_t v = (2 * sensor_y - settings.height) / (mfloat_t)settings.height;

        Ray camera_ray;
        get_camera_ray(&camera_ray, scene, u, v);

        integrate_Li_iterative(current_light, bvh, scene, camera_ray, &rand);
        vec3_add(total_light, total_light, current_light);
    }

    vec3_divide_f(total_light, total_light, (mfloat_t)settings.samples);

    mfloat_t* pixel = img[pixel_y * settings.width + pixel_x].v;

    mfloat_t total_samples = (mfloat_t)(settings.samples + previous_samples);
    vec3_multiply_f(pixel, pixel, previous_samples / total_samples);
    vec3_multiply_f(total_light, total_light, settings.samples / total_samples);
    vec3_add(pixel, pixel, total_light);
}

#endif
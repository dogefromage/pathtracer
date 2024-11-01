#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "renderer.h"
#include "scene.h"
#include "random.h"
#include "brdf.h"
#include "config.h"

static PLATFORM void
get_camera_ray(Ray* ray, const __restrict__ obj_scene_data* scene, 
    mfloat_t u, mfloat_t v, render_settings_t* settings) {
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

    vec3_multiply_f(U.v, U.v, 0.5 * settings->sensor_height);
    vec3_multiply_f(V.v, V.v, 0.5 * settings->sensor_height);
    vec3_multiply_f(W.v, W.v, settings->focal_length);

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

static PLATFORM void 
initialize_safe_ray(Ray* ray, mfloat_t* origin, mfloat_t* dir, mfloat_t* normal) {
    vec3_multiply_f(ray->o, normal, SAVE_RAY_EPS);
    vec3_add(ray->o, ray->o, origin);
    vec3_assign(ray->r, dir);
}

static PLATFORM void
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

#define RR_PROB_MAX 0.95

static PLATFORM void
integrate_Li_iterative(mfloat_t* light,
                       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
                       Ray ray, rand_state_t* rstate) {

    vec3_zero(light);

    mfloat_t throughput[3];
    vec3_one(throughput);

    for (int depth = 0;; depth++) {

        Intersection hit;
        intersect(bvh, scene, &ray, &hit);
        if (!hit.has_hit) {
            return;
        }

        mfloat_t Le[3];
        vec3_multiply(Le, hit.mat->emit, throughput);
        vec3_add(light, light, Le);

        mfloat_t rr_prob = MFMIN(vec3_max_component(throughput), RR_PROB_MAX);
        if (random_uniform(rstate) >= rr_prob) {
            break;  // ray dies
        }

        brdf_t brdf;
        sample_brdf(&brdf, ray.r, &hit, rstate);

        mfloat_t cosTheta = vec3_dot(hit.normal, brdf.omega_i);

        // set next ray
        initialize_safe_ray(&ray, hit.position, brdf.omega_i, hit.normal);

        vec3_multiply(throughput, throughput, brdf.brdf);
        vec3_multiply_f(throughput, throughput, cosTheta / (brdf.prob_i * rr_prob));
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
        get_camera_ray(&camera_ray, scene, u, v, &settings);

        integrate_Li_iterative(current_light, bvh, scene, camera_ray, &rand);
        vec3_add(total_light, total_light, current_light);
    }

    vec3_divide_f(total_light, total_light, (mfloat_t)settings.samples_per_round);

    mfloat_t* pixel = img[pixel_y * settings.width + pixel_x].v;

    mfloat_t total_samples = settings.samples_per_round + previous_samples;
    vec3_multiply_f(pixel, pixel, previous_samples / (mfloat_t)total_samples);
    vec3_multiply_f(total_light, total_light, settings.samples_per_round / (mfloat_t)total_samples);
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

    rand_state_t rstate;
    random_init(&rstate, settings.seed, curand_tid);

    mfloat_t current_light[3], total_light[3];
    vec3_zero(total_light);

    for (int i = 0; i < settings.samples_per_round; i++) {
        float sensor_variance = 0.33;
        mfloat_t sensor_x = (mfloat_t)pixel_x + sensor_variance * random_normal(&rstate);
        mfloat_t sensor_y = (mfloat_t)pixel_y + sensor_variance * random_normal(&rstate);

        mfloat_t u = (2 * sensor_x - settings.width) / (mfloat_t)settings.height;
        mfloat_t v = (2 * sensor_y - settings.height) / (mfloat_t)settings.height;

        Ray camera_ray;
        get_camera_ray(&camera_ray, scene, u, v, &settings);

        // integrate_Li_iterative(current_light, bvh, scene, camera_ray, &rand);
        integrate_Li_iterative(current_light, bvh, scene, camera_ray, &rstate);
        vec3_add(total_light, total_light, current_light);
    }

    vec3_divide_f(total_light, total_light, (mfloat_t)settings.samples_per_round);

    mfloat_t* pixel = img[pixel_y * settings.width + pixel_x].v;

    int total_samples = settings.samples_per_round + previous_samples;
    vec3_multiply_f(pixel, pixel, previous_samples / (mfloat_t)total_samples);
    vec3_multiply_f(total_light, total_light, settings.samples_per_round / (mfloat_t)total_samples);
    vec3_add(pixel, pixel, total_light);
}

#endif
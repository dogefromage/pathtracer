#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "renderer.h"
#include "scene.h"
#include "random.h"
#include "brdf.h"
#include "config.h"
#include "mathops.h"
#include <iostream>

static PLATFORM void
get_camera_ray(Ray& ray, const __restrict__ obj_scene_data* scene, 
    float u, float v, const render_settings_t& settings) {

    Vec3 P = scene->vertex_list[scene->camera.position];
    Vec3 T = scene->vertex_list[scene->camera.target];
    Vec3 Up = scene->vertex_normal_list[scene->camera.updir];

    Vec3 W = T - P;
    Vec3 U = W.cross(Up);
    Vec3 V = U.cross(W);

    U = U.normalized() * 0.5 * settings.sensor_height;
    V = V.normalized() * 0.5 * settings.sensor_height;
    W = W.normalized() * settings.focal_length;

    // U, V, W build orthogonal basis for camera ray direction
    // D = u*U + v*V + 1*W
    // build change of basis matrix A
    Mat3 A(
        U.x, V.x, W.x,
        U.y, V.y, W.y,
        U.z, V.z, W.z
    );

    // camera coordinate vector for pixel
    Vec3 x = {u, v, 1};

    ray.o = P;
    ray.r = A * x;
    ray.r.normalize();
}

// offset new ray slightly from triangle in normal dir
#define SAVE_RAY_EPS 1e-6

static PLATFORM void 
initialize_safe_ray(Ray& ray, const Vec3& origin, const Vec3& dir, const Vec3& normal) {
    ray.o = origin + SAVE_RAY_EPS * normal;
    ray.r = dir;
}

static PLATFORM void
intersect(const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
          const Ray& ray, intersection_t& hit) {
    // intersectionCalls++;
    hit.has_hit = 0;
    hit.distance = CLEAR_DISTANCE;
#ifdef USE_INTERSECT_CRUDE
    intersect_crude(scene, ray, hit);
#else
    bvh_intersect_iterative(bvh, scene, ray, hit);
    // bvh_intersect(bvh, BVH_ROOT_NODE, scene, ray, hit);
#endif
}

#define RR_PROB_MAX 0.95

static PLATFORM Vec3
integrate_Li_iterative(const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
                       Ray ray, rand_state_t& rstate) {

    Vec3 light = { 0, 0, 0 };
    Vec3 throughput = { 1, 1, 1 };

    for (int depth = 0;; depth++) {

        intersection_t hit;
        intersect(bvh, scene, ray, hit);
        if (!hit.has_hit) {
            break;
        }

        // direct light, attenuate using throughput which includes indirect lighting penalty
        Vec3 Le = hit.mat->emit * throughput;
        light += Le;


        float rr_prob = fminf(throughput.maxComponent(), RR_PROB_MAX);
        if (random_uniform(rstate) >= rr_prob) {
            break;  // ray dies
        }

        brdf_t brdf;
        sample_brdf(brdf, ray.r, hit, rstate);

        float cosTheta = hit.normal.dot(brdf.omega_i);

        // std::cout << "Ray before: " << ray << std::endl;

        // set next ray
        initialize_safe_ray(ray, hit.position, brdf.omega_i, hit.normal);

        // std::cout << "Ray after:  " << ray << std::endl;

        // find next throughput
        throughput *= brdf.brdf * (cosTheta / (brdf.prob_i * rr_prob));
    }
    
    return light;
}

#ifdef USE_CPU_RENDER

__host__ void
render_host(Vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       int pixel_x, int pixel_y,
       render_settings_t settings, int previous_samples) {

    uint64_t tid = pixel_y * settings.width + pixel_x;

    rand_state_t rstate;
    random_init(rstate, settings.seed, tid);

    Vec3 total_light = { 0, 0, 0 };

    for (size_t i = 0; i < settings.samples; i++) {
        // float sensor_variance = 0.33;
        float sensor_x = (float)pixel_x; /* TODO sensor variance */
        float sensor_y = (float)pixel_y; /* TODO sensor variance */

        float u = (2 * sensor_x - settings.width) / (float)settings.height;
        float v = (2 * sensor_y - settings.height) / (float)settings.height;

        Ray camera_ray;
        get_camera_ray(camera_ray, scene, u, v, settings);

        Vec3 current_light = integrate_Li_iterative(bvh, scene, camera_ray, rstate);
        total_light += current_light;
    }

    total_light /= (float)settings.samples_per_round;

    int total_samples = settings.samples_per_round + previous_samples;
    Vec3& pixel = img[pixel_y * settings.width + pixel_x];

    pixel = pixel * (previous_samples / (float)total_samples)
        + total_light * (settings.samples_per_round / (float)total_samples);
}

#else

__global__ void
render_kernel(Vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       render_settings_t settings, int previous_samples) {
        
    int pixel_x = threadIdx.x + blockDim.x * blockIdx.x;
    int pixel_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (pixel_x >= settings.width || pixel_y >= settings.height) {
        return;  // out of image
    }

    uint64_t curand_tid = pixel_y * settings.width + pixel_x;

    rand_state_t rstate;
    random_init(rstate, settings.seed, curand_tid);

    Vec3 total_light = { 0, 0, 0 };

    for (int i = 0; i < settings.samples_per_round; i++) {
        float sensor_variance = 0.33;
        float sensor_x = (float)pixel_x + sensor_variance * random_normal(rstate);
        float sensor_y = (float)pixel_y + sensor_variance * random_normal(rstate);

        float u = (2 * sensor_x - settings.width) / (float)settings.height;
        float v = (2 * sensor_y - settings.height) / (float)settings.height;

        Ray camera_ray;
        get_camera_ray(camera_ray, scene, u, v, settings);

        Vec3 current_light = integrate_Li_iterative(bvh, scene, camera_ray, rstate);
        total_light += current_light;
    }

    total_light /= (float)settings.samples_per_round;

    int total_samples = settings.samples_per_round + previous_samples;
    Vec3 last_pixel = img[pixel_y * settings.width + pixel_x];

    Vec3 next_pixel = last_pixel * (previous_samples / (float)total_samples)
        + total_light * (settings.samples_per_round / (float)total_samples);

    img[pixel_y * settings.width + pixel_x] = next_pixel;
}

#endif
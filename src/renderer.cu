#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <iostream>

#include "bsdf.h"
#include "config.h"
#include "lst.h"
#include "mathops.h"
#include "random.h"
#include "renderer.h"
#include "scene.h"

static PLATFORM void
get_camera_ray(Ray& ray, const __restrict__ obj_scene_data* scene,
               float u, float v, const settings_t& settings) {
    Vec3 P = scene->vertex_list[scene->camera.position];
    Vec3 T = scene->vertex_list[scene->camera.target];
    Vec3 Up = scene->vertex_normal_list[scene->camera.updir];

    Vec3 W = T - P;
    Vec3 U = W.cross(Up);
    Vec3 V = U.cross(W);

    U = U.normalized() * 0.5 * settings.camera.sensor_height;
    V = V.normalized() * 0.5 * settings.camera.sensor_height;
    W = W.normalized() * settings.camera.focal_length;

    // U, V, W build orthogonal basis for camera ray direction
    // D = u*U + v*V + 1*W
    // build change of basis matrix A
    Mat3 A(
        U.x, V.x, W.x,
        U.y, V.y, W.y,
        U.z, V.z, W.z);

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
    bool transmit = dir.dot(normal) < 0;
    ray.o = origin + SAVE_RAY_EPS * (transmit ? -normal : normal);
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

// static PLATFORM Vec3 L_skybox(const __restrict__ bvh_t* bvh,
//                               const __restrict__ obj_scene_data* scene,
//                               const __restrict__ lst_t* lst,
//                               Ray ray, rand_state_t& rstate, settings_t& settings) {
//     auto& sun = settings.world.sun;
//     float light_angle = acosf(ray.r.dot(sun.direction));

//     float a1 = 0.5 * sun.angular_diameter;
//     float a2 = a1 + sun.penumbra;

//     float light_amt = 0;

//     if (light_angle < a1) {
//         light_amt = 1;
//     } else if (light_angle < a2) {
//         light_amt = 1 - (light_angle - a1) / (a2 - a1);
//     }

//     return light_amt * sun.light + (1.0 - light_amt) * settings.world.clear_light;
// }

// static PLATFORM Vec3
// integrate_Ld(const __restrict__ bvh_t* bvh,
//              const __restrict__ obj_scene_data* scene,
//              const __restrict__ lst_t* lst,
//              Ray ray, rand_state_t& rstate, settings_t& settings) {
//     light_sample_t ls;
//     lst_sample(ls, lst, scene);
    
//     // calculate visibility

//     // find light blabla...

//     // https://www.youtube.com/watch?v=FU1dbi827LY&t=1189s
// }

#define RR_PROB_MAX 0.99

static PLATFORM Vec3
integrate_Li_iterative(const __restrict__ bvh_t* bvh,
                       const __restrict__ obj_scene_data* scene,
                       const __restrict__ lst_t* lst,
                       Ray ray, rand_state_t& rstate, settings_t& settings) {
    Vec3 light = {0, 0, 0};
    Vec3 throughput = {1, 1, 1};

    for (int depth = 0;; depth++) {
        intersection_t hit;
        intersect(bvh, scene, ray, hit);
        if (!hit.has_hit) {
            // // skybox light
            // Vec3 Lsky = throughput * L_skybox(bvh, scene, ray, rstate, settings);
            // light += Lsky;
            break;
        }

        // direct light, attenuate using throughput which includes indirect lighting penalty
        Vec3 Le = hit.mat->emit * throughput;
        light += Le;

        float rr_prob = fminf(throughput.maxComponent(), RR_PROB_MAX);
        if (random_uniform(rstate) >= rr_prob) {
            break;  // ray dies
        }

        bsdf_t bsdf;
        sample_bsdf(bsdf, ray.r, hit, rstate);

        float cosTheta = std::abs(hit.normal.dot(bsdf.omega_i));

        // set next ray
        initialize_safe_ray(ray, hit.position, bsdf.omega_i, hit.normal);

        // find next throughput
        throughput *= bsdf.bsdf * (cosTheta / (bsdf.prob_i * rr_prob));
    }

    return light;
}

#ifdef USE_CPU_RENDER

__host__ void
render_host(Vec3* img,
            const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
            int pixel_x, int pixel_y,
            settings_t settings, int previous_samples) {
    uint64_t tid = pixel_y * settings.width + pixel_x;

    rand_state_t rstate;
    random_init(rstate, settings.seed, tid);

    Vec3 total_light = {0, 0, 0};

    for (int i = 0; i < settings.samples; i++) {
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

    pixel = pixel * (previous_samples / (float)total_samples) + total_light * (settings.samples_per_round / (float)total_samples);
}

#else

__global__ void
render_kernel(Vec3* img,
              const __restrict__ bvh_t* bvh,
              const __restrict__ obj_scene_data* scene,
              const __restrict__ lst_t* lst,
              settings_t settings, int previous_samples) {
    int pixel_x = threadIdx.x + blockDim.x * blockIdx.x;
    int pixel_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (pixel_x >= settings.output.width || pixel_y >= settings.output.height) {
        return;  // out of image
    }

    settings.world.sun.direction = {1, 1, 1};
    settings.world.sun.direction.normalize();
    settings.world.sun.light = 3 * Vec3(1, 1, 1);
    settings.world.sun.angular_diameter = 1;
    settings.world.sun.penumbra = 0.1;
    // settings.world.sun.angular_diameter = 0.00872665;
    // settings.world.sun.penumbra = 0.001;
    settings.world.clear_light = 0.15 * Vec3(0.8, 0.8, 0.8);

    uint64_t curand_tid = pixel_y * settings.output.width + pixel_x;

    rand_state_t rstate;
    random_init(rstate, settings.sampling.seed, curand_tid);

    Vec3 total_light = {0, 0, 0};

    for (int i = 0; i < settings.sampling.samples_per_round; i++) {
        float sensor_variance = 0.33;
        float sensor_x = (float)pixel_x + sensor_variance * random_normal(rstate);
        float sensor_y = (float)pixel_y + sensor_variance * random_normal(rstate);

        float u = (2 * sensor_x - settings.output.width) / (float)settings.output.height;
        float v = (2 * sensor_y - settings.output.height) / (float)settings.output.height;

        Ray camera_ray;
        get_camera_ray(camera_ray, scene, u, v, settings);

        Vec3 current_light = integrate_Li_iterative(bvh, scene, lst, camera_ray, rstate, settings);
        total_light += current_light;
    }

    total_light /= (float)settings.sampling.samples_per_round;

    int total_samples = settings.sampling.samples_per_round + previous_samples;
    Vec3 last_pixel = img[pixel_y * settings.output.width + pixel_x];

    Vec3 next_pixel = last_pixel * (previous_samples / (float)total_samples) +
                      total_light * (settings.sampling.samples_per_round / (float)total_samples);

    img[pixel_y * settings.output.width + pixel_x] = next_pixel;
}

#endif
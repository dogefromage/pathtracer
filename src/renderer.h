#pragma once
#include <curand_kernel.h>

#include "bvh.h"
#include "mathc.h"
#include "scene.h"

// void hemi_sample_uniform(mfloat_t* out, mfloat_t* normal_unit);
// void sphere_sample_uniform(mfloat_t* out);
// void get_camera_ray(Ray* ray, obj_scene_data* scene, mfloat_t u, mfloat_t v);

__global__ void
render(struct vec3* img,
       const __restrict__ BVH* bvh, const __restrict__ obj_scene_data* scene,
       int width, int height, int seed, int samples, int previous_samples);

// void renderer_init();
// void renderer_print_info(uint64_t numPixels);
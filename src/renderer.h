#pragma once
#include <curand_kernel.h>

#include "bvh.h"
#include "mathc.h"
#include "scene.h"

typedef struct {
    size_t width, height,
        seed,
        samples, samples_per_round;
        
    mfloat_t sensor_height, focal_length;
} render_settings_t;

#ifdef USE_CPU_RENDER

__host__ void
render_host(struct vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       int pixel_x, int pixel_y,
       render_settings_t settings, int previous_samples);

#else

__global__ void
render_kernel(struct vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       render_settings_t settings, int previous_samples);

#endif
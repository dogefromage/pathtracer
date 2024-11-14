#pragma once
#include <curand_kernel.h>

#include "bvh.h"
#include "scene.h"
#include "settings.h"

#ifdef USE_CPU_RENDER

__host__ void
render_host(Vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       int pixel_x, int pixel_y,
       settings_t settings, int previous_samples);

#else

__global__ void
render_kernel(Vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ obj_scene_data* scene,
       settings_t settings, int previous_samples);

#endif
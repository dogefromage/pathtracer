#pragma once
#include <curand_kernel.h>

#include "lst.h"
#include "bvh.h"
#include "scene.h"
#include "settings.h"

#ifdef USE_CPU_RENDER

__host__ void
render_host(Vec3* img,
       const __restrict__ bvh_t* bvh, const __restrict__ scene_t* scene,
       int pixel_x, int pixel_y,
       settings_t settings, int previous_samples);

#else

__global__ void
render_kernel(Vec3* img,
       const __restrict__ bvh_t* bvh, 
       const __restrict__ scene_t* scene,
       const __restrict__ lst_t* lst,
       settings_t settings, int previousSamples, int currentSamples);

#endif
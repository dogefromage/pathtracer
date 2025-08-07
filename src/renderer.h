#pragma once
#include <curand_kernel.h>

#include "bvh.h"
#include "config.h"
#include "lst.h"
#include "scene.h"

#ifdef USE_CPU_RENDER

__host__ void render_host(Vec3 *img, const __restrict__ bvh_t *bvh,
                          const __restrict__ scene_t *scene, int pixel_x, int pixel_y,
                          config_t cfg, int previous_samples);

#else

__global__ void render_kernel(Vec3 *img, const __restrict__ bvh_t *bvh,
                              const __restrict__ scene_t *scene, const __restrict__ lst_t *lst,
                              config_t cfg, int previousSamples, int currentSamples);

#endif

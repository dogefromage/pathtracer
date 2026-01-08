#pragma once
#include <curand_kernel.h>

#include "bvh.h"
#include "config.h"
#include "lst.h"
#include "scene.h"

__global__ void render_kernel(Vec3 *img, const bvh_t *bvh, const Scene scene, const lst_t *lst, config_t cfg,
                              int previousSamples, int currentSamples);

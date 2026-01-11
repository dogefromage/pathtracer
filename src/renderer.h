#pragma once
#include <curand_kernel.h>

#include "bvh.h"
#include "config.h"
#include "lst.h"
#include "scene.h"

__global__ void render_kernel(Vec3 *img, const BVH bvh, const Scene scene, const LST lst, config_t cfg, int previous_samples,
                              int current_samples);

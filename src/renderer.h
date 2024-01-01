#pragma once
#include "bvh.h"
#include "intersect.h"
#include "mathc.h"
#include "obj_parser.h"

// void hemi_sample_uniform(mfloat_t* out, mfloat_t* normal_unit);
// void sphere_sample_uniform(mfloat_t* out);
// void get_camera_ray(Ray* ray, obj_scene_data* scene, mfloat_t u, mfloat_t v);

void render(mfloat_t* pixel, BVH* bvh, obj_scene_data* scene, mfloat_t u, mfloat_t v, int samples);

void renderer_init();
void renderer_print_info(uint64_t numPixels);
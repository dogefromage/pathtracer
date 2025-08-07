#pragma once
#include "bvh.h"
#include "renderer.h"
#include "scene.h"

#ifdef USE_CPU_RENDER

int render_image_host(obj_scene_data* h_scene, bvh_t* h_bvh, lst_t* h_lst, Vec3* h_img,
                      size_t img_size, settings_t settings, char* output_path);

#else

void render_image_device(scene_t* h_scene, bvh_t* h_bvh, lst_t* h_lst, Vec3* h_img,
                         size_t img_size, config_t* cfg);

#endif

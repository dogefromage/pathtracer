#pragma once
#include "bvh.h"
#include "scene.h"
#include "renderer.h"

#ifdef USE_CPU_RENDER

int render_image_host(obj_scene_data* h_scene, bvh_t* h_bvh, lst_t* h_lst, Vec3* h_img,
                      size_t img_size, settings_t settings);

#else
                      
int render_image_device(obj_scene_data* h_scene, bvh_t* h_bvh, lst_t* h_lst, Vec3* h_img,
                        size_t img_size, settings_t settings);

#endif

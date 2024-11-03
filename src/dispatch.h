#pragma once
#include "bvh.h"
#include "scene.h"
#include "renderer.h"

int render_image_host(obj_scene_data* h_scene, bvh_t* h_bvh, Vec3* h_img,
                      size_t img_size, render_settings_t settings);
                      
int render_image_device(obj_scene_data* h_scene, bvh_t* h_bvh, Vec3* h_img,
                        size_t img_size, render_settings_t settings);
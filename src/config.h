#pragma once
#include <cstdint>

#include "config.h"
#include "mathops.h"

typedef char path_t[256];

struct config_t {
    int resolution_x, resolution_y;
    float output_exposure;
    int samples, seed, samples_every_update;

    Vec3 world_clear_color;
    path_t world_clear_color_texture{}; // default all zeros

    int log_level, log_stdout;

    path_t path_gltf, dir_output, path_render;

    Vec3 default_camera_position, default_camera_target, default_camera_updir;
    float default_camera_yfov;
};

int load_config(config_t *cfg, int argc, char *argv[]);

#pragma once
#include <cstdint>

#include "config.h"
#include "mathops.h"

typedef char path_t[256];

struct config_t {
    int resolution_x, resolution_y;
    int samples, seed, samples_every_update;

    Vec3 world_clear_color;

    int log_level, log_stdout;

    path_t path_gltf, dir_output, path_render;
};

int load_config(config_t* cfg, int argc, char* argv[]);

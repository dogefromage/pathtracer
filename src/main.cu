#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cassert>

#include "config.h"
#include "dispatch.h"
#include "headers.h"
#include "logger.h"
#include "lst.h"

int main(int argc, char *argv[]) {

    // printf("argc = %d\n", argc);
    // for (int i = 0; argv[i]; i++) {
    //     printf("argv[%d] = %s\n", i, argv[i]);
    // }

    config_t cfg;

    if (load_config(&cfg, argc, argv)) {
        return 1;
    }

    path_t log_file;
    snprintf(log_file, sizeof(log_file), "%s/log.txt", cfg.dir_output);
    snprintf(cfg.path_render, sizeof(cfg.path_render), "%s/render.png", cfg.dir_output);

    if (log_init(cfg.log_level, log_file, cfg.log_stdout)) {
        return 1;
    }

    scene_t h_scene;
    scene_parse_gltf(h_scene, cfg.path_gltf);

    // bounding volume hierarchy
    bvh_t h_bvh;
    bvh_build(h_bvh, h_scene);

    // light source tree
    lst_t h_lst;
    lst_build(h_lst, h_scene);

    size_t img_size = sizeof(Vec3) * cfg.resolution_x * cfg.resolution_y;
    Vec3 *h_img = (Vec3 *)malloc(img_size);
    assert(h_img);

#ifdef USE_CPU_RENDER
    render_image_host(&h_scene, &h_bvh, &h_lst, h_img, img_size, settings, output_path);
#else
    render_image_device(&h_scene, &h_bvh, &h_lst, h_img, img_size, &cfg);
#endif

    free(h_img);
    h_img = NULL;
    bvh_free_host(h_bvh);
    lst_free_host(h_lst);
    scene_delete_host(h_scene);

    log_close();

    return 0;
}

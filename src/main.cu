#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cassert>

#include "dispatch.h"
#include "config.h"
#include "settings.h"
#include "lst.h"

bool doVerbosePrinting = false;

void print_help(int argc, char* argv[]) {
    fprintf(stderr, "Usage: %s [options] <path_to_obj>\n", argv[0]);
    fprintf(stderr, "Expects an .obj file with right handed coordinate system.\n");
    fprintf(stderr, "  -c <pathtracer.yaml>  Pathtracer render settings file.\n");
}

int main(int argc, char* argv[]) {
    
    settings_t settings;

    char output_path[PATH_MAX];
    strcpy(output_path, "output.png");

    int opt;
    while ((opt = getopt(argc, argv, "o:c:v")) != -1) {
        switch (opt) {
            case 'o':
                strcpy(output_path, optarg);
                break;
            case 'c':
                settings_parse_yaml(settings, optarg);
                break;
            case 'v':
                doVerbosePrinting = true;
                printf("Verbose printing enabled\n");
                break;
            default:
                print_help(argc, argv);
                exit(EXIT_FAILURE);
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Expected a path to an .obj file.\n");
        print_help(argc, argv);
        exit(EXIT_FAILURE);
    }

    settings_print(settings);

    char* obj_file_path = argv[optind];

    scene_t h_scene;
    scene_parse_gltf(h_scene, obj_file_path);

    // bounding volume hierarchy
    bvh_t h_bvh;
    bvh_build(h_bvh, h_scene);
    
    // light source tree
    lst_t h_lst;
    lst_build(h_lst, h_scene);

    size_t img_size = sizeof(Vec3) * settings.output.width * settings.output.height;

    Vec3* h_img = (Vec3*)malloc(img_size);
    assert(h_img);

#ifdef USE_CPU_RENDER
    render_image_host(&h_scene, &h_bvh, &h_lst, h_img, img_size, settings, output_path);
#else
    render_image_device(&h_scene, &h_bvh, &h_lst, h_img, img_size, settings, output_path);
#endif

    free(h_img);
    h_img = NULL;
    bvh_free_host(h_bvh);
    lst_free_host(h_lst);
    scene_delete_host(h_scene);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cassert>

#include "dispatch.h"
#include "config.h"

bool doVerbosePrinting = false;

void print_help(int argc, char* argv[], const render_settings_t& settings) {
    fprintf(stderr, "Usage: %s [options] <path_to_obj>\n", argv[0]);
    fprintf(stderr, "Expects an .obj file with right handed coordinate system.\n");
    fprintf(stderr, "  -w <width>           Sets width\n");
    fprintf(stderr, "  -h <height>          Sets height\n");
    fprintf(stderr, "  -s <size>            Sets both width and height\n");
    fprintf(stderr, "  -S <samples>         Sets image samples (default %d)\n", settings.samples);
    fprintf(stderr, "  -r <samples/call>    Sets image samples per kernel call (default %d)\n", settings.samples_per_round);
    fprintf(stderr, "  -d <seed>            Sets Seed (default %d)\n", settings.seed);
    fprintf(stderr, "  -f <focal_length>    Focal length (default %.2f)\n", settings.focal_length);
    fprintf(stderr, "  -c <sensor_height>   Sensor height (default %.2f)\n", settings.sensor_height);
    fprintf(stderr, "  -v                   Do verbose printing (default false)\n");
}

int main(int argc, char* argv[]) {
    
    render_settings_t settings;
    settings.width = -1;
    settings.height = -1;
    settings.samples = 100;
    settings.samples_per_round = 10;
    settings.seed = 42;
    settings.focal_length = 0.4;
    settings.sensor_height = 0.2;

    int opt;
    while ((opt = getopt(argc, argv, "w:h:s:S:r:d:f:c:v")) != -1) {
        switch (opt) {
            case 'w':
                settings.width = atoi(optarg);
                break;
            case 'h':
                settings.height = atoi(optarg);
                break;
            case 's':
                settings.width = settings.height = atoi(optarg);
                break;
            case 'S':
                settings.samples = atoi(optarg);
                break;
            case 'R':
                settings.samples_per_round = atoi(optarg);
                break;
            case 'd':
                settings.seed = atoi(optarg);
                break;
            case 'f':
                settings.focal_length = atof(optarg);
                break;
            case 'c':
                settings.sensor_height = atof(optarg);
                break;
            case 'v':
                doVerbosePrinting = true;
                break;
            default:
                print_help(argc, argv, settings);
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Expected a path to an .obj file.\n");
        print_help(argc, argv, settings);
        return 1;
    }

    if (settings.width < 0 || settings.height < 0) {
        fprintf(stderr, "Expected width and height for output image.\n");
        print_help(argc, argv, settings);
        return 1;
    }
    
    char* obj_file = argv[optind];
    printf("----------------------------------------\n");

    printf("General Settings:\n");
    printf("  Width:               %d\n", settings.width);
    printf("  Height:              %d\n", settings.height);
    printf("  Samples:             %d\n", settings.samples);
    printf("  Samples per round:   %d\n", settings.samples_per_round);
    printf("  Seed:                %d\n", settings.seed);

    printf("\nCamera Settings:\n");
    printf("  Focal Length:        %.2f\n", settings.focal_length);
    printf("  Sensor Height:       %.2f\n", settings.sensor_height);

    printf("\nFile Information:\n");
    printf("  .obj file path:       %s\n", obj_file);
    printf("----------------------------------------\n");

    if (doVerbosePrinting) {
        printf("\nVerbose printing enabled\n");
    }

    obj_scene_data h_scene;
    if (parse_obj_scene(&h_scene, obj_file)) {
        return 1;
    }

    bvh_t h_bvh;
    bvh_build(h_bvh, h_scene);

    size_t img_size = sizeof(Vec3) * settings.width * settings.height;

    Vec3* h_img = (Vec3*)malloc(img_size);
    assert(h_img);

#ifdef USE_CPU_RENDER
    if (render_image_host(&h_scene, &h_bvh, h_img, img_size, settings)) {
        return 1;
    }
#else
    if (render_image_device(&h_scene, &h_bvh, h_img, img_size, settings)) {
        return 1;
    }
#endif

    free(h_img);
    h_img = NULL;
    delete_obj_data(&h_scene);
    bvh_free_host(h_bvh);

    return 0;
}
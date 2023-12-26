#include <stdio.h>
#include <stdlib.h>

#include "assert.h"
#include "image.h"
#include "mathc.h"
#include "obj_parser.h"
#include "renderer.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Expected a path to an .obj file.\n");
        return 1;
    }
    char* obj_file = argv[1];
    obj_scene_data scene;
    if (parse_obj_scene(&scene, obj_file)) {
        return 1;
    }

    // render image
    int factor = 1;
    int width = 400 * factor;
    int height = 300 * factor;
    struct vec3* img = (struct vec3*)calloc(sizeof(struct vec3), width * height);
    assert(img);

    int y_start = 0, y_end = height, x_start = 0, x_end = width;
    // x_start = 150; x_end = 150 + 1;
    // y_start = 100; y_end = 100 + 1;

    for (int y = y_start; y < y_end; y++) {
        for (int x = x_start; x < x_end; x++) {
            double u = (2 * x - width) / (double)height;
            double v = (2 * y - height) / (double)height;
            render(&img[y * width + x], &scene, u, v);
        }

        if (y % 20 == 0) {
            write_bmp(img, width, height, "render.bmp");
        }
    }
    write_bmp(img, width, height, "render.bmp");

    free(img);
    delete_obj_data(&scene);

    return 0;
}
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

    int outer_samples = 50;
    int inner_samples = 10;

    for (int i = 0; i < outer_samples; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mfloat_t u = (2 * x - width) / (mfloat_t)height;
                mfloat_t v = (2 * y - height) / (mfloat_t)height;

                mfloat_t result[VEC3_SIZE];
                render(result, &scene, u, v, inner_samples);

                mfloat_t* pixel = img[y * width + x].v;

                vec3_multiply_f(pixel, pixel, i / (double)(i + 1));
                vec3_multiply_f(result, result, 1.0 / (double)(i + 1));
                vec3_add(pixel, pixel, result);
            }
        }
        write_bmp(img, width, height, "render.bmp");
    }

    free(img);
    delete_obj_data(&scene);

    return 0;
}
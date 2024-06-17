#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "assert.h"
#include "bvh.h"
#include "image.h"
#include "mathc.h"
#include "renderer.h"
#include "scene.h"
#include "utils.h"

#ifdef USE_CPU_RENDER

int render_image_host(obj_scene_data* h_scene, bvh_t* h_bvh, struct vec3* h_img,
                      size_t img_size, render_settings_t settings) {
    time_t start, end;

    printf("Launching host render... \n");
    printf("Rendering %ld samples in batches of %ld, img size (%ld, %ld)\n",
           settings.samples, settings.samples_per_round, settings.width, settings.height);

    time(&start);

    for (int s = 0; s < settings.samples;) {
        for (int y = 0; y < settings.height; y++) {
            for (int x = 0; x < settings.width; x++) {
                render_host(h_img, h_bvh, h_scene, x, y, settings, s);
            }
        }

        char filename[500];
        sprintf(filename, "render.bmp");
        // sprintf(filename, "render_%.4d.bmp", previous_samples);
        write_bmp(h_img, settings.width, settings.height, filename);

        time(&end);
        double elapsed_seconds = difftime(end, start);

        s += settings.samples_per_round;
        settings.seed++;

        printf("Rendered %d / %ld samples in %.0fs (%.2f samples/s)\n",
               s, settings.samples, elapsed_seconds, s / elapsed_seconds);
    }

    return 0;
}

#else

int render_image_device(obj_scene_data* h_scene, bvh_t* h_bvh, struct vec3* h_img,
                        size_t img_size, render_settings_t settings) {
    obj_scene_data* d_scene;
    if (scene_copy_to_device(&d_scene, h_scene)) {
        return 1;
    }

    bvh_t* d_bvh;
    if (bvh_copy_device(&d_bvh, h_bvh)) {
        return 1;
    }

    time_t start, end;
    cudaError_t err;

    struct vec3* d_img;
    err = cudaMalloc(&d_img, img_size);
    if (check_cuda_err(err)) return 1;

    dim3 threads_per_block(16, 16);  // #threads must be factor of 32 and <= 1024

    int grid_width = (settings.width + threads_per_block.x - 1) / threads_per_block.x;
    int grid_height = (settings.height + threads_per_block.y - 1) / threads_per_block.y;
    dim3 num_blocks(grid_width, grid_height);

    printf("Launching kernel... \n");
    printf("Rendering %ld samples in batches of %ld, img size (%ld, %ld)\n",
           settings.samples, settings.samples_per_round, settings.width, settings.height);
    printf("Kernel params <<<(%u,%u), (%u,%u)>>>\n",
           num_blocks.x, num_blocks.y, threads_per_block.x, threads_per_block.y);
    fflush(stdout);

    time(&start);

    for (int s = 0; s < settings.samples;) {
        render_kernel<<<num_blocks, threads_per_block>>>(d_img, d_bvh, d_scene,
                                                           settings, s);
        err = cudaDeviceSynchronize();
        if (check_cuda_err(err)) return 1;

        cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

        char filename[500];
        sprintf(filename, "render.bmp");
        // sprintf(filename, "render_%.4d.bmp", previous_samples);
        write_bmp(h_img, settings.width, settings.height, filename);

        time(&end);
        double elapsed_seconds = difftime(end, start);

        s += settings.samples_per_round;
        settings.seed++;

        printf("Rendered %d / %ld samples in %.0fs (%.2f samples/s)\n",
               s, settings.samples, elapsed_seconds, s / elapsed_seconds);
        fflush(stdout);
    }

    err = cudaFree(d_img);
    if (check_cuda_err(err)) return 1;

    if (free_device_scene(d_scene)) return 1;
    if (bvh_free_device(d_bvh)) return 1;

    return 0;
}

#endif

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Expected a path to an .obj file.\n");
        return 1;
    }

    char* obj_file = argv[1];
    obj_scene_data h_scene;
    if (parse_obj_scene(&h_scene, obj_file)) {
        return 1;
    }

    render_settings_t settings;
    // settings.width = 300;
    // settings.height = 300;
    settings.width = 1800;
    settings.height = 1800;
    settings.samples = 300;
    settings.samples_per_round = 20;
    settings.seed = 69;
    // camera
    // settings.focal_length = 1.5;
    settings.focal_length = 0.4;
    settings.sensor_height = 0.2;

    bvh_t h_bvh;
    bvh_build(&h_bvh, &h_scene);

    size_t img_size = sizeof(struct vec3) * settings.width * settings.height;

    struct vec3* h_img = (struct vec3*)malloc(img_size);
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
    bvh_free_host(&h_bvh);

    return 0;
}
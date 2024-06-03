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

int main(int argc, char* argv[]) {
    time_t start, end;

    if (argc != 2) {
        fprintf(stderr, "Expected a path to an .obj file.\n");
        return 1;
    }

    char* obj_file = argv[1];
    obj_scene_data h_scene;
    if (parse_obj_scene(&h_scene, obj_file)) {
        return 1;
    }

    obj_scene_data* d_scene;
    if (scene_copy_to_device(&d_scene, &h_scene)) {
        return 1;
    }

    BVH h_bvh;
    bvh_build(&h_bvh, &h_scene);

    BVH* d_bvh;
    if (bvh_copy_device(&d_bvh, &h_bvh)) {
        return 1;
    }

    // render image

    int width = 1500;
    int height = 1500;

    cudaError_t err;

    size_t img_size = sizeof(struct vec3) * width * height;

    struct vec3* d_img;
    err = cudaMalloc(&d_img, img_size);
    if (check_cuda_err(err)) return 1;

    struct vec3* h_img = (struct vec3*)malloc(img_size);
    assert(h_img);

    dim3 threads_per_block(16, 16);  // #threads must be factor of 32 and <= 1024

    int grid_width = (width + threads_per_block.x - 1) / threads_per_block.x;
    int grid_height = (height + threads_per_block.y - 1) / threads_per_block.y;
    dim3 num_blocks(grid_width, grid_height);

    int random_seed = 69;
    int samples = 20;
    int samples_per_round = 1;

    printf("Launching kernel... \n");
    printf("Rendering %d samples in batches of %d, img size (%d, %d)\n",
           samples, samples_per_round, width, height);
    printf("Kernel params <<<(%d,%d), (%d,%d)>>>\n",
           num_blocks.x, num_blocks.y, threads_per_block.x, threads_per_block.y);
    fflush(stdout);

    time(&start);

    for (int s = 0; s < samples;) {
        render<<<num_blocks, threads_per_block>>>(d_img, NULL, d_scene,
                                                  width, height, random_seed,
                                                  samples_per_round, s);
        err = cudaDeviceSynchronize();
        if (check_cuda_err(err)) return 1;

        cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

        char filename[500];
        sprintf(filename, "render.bmp");
        // sprintf(filename, "render_%.4d.bmp", previous_samples);
        write_bmp(h_img, width, height, filename);

        time(&end);
        double elapsed_seconds = difftime(end, start);

        s += samples_per_round;
        random_seed++;

        printf("Rendered %d / %d samples in %.0fs (%.2f samples/s)\n",
               s, samples, elapsed_seconds, s / elapsed_seconds);
        fflush(stdout);
    }

    free(h_img); h_img = NULL;
    cudaFree(d_img);
    delete_obj_data(&h_scene);
    free_device_scene(d_scene);
    bvh_free_host(&h_bvh);
    bvh_free_device(d_bvh);

    return 0;
}
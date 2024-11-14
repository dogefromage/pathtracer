#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "assert.h"
#include "dispatch.h"
#include "image.h"
#include "utils.h"
#include <chrono>

#ifdef USE_CPU_RENDER

int render_image_host(obj_scene_data* h_scene, bvh_t* h_bvh, Vec3* h_img,
                      size_t img_size, render_settings_t settings) {
    printf("Launching host render... \n");
    printf("Rendering %d samples in batches of %d, img size (%d, %d)\n",
           settings.samples, settings.samples_per_round, settings.width, settings.height);

    auto startTime = std::chrono::system_clock::now();

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

        auto endTime = std::chrono::system_clock::now();
        int elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>
            (endTime - startTime).count();
        float elapsedTime = elapsedMillis / (float)1000;

        s += settings.samples_per_round;
        settings.seed++;

        float ksns = settings.width * settings.height * s / (1'000.0 * elapsedTime);

        printf("Rendered %d / %d samples in %.0fs - %.2f samples/s - %.2f kSN/s\n",
               s, settings.samples, elapsedTime, s / elapsedTime, ksns);
    }

    return 0;
}

#else

int render_image_device(obj_scene_data* h_scene, bvh_t* h_bvh, Vec3* h_img,
                        size_t img_size, settings_t settings) {
    obj_scene_data* d_scene;
    if (scene_copy_to_device(&d_scene, h_scene)) {
        return 1;
    }

    bvh_t* d_bvh;
    if (bvh_copy_device(&d_bvh, h_bvh)) {
        return 1;
    }

    cudaError_t err;

    Vec3* d_img;
    err = cudaMalloc(&d_img, img_size);
    if (check_cuda_err(err)) return 1;

    dim3 threads_per_block(16, 16);  // #threads must be factor of 32 and <= 1024

    int grid_width = (settings.output.width + threads_per_block.x - 1) / threads_per_block.x;
    int grid_height = (settings.output.height + threads_per_block.y - 1) / threads_per_block.y;
    dim3 num_blocks(grid_width, grid_height);

    printf("Launching kernel... \n");
    printf("Rendering %d samples in batches of %d, img size (%d, %d)\n",
           settings.sampling.samples, settings.sampling.samples_per_round, settings.output.width, settings.output.height);
    printf("Kernel params <<<(%u,%u), (%u,%u)>>>\n",
           num_blocks.x, num_blocks.y, threads_per_block.x, threads_per_block.y);
    fflush(stdout);

    auto startTime = std::chrono::system_clock::now();

    for (int s = 0; s < settings.sampling.samples;) {
        render_kernel<<<num_blocks, threads_per_block>>>(d_img, d_bvh, d_scene,
                                                         settings, s);
        err = cudaDeviceSynchronize();
        if (check_cuda_err(err)) return 1;

        cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

        char filename[500];
        sprintf(filename, "render.bmp");
        // sprintf(filename, "render_%.4d.bmp", previous_samples);
        write_bmp(h_img, settings.output.width, settings.output.height, filename);

        auto endTime = std::chrono::system_clock::now();
        int elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>
            (endTime - startTime).count();
        float elapsedTime = elapsedMillis / (float)1000;

        s += settings.sampling.samples_per_round;
        settings.sampling.seed++;

        float megaPixelSamplesS = settings.output.width * settings.output.height * s / (1'000'000.0 * elapsedTime);

        printf("Rendered %d / %d samples in %.1fs - %.2f samples/s - %.2f MPS/s\n",
               s, settings.sampling.samples, elapsedTime, s / elapsedTime, megaPixelSamplesS);
        fflush(stdout);
    }

    err = cudaFree(d_img);
    if (check_cuda_err(err)) return 1;

    if (free_device_scene(d_scene)) return 1;
    if (bvh_free_device(d_bvh)) return 1;

    return 0;
}

#endif
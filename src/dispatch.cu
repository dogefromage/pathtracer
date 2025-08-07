#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <chrono>

#include "assert.h"
#include "dispatch.h"
#include "image.h"
#include "logger.h"
#include "lst.h"
#include "utils.h"

#ifdef USE_CPU_RENDER

int render_image_host(obj_scene_data *h_scene, bvh_t *h_bvh, Vec3 *h_img, size_t img_size,
                      render_settings_t settings, char *output_path) {
    printf("Launching host render... \n");
    printf("Rendering %d samples in batches of %d, img size "
           "(%d, %d)\n",
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
        // sprintf(filename, "render_%.4d.bmp",
        // previous_samples);
        write_bmp(h_img, settings.width, settings.height, filename);

        auto endTime = std::chrono::system_clock::now();
        int elapsedMillis =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        float elapsedTime = elapsedMillis / (float)1000;

        s += settings.samples_per_round;
        settings.seed++;

        float ksns = settings.width * settings.height * s / (1'000.0 * elapsedTime);

        printf("Rendered %d / %d samples in %.0fs - %.2f "
               "samples/s - %.2f kSN/s\n",
               s, settings.samples, elapsedTime, s / elapsedTime, ksns);
    }

    return 0;
}

#else

void render_image_device(scene_t *h_scene, bvh_t *h_bvh, lst_t *h_lst, Vec3 *h_img,
                         size_t img_size, config_t *cfg) {
    scene_t *d_scene;
    scene_copy_to_device(&d_scene, h_scene);

    bvh_t *d_bvh;
    bvh_copy_device(&d_bvh, h_bvh);

    lst_t *d_lst;
    lst_copy_device(&d_lst, h_lst);

    cudaError_t err;

    Vec3 *d_img;
    err = cudaMalloc(&d_img, img_size);
    if (check_cuda_err(err)) {
        exit(EXIT_FAILURE);
    }

    dim3 threads_per_block(16, 16); // #threads must be factor of 32 and <= 1024

    int grid_width = (cfg->resolution_x + threads_per_block.x - 1) / threads_per_block.x;
    int grid_height = (cfg->resolution_y + threads_per_block.y - 1) / threads_per_block.y;
    dim3 num_blocks(grid_width, grid_height);

    log_info("Launching kernel... \n");
    log_info("Rendering %d samples in batches of %d, img size "
             "(%d, %d)\n",
             cfg->samples, cfg->samples_every_update, cfg->resolution_x, cfg->resolution_y);
    log_info("Kernel params <<<(%u,%u), (%u,%u)>>>\n", num_blocks.x, num_blocks.y,
             threads_per_block.x, threads_per_block.y);

    auto startTime = std::chrono::system_clock::now();

    int renderedSamples = 0;
    while (renderedSamples < cfg->samples) {
        int currentSamples = min(cfg->samples_every_update, cfg->samples - renderedSamples);
        // clang-format off
        render_kernel<<<num_blocks, threads_per_block>>>(d_img, d_bvh, d_scene, d_lst, *cfg, renderedSamples, currentSamples);
        // clang-format on
        err = cudaDeviceSynchronize();
        if (check_cuda_err(err)) {
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
        write_image(h_img, cfg->resolution_x, cfg->resolution_y, cfg->path_render);

        renderedSamples += currentSamples;

        auto endTime = std::chrono::system_clock::now();
        int elapsedMillis =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        float elapsedTime = elapsedMillis / 1000.0f;

        cfg->seed++;

        float megaSamplesPerSecond = (float)cfg->resolution_x * cfg->resolution_y *
                                     renderedSamples / (1'000'000.0 * elapsedTime);

        float samplesPerPixelSecond = renderedSamples / elapsedTime;

        log_info("Rendered %d out of %d S/px in %.1fs - %.2f "
                 "S/px/s - %.2f MS/s\n",
                 renderedSamples, cfg->samples, elapsedTime, samplesPerPixelSecond,
                 megaSamplesPerSecond);
    }

    err = cudaFree(d_img);
    if (check_cuda_err(err)) {
        exit(EXIT_FAILURE);
    }

    free_device_scene(d_scene);
    bvh_free_device(d_bvh);
    lst_free_device(d_lst);
}

#endif

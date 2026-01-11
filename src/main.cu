#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cassert>
#include <chrono>

#include "bvh.h"
#include "config.h"
#include "headers.h"
#include "image.h"
#include "logger.h"
#include "lst.h"
#include "renderer.h"
#include "utils.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

    config_t cfg;

    if (load_config(&cfg, argc, argv)) {
        return 1;
    }

    struct stat st = {0};
    if (stat(cfg.dir_output, &st) == -1) {
        mkdir(cfg.dir_output, 0700);
    }

    path_t log_file;
    snprintf(log_file, sizeof(log_file), "%s/log.txt", cfg.dir_output);
    snprintf(cfg.path_render, sizeof(cfg.path_render), "%s/render.png", cfg.dir_output);

    if (log_init(cfg.log_level, log_file, cfg.log_stdout)) {
        return 1;
    }

    Scene h_scene, d_scene;
    h_scene.read_gltf(cfg.path_gltf);

    // bounding volume hierarchy
    BVH h_bvh, d_bvh;
    h_bvh.build(h_scene);

    // light source tree
    LST h_lst, d_lst;
    h_lst.build(h_scene);

    size_t img_size = sizeof(Vec3) * cfg.resolution_x * cfg.resolution_y;
    Vec3 *h_img = (Vec3 *)malloc(img_size);
    assert(h_img);

    // ########## DISPATCH #############

    d_scene.device_from_host(h_scene);
    d_bvh.device_from_host(h_bvh);
    d_lst.device_from_host(h_lst);

    cudaError_t err;
    Vec3 *d_img;
    err = cudaMalloc(&d_img, img_size);
    if (check_cuda_err(err)) {
        exit(EXIT_FAILURE);
    }

    dim3 threads_per_block(16, 16); // #threads must be factor of 32 and <= 1024

    int grid_width = (cfg.resolution_x + threads_per_block.x - 1) / threads_per_block.x;
    int grid_height = (cfg.resolution_y + threads_per_block.y - 1) / threads_per_block.y;
    dim3 num_blocks(grid_width, grid_height);

    log_info("Launching kernel... \n");
    log_info("Rendering %d samples in batches of %d, img size "
             "(%d, %d)\n",
             cfg.samples, cfg.samples_every_update, cfg.resolution_x, cfg.resolution_y);
    log_info("Kernel params <<<(%u,%u), (%u,%u)>>>\n", num_blocks.x, num_blocks.y, threads_per_block.x, threads_per_block.y);

    auto startTime = std::chrono::system_clock::now();

    int renderedSamples = 0;
    while (renderedSamples < cfg.samples) {
        int currentSamples = min(cfg.samples_every_update, cfg.samples - renderedSamples);
        // clang-format off
        render_kernel<<<num_blocks, threads_per_block>>>(d_img, d_bvh, d_scene, d_lst, cfg, renderedSamples, currentSamples);
        // clang-format on
        err = cudaDeviceSynchronize();
        if (check_cuda_err(err)) {
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
        write_image(h_img, cfg.resolution_x, cfg.resolution_y, cfg.path_render);
        log_trace("Updated image: %s\n", cfg.path_render);

        renderedSamples += currentSamples;

        auto endTime = std::chrono::system_clock::now();
        int elapsedMillis = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        float elapsedTime = elapsedMillis / 1000.0f;

        cfg.seed++;

        float megaSamplesPerSecond = (float)cfg.resolution_x * cfg.resolution_y * renderedSamples / (1'000'000.0 * elapsedTime);

        float samplesPerPixelSecond = renderedSamples / elapsedTime;

        log_info("Rendered %d out of %d S/px in %.1fs - %.2f "
                 "S/px/s - %.2f MS/s\n",
                 renderedSamples, cfg.samples, elapsedTime, samplesPerPixelSecond, megaSamplesPerSecond);
    }

    err = cudaFree(d_img);
    if (check_cuda_err(err)) {
        exit(EXIT_FAILURE);
    }

    // ########## DISPATCH #############

    free(h_img);
    h_img = NULL;

    h_scene._free();
    d_scene._free();
    h_bvh._free();
    d_bvh._free();
    h_lst._free();
    d_lst._free();

    log_close();

    return 0;
}

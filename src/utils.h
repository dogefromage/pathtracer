#pragma once

#include "logger.h"
#include "mathops.h"

void print_stacktrace();
cudaError_t check_cuda_err(cudaError_t err);

enum class CudaLocation { Host, Device };

template <typename T> size_t copy_device_struct(T **dev, const T *host) {
    cudaError_t err;
    size_t size = sizeof(T);

    err = cudaMalloc(dev, size);
    if (check_cuda_err(err))
        exit(EXIT_FAILURE);

    err = cudaMemcpy(*dev, host, size, cudaMemcpyHostToDevice);
    if (check_cuda_err(err))
        exit(EXIT_FAILURE);

    return size;
}

template <typename T> void copy_host_struct(T *host, const T *dev) {
    cudaError_t err;
    err = cudaMemcpy(host, dev, sizeof(T), cudaMemcpyDeviceToHost);
    if (check_cuda_err(err))
        exit(EXIT_FAILURE);
}

inline void device_free(void *devp) {
    cudaError_t err;
    err = cudaFree(devp);
    if (check_cuda_err(err)) {
        exit(EXIT_FAILURE);
    }
}

void human_readable_size(char *buf, size_t bytes);

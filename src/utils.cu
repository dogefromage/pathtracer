#include "utils.h"
#include <cstdio>
#include <execinfo.h>
#include <stdio.h>

#define MAX_STACK_LEVELS 20

void print_stacktrace() {
    void *buffer[MAX_STACK_LEVELS];
    int levels = backtrace(buffer, MAX_STACK_LEVELS);

    // print to stderr (fd = 2), and remove this function from the trace
    backtrace_symbols_fd(buffer + 1, levels - 1, 2);
}

cudaError_t check_cuda_err(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        print_stacktrace();
    }
    return err;
}

void human_readable_size(char *buf, size_t bytes) {
    if (bytes < 1'000) {
        sprintf(buf, "%luB", bytes);
    } else if (bytes < 1'000'000) {
        sprintf(buf, "%luKB", bytes / 1'000);
    } else if (bytes < 1'000'000'000) {
        sprintf(buf, "%luMB", bytes / 1'000'000);
    } else {
        // ???
        sprintf(buf, "%luGB", bytes / 1'000'000'000);
    }
}

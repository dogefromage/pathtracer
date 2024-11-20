#include "utils.h"
#include <stdio.h>
#include <cstdio>
#include <execinfo.h>

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
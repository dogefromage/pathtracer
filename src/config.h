#pragma once

// #define USE_INTERSECT_CRUDE
// #define USE_CPU_RENDER

extern bool doVerbosePrinting;

#define BVH_TRAVERSAL_STACK_SIZE 64

#ifdef USE_CPU_RENDER
#define PLATFORM __host__
#else
#define PLATFORM __device__
#endif
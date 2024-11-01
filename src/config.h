#pragma once

// #define USE_CPU_RENDER

#ifdef USE_CPU_RENDER
#define PLATFORM __host__
#else
#define PLATFORM __device__
#endif

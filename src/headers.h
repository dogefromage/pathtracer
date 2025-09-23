#pragma once

// #define USE_INTERSECT_CRUDE
// #define USE_CPU_RENDER

#ifdef USE_CPU_RENDER
#define PLATFORM __host__
#else
#define PLATFORM __device__
#endif

#define CHECK_VEC(v)                                                                           \
    assert(isfinite(v.x));                                                                     \
    assert(isfinite(v.y));                                                                     \
    assert(isfinite(v.z))

#define CHECK_FLOAT(x) assert(isfinite(x))

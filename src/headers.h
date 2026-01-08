#pragma once

// #define USE_INTERSECT_CRUDE

#define CHECK_VEC(v)                                                                                                           \
    assert(isfinite(v.x));                                                                                                     \
    assert(isfinite(v.y));                                                                                                     \
    assert(isfinite(v.z))

#define CHECK_FLOAT(x) assert(isfinite(x))

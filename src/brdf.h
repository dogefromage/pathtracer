#pragma once
#include "random.h"

typedef struct {
    Vec3 wi;
    Spectrum f_cos_theta;
    float pdf;
    bool is_delta;
} brdf_sample_t;

typedef struct {
    Vec3 base_color;
} brdf_params_t;

struct BRDF {

    Vec3 base_color;

    // Evaluate BRDF * cos(theta)
    __device__ Spectrum eval(const Vec3 &wo, const Vec3 &wi) const;

    __device__ float pdf(const Vec3 &wo, const Vec3 &wi) const;

    // Sample wi given wo
    __device__ brdf_sample_t sample(const Vec3 &wo, rand_state_t &rstate) const;
};

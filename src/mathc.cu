/*
Copyright © 2018 Felipe Ferreira da Silva

This software is provided 'as-is', without any express or implied warranty. In
no event will the authors be held liable for any damages arising from the use of
this software.

Permission is granted to anyone to use this software for any purpose, including
commercial applications, and to alter it and redistribute it freely, subject to
the following restrictions:

  1. The origin of this software must not be misrepresented; you must not claim
     that you wrote the original software. If you use this software in a
     product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "mathc.h"
#include <stdbool.h>

__host__ __device__ bool nearly_equal(mfloat_t a, mfloat_t b, mfloat_t epsilon) {
    bool result = false;
    if (a == b) {
        result = 1;
    } else if (MFABS(a - b) <= epsilon) {
        result = 1;
    }
    return result;
}

__host__ __device__ mfloat_t to_radians(mfloat_t degrees) {
    return MRADIANS(degrees);
}

__host__ __device__ mfloat_t to_degrees(mfloat_t radians) {
    return MDEGREES(radians);
}

__host__ __device__ mfloat_t clampf(mfloat_t value, mfloat_t min, mfloat_t max) {
    if (value < min) {
        value = min;
    } else if (value > max) {
        value = max;
    }
    return value;
}

__host__ __device__ bool vec2_is_zero(mfloat_t *v0) {
    return MFABS(v0[0]) < MFLT_EPSILON && MFABS(v0[1]) < MFLT_EPSILON;
}

__host__ __device__ bool vec2_is_equal(mfloat_t *v0, mfloat_t *v1) {
    return MFABS(v0[0] - v1[0]) < MFLT_EPSILON && MFABS(v0[1] - v1[1]) < MFLT_EPSILON;
}

__host__ __device__ mfloat_t *vec2(mfloat_t *result, mfloat_t x, mfloat_t y) {
    result[0] = x;
    result[1] = y;
    return result;
}

__host__ __device__ mfloat_t *vec2_assign(mfloat_t *result, mfloat_t *v0) {
    result[0] = v0[0];
    result[1] = v0[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_zero(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *vec2_one(mfloat_t *result) {
    result[0] = MFLOAT_C(1.0);
    result[1] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *vec2_sign(mfloat_t *result, mfloat_t *v0) {
    if (v0[0] > MFLOAT_C(0.0)) {
        result[0] = MFLOAT_C(1.0);
    } else if (v0[0] < MFLOAT_C(0.0)) {
        result[0] = -MFLOAT_C(1.0);
    } else {
        result[0] = MFLOAT_C(0.0);
    }
    if (v0[1] > MFLOAT_C(0.0)) {
        result[1] = MFLOAT_C(1.0);
    } else if (v0[1] < MFLOAT_C(0.0)) {
        result[1] = -MFLOAT_C(1.0);
    } else {
        result[1] = MFLOAT_C(0.0);
    }
    return result;
}

__host__ __device__ mfloat_t *vec2_add(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] + v1[0];
    result[1] = v0[1] + v1[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_add_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] + f;
    result[1] = v0[1] + f;
    return result;
}

__host__ __device__ mfloat_t *vec2_subtract(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] - v1[0];
    result[1] = v0[1] - v1[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_subtract_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] - f;
    result[1] = v0[1] - f;
    return result;
}

__host__ __device__ mfloat_t *vec2_multiply(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] * v1[0];
    result[1] = v0[1] * v1[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_multiply_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] * f;
    result[1] = v0[1] * f;
    return result;
}

__host__ __device__ mfloat_t *vec2_multiply_mat2(mfloat_t *result, mfloat_t *v0, mfloat_t *m0) {
    mfloat_t x = v0[0];
    mfloat_t y = v0[1];
    result[0] = m0[0] * x + m0[2] * y;
    result[1] = m0[1] * x + m0[3] * y;
    return result;
}

__host__ __device__ mfloat_t *vec2_divide(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] / v1[0];
    result[1] = v0[1] / v1[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_divide_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] / f;
    result[1] = v0[1] / f;
    return result;
}

__host__ __device__ mfloat_t *vec2_snap(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = MFLOOR(v0[0] / v1[0]) * v1[0];
    result[1] = MFLOOR(v0[1] / v1[1]) * v1[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_snap_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = MFLOOR(v0[0] / f) * f;
    result[1] = MFLOOR(v0[1] / f) * f;
    return result;
}

__host__ __device__ mfloat_t *vec2_negative(mfloat_t *result, mfloat_t *v0) {
    result[0] = -v0[0];
    result[1] = -v0[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_abs(mfloat_t *result, mfloat_t *v0) {
    result[0] = MFABS(v0[0]);
    result[1] = MFABS(v0[1]);
    return result;
}

__host__ __device__ mfloat_t *vec2_floor(mfloat_t *result, mfloat_t *v0) {
    result[0] = MFLOOR(v0[0]);
    result[1] = MFLOOR(v0[1]);
    return result;
}

__host__ __device__ mfloat_t *vec2_ceil(mfloat_t *result, mfloat_t *v0) {
    result[0] = MCEIL(v0[0]);
    result[1] = MCEIL(v0[1]);
    return result;
}

__host__ __device__ mfloat_t *vec2_round(mfloat_t *result, mfloat_t *v0) {
    result[0] = MROUND(v0[0]);
    result[1] = MROUND(v0[1]);
    return result;
}

__host__ __device__ mfloat_t *vec2_max(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = MFMAX(v0[0], v1[0]);
    result[1] = MFMAX(v0[1], v1[1]);
    return result;
}

__host__ __device__ mfloat_t *vec2_min(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = MFMIN(v0[0], v1[0]);
    result[1] = MFMIN(v0[1], v1[1]);
    return result;
}

__host__ __device__ mfloat_t *vec2_clamp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2) {
    vec2_min(result, v0, v1);
    vec2_max(result, v0, v2);
    return result;
}

__host__ __device__ mfloat_t *vec2_normalize(mfloat_t *result, mfloat_t *v0) {
    mfloat_t l = MSQRT(v0[0] * v0[0] + v0[1] * v0[1]);
    result[0] = v0[0] / l;
    result[1] = v0[1] / l;
    return result;
}

__host__ __device__ mfloat_t vec2_dot(mfloat_t *v0, mfloat_t *v1) {
    return v0[0] * v1[0] + v0[1] * v1[1];
}

__host__ __device__ mfloat_t *vec2_project(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    mfloat_t d = vec2_dot(v1, v1);
    mfloat_t s = vec2_dot(v0, v1) / d;
    result[0] = v1[0] * s;
    result[1] = v1[1] * s;
    return result;
}

__host__ __device__ mfloat_t *vec2_slide(mfloat_t *result, mfloat_t *v0, mfloat_t *normal) {
    mfloat_t d = vec2_dot(v0, normal);
    result[0] = v0[0] - normal[0] * d;
    result[1] = v0[1] - normal[1] * d;
    return result;
}

__host__ __device__ mfloat_t *vec2_reflect(mfloat_t *result, mfloat_t *v0, mfloat_t *normal) {
    mfloat_t d = MFLOAT_C(2.0) * vec2_dot(v0, normal);
    result[0] = normal[0] * d - v0[0];
    result[1] = normal[1] * d - v0[1];
    return result;
}

__host__ __device__ mfloat_t *vec2_tangent(mfloat_t *result, mfloat_t *v0) {
    mfloat_t a0 = v0[0];
    mfloat_t a1 = v0[1];
    result[0] = a1;
    result[1] = -a0;
    return result;
}

__host__ __device__ mfloat_t *vec2_rotate(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    mfloat_t cs = MCOS(f);
    mfloat_t sn = MSIN(f);
    mfloat_t x = v0[0];
    mfloat_t y = v0[1];
    result[0] = x * cs - y * sn;
    result[1] = x * sn + y * cs;
    return result;
}

__host__ __device__ mfloat_t *vec2_lerp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t f) {
    result[0] = v0[0] + (v1[0] - v0[0]) * f;
    result[1] = v0[1] + (v1[1] - v0[1]) * f;
    return result;
}

__host__ __device__ mfloat_t *vec2_bezier3(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2, mfloat_t f) {
    mfloat_t tmp0[VEC2_SIZE];
    mfloat_t tmp1[VEC2_SIZE];
    vec2_lerp(tmp0, v0, v1, f);
    vec2_lerp(tmp1, v1, v2, f);
    vec2_lerp(result, tmp0, tmp1, f);
    return result;
}

__host__ __device__ mfloat_t *vec2_bezier4(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2, mfloat_t *v3, mfloat_t f) {
    mfloat_t tmp0[VEC2_SIZE];
    mfloat_t tmp1[VEC2_SIZE];
    mfloat_t tmp2[VEC2_SIZE];
    mfloat_t tmp3[VEC2_SIZE];
    mfloat_t tmp4[VEC2_SIZE];
    vec2_lerp(tmp0, v0, v1, f);
    vec2_lerp(tmp1, v1, v2, f);
    vec2_lerp(tmp2, v2, v3, f);
    vec2_lerp(tmp3, tmp0, tmp1, f);
    vec2_lerp(tmp4, tmp1, tmp2, f);
    vec2_lerp(result, tmp3, tmp4, f);
    return result;
}

__host__ __device__ mfloat_t vec2_angle(mfloat_t *v0) {
    return MATAN2(v0[1], v0[0]);
}

__host__ __device__ mfloat_t vec2_length(mfloat_t *v0) {
    return MSQRT(v0[0] * v0[0] + v0[1] * v0[1]);
}

__host__ __device__ mfloat_t vec2_length_squared(mfloat_t *v0) {
    return v0[0] * v0[0] + v0[1] * v0[1];
}

__host__ __device__ mfloat_t vec2_distance(mfloat_t *v0, mfloat_t *v1) {
    return MSQRT((v0[0] - v1[0]) * (v0[0] - v1[0]) + (v0[1] - v1[1]) * (v0[1] - v1[1]));
}

__host__ __device__ mfloat_t vec2_distance_squared(mfloat_t *v0, mfloat_t *v1) {
    return (v0[0] - v1[0]) * (v0[0] - v1[0]) + (v0[1] - v1[1]) * (v0[1] - v1[1]);
}

__host__ __device__ bool vec2_linear_independent(mfloat_t *v0, mfloat_t *v1) {
    return (v0[0] * v1[1] - v1[0] * v0[1]) != 0;
}

__host__ __device__ mfloat_t **vec2_orthonormalization(mfloat_t result[2][2], mfloat_t basis[2][2]) {
    mfloat_t v0[2];
    mfloat_t v1[2];

    for (int32_t i = 0; i < 2; ++i) {
        v0[i] = basis[0][i];
        v1[i] = basis[1][i];
    }

    if (!vec2_linear_independent(v0, v1)) {
        return (mfloat_t **)result;
    }

    mfloat_t proju1[2];
    mfloat_t u0[2];
    mfloat_t u1[2];

    for (int32_t i = 0; i < 2; ++i) {
        u0[i] = v0[i];
    }

    vec2_project(proju1, v1, v0);
    vec2_subtract(u1, v1, proju1);
    vec2_normalize(result[0], u0);
    vec2_normalize(result[1], u1);

    return (mfloat_t **)result;
}

__host__ __device__ bool vec3_is_zero(const mfloat_t *v0) {
    return MFABS(v0[0]) < MFLT_EPSILON && MFABS(v0[1]) < MFLT_EPSILON && MFABS(v0[2]) < MFLT_EPSILON;
}

__host__ __device__ bool vec3_is_equal(const mfloat_t *v0, const mfloat_t *v1) {
    return MFABS(v0[0] - v1[0]) < MFLT_EPSILON && MFABS(v0[1] - v1[1]) < MFLT_EPSILON && MFABS(v0[2] - v1[2]) < MFLT_EPSILON;
}

__host__ __device__ mfloat_t *vec3(mfloat_t *result, mfloat_t x, mfloat_t y, mfloat_t z) {
    result[0] = x;
    result[1] = y;
    result[2] = z;
    return result;
}

__host__ __device__ mfloat_t *vec3_assign(mfloat_t *result, const mfloat_t *v0) {
    result[0] = v0[0];
    result[1] = v0[1];
    result[2] = v0[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_zero(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *vec3_const(mfloat_t *result, const mfloat_t v) {
    result[0] = MFLOAT_C(v);
    result[1] = MFLOAT_C(v);
    result[2] = MFLOAT_C(v);
    return result;
}

__host__ __device__ mfloat_t *vec3_one(mfloat_t *result) {
    result[0] = MFLOAT_C(1.0);
    result[1] = MFLOAT_C(1.0);
    result[2] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *vec3_sign(mfloat_t *result, const mfloat_t *v0) {
    if (v0[0] > MFLOAT_C(0.0)) {
        result[0] = MFLOAT_C(1.0);
    } else if (v0[0] < 0) {
        result[0] = -MFLOAT_C(1.0);
    } else {
        result[0] = MFLOAT_C(0.0);
    }
    if (v0[1] > MFLOAT_C(0.0)) {
        result[1] = MFLOAT_C(1.0);
    } else if (v0[1] < 0) {
        result[1] = -MFLOAT_C(1.0);
    } else {
        result[1] = MFLOAT_C(0.0);
    }
    if (v0[2] > MFLOAT_C(0.0)) {
        result[2] = MFLOAT_C(1.0);
    } else if (v0[2] < 0) {
        result[2] = -MFLOAT_C(1.0);
    } else {
        result[2] = MFLOAT_C(0.0);
    }
    return result;
}

__host__ __device__ mfloat_t *vec3_add(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    result[0] = v0[0] + v1[0];
    result[1] = v0[1] + v1[1];
    result[2] = v0[2] + v1[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_add_f(mfloat_t *result, const mfloat_t *v0, const mfloat_t f) {
    result[0] = v0[0] + f;
    result[1] = v0[1] + f;
    result[2] = v0[2] + f;
    return result;
}

__host__ __device__ mfloat_t *vec3_subtract(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    result[0] = v0[0] - v1[0];
    result[1] = v0[1] - v1[1];
    result[2] = v0[2] - v1[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_subtract_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] - f;
    result[1] = v0[1] - f;
    result[2] = v0[2] - f;
    return result;
}

__host__ __device__ mfloat_t *vec3_multiply(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    result[0] = v0[0] * v1[0];
    result[1] = v0[1] * v1[1];
    result[2] = v0[2] * v1[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_multiply_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] * f;
    result[1] = v0[1] * f;
    result[2] = v0[2] * f;
    return result;
}

__host__ __device__ mfloat_t *vec3_multiply_mat3(mfloat_t *result, const mfloat_t *v0, const mfloat_t *m0) {
    mfloat_t x = v0[0];
    mfloat_t y = v0[1];
    mfloat_t z = v0[2];
    result[0] = m0[0] * x + m0[3] * y + m0[6] * z;
    result[1] = m0[1] * x + m0[4] * y + m0[7] * z;
    result[2] = m0[2] * x + m0[5] * y + m0[8] * z;
    return result;
}

__host__ __device__ mfloat_t *vec3_divide(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    result[0] = v0[0] / v1[0];
    result[1] = v0[1] / v1[1];
    result[2] = v0[2] / v1[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_divide_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] / f;
    result[1] = v0[1] / f;
    result[2] = v0[2] / f;
    return result;
}

__host__ __device__ mfloat_t *vec3_snap(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    result[0] = MFLOOR(v0[0] / v1[0]) * v1[0];
    result[1] = MFLOOR(v0[1] / v1[1]) * v1[1];
    result[2] = MFLOOR(v0[2] / v1[2]) * v1[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_snap_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f) {
    result[0] = MFLOOR(v0[0] / f) * f;
    result[1] = MFLOOR(v0[1] / f) * f;
    result[2] = MFLOOR(v0[2] / f) * f;
    return result;
}

__host__ __device__ mfloat_t *vec3_negative(mfloat_t *result, const mfloat_t *v0) {
    result[0] = -v0[0];
    result[1] = -v0[1];
    result[2] = -v0[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_abs(mfloat_t *result, const mfloat_t *v0) {
    result[0] = MFABS(v0[0]);
    result[1] = MFABS(v0[1]);
    result[2] = MFABS(v0[2]);
    return result;
}

__host__ __device__ mfloat_t *vec3_floor(mfloat_t *result, const mfloat_t *v0) {
    result[0] = MFLOOR(v0[0]);
    result[1] = MFLOOR(v0[1]);
    result[2] = MFLOOR(v0[2]);
    return result;
}

__host__ __device__ mfloat_t *vec3_ceil(mfloat_t *result, const mfloat_t *v0) {
    result[0] = MCEIL(v0[0]);
    result[1] = MCEIL(v0[1]);
    result[2] = MCEIL(v0[2]);
    return result;
}

__host__ __device__ mfloat_t *vec3_round(mfloat_t *result, const mfloat_t *v0) {
    result[0] = MROUND(v0[0]);
    result[1] = MROUND(v0[1]);
    result[2] = MROUND(v0[2]);
    return result;
}

__host__ __device__ mfloat_t *vec3_max(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    result[0] = MFMAX(v0[0], v1[0]);
    result[1] = MFMAX(v0[1], v1[1]);
    result[2] = MFMAX(v0[2], v1[2]);
    return result;
}

__host__ __device__ mfloat_t *vec3_min(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    result[0] = MFMIN(v0[0], v1[0]);
    result[1] = MFMIN(v0[1], v1[1]);
    result[2] = MFMIN(v0[2], v1[2]);
    return result;
}

__host__ __device__ mfloat_t *vec3_clamp(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2) {
    vec3_min(result, v0, v1);
    vec3_max(result, v0, v2);
    return result;
}

__host__ __device__ mfloat_t *vec3_cross(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    mfloat_t cross[VEC3_SIZE];
    cross[0] = v0[1] * v1[2] - v0[2] * v1[1];
    cross[1] = v0[2] * v1[0] - v0[0] * v1[2];
    cross[2] = v0[0] * v1[1] - v0[1] * v1[0];
    result[0] = cross[0];
    result[1] = cross[1];
    result[2] = cross[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_normalize(mfloat_t *result, const mfloat_t *v0) {
    mfloat_t invL = 1.0 / (MSQRT(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]));
    result[0] = v0[0] * invL;
    result[1] = v0[1] * invL;
    result[2] = v0[2] * invL;
    return result;
}

__host__ __device__ mfloat_t vec3_dot(const mfloat_t *v0, const mfloat_t *v1) {
    return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
}

__host__ __device__ mfloat_t *vec3_project(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1) {
    mfloat_t d = vec3_dot(v1, v1);
    mfloat_t s = vec3_dot(v0, v1) / d;
    result[0] = v1[0] * s;
    result[1] = v1[1] * s;
    result[2] = v1[2] * s;
    return result;
}

__host__ __device__ mfloat_t *vec3_slide(mfloat_t *result, const mfloat_t *v0, const mfloat_t *normal) {
    mfloat_t d = vec3_dot(v0, normal);
    result[0] = v0[0] - normal[0] * d;
    result[1] = v0[1] - normal[1] * d;
    result[2] = v0[2] - normal[2] * d;
    return result;
}

__host__ __device__ mfloat_t *vec3_reflect(mfloat_t *result, const mfloat_t *v0, const mfloat_t *normal) {
    mfloat_t d = MFLOAT_C(2.0) * vec3_dot(v0, normal);
    result[0] = normal[0] * d - v0[0];
    result[1] = normal[1] * d - v0[1];
    result[2] = normal[2] * d - v0[2];
    return result;
}

__host__ __device__ mfloat_t *vec3_rotate(mfloat_t *result, const mfloat_t *v0, const mfloat_t *ra, mfloat_t f) {
    mfloat_t cs;
    mfloat_t sn;
    mfloat_t x;
    mfloat_t y;
    mfloat_t z;
    mfloat_t rx;
    mfloat_t ry;
    mfloat_t rz;
    cs = MCOS(f);
    sn = MSIN(f);
    x = v0[0];
    y = v0[1];
    z = v0[2];
    mfloat_t nra[3];
    vec3_normalize(nra, ra);
    rx = nra[0];
    ry = nra[1];
    rz = nra[2];
    result[0] = x * (cs + rx * rx * (1 - cs)) + y * (rx * ry * (1 - cs) - rz * sn) + z * (rx * rz * (1 - cs) + ry * sn);
    result[1] = x * (ry * rx * (1 - cs) + rz * sn) + y * (cs + ry * ry * (1 - cs)) + z * (ry * rz * (1 - cs) - rx * sn);
    result[2] = x * (rz * rx * (1 - cs) - ry * sn) + y * (rz * ry * (1 - cs) + rx * sn) + z * (cs + rz * rz * (1 - cs));
    return result;
}

__host__ __device__ mfloat_t *vec3_lerp(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, mfloat_t f) {
    result[0] = v0[0] + (v1[0] - v0[0]) * f;
    result[1] = v0[1] + (v1[1] - v0[1]) * f;
    result[2] = v0[2] + (v1[2] - v0[2]) * f;
    return result;
}

__host__ __device__ mfloat_t *vec3_bezier3(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2, mfloat_t f) {
    mfloat_t tmp0[VEC3_SIZE];
    mfloat_t tmp1[VEC3_SIZE];
    vec3_lerp(tmp0, v0, v1, f);
    vec3_lerp(tmp1, v1, v2, f);
    vec3_lerp(result, tmp0, tmp1, f);
    return result;
}

__host__ __device__ mfloat_t *vec3_bezier4(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2, const mfloat_t *v3, mfloat_t f) {
    mfloat_t tmp0[VEC3_SIZE];
    mfloat_t tmp1[VEC3_SIZE];
    mfloat_t tmp2[VEC3_SIZE];
    mfloat_t tmp3[VEC3_SIZE];
    mfloat_t tmp4[VEC3_SIZE];
    vec3_lerp(tmp0, v0, v1, f);
    vec3_lerp(tmp1, v1, v2, f);
    vec3_lerp(tmp2, v2, v3, f);
    vec3_lerp(tmp3, tmp0, tmp1, f);
    vec3_lerp(tmp4, tmp1, tmp2, f);
    vec3_lerp(result, tmp3, tmp4, f);
    return result;
}

__host__ __device__ mfloat_t vec3_max_component(const mfloat_t *v) {
    return MFMAX(MFMAX(v[0], v[1]), v[2]);
}

__host__ __device__ mfloat_t vec3_length(const mfloat_t *v0) {
    return MSQRT(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
}

__host__ __device__ mfloat_t vec3_length_squared(const mfloat_t *v0) {
    return v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
}

__host__ __device__ mfloat_t vec3_distance(const mfloat_t *v0, const mfloat_t *v1) {
    return MSQRT((v0[0] - v1[0]) * (v0[0] - v1[0]) + (v0[1] - v1[1]) * (v0[1] - v1[1]) + (v0[2] - v1[2]) * (v0[2] - v1[2]));
}

__host__ __device__ mfloat_t vec3_distance_squared(const mfloat_t *v0, const mfloat_t *v1) {
    return (v0[0] - v1[0]) * (v0[0] - v1[0]) + (v0[1] - v1[1]) * (v0[1] - v1[1]) + (v0[2] - v1[2]) * (v0[2] - v1[2]);
}

__host__ __device__ bool vec3_linear_independent(const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2) {
    return v0[0] * v1[1] * v2[2] + v0[1] * v1[2] * v2[0] + v0[2] * v1[0] * v2[1] - v0[2] * v1[1] * v2[0] - v0[1] * v1[0] * v2[2] - v0[0] * v1[2] * v2[1];
}

__host__ __device__ mfloat_t **vec3_orthonormalization(mfloat_t result[3][3], const mfloat_t basis[3][3]) {
    mfloat_t v0[3];
    mfloat_t v1[3];
    mfloat_t v2[3];

    for (int32_t i = 0; i < 3; ++i) {
        v0[i] = basis[0][i];
        v1[i] = basis[1][i];
        v2[i] = basis[2][i];
    }

    if (!vec3_linear_independent(v0, v1, v2)) {
        return (mfloat_t **)result;
    }

    mfloat_t proj[3];
    mfloat_t u0[3];
    mfloat_t u1[3];
    mfloat_t u2[3];

    for (int32_t i = 0; i < 3; ++i) {
        u0[i] = v0[i];
    }

    vec3_project(proj, v1, u0);
    vec3_subtract(u1, v1, proj);

    vec3_project(proj, v2, u0);
    vec3_subtract(u2, v2, proj);
    vec3_project(proj, v2, u1);
    vec3_subtract(u2, u2, proj);

    vec3_normalize(result[0], u0);
    vec3_normalize(result[1], u1);
    vec3_normalize(result[2], u2);

    return (mfloat_t **)result;
}

__host__ __device__ bool vec4_is_zero(mfloat_t *v0) {
    return MFABS(v0[0]) < MFLT_EPSILON && MFABS(v0[1]) < MFLT_EPSILON && MFABS(v0[2]) < MFLT_EPSILON && MFABS(v0[3]) < MFLT_EPSILON;
}

__host__ __device__ bool vec4_is_equal(mfloat_t *v0, mfloat_t *v1) {
    return MFABS(v0[0] - v1[0]) < MFLT_EPSILON && MFABS(v0[1] - v1[1]) < MFLT_EPSILON && MFABS(v0[2] - v1[2]) < MFLT_EPSILON && MFABS(v0[3] - v1[3]) < MFLT_EPSILON;
}

__host__ __device__ mfloat_t *vec4(mfloat_t *result, mfloat_t x, mfloat_t y, mfloat_t z, mfloat_t w) {
    result[0] = x;
    result[1] = y;
    result[2] = z;
    result[3] = w;
    return result;
}

__host__ __device__ mfloat_t *vec4_assign(mfloat_t *result, mfloat_t *v0) {
    result[0] = v0[0];
    result[1] = v0[1];
    result[2] = v0[2];
    result[3] = v0[3];
    return result;
}

__host__ __device__ mfloat_t *vec4_zero(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *vec4_one(mfloat_t *result) {
    result[0] = MFLOAT_C(1.0);
    result[1] = MFLOAT_C(1.0);
    result[2] = MFLOAT_C(1.0);
    result[3] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *vec4_sign(mfloat_t *result, mfloat_t *v0) {
    if (v0[0] > MFLOAT_C(0.0)) {
        result[0] = MFLOAT_C(1.0);
    } else if (v0[0] < 0) {
        result[0] = -MFLOAT_C(1.0);
    } else {
        result[0] = MFLOAT_C(0.0);
    }
    if (v0[1] > MFLOAT_C(0.0)) {
        result[1] = MFLOAT_C(1.0);
    } else if (v0[1] < 0) {
        result[1] = -MFLOAT_C(1.0);
    } else {
        result[1] = MFLOAT_C(0.0);
    }
    if (v0[2] > MFLOAT_C(0.0)) {
        result[2] = MFLOAT_C(1.0);
    } else if (v0[2] < 0) {
        result[2] = -MFLOAT_C(1.0);
    } else {
        result[2] = MFLOAT_C(0.0);
    }
    if (v0[3] > MFLOAT_C(0.0)) {
        result[3] = MFLOAT_C(1.0);
    } else if (v0[3] < 0) {
        result[3] = -MFLOAT_C(1.0);
    } else {
        result[3] = MFLOAT_C(0.0);
    }
    return result;
}

__host__ __device__ mfloat_t *vec4_add(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] + v1[0];
    result[1] = v0[1] + v1[1];
    result[2] = v0[2] + v1[2];
    result[3] = v0[3] + v1[3];
    return result;
}

__host__ __device__ mfloat_t *vec4_add_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] + f;
    result[1] = v0[1] + f;
    result[2] = v0[2] + f;
    result[3] = v0[3] + f;
    return result;
}

__host__ __device__ mfloat_t *vec4_subtract(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] - v1[0];
    result[1] = v0[1] - v1[1];
    result[2] = v0[2] - v1[2];
    result[3] = v0[3] - v1[3];
    return result;
}

__host__ __device__ mfloat_t *vec4_subtract_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] - f;
    result[1] = v0[1] - f;
    result[2] = v0[2] - f;
    result[3] = v0[3] - f;
    return result;
}

__host__ __device__ mfloat_t *vec4_multiply(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] * v1[0];
    result[1] = v0[1] * v1[1];
    result[2] = v0[2] * v1[2];
    result[3] = v0[3] * v1[3];
    return result;
}

__host__ __device__ mfloat_t *vec4_multiply_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] * f;
    result[1] = v0[1] * f;
    result[2] = v0[2] * f;
    result[3] = v0[3] * f;
    return result;
}

__host__ __device__ mfloat_t *vec4_multiply_mat4(mfloat_t *result, mfloat_t *v0, mfloat_t *m0) {
    mfloat_t x = v0[0];
    mfloat_t y = v0[1];
    mfloat_t z = v0[2];
    mfloat_t w = v0[3];
    result[0] = m0[0] * x + m0[4] * y + m0[8] * z + m0[12] * w;
    result[1] = m0[1] * x + m0[5] * y + m0[9] * z + m0[13] * w;
    result[2] = m0[2] * x + m0[6] * y + m0[10] * z + m0[14] * w;
    result[3] = m0[3] * x + m0[7] * y + m0[11] * z + m0[15] * w;
    return result;
}

__host__ __device__ mfloat_t *vec4_divide(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = v0[0] / v1[0];
    result[1] = v0[1] / v1[1];
    result[2] = v0[2] / v1[2];
    result[3] = v0[3] / v1[3];
    return result;
}

__host__ __device__ mfloat_t *vec4_divide_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = v0[0] / f;
    result[1] = v0[1] / f;
    result[2] = v0[2] / f;
    result[3] = v0[3] / f;
    return result;
}

__host__ __device__ mfloat_t *vec4_snap(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = MFLOOR(v0[0] / v1[0]) * v1[0];
    result[1] = MFLOOR(v0[1] / v1[1]) * v1[1];
    result[2] = MFLOOR(v0[2] / v1[2]) * v1[2];
    result[3] = MFLOOR(v0[3] / v1[3]) * v1[3];
    return result;
}

__host__ __device__ mfloat_t *vec4_snap_f(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    result[0] = MFLOOR(v0[0] / f) * f;
    result[1] = MFLOOR(v0[1] / f) * f;
    result[2] = MFLOOR(v0[2] / f) * f;
    result[3] = MFLOOR(v0[3] / f) * f;
    return result;
}

__host__ __device__ mfloat_t *vec4_negative(mfloat_t *result, mfloat_t *v0) {
    result[0] = -v0[0];
    result[1] = -v0[1];
    result[2] = -v0[2];
    result[3] = -v0[3];
    return result;
}

__host__ __device__ mfloat_t *vec4_abs(mfloat_t *result, mfloat_t *v0) {
    result[0] = MFABS(v0[0]);
    result[1] = MFABS(v0[1]);
    result[2] = MFABS(v0[2]);
    result[3] = MFABS(v0[3]);
    return result;
}

__host__ __device__ mfloat_t *vec4_floor(mfloat_t *result, mfloat_t *v0) {
    result[0] = MFLOOR(v0[0]);
    result[1] = MFLOOR(v0[1]);
    result[2] = MFLOOR(v0[2]);
    result[3] = MFLOOR(v0[3]);
    return result;
}

__host__ __device__ mfloat_t *vec4_ceil(mfloat_t *result, mfloat_t *v0) {
    result[0] = MCEIL(v0[0]);
    result[1] = MCEIL(v0[1]);
    result[2] = MCEIL(v0[2]);
    result[3] = MCEIL(v0[3]);
    return result;
}

__host__ __device__ mfloat_t *vec4_round(mfloat_t *result, mfloat_t *v0) {
    result[0] = MROUND(v0[0]);
    result[1] = MROUND(v0[1]);
    result[2] = MROUND(v0[2]);
    result[3] = MROUND(v0[3]);
    return result;
}

__host__ __device__ mfloat_t *vec4_max(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = MFMAX(v0[0], v1[0]);
    result[1] = MFMAX(v0[1], v1[1]);
    result[2] = MFMAX(v0[2], v1[2]);
    result[3] = MFMAX(v0[3], v1[3]);
    return result;
}

__host__ __device__ mfloat_t *vec4_min(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    result[0] = MFMIN(v0[0], v1[0]);
    result[1] = MFMIN(v0[1], v1[1]);
    result[2] = MFMIN(v0[2], v1[2]);
    result[3] = MFMIN(v0[3], v1[3]);
    return result;
}

__host__ __device__ mfloat_t *vec4_clamp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2) {
    vec4_min(result, v0, v1);
    vec4_max(result, v0, v2);
    return result;
}

__host__ __device__ mfloat_t *vec4_normalize(mfloat_t *result, mfloat_t *v0) {
    mfloat_t l = MSQRT(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2] + v0[3] * v0[3]);
    result[0] = v0[0] / l;
    result[1] = v0[1] / l;
    result[2] = v0[2] / l;
    result[3] = v0[3] / l;
    return result;
}

__host__ __device__ mfloat_t *vec4_lerp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t f) {
    result[0] = v0[0] + (v1[0] - v0[0]) * f;
    result[1] = v0[1] + (v1[1] - v0[1]) * f;
    result[2] = v0[2] + (v1[2] - v0[2]) * f;
    result[3] = v0[3] + (v1[3] - v0[3]) * f;
    return result;
}

__host__ __device__ bool quat_is_zero(mfloat_t *q0) {
    return MFABS(q0[0]) < MFLT_EPSILON && MFABS(q0[1]) < MFLT_EPSILON && MFABS(q0[2]) < MFLT_EPSILON && MFABS(q0[3]) < MFLT_EPSILON;
}

__host__ __device__ bool quat_is_equal(mfloat_t *q0, mfloat_t *q1) {
    return MFABS(q0[0] - q1[0]) < MFLT_EPSILON && MFABS(q0[1] - q1[1]) < MFLT_EPSILON && MFABS(q0[2] - q1[2]) < MFLT_EPSILON && MFABS(q0[3] - q1[3]) < MFLT_EPSILON;
}

__host__ __device__ mfloat_t *quat(mfloat_t *result, mfloat_t x, mfloat_t y, mfloat_t z, mfloat_t w) {
    result[0] = x;
    result[1] = y;
    result[2] = z;
    result[3] = w;
    return result;
}

__host__ __device__ mfloat_t *quat_assign(mfloat_t *result, mfloat_t *q0) {
    result[0] = q0[0];
    result[1] = q0[1];
    result[2] = q0[2];
    result[3] = q0[3];
    return result;
}

__host__ __device__ mfloat_t *quat_zero(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *quat_null(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *quat_multiply(mfloat_t *result, mfloat_t *q0, mfloat_t *q1) {
    result[0] = q0[3] * q1[0] + q0[0] * q1[3] + q0[1] * q1[2] - q0[2] * q1[1];
    result[1] = q0[3] * q1[1] + q0[1] * q1[3] + q0[2] * q1[0] - q0[0] * q1[2];
    result[2] = q0[3] * q1[2] + q0[2] * q1[3] + q0[0] * q1[1] - q0[1] * q1[0];
    result[3] = q0[3] * q1[3] - q0[0] * q1[0] - q0[1] * q1[1] - q0[2] * q1[2];
    return result;
}

__host__ __device__ mfloat_t *quat_multiply_f(mfloat_t *result, mfloat_t *q0, mfloat_t f) {
    result[0] = q0[0] * f;
    result[1] = q0[1] * f;
    result[2] = q0[2] * f;
    result[3] = q0[3] * f;
    return result;
}

__host__ __device__ mfloat_t *quat_divide(mfloat_t *result, mfloat_t *q0, mfloat_t *q1) {
    mfloat_t x = q0[0];
    mfloat_t y = q0[1];
    mfloat_t z = q0[2];
    mfloat_t w = q0[3];
    mfloat_t ls = q1[0] * q1[0] + q1[1] * q1[1] + q1[8] * q1[8] + q1[3] * q1[3];
    mfloat_t normalized_x = -q1[0] / ls;
    mfloat_t normalized_y = -q1[1] / ls;
    mfloat_t normalized_z = -q1[8] / ls;
    mfloat_t normalized_w = q1[3] / ls;
    result[0] = x * normalized_w + normalized_x * w + (y * normalized_z - z * normalized_y);
    result[1] = y * normalized_w + normalized_y * w + (z * normalized_x - x * normalized_z);
    result[2] = z * normalized_w + normalized_z * w + (x * normalized_y - y * normalized_x);
    result[3] = w * normalized_w - (x * normalized_x + y * normalized_y + z * normalized_z);
    return result;
}

__host__ __device__ mfloat_t *quat_divide_f(mfloat_t *result, mfloat_t *q0, mfloat_t f) {
    result[0] = q0[0] / f;
    result[1] = q0[1] / f;
    result[2] = q0[2] / f;
    result[3] = q0[3] / f;
    return result;
}

__host__ __device__ mfloat_t *quat_negative(mfloat_t *result, mfloat_t *q0) {
    result[0] = -q0[0];
    result[1] = -q0[1];
    result[2] = -q0[2];
    result[3] = -q0[3];
    return result;
}

__host__ __device__ mfloat_t *quat_conjugate(mfloat_t *result, mfloat_t *q0) {
    result[0] = -q0[0];
    result[1] = -q0[1];
    result[2] = -q0[2];
    result[3] = q0[3];
    return result;
}

__host__ __device__ mfloat_t *quat_inverse(mfloat_t *result, mfloat_t *q0) {
    mfloat_t l = MFLOAT_C(1.0) / (q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3]);
    result[0] = -q0[0] * l;
    result[1] = -q0[1] * l;
    result[2] = -q0[2] * l;
    result[3] = q0[3] * l;
    return result;
}

__host__ __device__ mfloat_t *quat_normalize(mfloat_t *result, mfloat_t *q0) {
    mfloat_t l = MFLOAT_C(1.0) / MSQRT(q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3]);
    result[0] = q0[0] * l;
    result[1] = q0[1] * l;
    result[2] = q0[2] * l;
    result[3] = q0[3] * l;
    return result;
}

__host__ __device__ mfloat_t quat_dot(mfloat_t *q0, mfloat_t *q1) {
    return q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
}

__host__ __device__ mfloat_t *quat_power(mfloat_t *result, mfloat_t *q0, mfloat_t exponent) {
    if (MFABS(q0[3]) < MFLOAT_C(1.0) - MFLT_EPSILON) {
        mfloat_t alpha = MACOS(q0[3]);
        mfloat_t new_alpha = alpha * exponent;
        mfloat_t s = MSIN(new_alpha) / MSIN(alpha);
        result[0] = result[0] * s;
        result[1] = result[1] * s;
        result[2] = result[2] * s;
        result[3] = MCOS(new_alpha);
    } else {
        result[0] = q0[0];
        result[1] = q0[1];
        result[2] = q0[1];
        result[3] = q0[3];
    }
    return result;
}

__host__ __device__ mfloat_t *quat_from_axis_angle(mfloat_t *result, mfloat_t *v0, mfloat_t angle) {
    mfloat_t half = angle * MFLOAT_C(0.5);
    mfloat_t s = MSIN(half);
    result[0] = v0[0] * s;
    result[1] = v0[1] * s;
    result[2] = v0[2] * s;
    result[3] = MCOS(half);
    return result;
}

__host__ __device__ mfloat_t *quat_from_vec3(mfloat_t *result, mfloat_t *v0, mfloat_t *v1) {
    mfloat_t cross[VEC3_SIZE];
    mfloat_t d = vec3_dot(v0, v1);
    mfloat_t a_ls = vec3_length_squared(v0);
    mfloat_t b_ls = vec3_length_squared(v0);
    vec3_cross(cross, v0, v1);
    quat(result, cross[0], cross[1], cross[1], d + MSQRT(a_ls * b_ls));
    quat_normalize(result, result);
    return result;
}

__host__ __device__ mfloat_t *quat_from_mat4(mfloat_t *result, mfloat_t *m0) {
    mfloat_t scale = m0[0] + m0[5] + m0[10];
    if (scale > MFLOAT_C(0.0)) {
        mfloat_t sr = MSQRT(scale + MFLOAT_C(1.0));
        result[3] = sr * MFLOAT_C(0.5);
        sr = MFLOAT_C(0.5) / sr;
        result[0] = (m0[9] - m0[6]) * sr;
        result[1] = (m0[2] - m0[8]) * sr;
        result[2] = (m0[4] - m0[1]) * sr;
    } else if ((m0[0] >= m0[5]) && (m0[0] >= m0[10])) {
        mfloat_t sr = MSQRT(MFLOAT_C(1.0) + m0[0] - m0[5] - m0[10]);
        mfloat_t half = MFLOAT_C(0.5) / sr;
        result[0] = MFLOAT_C(0.5) * sr;
        result[1] = (m0[4] + m0[1]) * half;
        result[2] = (m0[8] + m0[2]) * half;
        result[3] = (m0[9] - m0[6]) * half;
    } else if (m0[5] > m0[10]) {
        mfloat_t sr = MSQRT(MFLOAT_C(1.0) + m0[5] - m0[0] - m0[10]);
        mfloat_t half = MFLOAT_C(0.5) / sr;
        result[0] = (m0[1] + m0[4]) * half;
        result[1] = MFLOAT_C(0.5) * sr;
        result[2] = (m0[6] + m0[9]) * half;
        result[3] = (m0[2] - m0[8]) * half;
    } else {
        mfloat_t sr = MSQRT(MFLOAT_C(1.0) + m0[10] - m0[0] - m0[5]);
        mfloat_t half = MFLOAT_C(0.5) / sr;
        result[0] = (m0[2] + m0[8]) * half;
        result[1] = (m0[6] + m0[9]) * half;
        result[2] = MFLOAT_C(0.5) * sr;
        result[3] = (m0[4] - m0[1]) * half;
    }
    return result;
}

__host__ __device__ mfloat_t *quat_lerp(mfloat_t *result, mfloat_t *q0, mfloat_t *q1, mfloat_t f) {
    result[0] = q0[0] + (q1[0] - q0[0]) * f;
    result[1] = q0[1] + (q1[1] - q0[1]) * f;
    result[2] = q0[2] + (q1[2] - q0[2]) * f;
    result[3] = q0[3] + (q1[3] - q0[3]) * f;
    return result;
}

__host__ __device__ mfloat_t *quat_slerp(mfloat_t *result, mfloat_t *q0, mfloat_t *q1, mfloat_t f) {
    mfloat_t tmp1[QUAT_SIZE];
    mfloat_t d = quat_dot(q0, q1);
    mfloat_t f0;
    mfloat_t f1;
    quat_assign(tmp1, q1);
    if (d < MFLOAT_C(0.0)) {
        quat_negative(tmp1, tmp1);
        d = -d;
    }
    if (d > MFLOAT_C(0.9995)) {
        f0 = MFLOAT_C(1.0) - f;
        f1 = f;
    } else {
        mfloat_t theta = MACOS(d);
        mfloat_t sin_theta = MSIN(theta);
        f0 = MSIN((MFLOAT_C(1.0) - f) * theta) / sin_theta;
        f1 = MSIN(f * theta) / sin_theta;
    }
    result[0] = q0[0] * f0 + tmp1[0] * f1;
    result[1] = q0[1] * f0 + tmp1[1] * f1;
    result[2] = q0[2] * f0 + tmp1[2] * f1;
    result[3] = q0[3] * f0 + tmp1[3] * f1;
    return result;
}

__host__ __device__ mfloat_t quat_length(mfloat_t *q0) {
    return MSQRT(q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3]);
}

__host__ __device__ mfloat_t quat_length_squared(mfloat_t *q0) {
    return q0[0] * q0[0] + q0[1] * q0[1] + q0[2] * q0[2] + q0[3] * q0[3];
}

__host__ __device__ mfloat_t quat_angle(mfloat_t *q0, mfloat_t *q1) {
    mfloat_t s = MSQRT(quat_length_squared(q0) * quat_length_squared(q1));
    s = MFLOAT_C(1.0) / s;
    return MACOS(quat_dot(q0, q1) * s);
}

__host__ __device__ mfloat_t *mat2(mfloat_t *result, mfloat_t m11, mfloat_t m12, mfloat_t m21, mfloat_t m22) {
    result[0] = m11;
    result[1] = m21;
    result[2] = m12;
    result[3] = m22;
    return result;
}

__host__ __device__ mfloat_t *mat2_zero(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *mat2_identity(mfloat_t *result) {
    result[0] = MFLOAT_C(1.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t mat2_determinant(mfloat_t *m0) {
    return m0[0] * m0[3] - m0[2] * m0[1];
}

__host__ __device__ mfloat_t *mat2_assign(mfloat_t *result, mfloat_t *m0) {
    result[0] = m0[0];
    result[1] = m0[1];
    result[2] = m0[2];
    result[3] = m0[3];
    return result;
}

__host__ __device__ mfloat_t *mat2_negative(mfloat_t *result, mfloat_t *m0) {
    result[0] = -m0[0];
    result[1] = -m0[1];
    result[2] = -m0[2];
    result[3] = -m0[3];
    return result;
}

__host__ __device__ mfloat_t *mat2_transpose(mfloat_t *result, mfloat_t *m0) {
    mfloat_t transposed[MAT2_SIZE];
    transposed[0] = m0[0];
    transposed[1] = m0[2];
    transposed[2] = m0[1];
    transposed[3] = m0[3];
    result[0] = transposed[0];
    result[1] = transposed[1];
    result[2] = transposed[2];
    result[3] = transposed[3];
    return result;
}

__host__ __device__ mfloat_t *mat2_cofactor(mfloat_t *result, mfloat_t *m0) {
    mfloat_t cofactor[MAT2_SIZE];
    cofactor[0] = m0[3];
    cofactor[1] = -m0[2];
    cofactor[2] = -m0[1];
    cofactor[3] = m0[0];
    result[0] = cofactor[0];
    result[1] = cofactor[1];
    result[2] = cofactor[2];
    result[3] = cofactor[3];
    return result;
}

__host__ __device__ mfloat_t *mat2_adjugate(mfloat_t *result, mfloat_t *m0) {
    mfloat_t adjugate[MAT2_SIZE];
    adjugate[0] = m0[3];
    adjugate[1] = -m0[1];
    adjugate[2] = -m0[2];
    adjugate[3] = m0[0];
    result[0] = adjugate[0];
    result[1] = adjugate[1];
    result[2] = adjugate[2];
    result[3] = adjugate[3];
    return result;
}

__host__ __device__ mfloat_t *mat2_multiply(mfloat_t *result, mfloat_t *m0, mfloat_t *m1) {
    mfloat_t multiplied[MAT3_SIZE];
    multiplied[0] = m0[0] * m1[0] + m0[2] * m1[1];
    multiplied[1] = m0[1] * m1[0] + m0[3] * m1[1];
    multiplied[2] = m0[0] * m1[2] + m0[2] * m1[3];
    multiplied[3] = m0[1] * m1[2] + m0[3] * m1[3];
    result[0] = multiplied[0];
    result[1] = multiplied[1];
    result[2] = multiplied[2];
    result[3] = multiplied[3];
    return result;
}

__host__ __device__ mfloat_t *mat2_multiply_f(mfloat_t *result, mfloat_t *m0, mfloat_t f) {
    result[0] = m0[0] * f;
    result[1] = m0[1] * f;
    result[2] = m0[2] * f;
    result[3] = m0[3] * f;
    return result;
}

__host__ __device__ mfloat_t *mat2_inverse(mfloat_t *result, mfloat_t *m0) {
    mfloat_t inverse[MAT2_SIZE];
    mfloat_t det = mat2_determinant(m0);
    mat2_cofactor(inverse, m0);
    mat2_multiply_f(inverse, inverse, MFLOAT_C(1.0) / det);
    result[0] = inverse[0];
    result[1] = inverse[1];
    result[2] = inverse[2];
    result[3] = inverse[3];
    return result;
}

__host__ __device__ mfloat_t *mat2_scaling(mfloat_t *result, mfloat_t *v0) {
    result[0] = v0[0];
    result[3] = v0[1];
    return result;
}

__host__ __device__ mfloat_t *mat2_scale(mfloat_t *result, mfloat_t *m0, mfloat_t *v0) {
    result[0] = m0[0] * v0[0];
    result[3] = m0[3] * v0[1];
    return result;
}

__host__ __device__ mfloat_t *mat2_rotation_z(mfloat_t *result, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    result[0] = c;
    result[1] = s;
    result[2] = -s;
    result[3] = c;
    return result;
}

__host__ __device__ mfloat_t *mat2_lerp(mfloat_t *result, mfloat_t *m0, mfloat_t *m1, mfloat_t f) {
    result[0] = m0[0] + (m1[0] - m0[0]) * f;
    result[1] = m0[1] + (m1[1] - m0[1]) * f;
    result[2] = m0[2] + (m1[2] - m0[2]) * f;
    result[3] = m0[3] + (m1[3] - m0[3]) * f;
    return result;
}

__host__ __device__ mfloat_t *mat3(mfloat_t *result, mfloat_t m11, mfloat_t m12, mfloat_t m13, mfloat_t m21, mfloat_t m22, mfloat_t m23, mfloat_t m31, mfloat_t m32, mfloat_t m33) {
    result[0] = m11;
    result[1] = m21;
    result[2] = m31;
    result[3] = m12;
    result[4] = m22;
    result[5] = m32;
    result[6] = m13;
    result[7] = m23;
    result[8] = m33;
    return result;
}

__host__ __device__ mfloat_t *mat3_zero(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(0.0);
    result[5] = MFLOAT_C(0.0);
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *mat3_identity(mfloat_t *result) {
    result[0] = MFLOAT_C(1.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(1.0);
    result[5] = MFLOAT_C(0.0);
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t mat3_determinant(mfloat_t *m0) {
    mfloat_t m11 = m0[0];
    mfloat_t m21 = m0[1];
    mfloat_t m31 = m0[2];
    mfloat_t m12 = m0[3];
    mfloat_t m22 = m0[4];
    mfloat_t m32 = m0[5];
    mfloat_t m13 = m0[6];
    mfloat_t m23 = m0[7];
    mfloat_t m33 = m0[8];
    mfloat_t determinant = m11 * m22 * m33 + m12 * m23 * m31 + m13 * m21 * m32 - m11 * m23 * m32 - m12 * m21 * m33 - m13 * m22 * m31;
    return determinant;
}

__host__ __device__ mfloat_t *mat3_assign(mfloat_t *result, mfloat_t *m0) {
    result[0] = m0[0];
    result[1] = m0[1];
    result[2] = m0[2];
    result[3] = m0[3];
    result[4] = m0[4];
    result[5] = m0[5];
    result[6] = m0[6];
    result[7] = m0[7];
    result[8] = m0[8];
    return result;
}

__host__ __device__ mfloat_t *mat3_negative(mfloat_t *result, mfloat_t *m0) {
    result[0] = -m0[0];
    result[1] = -m0[1];
    result[2] = -m0[2];
    result[3] = -m0[3];
    result[4] = -m0[4];
    result[5] = -m0[5];
    result[6] = -m0[6];
    result[7] = -m0[7];
    result[8] = -m0[8];
    return result;
}

__host__ __device__ mfloat_t *mat3_transpose(mfloat_t *result, mfloat_t *m0) {
    mfloat_t transposed[MAT4_SIZE];
    transposed[0] = m0[0];
    transposed[1] = m0[3];
    transposed[2] = m0[6];
    transposed[3] = m0[1];
    transposed[4] = m0[4];
    transposed[5] = m0[7];
    transposed[6] = m0[2];
    transposed[7] = m0[5];
    transposed[8] = m0[8];
    result[0] = transposed[0];
    result[1] = transposed[1];
    result[2] = transposed[2];
    result[3] = transposed[3];
    result[4] = transposed[4];
    result[5] = transposed[5];
    result[6] = transposed[6];
    result[7] = transposed[7];
    result[8] = transposed[8];
    return result;
}

__host__ __device__ mfloat_t *mat3_cofactor(mfloat_t *result, mfloat_t *m0) {
    mfloat_t cofactor[MAT3_SIZE];
    mfloat_t minor[MAT2_SIZE];
    minor[0] = m0[4];
    minor[1] = m0[5];
    minor[2] = m0[7];
    minor[3] = m0[8];
    cofactor[0] = mat2_determinant(minor);
    minor[0] = m0[3];
    minor[1] = m0[5];
    minor[2] = m0[6];
    minor[3] = m0[8];
    cofactor[1] = -mat2_determinant(minor);
    minor[0] = m0[3];
    minor[1] = m0[4];
    minor[2] = m0[6];
    minor[3] = m0[7];
    cofactor[2] = mat2_determinant(minor);
    minor[0] = m0[1];
    minor[1] = m0[2];
    minor[2] = m0[7];
    minor[3] = m0[8];
    cofactor[3] = -mat2_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[2];
    minor[2] = m0[6];
    minor[3] = m0[8];
    cofactor[4] = mat2_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[6];
    minor[3] = m0[7];
    cofactor[5] = -mat2_determinant(minor);
    minor[0] = m0[1];
    minor[1] = m0[2];
    minor[2] = m0[4];
    minor[3] = m0[5];
    cofactor[6] = mat2_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[2];
    minor[2] = m0[3];
    minor[3] = m0[5];
    cofactor[7] = -mat2_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[3];
    minor[3] = m0[4];
    cofactor[8] = mat2_determinant(minor);
    result[0] = cofactor[0];
    result[1] = cofactor[1];
    result[2] = cofactor[2];
    result[3] = cofactor[3];
    result[4] = cofactor[4];
    result[5] = cofactor[5];
    result[6] = cofactor[6];
    result[7] = cofactor[7];
    result[8] = cofactor[8];
    return result;
}

__host__ __device__ mfloat_t *mat3_multiply(mfloat_t *result, mfloat_t *m0, mfloat_t *m1) {
    mfloat_t multiplied[MAT3_SIZE];
    multiplied[0] = m0[0] * m1[0] + m0[3] * m1[1] + m0[6] * m1[2];
    multiplied[1] = m0[1] * m1[0] + m0[4] * m1[1] + m0[7] * m1[2];
    multiplied[2] = m0[2] * m1[0] + m0[5] * m1[1] + m0[8] * m1[2];
    multiplied[3] = m0[0] * m1[3] + m0[3] * m1[4] + m0[6] * m1[5];
    multiplied[4] = m0[1] * m1[3] + m0[4] * m1[4] + m0[7] * m1[5];
    multiplied[5] = m0[2] * m1[3] + m0[5] * m1[4] + m0[8] * m1[5];
    multiplied[6] = m0[0] * m1[6] + m0[3] * m1[7] + m0[6] * m1[8];
    multiplied[7] = m0[1] * m1[6] + m0[4] * m1[7] + m0[7] * m1[8];
    multiplied[8] = m0[2] * m1[6] + m0[5] * m1[7] + m0[8] * m1[8];
    result[0] = multiplied[0];
    result[1] = multiplied[1];
    result[2] = multiplied[2];
    result[3] = multiplied[3];
    result[4] = multiplied[4];
    result[5] = multiplied[5];
    result[6] = multiplied[6];
    result[7] = multiplied[7];
    result[8] = multiplied[8];
    return result;
}

__host__ __device__ mfloat_t *mat3_multiply_f(mfloat_t *result, mfloat_t *m0, mfloat_t f) {
    result[0] = m0[0] * f;
    result[1] = m0[1] * f;
    result[2] = m0[2] * f;
    result[3] = m0[3] * f;
    result[4] = m0[4] * f;
    result[5] = m0[5] * f;
    result[6] = m0[6] * f;
    result[7] = m0[7] * f;
    result[8] = m0[8] * f;
    return result;
}

__host__ __device__ mfloat_t *mat3_inverse(mfloat_t *result, mfloat_t *m0) {
    result = m0;
    return result;
}

__host__ __device__ mfloat_t *mat3_scaling(mfloat_t *result, mfloat_t *v0) {
    result[0] = v0[0];
    result[4] = v0[1];
    result[8] = v0[2];
    return result;
}

__host__ __device__ mfloat_t *mat3_scale(mfloat_t *result, mfloat_t *m0, mfloat_t *v0) {
    result[0] = m0[0] * v0[0];
    result[4] = m0[4] * v0[1];
    result[8] = m0[8] * v0[2];
    return result;
}

__host__ __device__ mfloat_t *mat3_rotation_x(mfloat_t *result, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    result[4] = c;
    result[5] = s;
    result[7] = -s;
    result[8] = c;
    return result;
}

__host__ __device__ mfloat_t *mat3_rotation_y(mfloat_t *result, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    result[0] = c;
    result[2] = -s;
    result[6] = s;
    result[8] = c;
    return result;
}

__host__ __device__ mfloat_t *mat3_rotation_z(mfloat_t *result, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    result[0] = c;
    result[1] = s;
    result[3] = -s;
    result[4] = c;
    return result;
}

__host__ __device__ mfloat_t *mat3_rotation_axis(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    mfloat_t one_c = MFLOAT_C(1.0) - c;
    mfloat_t x = v0[0];
    mfloat_t y = v0[4];
    mfloat_t z = v0[8];
    mfloat_t xx = x * x;
    mfloat_t xy = x * y;
    mfloat_t xz = x * z;
    mfloat_t yy = y * y;
    mfloat_t yz = y * z;
    mfloat_t zz = z * z;
    mfloat_t l = xx + yy + zz;
    mfloat_t sqrt_l = MSQRT(l);
    result[0] = (xx + (yy + zz) * c) / l;
    result[1] = (xy * one_c + v0[2] * sqrt_l * s) / l;
    result[2] = (xz * one_c - v0[1] * sqrt_l * s) / l;
    result[3] = (xy * one_c - v0[2] * sqrt_l * s) / l;
    result[4] = (yy + (xx + zz) * c) / l;
    result[5] = (yz * one_c + v0[0] * sqrt_l * s) / l;
    result[6] = (xz * one_c + v0[1] * sqrt_l * s) / l;
    result[7] = (yz * one_c - v0[0] * sqrt_l * s) / l;
    result[8] = (zz + (xx + yy) * c) / l;
    return result;
}

__host__ __device__ mfloat_t *mat3_rotation_quat(mfloat_t *result, mfloat_t *q0) {
    mfloat_t xx = q0[0] * q0[0];
    mfloat_t yy = q0[1] * q0[1];
    mfloat_t zz = q0[2] * q0[2];
    mfloat_t xy = q0[0] * q0[1];
    mfloat_t zw = q0[2] * q0[3];
    mfloat_t xz = q0[8] * q0[0];
    mfloat_t yw = q0[1] * q0[3];
    mfloat_t yz = q0[1] * q0[2];
    mfloat_t xw = q0[0] * q0[3];
    result[0] = MFLOAT_C(1.0) - MFLOAT_C(2.0) * (yy - zz);
    result[1] = MFLOAT_C(2.0) * (xy + zw);
    result[2] = MFLOAT_C(2.0) * (xz - yw);
    result[3] = MFLOAT_C(2.0) * (xy - zw);
    result[4] = MFLOAT_C(1.0) - MFLOAT_C(2.0) * (xx - zz);
    result[5] = MFLOAT_C(2.0) * (yz + xw);
    result[6] = MFLOAT_C(2.0) * (xz + yw);
    result[7] = MFLOAT_C(2.0) * (yz - xw);
    result[8] = MFLOAT_C(1.0) - MFLOAT_C(2.0) * (xx - yy);
    return result;
}

__host__ __device__ mfloat_t *mat3_lerp(mfloat_t *result, mfloat_t *m0, mfloat_t *m1, mfloat_t f) {
    result[0] = m0[0] + (m1[0] - m0[0]) * f;
    result[1] = m0[1] + (m1[1] - m0[1]) * f;
    result[2] = m0[2] + (m1[2] - m0[2]) * f;
    result[3] = m0[3] + (m1[3] - m0[3]) * f;
    result[4] = m0[4] + (m1[4] - m0[4]) * f;
    result[5] = m0[5] + (m1[5] - m0[5]) * f;
    result[6] = m0[6] + (m1[6] - m0[6]) * f;
    result[7] = m0[7] + (m1[7] - m0[7]) * f;
    result[8] = m0[8] + (m1[8] - m0[8]) * f;
    return result;
}

__host__ __device__ mfloat_t *mat4(mfloat_t *result, mfloat_t m11, mfloat_t m12, mfloat_t m13, mfloat_t m14, mfloat_t m21, mfloat_t m22, mfloat_t m23, mfloat_t m24, mfloat_t m31, mfloat_t m32, mfloat_t m33, mfloat_t m34, mfloat_t m41, mfloat_t m42, mfloat_t m43, mfloat_t m44) {
    result[0] = m11;
    result[1] = m21;
    result[2] = m31;
    result[3] = m41;
    result[4] = m12;
    result[5] = m22;
    result[6] = m32;
    result[7] = m42;
    result[8] = m13;
    result[9] = m23;
    result[10] = m33;
    result[11] = m43;
    result[12] = m14;
    result[13] = m24;
    result[14] = m34;
    result[15] = m44;
    return result;
}

__host__ __device__ mfloat_t *mat4_zero(mfloat_t *result) {
    result[0] = MFLOAT_C(0.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(0.0);
    result[5] = MFLOAT_C(0.0);
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(0.0);
    result[9] = MFLOAT_C(0.0);
    result[10] = MFLOAT_C(0.0);
    result[11] = MFLOAT_C(0.0);
    result[12] = MFLOAT_C(0.0);
    result[13] = MFLOAT_C(0.0);
    result[14] = MFLOAT_C(0.0);
    result[15] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *mat4_identity(mfloat_t *result) {
    result[0] = MFLOAT_C(1.0);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(0.0);
    result[5] = MFLOAT_C(1.0);
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(0.0);
    result[9] = MFLOAT_C(0.0);
    result[10] = MFLOAT_C(1.0);
    result[11] = MFLOAT_C(0.0);
    result[12] = MFLOAT_C(0.0);
    result[13] = MFLOAT_C(0.0);
    result[14] = MFLOAT_C(0.0);
    result[15] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t mat4_determinant(mfloat_t *m0) {
    mfloat_t m11 = m0[0];
    mfloat_t m21 = m0[1];
    mfloat_t m31 = m0[2];
    mfloat_t m41 = m0[3];
    mfloat_t m12 = m0[4];
    mfloat_t m22 = m0[5];
    mfloat_t m32 = m0[6];
    mfloat_t m42 = m0[7];
    mfloat_t m13 = m0[8];
    mfloat_t m23 = m0[9];
    mfloat_t m33 = m0[10];
    mfloat_t m43 = m0[11];
    mfloat_t m14 = m0[12];
    mfloat_t m24 = m0[13];
    mfloat_t m34 = m0[14];
    mfloat_t m44 = m0[15];
    mfloat_t determinant = m14 * m23 * m32 * m41 - m13 * m24 * m32 * m41 - m14 * m22 * m33 * m41 + m12 * m24 * m33 * m41 + m13 * m22 * m34 * m41 - m12 * m23 * m34 * m41 - m14 * m23 * m31 * m42 + m13 * m24 * m31 * m42 + m14 * m21 * m33 * m42 - m11 * m24 * m33 * m42 - m13 * m21 * m34 * m42 + m11 * m23 * m34 * m42 + m14 * m22 * m31 * m43 - m12 * m24 * m31 * m43 - m14 * m21 * m32 * m43 + m11 * m24 * m32 * m43 + m12 * m21 * m34 * m43 - m11 * m22 * m34 * m43 - m13 * m22 * m31 * m44 + m12 * m23 * m31 * m44 + m13 * m21 * m32 * m44 - m11 * m23 * m32 * m44 - m12 * m21 * m33 * m44 + m11 * m22 * m33 * m44;
    return determinant;
}

__host__ __device__ mfloat_t *mat4_assign(mfloat_t *result, mfloat_t *m0) {
    result[0] = m0[0];
    result[1] = m0[1];
    result[2] = m0[2];
    result[3] = m0[3];
    result[4] = m0[4];
    result[5] = m0[5];
    result[6] = m0[6];
    result[7] = m0[7];
    result[8] = m0[8];
    result[9] = m0[9];
    result[10] = m0[10];
    result[11] = m0[11];
    result[12] = m0[12];
    result[13] = m0[13];
    result[14] = m0[14];
    result[15] = m0[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_negative(mfloat_t *result, mfloat_t *m0) {
    result[0] = -m0[0];
    result[1] = -m0[1];
    result[2] = -m0[2];
    result[3] = -m0[3];
    result[4] = -m0[4];
    result[5] = -m0[5];
    result[6] = -m0[6];
    result[7] = -m0[7];
    result[8] = -m0[8];
    result[9] = -m0[9];
    result[10] = -m0[10];
    result[11] = -m0[11];
    result[12] = -m0[12];
    result[13] = -m0[13];
    result[14] = -m0[14];
    result[15] = -m0[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_transpose(mfloat_t *result, mfloat_t *m0) {
    mfloat_t transposed[MAT4_SIZE];
    transposed[0] = m0[0];
    transposed[1] = m0[4];
    transposed[2] = m0[8];
    transposed[3] = m0[12];
    transposed[4] = m0[1];
    transposed[5] = m0[5];
    transposed[6] = m0[9];
    transposed[7] = m0[13];
    transposed[8] = m0[2];
    transposed[9] = m0[6];
    transposed[10] = m0[10];
    transposed[11] = m0[14];
    transposed[12] = m0[3];
    transposed[13] = m0[7];
    transposed[14] = m0[11];
    transposed[15] = m0[15];
    result[0] = transposed[0];
    result[1] = transposed[1];
    result[2] = transposed[2];
    result[3] = transposed[3];
    result[4] = transposed[4];
    result[5] = transposed[5];
    result[6] = transposed[6];
    result[7] = transposed[7];
    result[8] = transposed[8];
    result[9] = transposed[9];
    result[10] = transposed[10];
    result[11] = transposed[11];
    result[12] = transposed[12];
    result[13] = transposed[13];
    result[14] = transposed[14];
    result[15] = transposed[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_cofactor(mfloat_t *result, mfloat_t *m0) {
    mfloat_t cofactor[MAT4_SIZE];
    mfloat_t minor[MAT3_SIZE];
    minor[0] = m0[5];
    minor[1] = m0[6];
    minor[2] = m0[7];
    minor[3] = m0[9];
    minor[4] = m0[10];
    minor[5] = m0[11];
    minor[6] = m0[13];
    minor[7] = m0[14];
    minor[8] = m0[15];
    cofactor[0] = mat3_determinant(minor);
    minor[0] = m0[4];
    minor[1] = m0[6];
    minor[2] = m0[7];
    minor[3] = m0[8];
    minor[4] = m0[10];
    minor[5] = m0[11];
    minor[6] = m0[12];
    minor[7] = m0[14];
    minor[8] = m0[15];
    cofactor[1] = -mat3_determinant(minor);
    minor[0] = m0[4];
    minor[1] = m0[5];
    minor[2] = m0[7];
    minor[3] = m0[8];
    minor[4] = m0[9];
    minor[5] = m0[11];
    minor[6] = m0[12];
    minor[7] = m0[13];
    minor[8] = m0[15];
    cofactor[2] = mat3_determinant(minor);
    minor[0] = m0[4];
    minor[1] = m0[5];
    minor[2] = m0[6];
    minor[3] = m0[8];
    minor[4] = m0[9];
    minor[5] = m0[10];
    minor[6] = m0[12];
    minor[7] = m0[13];
    minor[8] = m0[14];
    cofactor[3] = -mat3_determinant(minor);
    minor[0] = m0[1];
    minor[1] = m0[2];
    minor[2] = m0[3];
    minor[3] = m0[9];
    minor[4] = m0[10];
    minor[5] = m0[11];
    minor[6] = m0[13];
    minor[7] = m0[14];
    minor[8] = m0[15];
    cofactor[4] = -mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[2];
    minor[2] = m0[3];
    minor[3] = m0[8];
    minor[4] = m0[10];
    minor[5] = m0[11];
    minor[6] = m0[12];
    minor[7] = m0[14];
    minor[8] = m0[15];
    cofactor[5] = mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[3];
    minor[3] = m0[8];
    minor[4] = m0[9];
    minor[5] = m0[11];
    minor[6] = m0[12];
    minor[7] = m0[13];
    minor[8] = m0[15];
    cofactor[6] = -mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[2];
    minor[3] = m0[8];
    minor[4] = m0[9];
    minor[5] = m0[10];
    minor[6] = m0[12];
    minor[7] = m0[13];
    minor[8] = m0[14];
    cofactor[7] = mat3_determinant(minor);
    minor[0] = m0[1];
    minor[1] = m0[2];
    minor[2] = m0[3];
    minor[3] = m0[5];
    minor[4] = m0[6];
    minor[5] = m0[7];
    minor[6] = m0[13];
    minor[7] = m0[14];
    minor[8] = m0[15];
    cofactor[8] = mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[2];
    minor[2] = m0[3];
    minor[3] = m0[4];
    minor[4] = m0[6];
    minor[5] = m0[7];
    minor[6] = m0[12];
    minor[7] = m0[14];
    minor[8] = m0[15];
    cofactor[9] = -mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[3];
    minor[3] = m0[4];
    minor[4] = m0[5];
    minor[5] = m0[7];
    minor[6] = m0[12];
    minor[7] = m0[13];
    minor[8] = m0[15];
    cofactor[10] = mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[2];
    minor[3] = m0[4];
    minor[4] = m0[5];
    minor[5] = m0[6];
    minor[6] = m0[12];
    minor[7] = m0[13];
    minor[8] = m0[14];
    cofactor[11] = -mat3_determinant(minor);
    minor[0] = m0[1];
    minor[1] = m0[2];
    minor[2] = m0[3];
    minor[3] = m0[5];
    minor[4] = m0[6];
    minor[5] = m0[7];
    minor[6] = m0[9];
    minor[7] = m0[10];
    minor[8] = m0[11];
    cofactor[12] = -mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[2];
    minor[2] = m0[3];
    minor[3] = m0[4];
    minor[4] = m0[6];
    minor[5] = m0[7];
    minor[6] = m0[8];
    minor[7] = m0[10];
    minor[8] = m0[11];
    cofactor[13] = mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[3];
    minor[3] = m0[4];
    minor[4] = m0[5];
    minor[5] = m0[7];
    minor[6] = m0[8];
    minor[7] = m0[9];
    minor[8] = m0[11];
    cofactor[14] = -mat3_determinant(minor);
    minor[0] = m0[0];
    minor[1] = m0[1];
    minor[2] = m0[2];
    minor[3] = m0[4];
    minor[4] = m0[5];
    minor[5] = m0[6];
    minor[6] = m0[8];
    minor[7] = m0[9];
    minor[8] = m0[10];
    cofactor[15] = mat3_determinant(minor);
    result[0] = cofactor[0];
    result[1] = cofactor[1];
    result[2] = cofactor[2];
    result[3] = cofactor[3];
    result[4] = cofactor[4];
    result[5] = cofactor[5];
    result[6] = cofactor[6];
    result[7] = cofactor[7];
    result[8] = cofactor[8];
    result[9] = cofactor[9];
    result[10] = cofactor[10];
    result[11] = cofactor[11];
    result[12] = cofactor[12];
    result[13] = cofactor[13];
    result[14] = cofactor[14];
    result[15] = cofactor[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_rotation_x(mfloat_t *result, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    result[5] = c;
    result[6] = s;
    result[9] = -s;
    result[10] = c;
    return result;
}

__host__ __device__ mfloat_t *mat4_rotation_y(mfloat_t *result, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    result[0] = c;
    result[2] = -s;
    result[8] = s;
    result[10] = c;
    return result;
}

__host__ __device__ mfloat_t *mat4_rotation_z(mfloat_t *result, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    result[0] = c;
    result[1] = s;
    result[4] = -s;
    result[5] = c;
    return result;
}

__host__ __device__ mfloat_t *mat4_rotation_axis(mfloat_t *result, mfloat_t *v0, mfloat_t f) {
    mfloat_t c = MCOS(f);
    mfloat_t s = MSIN(f);
    mfloat_t one_c = MFLOAT_C(1.0) - c;
    mfloat_t x = v0[0];
    mfloat_t y = v0[1];
    mfloat_t z = v0[2];
    mfloat_t xx = x * x;
    mfloat_t xy = x * y;
    mfloat_t xz = x * z;
    mfloat_t yy = y * y;
    mfloat_t yz = y * z;
    mfloat_t zz = z * z;
    mfloat_t l = xx + yy + zz;
    mfloat_t sqrt_l = MSQRT(l);
    result[0] = (xx + (yy + zz) * c) / l;
    result[1] = (xy * one_c + v0[2] * sqrt_l * s) / l;
    result[2] = (xz * one_c - v0[1] * sqrt_l * s) / l;
    result[3] = MFLOAT_C(0.0);
    result[4] = (xy * one_c - v0[2] * sqrt_l * s) / l;
    result[5] = (yy + (xx + zz) * c) / l;
    result[6] = (yz * one_c + v0[0] * sqrt_l * s) / l;
    result[7] = MFLOAT_C(0.0);
    result[8] = (xz * one_c + v0[1] * sqrt_l * s) / l;
    result[9] = (yz * one_c - v0[0] * sqrt_l * s) / l;
    result[10] = (zz + (xx + yy) * c) / l;
    result[11] = MFLOAT_C(0.0);
    result[12] = MFLOAT_C(0.0);
    result[13] = MFLOAT_C(0.0);
    result[14] = MFLOAT_C(0.0);
    result[15] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *mat4_rotation_quat(mfloat_t *result, mfloat_t *q0) {
    mfloat_t xx = q0[0] * q0[0];
    mfloat_t yy = q0[1] * q0[1];
    mfloat_t zz = q0[2] * q0[2];
    mfloat_t xy = q0[0] * q0[1];
    mfloat_t zw = q0[2] * q0[3];
    mfloat_t xz = q0[0] * q0[2];
    mfloat_t yw = q0[1] * q0[3];
    mfloat_t yz = q0[1] * q0[2];
    mfloat_t xw = q0[0] * q0[3];
    result[0] = MFLOAT_C(1.0) - MFLOAT_C(2.0) * (yy + zz);
    result[1] = MFLOAT_C(2.0) * (xy + zw);
    result[2] = MFLOAT_C(2.0) * (xz - yw);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(2.0) * (xy - zw);
    result[5] = MFLOAT_C(1.0) - MFLOAT_C(2.0) * (xx + zz);
    result[6] = MFLOAT_C(2.0) * (yz + xw);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(2.0) * (xz + yw);
    result[9] = MFLOAT_C(2.0) * (yz - xw);
    result[10] = MFLOAT_C(1.0) - MFLOAT_C(2.0) * (xx + yy);
    result[11] = MFLOAT_C(0.0);
    result[12] = MFLOAT_C(0.0);
    result[13] = MFLOAT_C(0.0);
    result[14] = MFLOAT_C(0.0);
    result[15] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *mat4_translation(mfloat_t *result, mfloat_t *m0, mfloat_t *v0) {
    result[0] = m0[0];
    result[1] = m0[1];
    result[2] = m0[2];
    result[3] = m0[3];
    result[4] = m0[4];
    result[5] = m0[5];
    result[6] = m0[6];
    result[7] = m0[7];
    result[8] = m0[8];
    result[9] = m0[9];
    result[10] = m0[10];
    result[11] = m0[11];
    result[12] = v0[0];
    result[13] = v0[1];
    result[14] = v0[2];
    result[15] = m0[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_translate(mfloat_t *result, mfloat_t *m0, mfloat_t *v0) {
    result[0] = m0[0];
    result[1] = m0[1];
    result[2] = m0[2];
    result[3] = m0[3];
    result[4] = m0[4];
    result[5] = m0[5];
    result[6] = m0[6];
    result[7] = m0[7];
    result[8] = m0[8];
    result[9] = m0[9];
    result[10] = m0[10];
    result[11] = m0[11];
    result[12] = m0[12] + v0[0];
    result[13] = m0[13] + v0[1];
    result[14] = m0[14] + v0[2];
    result[15] = m0[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_scaling(mfloat_t *result, mfloat_t *m0, mfloat_t *v0) {
    result[0] = v0[0];
    result[1] = m0[1];
    result[2] = m0[2];
    result[3] = m0[3];
    result[4] = m0[4];
    result[5] = v0[1];
    result[6] = m0[6];
    result[7] = m0[7];
    result[8] = m0[8];
    result[9] = m0[9];
    result[10] = v0[2];
    result[11] = m0[11];
    result[12] = m0[12];
    result[13] = m0[13];
    result[14] = m0[14];
    result[15] = m0[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_scale(mfloat_t *result, mfloat_t *m0, mfloat_t *v0) {
    result[0] = m0[0] * v0[0];
    result[1] = m0[1];
    result[2] = m0[2];
    result[3] = m0[3];
    result[4] = m0[4];
    result[5] = m0[5] * v0[1];
    result[6] = m0[6];
    result[7] = m0[7];
    result[8] = m0[8];
    result[9] = m0[9];
    result[10] = m0[10] * v0[2];
    result[11] = m0[11];
    result[12] = m0[12];
    result[13] = m0[13];
    result[14] = m0[14];
    result[15] = m0[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_multiply(mfloat_t *result, mfloat_t *m0, mfloat_t *m1) {
    mfloat_t multiplied[MAT4_SIZE];
    multiplied[0] = m0[0] * m1[0] + m0[4] * m1[1] + m0[8] * m1[2] + m0[12] * m1[3];
    multiplied[1] = m0[1] * m1[0] + m0[5] * m1[1] + m0[9] * m1[2] + m0[13] * m1[3];
    multiplied[2] = m0[2] * m1[0] + m0[6] * m1[1] + m0[10] * m1[2] + m0[14] * m1[3];
    multiplied[3] = m0[3] * m1[0] + m0[7] * m1[1] + m0[11] * m1[2] + m0[15] * m1[3];
    multiplied[4] = m0[0] * m1[4] + m0[4] * m1[5] + m0[8] * m1[6] + m0[12] * m1[7];
    multiplied[5] = m0[1] * m1[4] + m0[5] * m1[5] + m0[9] * m1[6] + m0[13] * m1[7];
    multiplied[6] = m0[2] * m1[4] + m0[6] * m1[5] + m0[10] * m1[6] + m0[14] * m1[7];
    multiplied[7] = m0[3] * m1[4] + m0[7] * m1[5] + m0[11] * m1[6] + m0[15] * m1[7];
    multiplied[8] = m0[0] * m1[8] + m0[4] * m1[9] + m0[8] * m1[10] + m0[12] * m1[11];
    multiplied[9] = m0[1] * m1[8] + m0[5] * m1[9] + m0[9] * m1[10] + m0[13] * m1[11];
    multiplied[10] = m0[2] * m1[8] + m0[6] * m1[9] + m0[10] * m1[10] + m0[14] * m1[11];
    multiplied[11] = m0[3] * m1[8] + m0[7] * m1[9] + m0[11] * m1[10] + m0[15] * m1[11];
    multiplied[12] = m0[0] * m1[12] + m0[4] * m1[13] + m0[8] * m1[14] + m0[12] * m1[15];
    multiplied[13] = m0[1] * m1[12] + m0[5] * m1[13] + m0[9] * m1[14] + m0[13] * m1[15];
    multiplied[14] = m0[2] * m1[12] + m0[6] * m1[13] + m0[10] * m1[14] + m0[14] * m1[15];
    multiplied[15] = m0[3] * m1[12] + m0[7] * m1[13] + m0[11] * m1[14] + m0[15] * m1[15];
    result[0] = multiplied[0];
    result[1] = multiplied[1];
    result[2] = multiplied[2];
    result[3] = multiplied[3];
    result[4] = multiplied[4];
    result[5] = multiplied[5];
    result[6] = multiplied[6];
    result[7] = multiplied[7];
    result[8] = multiplied[8];
    result[9] = multiplied[9];
    result[10] = multiplied[10];
    result[11] = multiplied[11];
    result[12] = multiplied[12];
    result[13] = multiplied[13];
    result[14] = multiplied[14];
    result[15] = multiplied[15];
    return result;
}

__host__ __device__ mfloat_t *mat4_multiply_f(mfloat_t *result, mfloat_t *m0, mfloat_t f) {
    result[0] = m0[0] * f;
    result[1] = m0[1] * f;
    result[2] = m0[2] * f;
    result[3] = m0[3] * f;
    result[4] = m0[4] * f;
    result[5] = m0[5] * f;
    result[6] = m0[6] * f;
    result[7] = m0[7] * f;
    result[8] = m0[8] * f;
    result[9] = m0[9] * f;
    result[10] = m0[10] * f;
    result[11] = m0[11] * f;
    result[12] = m0[12] * f;
    result[13] = m0[13] * f;
    result[14] = m0[14] * f;
    result[15] = m0[15] * f;
    return result;
}

__host__ __device__ mfloat_t *mat4_inverse(mfloat_t *result, mfloat_t *m0) {
    mfloat_t inverse[MAT4_SIZE];
    mfloat_t inverted_determinant;
    mfloat_t m11 = m0[0];
    mfloat_t m21 = m0[1];
    mfloat_t m31 = m0[2];
    mfloat_t m41 = m0[3];
    mfloat_t m12 = m0[4];
    mfloat_t m22 = m0[5];
    mfloat_t m32 = m0[6];
    mfloat_t m42 = m0[7];
    mfloat_t m13 = m0[8];
    mfloat_t m23 = m0[9];
    mfloat_t m33 = m0[10];
    mfloat_t m43 = m0[11];
    mfloat_t m14 = m0[12];
    mfloat_t m24 = m0[13];
    mfloat_t m34 = m0[14];
    mfloat_t m44 = m0[15];
    inverse[0] = m22 * m33 * m44 - m22 * m43 * m34 - m23 * m32 * m44 + m23 * m42 * m34 + m24 * m32 * m43 - m24 * m42 * m33;
    inverse[4] = -m12 * m33 * m44 + m12 * m43 * m34 + m13 * m32 * m44 - m13 * m42 * m34 - m14 * m32 * m43 + m14 * m42 * m33;
    inverse[8] = m12 * m23 * m44 - m12 * m43 * m24 - m13 * m22 * m44 + m13 * m42 * m24 + m14 * m22 * m43 - m14 * m42 * m23;
    inverse[12] = -m12 * m23 * m34 + m12 * m33 * m24 + m13 * m22 * m34 - m13 * m32 * m24 - m14 * m22 * m33 + m14 * m32 * m23;
    inverse[1] = -m21 * m33 * m44 + m21 * m43 * m34 + m23 * m31 * m44 - m23 * m41 * m34 - m24 * m31 * m43 + m24 * m41 * m33;
    inverse[5] = m11 * m33 * m44 - m11 * m43 * m34 - m13 * m31 * m44 + m13 * m41 * m34 + m14 * m31 * m43 - m14 * m41 * m33;
    inverse[9] = -m11 * m23 * m44 + m11 * m43 * m24 + m13 * m21 * m44 - m13 * m41 * m24 - m14 * m21 * m43 + m14 * m41 * m23;
    inverse[13] = m11 * m23 * m34 - m11 * m33 * m24 - m13 * m21 * m34 + m13 * m31 * m24 + m14 * m21 * m33 - m14 * m31 * m23;
    inverse[2] = m21 * m32 * m44 - m21 * m42 * m34 - m22 * m31 * m44 + m22 * m41 * m34 + m24 * m31 * m42 - m24 * m41 * m32;
    inverse[6] = -m11 * m32 * m44 + m11 * m42 * m34 + m12 * m31 * m44 - m12 * m41 * m34 - m14 * m31 * m42 + m14 * m41 * m32;
    inverse[10] = m11 * m22 * m44 - m11 * m42 * m24 - m12 * m21 * m44 + m12 * m41 * m24 + m14 * m21 * m42 - m14 * m41 * m22;
    inverse[14] = -m11 * m22 * m34 + m11 * m32 * m24 + m12 * m21 * m34 - m12 * m31 * m24 - m14 * m21 * m32 + m14 * m31 * m22;
    inverse[3] = -m21 * m32 * m43 + m21 * m42 * m33 + m22 * m31 * m43 - m22 * m41 * m33 - m23 * m31 * m42 + m23 * m41 * m32;
    inverse[7] = m11 * m32 * m43 - m11 * m42 * m33 - m12 * m31 * m43 + m12 * m41 * m33 + m13 * m31 * m42 - m13 * m41 * m32;
    inverse[11] = -m11 * m22 * m43 + m11 * m42 * m23 + m12 * m21 * m43 - m12 * m41 * m23 - m13 * m21 * m42 + m13 * m41 * m22;
    inverse[15] = m11 * m22 * m33 - m11 * m32 * m23 - m12 * m21 * m33 + m12 * m31 * m23 + m13 * m21 * m32 - m13 * m31 * m22;
    inverted_determinant = MFLOAT_C(1.0) / (m11 * inverse[0] + m21 * inverse[4] + m31 * inverse[8] + m41 * inverse[12]);
    result[0] = inverse[0] * inverted_determinant;
    result[1] = inverse[1] * inverted_determinant;
    result[2] = inverse[2] * inverted_determinant;
    result[3] = inverse[3] * inverted_determinant;
    result[4] = inverse[4] * inverted_determinant;
    result[5] = inverse[5] * inverted_determinant;
    result[6] = inverse[6] * inverted_determinant;
    result[7] = inverse[7] * inverted_determinant;
    result[8] = inverse[8] * inverted_determinant;
    result[9] = inverse[9] * inverted_determinant;
    result[10] = inverse[10] * inverted_determinant;
    result[11] = inverse[11] * inverted_determinant;
    result[12] = inverse[12] * inverted_determinant;
    result[13] = inverse[13] * inverted_determinant;
    result[14] = inverse[14] * inverted_determinant;
    result[15] = inverse[15] * inverted_determinant;
    return result;
}

__host__ __device__ mfloat_t *mat4_lerp(mfloat_t *result, mfloat_t *m0, mfloat_t *m1, mfloat_t f) {
    result[0] = m0[0] + (m1[0] - m0[0]) * f;
    result[1] = m0[1] + (m1[1] - m0[1]) * f;
    result[2] = m0[2] + (m1[2] - m0[2]) * f;
    result[3] = m0[3] + (m1[3] - m0[3]) * f;
    result[4] = m0[4] + (m1[4] - m0[4]) * f;
    result[5] = m0[5] + (m1[5] - m0[5]) * f;
    result[6] = m0[6] + (m1[6] - m0[6]) * f;
    result[7] = m0[7] + (m1[7] - m0[7]) * f;
    result[8] = m0[8] + (m1[8] - m0[8]) * f;
    result[9] = m0[9] + (m1[9] - m0[9]) * f;
    result[10] = m0[10] + (m1[10] - m0[10]) * f;
    result[11] = m0[11] + (m1[11] - m0[11]) * f;
    result[12] = m0[12] + (m1[12] - m0[12]) * f;
    result[13] = m0[13] + (m1[13] - m0[13]) * f;
    result[14] = m0[14] + (m1[14] - m0[14]) * f;
    result[15] = m0[15] + (m1[15] - m0[15]) * f;
    return result;
}

__host__ __device__ mfloat_t *mat4_look_at(mfloat_t *result, mfloat_t *position, mfloat_t *target, mfloat_t *up) {
    mfloat_t tmp_forward[VEC3_SIZE];
    mfloat_t tmp_side[VEC3_SIZE];
    mfloat_t tmp_up[VEC3_SIZE];
    vec3_subtract(tmp_forward, target, position);
    vec3_normalize(tmp_forward, tmp_forward);
    vec3_cross(tmp_side, tmp_forward, up);
    vec3_normalize(tmp_side, tmp_side);
    vec3_cross(tmp_up, tmp_side, tmp_forward);
    result[0] = tmp_side[0];
    result[1] = tmp_up[0];
    result[2] = -tmp_forward[0];
    result[3] = MFLOAT_C(0.0);
    result[4] = tmp_side[1];
    result[5] = tmp_up[1];
    result[6] = -tmp_forward[1];
    result[7] = MFLOAT_C(0.0);
    result[8] = tmp_side[2];
    result[9] = tmp_up[2];
    result[10] = -tmp_forward[2];
    result[11] = MFLOAT_C(0.0);
    result[12] = -vec3_dot(tmp_side, position);
    result[13] = -vec3_dot(tmp_up, position);
    result[14] = vec3_dot(tmp_forward, position);
    result[15] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *mat4_ortho(mfloat_t *result, mfloat_t l, mfloat_t r, mfloat_t b, mfloat_t t, mfloat_t n, mfloat_t f) {
    result[0] = MFLOAT_C(2.0) / (r - l);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(0.0);
    result[5] = MFLOAT_C(2.0) / (t - b);
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(0.0);
    result[9] = MFLOAT_C(0.0);
    result[10] = -MFLOAT_C(2.0) / (f - n);
    result[11] = MFLOAT_C(0.0);
    result[12] = -((r + l) / (r - l));
    result[13] = -((t + b) / (t - b));
    result[14] = -((f + n) / (f - n));
    result[15] = MFLOAT_C(1.0);
    return result;
}

__host__ __device__ mfloat_t *mat4_perspective(mfloat_t *result, mfloat_t fov_y, mfloat_t aspect, mfloat_t n, mfloat_t f) {
    mfloat_t tan_half_fov_y = MFLOAT_C(1.0) / MTAN(fov_y * MFLOAT_C(0.5));
    result[0] = MFLOAT_C(1.0) / aspect * tan_half_fov_y;
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(0.0);
    result[5] = MFLOAT_C(1.0) / tan_half_fov_y;
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(0.0);
    result[9] = MFLOAT_C(0.0);
    result[10] = f / (n - f);
    result[11] = -MFLOAT_C(1.0);
    result[12] = MFLOAT_C(0.0);
    result[13] = MFLOAT_C(0.0);
    result[14] = -(f * n) / (f - n);
    result[15] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *mat4_perspective_fov(mfloat_t *result, mfloat_t fov, mfloat_t w, mfloat_t h, mfloat_t n, mfloat_t f) {
    mfloat_t h2 = MCOS(fov * MFLOAT_C(0.5)) / MSIN(fov * MFLOAT_C(0.5));
    mfloat_t w2 = h2 * h / w;
    result[0] = w2;
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(0.0);
    result[5] = h2;
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(0.0);
    result[9] = MFLOAT_C(0.0);
    result[10] = f / (n - f);
    result[11] = -MFLOAT_C(1.0);
    result[12] = MFLOAT_C(0.0);
    result[13] = MFLOAT_C(0.0);
    result[14] = -(f * n) / (f - n);
    result[15] = MFLOAT_C(0.0);
    return result;
}

__host__ __device__ mfloat_t *mat4_perspective_infinite(mfloat_t *result, mfloat_t fov_y, mfloat_t aspect, mfloat_t n) {
    mfloat_t range = MTAN(fov_y * MFLOAT_C(0.5)) * n;
    mfloat_t left = -range * aspect;
    mfloat_t right = range * aspect;
    mfloat_t top = range;
    mfloat_t bottom = -range;
    result[0] = MFLOAT_C(2.0) * n / (right - left);
    result[1] = MFLOAT_C(0.0);
    result[2] = MFLOAT_C(0.0);
    result[3] = MFLOAT_C(0.0);
    result[4] = MFLOAT_C(0.0);
    result[5] = MFLOAT_C(2.0) * n / (top - bottom);
    result[6] = MFLOAT_C(0.0);
    result[7] = MFLOAT_C(0.0);
    result[8] = MFLOAT_C(0.0);
    result[9] = MFLOAT_C(0.0);
    result[10] = -MFLOAT_C(1.0);
    result[11] = -MFLOAT_C(1.0);
    result[12] = MFLOAT_C(0.0);
    result[13] = MFLOAT_C(0.0);
    result[14] = -MFLOAT_C(2.0) * n;
    result[15] = MFLOAT_C(0.0);
    return result;
}

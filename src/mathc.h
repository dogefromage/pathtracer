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

#ifndef MATHC_H
#define MATHC_H

#include <stdbool.h>
#include <stdint.h>
#include <float.h>

#define MATHC_USE_FLOATING_POINT
#define MATHC_USE_POINTER_STRUCT_FUNCTIONS

#define VEC2_SIZE 2
#define VEC3_SIZE 3
#define VEC4_SIZE 4
#define QUAT_SIZE 4
#define MAT2_SIZE 4
#define MAT3_SIZE 9
#define MAT4_SIZE 16

typedef float mfloat_t;

#define MPI 3.1415926536f
#define MPI_2 1.5707963268f
#define MPI_4 0.7853981634f
#define MFLT_EPSILON FLT_EPSILON
#define MFABS fabsf
#define MFMIN fminf
#define MFMAX fmaxf
#define MSQRT sqrtf
#define MSIN sinf
#define MCOS cosf
#define MACOS acosf
#define MASIN asinf
#define MTAN tanf
#define MATAN2 atan2f
#define MPOW powf
#define MFLOOR floorf
#define MCEIL ceilf
#define MROUND roundf
#define MFLOAT_C(c) ((mfloat_t)c)

struct vec2 {
	union {
		struct {
			mfloat_t x;
			mfloat_t y;
		};
		mfloat_t v[VEC2_SIZE];
	};
};

struct vec3 {
	union {
		struct {
			mfloat_t x;
			mfloat_t y;
			mfloat_t z;
		};
		mfloat_t v[VEC3_SIZE];
        struct {
			mfloat_t r;
			mfloat_t g;
			mfloat_t b;
        };
	};
};

struct vec4 {
	union {
		struct {
			mfloat_t x;
			mfloat_t y;
			mfloat_t z;
			mfloat_t w;
		};
		mfloat_t v[VEC4_SIZE];
	};
};

struct quat {
	union {
		struct {
			mfloat_t x;
			mfloat_t y;
			mfloat_t z;
			mfloat_t w;
		};
		mfloat_t v[QUAT_SIZE];
	};
};

/*
Matrix 2×2 representation:
0/m11 2/m12
1/m21 3/m22
*/
struct mat2 {
	union {
		struct {
			mfloat_t m11;
			mfloat_t m21;
			mfloat_t m12;
			mfloat_t m22;
		};
		mfloat_t v[MAT2_SIZE];
	};
};

/*
Matrix 3×3 representation:
0/m11 3/m12 6/m13
1/m21 4/m22 7/m23
2/m31 5/m32 8/m33
*/
struct mat3 {
	union {
		struct {
			mfloat_t m11;
			mfloat_t m21;
			mfloat_t m31;
			mfloat_t m12;
			mfloat_t m22;
			mfloat_t m32;
			mfloat_t m13;
			mfloat_t m23;
			mfloat_t m33;
		};
		mfloat_t v[MAT3_SIZE];
	};
};

/*
Matrix 4×4 representation:
0/m11 4/m12  8/m13 12/m14
1/m21 5/m22  9/m23 13/m24
2/m31 6/m32 10/m33 14/m34
3/m41 7/m42 11/m43 15/m44
*/
struct mat4 {
	union {
		struct {
			mfloat_t m11;
			mfloat_t m21;
			mfloat_t m31;
			mfloat_t m41;
			mfloat_t m12;
			mfloat_t m22;
			mfloat_t m32;
			mfloat_t m42;
			mfloat_t m13;
			mfloat_t m23;
			mfloat_t m33;
			mfloat_t m43;
			mfloat_t m14;
			mfloat_t m24;
			mfloat_t m34;
			mfloat_t m44;
		};
		mfloat_t v[MAT4_SIZE];
	};
};

#define MRADIANS(degrees) (degrees * MPI / MFLOAT_C(180.0))
#define MDEGREES(radians) (radians * MFLOAT_C(180.0) / MPI)

__host__ __device__ bool nearly_equal(mfloat_t a, mfloat_t b, mfloat_t epsilon);
__host__ __device__ mfloat_t to_radians(mfloat_t degrees);
__host__ __device__ mfloat_t to_degrees(mfloat_t radians);
__host__ __device__ mfloat_t clampf(mfloat_t value, mfloat_t min, mfloat_t max);
__host__ __device__ bool vec2_is_zero(mfloat_t *v0);
__host__ __device__ bool vec2_is_equal(mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2(mfloat_t *result, mfloat_t x, mfloat_t y);
__host__ __device__ mfloat_t *vec2_assign(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_zero(mfloat_t *result);
__host__ __device__ mfloat_t *vec2_one(mfloat_t *result);
__host__ __device__ mfloat_t *vec2_sign(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_add(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_add_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec2_subtract(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_subtract_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec2_multiply(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_multiply_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec2_multiply_mat2(mfloat_t *result, mfloat_t *v0, mfloat_t *m0);
__host__ __device__ mfloat_t *vec2_divide(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_divide_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec2_snap(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_snap_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec2_negative(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_abs(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_floor(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_ceil(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_round(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_max(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_min(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_clamp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2);
__host__ __device__ mfloat_t *vec2_normalize(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t vec2_dot(mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_project(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec2_slide(mfloat_t *result, mfloat_t *v0, mfloat_t *normal);
__host__ __device__ mfloat_t *vec2_reflect(mfloat_t *result, mfloat_t *v0, mfloat_t *normal);
__host__ __device__ mfloat_t *vec2_tangent(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec2_rotate(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec2_lerp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t f);
__host__ __device__ mfloat_t *vec2_bezier3(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2, mfloat_t f);
__host__ __device__ mfloat_t *vec2_bezier4(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2, mfloat_t *v3, mfloat_t f);
__host__ __device__ mfloat_t vec2_angle(mfloat_t *v0);
__host__ __device__ mfloat_t vec2_length(mfloat_t *v0);
__host__ __device__ mfloat_t vec2_length_squared(mfloat_t *v0);
__host__ __device__ mfloat_t vec2_distance(mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t vec2_distance_squared(mfloat_t *v0, mfloat_t *v1);
__host__ __device__ bool vec2_linear_independent(mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t** vec2_orthonormalization(mfloat_t result[2][2], mfloat_t basis[2][2]);
__host__ __device__ bool vec3_is_zero(const mfloat_t *v0);
__host__ __device__ bool vec3_is_equal(const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3(mfloat_t *result, mfloat_t x, mfloat_t y, mfloat_t z);
__host__ __device__ mfloat_t *vec3_assign(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t *vec3_zero(mfloat_t *result);
__host__ __device__ mfloat_t *vec3_const(mfloat_t *result, mfloat_t v);
__host__ __device__ mfloat_t *vec3_one(mfloat_t *result);
__host__ __device__ mfloat_t *vec3_sign(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t *vec3_add(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_add_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec3_subtract(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_subtract_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec3_multiply(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_multiply_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec3_multiply_mat3(mfloat_t *result, const mfloat_t *v0, const mfloat_t *m0);
__host__ __device__ mfloat_t *vec3_divide(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_divide_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec3_snap(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_snap_f(mfloat_t *result, const mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec3_negative(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t *vec3_abs(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t *vec3_floor(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t *vec3_ceil(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t *vec3_round(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t *vec3_max(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_min(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_clamp(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2);
__host__ __device__ mfloat_t *vec3_cross(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_normalize(mfloat_t *result, const mfloat_t *v0);
__host__ __device__ mfloat_t vec3_dot(const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_project(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t *vec3_slide(mfloat_t *result, const mfloat_t *v0, const mfloat_t *normal);
__host__ __device__ mfloat_t *vec3_reflect(mfloat_t *result, const mfloat_t *v0, const mfloat_t *normal);
__host__ __device__ mfloat_t *vec3_rotate(mfloat_t *result, const mfloat_t *v0, const mfloat_t *ra, mfloat_t f);
__host__ __device__ mfloat_t *vec3_lerp(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, mfloat_t f);
__host__ __device__ mfloat_t *vec3_bezier3(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2, mfloat_t f);
__host__ __device__ mfloat_t *vec3_bezier4(mfloat_t *result, const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2, const mfloat_t *v3, mfloat_t f);
__host__ __device__ mfloat_t vec3_max_component(const mfloat_t *v);
__host__ __device__ mfloat_t vec3_length(const mfloat_t *v0);
__host__ __device__ mfloat_t vec3_length_squared(const mfloat_t *v0);
__host__ __device__ mfloat_t vec3_distance(const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ mfloat_t vec3_distance_squared(const mfloat_t *v0, const mfloat_t *v1);
__host__ __device__ bool vec3_linear_independent(const mfloat_t *v0, const mfloat_t *v1, const mfloat_t *v2);
__host__ __device__ mfloat_t** vec3_orthonormalization(mfloat_t result[3][3], const mfloat_t basis[3][3]);
__host__ __device__ bool vec4_is_zero(mfloat_t *v0);
__host__ __device__ bool vec4_is_equal(mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4(mfloat_t *result, mfloat_t x, mfloat_t y, mfloat_t z, mfloat_t w);
__host__ __device__ mfloat_t *vec4_assign(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_zero(mfloat_t *result);
__host__ __device__ mfloat_t *vec4_one(mfloat_t *result);
__host__ __device__ mfloat_t *vec4_sign(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_add(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4_add_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec4_subtract(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4_subtract_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec4_multiply(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4_multiply_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec4_multiply_mat4(mfloat_t *result, mfloat_t *v0, mfloat_t *m0);
__host__ __device__ mfloat_t *vec4_divide(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4_divide_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec4_snap(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4_snap_f(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *vec4_negative(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_abs(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_floor(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_ceil(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_round(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_max(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4_min(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *vec4_clamp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t *v2);
__host__ __device__ mfloat_t *vec4_normalize(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *vec4_lerp(mfloat_t *result, mfloat_t *v0, mfloat_t *v1, mfloat_t f);
__host__ __device__ bool quat_is_zero(mfloat_t *q0);
__host__ __device__ bool quat_is_equal(mfloat_t *q0, mfloat_t *q1);
__host__ __device__ mfloat_t *quat(mfloat_t *result, mfloat_t x, mfloat_t y, mfloat_t z, mfloat_t w);
__host__ __device__ mfloat_t *quat_assign(mfloat_t *result, mfloat_t *q0);
__host__ __device__ mfloat_t *quat_zero(mfloat_t *result);
__host__ __device__ mfloat_t *quat_null(mfloat_t *result);
__host__ __device__ mfloat_t *quat_multiply(mfloat_t *result, mfloat_t *q0, mfloat_t *q1);
__host__ __device__ mfloat_t *quat_multiply_f(mfloat_t *result, mfloat_t *q0, mfloat_t f);
__host__ __device__ mfloat_t *quat_divide(mfloat_t *result, mfloat_t *q0, mfloat_t *q1);
__host__ __device__ mfloat_t *quat_divide_f(mfloat_t *result, mfloat_t *q0, mfloat_t f);
__host__ __device__ mfloat_t *quat_negative(mfloat_t *result, mfloat_t *q0);
__host__ __device__ mfloat_t *quat_conjugate(mfloat_t *result, mfloat_t *q0);
__host__ __device__ mfloat_t *quat_inverse(mfloat_t *result, mfloat_t *q0);
__host__ __device__ mfloat_t *quat_normalize(mfloat_t *result, mfloat_t *q0);
__host__ __device__ mfloat_t quat_dot(mfloat_t *q0, mfloat_t *q1);
__host__ __device__ mfloat_t *quat_power(mfloat_t *result, mfloat_t *q0, mfloat_t exponent);
__host__ __device__ mfloat_t *quat_from_axis_angle(mfloat_t *result, mfloat_t *v0, mfloat_t angle);
__host__ __device__ mfloat_t *quat_from_vec3(mfloat_t *result, mfloat_t *v0, mfloat_t *v1);
__host__ __device__ mfloat_t *quat_from_mat4(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *quat_lerp(mfloat_t *result, mfloat_t *q0, mfloat_t *q1, mfloat_t f);
__host__ __device__ mfloat_t *quat_slerp(mfloat_t *result, mfloat_t *q0, mfloat_t *q1, mfloat_t f);
__host__ __device__ mfloat_t quat_length(mfloat_t *q0);
__host__ __device__ mfloat_t quat_length_squared(mfloat_t *q0);
__host__ __device__ mfloat_t quat_angle(mfloat_t *q0, mfloat_t *q1);
__host__ __device__ mfloat_t *mat2(mfloat_t *result, mfloat_t m11, mfloat_t m12, mfloat_t m21, mfloat_t m22);
__host__ __device__ mfloat_t *mat2_zero(mfloat_t *result);
__host__ __device__ mfloat_t *mat2_identity(mfloat_t *result);
__host__ __device__ mfloat_t mat2_determinant(mfloat_t *m0);
__host__ __device__ mfloat_t *mat2_assign(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat2_negative(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat2_transpose(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat2_cofactor(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat2_adjugate(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat2_multiply(mfloat_t *result, mfloat_t *m0, mfloat_t *m1);
__host__ __device__ mfloat_t *mat2_multiply_f(mfloat_t *result, mfloat_t *m0, mfloat_t f);
__host__ __device__ mfloat_t *mat2_inverse(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat2_scaling(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *mat2_scale(mfloat_t *result, mfloat_t *m0, mfloat_t *v0);
__host__ __device__ mfloat_t *mat2_rotation_z(mfloat_t *result, mfloat_t f);
__host__ __device__ mfloat_t *mat2_lerp(mfloat_t *result, mfloat_t *m0, mfloat_t *m1, mfloat_t f);
__host__ __device__ mfloat_t *mat3(mfloat_t *result, mfloat_t m11, mfloat_t m12, mfloat_t m13, mfloat_t m21, mfloat_t m22, mfloat_t m23, mfloat_t m31, mfloat_t m32, mfloat_t m33);
__host__ __device__ mfloat_t *mat3_zero(mfloat_t *result);
__host__ __device__ mfloat_t *mat3_identity(mfloat_t *result);
__host__ __device__ mfloat_t mat3_determinant(mfloat_t *m0);
__host__ __device__ mfloat_t *mat3_assign(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat3_negative(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat3_transpose(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat3_cofactor(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat3_multiply(mfloat_t *result, mfloat_t *m0, mfloat_t *m1);
__host__ __device__ mfloat_t *mat3_multiply_f(mfloat_t *result, mfloat_t *m0, mfloat_t f);
__host__ __device__ mfloat_t *mat3_inverse(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat3_scaling(mfloat_t *result, mfloat_t *v0);
__host__ __device__ mfloat_t *mat3_scale(mfloat_t *result, mfloat_t *m0, mfloat_t *v0);
__host__ __device__ mfloat_t *mat3_rotation_x(mfloat_t *result, mfloat_t f);
__host__ __device__ mfloat_t *mat3_rotation_y(mfloat_t *result, mfloat_t f);
__host__ __device__ mfloat_t *mat3_rotation_z(mfloat_t *result, mfloat_t f);
__host__ __device__ mfloat_t *mat3_rotation_axis(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *mat3_rotation_quat(mfloat_t *result, mfloat_t *q0);
__host__ __device__ mfloat_t *mat3_lerp(mfloat_t *result, mfloat_t *m0, mfloat_t *m1, mfloat_t f);
__host__ __device__ mfloat_t *mat4(mfloat_t *result, mfloat_t m11, mfloat_t m12, mfloat_t m13, mfloat_t m14, mfloat_t m21, mfloat_t m22, mfloat_t m23, mfloat_t m24, mfloat_t m31, mfloat_t m32, mfloat_t m33, mfloat_t m34, mfloat_t m41, mfloat_t m42, mfloat_t m43, mfloat_t m44);
__host__ __device__ mfloat_t *mat4_zero(mfloat_t *result);
__host__ __device__ mfloat_t *mat4_identity(mfloat_t *result);
__host__ __device__ mfloat_t mat4_determinant(mfloat_t *m0);
__host__ __device__ mfloat_t *mat4_assign(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat4_negative(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat4_transpose(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat4_cofactor(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat4_rotation_x(mfloat_t *result, mfloat_t f);
__host__ __device__ mfloat_t *mat4_rotation_y(mfloat_t *result, mfloat_t f);
__host__ __device__ mfloat_t *mat4_rotation_z(mfloat_t *result, mfloat_t f);
__host__ __device__ mfloat_t *mat4_rotation_axis(mfloat_t *result, mfloat_t *v0, mfloat_t f);
__host__ __device__ mfloat_t *mat4_rotation_quat(mfloat_t *result, mfloat_t *q0);
__host__ __device__ mfloat_t *mat4_translation(mfloat_t *result, mfloat_t *m0, mfloat_t *v0);
__host__ __device__ mfloat_t *mat4_translate(mfloat_t *result, mfloat_t *m0, mfloat_t *v0);
__host__ __device__ mfloat_t *mat4_scaling(mfloat_t *result, mfloat_t *m0, mfloat_t *v0);
__host__ __device__ mfloat_t *mat4_scale(mfloat_t *result, mfloat_t *m0, mfloat_t *v0);
__host__ __device__ mfloat_t *mat4_multiply(mfloat_t *result, mfloat_t *m0, mfloat_t *m1);
__host__ __device__ mfloat_t *mat4_multiply_f(mfloat_t *result, mfloat_t *m0, mfloat_t f);
__host__ __device__ mfloat_t *mat4_inverse(mfloat_t *result, mfloat_t *m0);
__host__ __device__ mfloat_t *mat4_lerp(mfloat_t *result, mfloat_t *m0, mfloat_t *m1, mfloat_t f);
__host__ __device__ mfloat_t *mat4_look_at(mfloat_t *result, mfloat_t *position, mfloat_t *target, mfloat_t *up);
__host__ __device__ mfloat_t *mat4_ortho(mfloat_t *result, mfloat_t l, mfloat_t r, mfloat_t b, mfloat_t t, mfloat_t n, mfloat_t f);
__host__ __device__ mfloat_t *mat4_perspective(mfloat_t *result, mfloat_t fov_y, mfloat_t aspect, mfloat_t n, mfloat_t f);
__host__ __device__ mfloat_t *mat4_perspective_fov(mfloat_t *result, mfloat_t fov, mfloat_t w, mfloat_t h, mfloat_t n, mfloat_t f);
__host__ __device__ mfloat_t *mat4_perspective_infinite(mfloat_t *result, mfloat_t fov_y, mfloat_t aspect, mfloat_t n);

#endif
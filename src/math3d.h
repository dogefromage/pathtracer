/**

Math 3D v1.0
By Stephan Soller <stephan.soller@helionweb.de> and Tobias Malmsheimer
Licensed under the MIT license

Math 3D is a compact C99 library meant to be used with OpenGL. It provides basic
3D vector and 4x4 matrix operations as well as functions to create transformation
and projection matrices. The OpenGL binary layout is used so you can just upload
vectors and matrices into shaders and work with them without any conversions.

It's an stb style single header file library. Define MATH_3D_IMPLEMENTATION
before you include this file in *one* C file to create the implementation.


QUICK NOTES

- If not explicitly stated by a parameter name all angles are in radians.
- The matrices use column-major indices. This is the same as in OpenGL and GLSL.
  The matrix documentation below for details.
- Matrices are passed by value. This is probably a bit inefficient but
  simplifies code quite a bit. Most operations will be inlined by the compiler
  anyway so the difference shouldn't matter that much. A matrix fits into 4 of
  the 16 SSE2 registers anyway. If profiling shows significant slowdowns the
  matrix type might change but ease of use is more important than every last
  percent of performance.
- When combining matrices with multiplication the effects apply right to left.
  This is the convention used in mathematics and OpenGL. Source:
  https://en.wikipedia.org/wiki/Transformation_matrix#Composing_and_inverting_transformations
  Direct3D does it differently.
- The `m4_mul_pos()` and `m4_mul_dir()` functions do a correct perspective
  divide (division by w) when necessary. This is a bit slower but ensures that
  the functions will properly work with projection matrices. If profiling shows
  this is a bottleneck special functions without perspective division can be
  added. But the normal multiplications should avoid any surprises.
- The library consistently uses a right-handed coordinate system. The old
  `glOrtho()` broke that rule and `m4_ortho()` has be slightly modified so you
  can always think of right-handed cubes that are projected into OpenGLs
  normalized device coordinates.
- Special care has been taken to document all complex operations and important
  sources. Most code is covered by test cases that have been manually calculated
  and checked on the whiteboard. Since indices and math code is prone to be
  confusing we used pair programming to avoid mistakes.


FURTHER IDEARS

These are ideas for future work on the library. They're implemented as soon as
there is a proper use case and we can find good names for them.

- bool v3_is_null(vec3_t v, float epsilon)
  To check if the length of a vector is smaller than `epsilon`.
- vec3_t v3_length_default(vec3_t v, float default_length, float epsilon)
  Returns `default_length` if the length of `v` is smaller than `epsilon`.
  Otherwise same as `v3_length()`.
- vec3_t v3_norm_default(vec3_t v, vec3_t default_vector, float epsilon)
  Returns `default_vector` if the length of `v` is smaller than `epsilon`.
  Otherwise the same as `v3_norm()`.
- mat4_t m4_invert(mat4_t matrix)
  Matrix inversion that works with arbitrary matrices. `m4_invert_affine()` can
  already invert translation, rotation, scaling, mirroring, reflection and
  shearing matrices. So a general inversion might only be useful to invert
  projection matrices for picking. But with orthographic and perspective
  projection it's probably simpler to calculate the ray into the scene directly
  based on the screen coordinates.


VERSION HISTORY

v1.0  2016-02-15  Initial release

**/

#pragma once

#include <math.h>
#include <stdio.h>

// Define PI directly because we would need to define the _BSD_SOURCE or
// _XOPEN_SOURCE feature test macros to get it from math.h. That would be a
// rather harsh dependency. So we define it directly if necessary.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//
// 3D vectors
//
// Use the `vec3()` function to create vectors. All other vector functions start
// with the `v3_` prefix.
//
// The binary layout is the same as in GLSL and everything else (just 3 floats).
// So you can just upload the vectors into shaders as they are.
//

typedef struct {
    float x, y, z;
    // maybe add rgb
} vec3_t;
static inline vec3_t vec3(float x, float y, float z) { return (vec3_t){x, y, z}; }

static inline vec3_t v3_add(vec3_t a, vec3_t b) { return (vec3_t){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline vec3_t v3_adds(vec3_t a, float s) { return (vec3_t){a.x + s, a.y + s, a.z + s}; }
static inline vec3_t v3_sub(vec3_t a, vec3_t b) { return (vec3_t){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline vec3_t v3_subs(vec3_t a, float s) { return (vec3_t){a.x - s, a.y - s, a.z - s}; }
static inline vec3_t v3_mul(vec3_t a, vec3_t b) { return (vec3_t){a.x * b.x, a.y * b.y, a.z * b.z}; }
static inline vec3_t v3_muls(vec3_t a, float s) { return (vec3_t){a.x * s, a.y * s, a.z * s}; }
static inline vec3_t v3_div(vec3_t a, vec3_t b) { return (vec3_t){a.x / b.x, a.y / b.y, a.z / b.z}; }
static inline vec3_t v3_divs(vec3_t a, float s) { return (vec3_t){a.x / s, a.y / s, a.z / s}; }
static inline float v3_length(vec3_t v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline vec3_t v3_norm(vec3_t v);
static inline float v3_dot(vec3_t a, vec3_t b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3_t v3_proj(vec3_t v, vec3_t onto);
static inline vec3_t v3_cross(vec3_t a, vec3_t b);
static inline float v3_angle_between(vec3_t a, vec3_t b);

//
// 4x4 matrices
//
// Use the `mat4()` function to create a matrix. You can write the matrix
// members in the same way as you would write them on paper or on a whiteboard:
//
// mat4_t m = mat4(
//     1,  0,  0,  7,
//     0,  1,  0,  5,
//     0,  0,  1,  3,
//     0,  0,  0,  1
// )
//
// This creates a matrix that translates points by vec3(7, 5, 3). All other
// matrix functions start with the `m4_` prefix. Among them functions to create
// identity, translation, rotation, scaling and projection matrices.
//
// The matrix is stored in column-major order, just as OpenGL expects. Members
// can be accessed by indices or member names. When you write a matrix on paper
// or on the whiteboard the indices and named members correspond to these
// positions:
//
// | m[0][0]  m[1][0]  m[2][0]  m[3][0] |
// | m[0][1]  m[1][1]  m[2][1]  m[3][1] |
// | m[0][2]  m[1][2]  m[2][2]  m[3][2] |
// | m[0][3]  m[1][3]  m[2][3]  m[3][3] |
//
// | m00  m10  m20  m30 |
// | m01  m11  m21  m31 |
// | m02  m12  m22  m32 |
// | m03  m13  m23  m33 |
//
// The first index or number in a name denotes the column, the second the row.
// So m[i][j] denotes the member in the ith column and the jth row. This is the
// same as in GLSL (source: GLSL v1.3 specification, 5.6 Matrix Components).
//

typedef union {
    // The first index is the column index, the second the row index. The memory
    // layout of nested arrays in C matches the memory layout expected by OpenGL.
    float m[4][4];
    // OpenGL expects the first 4 floats to be the first column of the matrix.
    // So we need to define the named members column by column for the names to
    // match the memory locations of the array elements.
    struct {
        float m00, m01, m02, m03;
        float m10, m11, m12, m13;
        float m20, m21, m22, m23;
        float m30, m31, m32, m33;
    };
} mat4_t;

static inline mat4_t mat4(
    float m00, float m10, float m20, float m30,
    float m01, float m11, float m21, float m31,
    float m02, float m12, float m22, float m32,
    float m03, float m13, float m23, float m33);

static inline mat4_t m4_identity();
static inline mat4_t m4_translation(vec3_t offset);
static inline mat4_t m4_scaling(vec3_t scale);
static inline mat4_t m4_rotation_x(float angle_in_rad);
static inline mat4_t m4_rotation_y(float angle_in_rad);
static inline mat4_t m4_rotation_z(float angle_in_rad);
mat4_t m4_rotation(float angle_in_rad, vec3_t axis);

mat4_t m4_ortho(float left, float right, float bottom, float top, float back, float front);
mat4_t m4_perspective(float vertical_field_of_view_in_deg, float aspect_ratio, float near_view_distance, float far_view_distance);
mat4_t m4_look_at(vec3_t from, vec3_t to, vec3_t up);

static inline mat4_t m4_transpose(mat4_t matrix);
static inline mat4_t m4_mul(mat4_t a, mat4_t b);
mat4_t m4_invert_affine(mat4_t matrix);
vec3_t m4_mul_pos(mat4_t matrix, vec3_t position);
vec3_t m4_mul_dir(mat4_t matrix, vec3_t direction);

void m4_print(mat4_t matrix);
void m4_printp(mat4_t matrix, int width, int precision);
void m4_fprint(FILE* stream, mat4_t matrix);
void m4_fprintp(FILE* stream, mat4_t matrix, int width, int precision);

//
// 3D vector functions header implementation
//

static inline vec3_t v3_norm(vec3_t v) {
    float len = v3_length(v);
    if (len > 0)
        return (vec3_t){v.x / len, v.y / len, v.z / len};
    else
        return (vec3_t){0, 0, 0};
}

static inline vec3_t v3_proj(vec3_t v, vec3_t onto) {
    return v3_muls(onto, v3_dot(v, onto) / v3_dot(onto, onto));
}

static inline vec3_t v3_cross(vec3_t a, vec3_t b) {
    return (vec3_t){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

static inline float v3_angle_between(vec3_t a, vec3_t b) {
    return acosf(v3_dot(a, b) / (v3_length(a) * v3_length(b)));
}

//
// Matrix functions header implementation
//

static inline mat4_t mat4(
    float m00, float m10, float m20, float m30,
    float m01, float m11, float m21, float m31,
    float m02, float m12, float m22, float m32,
    float m03, float m13, float m23, float m33) {
    return (mat4_t){
        // column major (transposed matrix)
        .m={
            {m00, m01, m02, m03},
            {m10, m11, m12, m13},
            {m20, m21, m22, m23},
            {m30, m31, m32, m33},
        }
    };

    // return (mat4_t){
    // .m={
    //     { m00, m10, m20, m30 },
    //     { m01, m11, m21, m31 },
    //     { m02, m12, m22, m32 },
    //     { m03, m13, m23, m33 },
    // }

    // .m[0][0] = m00, .m[1][0] = m10, .m[2][0] = m20, .m[3][0] = m30,
    // .m[0][1] = m01, .m[1][1] = m11, .m[2][1] = m21, .m[3][1] = m31,
    // .m[0][2] = m02, .m[1][2] = m12, .m[2][2] = m22, .m[3][2] = m32,
    // .m[0][3] = m03, .m[1][3] = m13, .m[2][3] = m23, .m[3][3] = m33

    // .m00 = m00, .m10 = m10, .m20 = m20, .m30 = m30,
    // .m01 = m01, .m11 = m11, .m21 = m21, .m31 = m31,
    // .m02 = m02, .m12 = m12, .m22 = m22, .m32 = m32,
    // .m03 = m03, .m13 = m13, .m23 = m23, .m33 = m33

    // };
}

static inline mat4_t m4_identity() {
    return mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

static inline mat4_t m4_translation(vec3_t offset) {
    return mat4(
        1, 0, 0, offset.x,
        0, 1, 0, offset.y,
        0, 0, 1, offset.z,
        0, 0, 0, 1);
}

static inline mat4_t m4_scaling(vec3_t scale) {
    float x = scale.x, y = scale.y, z = scale.z;
    return mat4(
        x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1);
}

static inline mat4_t m4_rotation_x(float angle_in_rad) {
    float s = sinf(angle_in_rad), c = cosf(angle_in_rad);
    return mat4(
        1, 0, 0, 0,
        0, c, -s, 0,
        0, s, c, 0,
        0, 0, 0, 1);
}

static inline mat4_t m4_rotation_y(float angle_in_rad) {
    float s = sinf(angle_in_rad), c = cosf(angle_in_rad);
    return mat4(
        c, 0, s, 0,
        0, 1, 0, 0,
        -s, 0, c, 0,
        0, 0, 0, 1);
}

static inline mat4_t m4_rotation_z(float angle_in_rad) {
    float s = sinf(angle_in_rad), c = cosf(angle_in_rad);
    return mat4(
        c, -s, 0, 0,
        s, c, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

static inline mat4_t m4_transpose(mat4_t matrix) {
    return mat4(
        matrix.m00, matrix.m01, matrix.m02, matrix.m03,
        matrix.m10, matrix.m11, matrix.m12, matrix.m13,
        matrix.m20, matrix.m21, matrix.m22, matrix.m23,
        matrix.m30, matrix.m31, matrix.m32, matrix.m33);
}

/**
 * Multiplication of two 4x4 matrices.
 *
 * Implemented by following the row times column rule and illustrating it on a
 * whiteboard with the proper indices in mind.
 *
 * Further reading: https://en.wikipedia.org/wiki/Matrix_multiplication
 * But note that the article use the first index for rows and the second for
 * columns.
 */
static inline mat4_t m4_mul(mat4_t a, mat4_t b) {
    mat4_t result;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) {
                sum += a.m[k][j] * b.m[i][k];
            }
            result.m[i][j] = sum;
        }
    }

    return result;
}

mat4_t m4_rotation(float angle_in_rad, vec3_t axis);
mat4_t m4_ortho(float left, float right, float bottom, float top, float back, float front);
mat4_t m4_perspective(float vertical_field_of_view_in_deg, float aspect_ratio, float near_view_distance, float far_view_distance);
mat4_t m4_look_at(vec3_t from, vec3_t to, vec3_t up);
mat4_t m4_invert_affine(mat4_t matrix);
vec3_t m4_mul_pos(mat4_t matrix, vec3_t position);
vec3_t m4_mul_dir(mat4_t matrix, vec3_t direction);
void m4_print(mat4_t matrix);
void m4_printp(mat4_t matrix, int width, int precision);
void m4_fprint(FILE* stream, mat4_t matrix);
void m4_fprintp(FILE* stream, mat4_t matrix, int width, int precision);
void v3_fprintf(FILE* stream, vec3_t v);
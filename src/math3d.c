#include "math3d.h"

/**
 * Creates a matrix to rotate around an axis by a given angle. The axis doesn't
 * need to be normalized.
 *
 * Sources:
 *
 * https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
 */
mat4_t m4_rotation(float angle_in_rad, vec3_t axis) {
    vec3_t normalized_axis = v3_norm(axis);
    float x = normalized_axis.x, y = normalized_axis.y, z = normalized_axis.z;
    float c = cosf(angle_in_rad), s = sinf(angle_in_rad);

    return mat4(
        c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s, 0,
        y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s, 0,
        z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c), 0,
        0, 0, 0, 1);
}

/**
 * Creates an orthographic projection matrix. It maps the right handed cube
 * defined by left, right, bottom, top, back and front onto the screen and
 * z-buffer. You can think of it as a cube you move through world or camera
 * space and everything inside is visible.
 *
 * This is slightly different from the traditional glOrtho() and from the linked
 * sources. These functions require the user to negate the last two arguments
 * (creating a left-handed coordinate system). We avoid that here so you can
 * think of this function as moving a right-handed cube through world space.
 *
 * The arguments are ordered in a way that for each axis you specify the minimum
 * followed by the maximum. Thats why it's bottom to top and back to front.
 *
 * Implementation details:
 *
 * To be more exact the right-handed cube is mapped into normalized device
 * coordinates, a left-handed cube where (-1 -1) is the lower left corner,
 * (1, 1) the upper right corner and a z-value of -1 is the nearest point and
 * 1 the furthest point. OpenGL takes it from there and puts it on the screen
 * and into the z-buffer.
 *
 * Sources:
 *
 * https://msdn.microsoft.com/en-us/library/windows/desktop/dd373965(v=vs.85).aspx
 * https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/
 */
mat4_t m4_ortho(float left, float right, float bottom, float top, float back, float front) {
    float l = left, r = right, b = bottom, t = top, n = front, f = back;
    float tx = -(r + l) / (r - l);
    float ty = -(t + b) / (t - b);
    float tz = -(f + n) / (f - n);
    return mat4(
        2 / (r - l), 0, 0, tx,
        0, 2 / (t - b), 0, ty,
        0, 0, 2 / (f - n), tz,
        0, 0, 0, 1);
}

/**
 * Creates a perspective projection matrix for a camera.
 *
 * The camera is at the origin and looks in the direction of the negative Z axis.
 * `near_view_distance` and `far_view_distance` have to be positive and > 0.
 * They are distances from the camera eye, not values on an axis.
 *
 * `near_view_distance` can be small but not 0. 0 breaks the projection and
 * everything ends up at the max value (far end) of the z-buffer. Making the
 * z-buffer useless.
 *
 * The matrix is the same as `gluPerspective()` builds. The view distance is
 * mapped to the z-buffer with a reciprocal function (1/x). Therefore the z-buffer
 * resolution for near objects is very good while resolution for far objects is
 * limited.
 *
 * Sources:
 *
 * https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/
 */
mat4_t m4_perspective(float vertical_field_of_view_in_deg, float aspect_ratio, float near_view_distance, float far_view_distance) {
    float fovy_in_rad = vertical_field_of_view_in_deg / 180 * M_PI;
    float f = 1.0f / tanf(fovy_in_rad / 2.0f);
    float ar = aspect_ratio;
    float nd = near_view_distance, fd = far_view_distance;

    return mat4(
        f / ar, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (fd + nd) / (nd - fd), (2 * fd * nd) / (nd - fd),
        0, 0, -1, 0);
}

/**
 * Builds a transformation matrix for a camera that looks from `from` towards
 * `to`. `up` defines the direction that's upwards for the camera. All three
 * vectors are given in world space and `up` doesn't need to be normalized.
 *
 * Sources: Derived on whiteboard.
 *
 * Implementation details:
 *
 * x, y and z are the right-handed base vectors of the cameras subspace.
 * x has to be normalized because the cross product only produces a normalized
 *   output vector if both input vectors are orthogonal to each other. And up
 *   probably isn't orthogonal to z.
 *
 * These vectors are then used to build a 3x3 rotation matrix. This matrix
 * rotates a vector by the same amount the camera is rotated. But instead we
 * need to rotate all incoming vertices backwards by that amount. That's what a
 * camera matrix is for: To move the world so that the camera is in the origin.
 * So we take the inverse of that rotation matrix and in case of an rotation
 * matrix this is just the transposed matrix. That's why the 3x3 part of the
 * matrix are the x, y and z vectors but written horizontally instead of
 * vertically.
 *
 * The translation is derived by creating a translation matrix to move the world
 * into the origin (thats translate by minus `from`). The complete lookat matrix
 * is then this translation followed by the rotation. Written as matrix
 * multiplication:
 *
 *   lookat = rotation * translation
 *
 * Since we're right-handed this equals to first doing the translation and after
 * that doing the rotation. During that multiplication the rotation 3x3 part
 * doesn't change but the translation vector is multiplied with each rotation
 * axis. The dot product is just a more compact way to write the actual
 * multiplications.
 */
mat4_t m4_look_at(vec3_t from, vec3_t to, vec3_t up) {
    vec3_t z = v3_muls(v3_norm(v3_sub(to, from)), -1);
    vec3_t x = v3_norm(v3_cross(up, z));
    vec3_t y = v3_cross(z, x);

    return mat4(
        x.x, x.y, x.z, -v3_dot(from, x),
        y.x, y.y, y.z, -v3_dot(from, y),
        z.x, z.y, z.z, -v3_dot(from, z),
        0, 0, 0, 1);
}

/**
 * Inverts an affine transformation matrix. That are translation, scaling,
 * mirroring, reflection, rotation and shearing matrices or any combination of
 * them.
 *
 * Implementation details:
 *
 * - Invert the 3x3 part of the 4x4 matrix to handle rotation, scaling, etc.
 *   correctly (see source).
 * - Invert the translation part of the 4x4 matrix by multiplying it with the
 *   inverted rotation matrix and negating it.
 *
 * When a 3D point is multiplied with a transformation matrix it is first
 * rotated and then translated. The inverted transformation matrix is the
 * inverse translation followed by the inverse rotation. Written as a matrix
 * multiplication (remember, the effect applies right to left):
 *
 *   inv(matrix) = inv(rotation) * inv(translation)
 *
 * The inverse translation is a translation into the opposite direction, just
 * the negative translation. The rotation part isn't changed by that
 * multiplication but the translation part is multiplied by the inverse rotation
 * matrix. It's the same situation as with `m4_look_at()`. But since we don't
 * store the rotation matrix as 3D vectors we can't use the dot product and have
 * to write the matrix multiplication operations by hand.
 *
 * Sources for 3x3 matrix inversion:
 *
 * https://www.khanacademy.org/math/precalculus/precalc-matrices/determinants-and-inverses-of-large-matrices/v/inverting-3x3-part-2-determinant-and-adjugate-of-a-matrix
 */
mat4_t m4_invert_affine(mat4_t matrix) {
    // Create shorthands to access matrix members
    float m00 = matrix.m00, m10 = matrix.m10, m20 = matrix.m20, m30 = matrix.m30;
    float m01 = matrix.m01, m11 = matrix.m11, m21 = matrix.m21, m31 = matrix.m31;
    float m02 = matrix.m02, m12 = matrix.m12, m22 = matrix.m22, m32 = matrix.m32;

    // Invert 3x3 part of the 4x4 matrix that contains the rotation, etc.
    // That part is called R from here on.

    // Calculate cofactor matrix of R
    float c00 = m11 * m22 - m12 * m21, c10 = -(m01 * m22 - m02 * m21), c20 = m01 * m12 - m02 * m11;
    float c01 = -(m10 * m22 - m12 * m20), c11 = m00 * m22 - m02 * m20, c21 = -(m00 * m12 - m02 * m10);
    float c02 = m10 * m21 - m11 * m20, c12 = -(m00 * m21 - m01 * m20), c22 = m00 * m11 - m01 * m10;

    // Caclculate the determinant by using the already calculated determinants
    // in the cofactor matrix.
    // Second sign is already minus from the cofactor matrix.
    float det = m00 * c00 + m10 * c10 + m20 * c20;
    if (fabsf(det) < 0.00001)
        return m4_identity();

    // Calcuate inverse of R by dividing the transposed cofactor matrix by the
    // determinant.
    float i00 = c00 / det, i10 = c01 / det, i20 = c02 / det;
    float i01 = c10 / det, i11 = c11 / det, i21 = c12 / det;
    float i02 = c20 / det, i12 = c21 / det, i22 = c22 / det;

    // Combine the inverted R with the inverted translation
    return mat4(
        i00, i10, i20, -(i00 * m30 + i10 * m31 + i20 * m32),
        i01, i11, i21, -(i01 * m30 + i11 * m31 + i21 * m32),
        i02, i12, i22, -(i02 * m30 + i12 * m31 + i22 * m32),
        0, 0, 0, 1);
}

/**
 * Multiplies a 4x4 matrix with a 3D vector representing a point in 3D space.
 *
 * Before the matrix multiplication the vector is first expanded to a 4D vector
 * (x, y, z, 1). After the multiplication the vector is reduced to 3D again by
 * dividing through the 4th component (if it's not 0 or 1).
 */
vec3_t m4_mul_pos(mat4_t matrix, vec3_t position) {
    vec3_t result = vec3(
        matrix.m00 * position.x + matrix.m10 * position.y + matrix.m20 * position.z + matrix.m30,
        matrix.m01 * position.x + matrix.m11 * position.y + matrix.m21 * position.z + matrix.m31,
        matrix.m02 * position.x + matrix.m12 * position.y + matrix.m22 * position.z + matrix.m32);

    float w = matrix.m03 * position.x + matrix.m13 * position.y + matrix.m23 * position.z + matrix.m33;
    if (w != 0 && w != 1)
        return vec3(result.x / w, result.y / w, result.z / w);

    return result;
}

/**
 * Multiplies a 4x4 matrix with a 3D vector representing a direction in 3D space.
 *
 * Before the matrix multiplication the vector is first expanded to a 4D vector
 * (x, y, z, 0). For directions the 4th component is set to 0 because directions
 * are only rotated, not translated. After the multiplication the vector is
 * reduced to 3D again by dividing through the 4th component (if it's not 0 or
 * 1). This is necessary because the matrix might contains something other than
 * (0, 0, 0, 1) in the bottom row which might set w to something other than 0
 * or 1.
 */
vec3_t m4_mul_dir(mat4_t matrix, vec3_t direction) {
    vec3_t result = vec3(
        matrix.m00 * direction.x + matrix.m10 * direction.y + matrix.m20 * direction.z,
        matrix.m01 * direction.x + matrix.m11 * direction.y + matrix.m21 * direction.z,
        matrix.m02 * direction.x + matrix.m12 * direction.y + matrix.m22 * direction.z);

    float w = matrix.m03 * direction.x + matrix.m13 * direction.y + matrix.m23 * direction.z;
    if (w != 0 && w != 1)
        return vec3(result.x / w, result.y / w, result.z / w);

    return result;
}

void m4_print(mat4_t matrix) {
    m4_fprintp(stdout, matrix, 6, 2);
}

void m4_printp(mat4_t matrix, int width, int precision) {
    m4_fprintp(stdout, matrix, width, precision);
}

void m4_fprint(FILE* stream, mat4_t matrix) {
    m4_fprintp(stream, matrix, 6, 2);
}

void m4_fprintp(FILE* stream, mat4_t matrix, int width, int precision) {
    mat4_t m = matrix;
    int w = width, p = precision;
    for (int r = 0; r < 4; r++) {
        fprintf(stream, "| %*.*f %*.*f %*.*f %*.*f |\n",
                w, p, m.m[0][r], w, p, m.m[1][r], w, p, m.m[2][r], w, p, m.m[3][r]);
    }
}

void v3_fprintf(FILE* stream, vec3_t v) {
    fprintf(stream, "(%.2f,%.2f,%.2f)\n", v.x, v.y, v.z);
}

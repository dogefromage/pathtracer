#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cmath>
#include <ostream>
#include <vector>

#include "headers.h"

#define TEST(x) (x)
// #define TEST(x) test_finite(x)

// #ifdef USE_CPU_RENDER
#define ADD(x, y) TEST((x) + (y))
#define SUB(x, y) TEST((x) - (y))
#define MUL(x, y) TEST((x) * (y))
#define DIV(x, y) TEST((x) / (y))
#define FMADD(x, y, z) TEST((x) * (y) + (z)) // Fused multiply-add
#define SQRT(x) TEST(sqrtf(x))
// #else
// #define ADD(x, y) (__fadd_rn((x), (y)))
// #define SUB(x, y) (__fsub_rn((x), (y)))
// #define MUL(x, y) (__fmul_rn((x), (y)))
// #define DIV(x, y) (__frcp_rn((y)) * (x))  // Fast reciprocal multiply
// #define FMADD(x, y, z) (__fmaf_rn((x), (y), (z)))
// #define SQRT(x) (sqrtf(x))
// #endif

#define MATH_PLATFORM __host__ __device__

MATH_PLATFORM inline float test_finite(float x) {
    assert(isfinite(x));
    return x;
}

template <typename T> struct fixed_array {
    uint32_t count;
    T *items;

    MATH_PLATFORM const T &operator[](size_t k) const {
        assert(k < count);
        return items[k];
    }

    MATH_PLATFORM T &operator[](size_t k) {
        assert(k < count);
        return items[k];
    }
};

template <typename T>
void fixed_array_from_vector(fixed_array<T> &dest, const std::vector<T> &src) {
    dest.count = src.size();
    dest.items = (T *)malloc(sizeof(T) * dest.count);
    std::copy(src.begin(), src.end(), dest.items);
}

struct Vec3 {
    float x, y, z;

    MATH_PLATFORM static Vec3 Zero() {
        return {0, 0, 0};
    }
    MATH_PLATFORM static Vec3 Const(float c) {
        return {c, c, c};
    }

    MATH_PLATFORM Vec3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {
    }

    MATH_PLATFORM void set(float a) {
        x = y = z = a;
    }

    MATH_PLATFORM void snprint(char *buffer, int n) {
        snprintf(buffer, n, "(%.2f, %.2f, %.2f)\n", x, y, z);
    }

    MATH_PLATFORM void print() const {
        printf("(%.2f, %.2f, %.2f)\n", x, y, z);
    }

    MATH_PLATFORM Vec3 operator+(const Vec3 &other) const {
        return Vec3(ADD(x, other.x), ADD(y, other.y), ADD(z, other.z));
    }

    MATH_PLATFORM Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    MATH_PLATFORM Vec3 operator-(const Vec3 &other) const {
        return Vec3(SUB(x, other.x), SUB(y, other.y), SUB(z, other.z));
    }

    MATH_PLATFORM Vec3 operator*(const Vec3 &other) const {
        return Vec3(MUL(x, other.x), MUL(y, other.y), MUL(z, other.z));
    }

    MATH_PLATFORM Vec3 operator*(float scalar) const {
        return Vec3(MUL(x, scalar), MUL(y, scalar), MUL(z, scalar));
    }

    MATH_PLATFORM Vec3 operator/(const Vec3 &other) const {
        return Vec3(DIV(x, other.x), DIV(y, other.y), DIV(z, other.z));
    }

    MATH_PLATFORM Vec3 operator/(float scalar) const {
        float t = DIV(1.0, scalar);
        return Vec3(MUL(x, t), MUL(y, t), MUL(z, t));
    }

    MATH_PLATFORM float dot(const Vec3 &other) const {
        return FMADD(x, other.x, FMADD(y, other.y, MUL(z, other.z)));
    }

    MATH_PLATFORM Vec3 cross(const Vec3 &other) const {
        return Vec3(SUB(MUL(y, other.z), MUL(z, other.y)),
                    SUB(MUL(z, other.x), MUL(x, other.z)),
                    SUB(MUL(x, other.y), MUL(y, other.x)));
    }

    MATH_PLATFORM float magnitude() const {
        return SQRT(dot(*this));
    }

    MATH_PLATFORM Vec3 normalized() const {
        float t = DIV(1.0, magnitude());
        return Vec3(MUL(x, t), MUL(y, t), MUL(z, t));
    }

    MATH_PLATFORM void normalize() {
        float t = DIV(1.0, magnitude());
        *this = Vec3(MUL(x, t), MUL(y, t), MUL(z, t));
    }

    MATH_PLATFORM bool epsilonEquals(const Vec3 &other, float epsilon = 1e-6f) const {
        return (fabs(SUB(x, other.x)) < epsilon) && (fabs(SUB(y, other.y)) < epsilon) &&
               (fabs(SUB(z, other.z)) < epsilon);
    }

    MATH_PLATFORM float operator[](int k) const {
        switch (k) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        }
        return 0.0;
    }

    MATH_PLATFORM float &operator[](int k) {
        switch (k) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        }
        return x;
    }

    MATH_PLATFORM bool operator==(const Vec3 &other) const {
        return epsilonEquals(other);
    }

    MATH_PLATFORM bool operator!=(const Vec3 &other) const {
        return !(*this == other);
    }

    MATH_PLATFORM static Vec3 min(const Vec3 &a, const Vec3 &b) {
        return Vec3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
    }

    MATH_PLATFORM static Vec3 max(const Vec3 &a, const Vec3 &b) {
        return Vec3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
    }

    MATH_PLATFORM float minComponent() const {
        return (x < y && x < z) ? x : (y < z ? y : z);
    }

    MATH_PLATFORM float maxComponent() const {
        return (x > y && x > z) ? x : (y > z ? y : z);
    }

    MATH_PLATFORM Vec3 map(float f(float)) const {
        return {f(x), f(y), f(z)};
    }

    MATH_PLATFORM void checkFinite() const {
        assert(isfinite(x));
        assert(isfinite(y));
        assert(isfinite(z));
    }
};

inline MATH_PLATFORM void operator+=(Vec3 &lhs, const Vec3 &rhs) {
    lhs = lhs + rhs;
}

inline MATH_PLATFORM void operator-=(Vec3 &lhs, const Vec3 &rhs) {
    lhs = lhs - rhs;
}

inline MATH_PLATFORM void operator*=(Vec3 &lhs, const Vec3 &rhs) {
    lhs = lhs * rhs;
}

inline MATH_PLATFORM void operator*=(Vec3 &lhs, float a) {
    lhs = lhs * a;
}

inline MATH_PLATFORM void operator/=(Vec3 &lhs, float a) {
    lhs = lhs / a;
}

inline MATH_PLATFORM Vec3 operator+(float a, Vec3 v) {
    return v + a;
}

inline MATH_PLATFORM Vec3 operator-(float a, Vec3 v) {
    return a + (-v);
}

inline MATH_PLATFORM Vec3 operator*(float a, Vec3 v) {
    return v * a;
}

inline MATH_PLATFORM Vec3 operator/(float a, Vec3 v) {
    return Vec3::Const(a) / v;
}

inline std::ostream &operator<<(std::ostream &os, const Vec3 &a) {
    char buf[128];
    snprintf(buf, 128, "(%.2f, %.2f, %.2f)\n", a.x, a.y, a.z);
    os << buf;
    return os;
}

struct Mat3 {
    float m[3][3];

    MATH_PLATFORM Mat3(float m00, float m01, float m02, float m10, float m11, float m12,
                       float m20, float m21, float m22) {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
    }

    MATH_PLATFORM Vec3 operator*(const Vec3 &v) const {
        return Vec3(ADD(ADD(MUL(m[0][0], v.x), MUL(m[0][1], v.y)), MUL(m[0][2], v.z)),
                    ADD(ADD(MUL(m[1][0], v.x), MUL(m[1][1], v.y)), MUL(m[1][2], v.z)),
                    ADD(ADD(MUL(m[2][0], v.x), MUL(m[2][1], v.y)), MUL(m[2][2], v.z)));
    }
};

struct Vec4 {
    float x, y, z, w;

    // Constructor
    MATH_PLATFORM Vec4(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f)
        : x(x), y(y), z(z), w(w) {
    }

    // Set all components to the same value
    MATH_PLATFORM void set(float a) {
        x = y = z = w = a;
    }

    // Print vector components
    void print() const {
        printf("(%.2f, %.2f, %.2f, %.2f)\n", x, y, z, w);
    }

    // Add two Vec4
    MATH_PLATFORM Vec4 operator+(const Vec4 &other) const {
        return Vec4(ADD(x, other.x), ADD(y, other.y), ADD(z, other.z), ADD(w, other.w));
    }

    // Negate the vector
    MATH_PLATFORM Vec4 operator-() const {
        return Vec4(-x, -y, -z, -w);
    }

    // Subtract two Vec4
    MATH_PLATFORM Vec4 operator-(const Vec4 &other) const {
        return Vec4(SUB(x, other.x), SUB(y, other.y), SUB(z, other.z), SUB(w, other.w));
    }

    // Multiply component-wise
    MATH_PLATFORM Vec4 operator*(const Vec4 &other) const {
        return Vec4(MUL(x, other.x), MUL(y, other.y), MUL(z, other.z), MUL(w, other.w));
    }

    // Scale by a scalar
    MATH_PLATFORM Vec4 operator*(float scalar) const {
        return Vec4(MUL(x, scalar), MUL(y, scalar), MUL(z, scalar), MUL(w, scalar));
    }

    // Divide component-wise
    MATH_PLATFORM Vec4 operator/(const Vec4 &other) const {
        return Vec4(DIV(x, other.x), DIV(y, other.y), DIV(z, other.z), DIV(w, other.w));
    }

    // Divide by scalar
    MATH_PLATFORM Vec4 operator/(float scalar) const {
        float t = DIV(1.0, scalar);
        return Vec4(MUL(x, t), MUL(y, t), MUL(z, t), MUL(w, t));
    }

    // Dot product
    MATH_PLATFORM float dot(const Vec4 &other) const {
        return FMADD(x, other.x, FMADD(y, other.y, FMADD(z, other.z, MUL(w, other.w))));
    }

    // Magnitude (length)
    MATH_PLATFORM float magnitude() const {
        return SQRT(dot(*this));
    }

    // Normalized vector
    MATH_PLATFORM Vec4 normalized() const {
        float t = DIV(1.0, magnitude());
        return Vec4(MUL(x, t), MUL(y, t), MUL(z, t), MUL(w, t));
    }

    // Normalize this vector
    MATH_PLATFORM void normalize() {
        float t = DIV(1.0, magnitude());
        *this = Vec4(MUL(x, t), MUL(y, t), MUL(z, t), MUL(w, t));
    }

    // Epsilon-based equality check
    MATH_PLATFORM bool epsilonEquals(const Vec4 &other, float epsilon = 1e-5f) const {
        return (fabs(SUB(x, other.x)) < epsilon) && (fabs(SUB(y, other.y)) < epsilon) &&
               (fabs(SUB(z, other.z)) < epsilon) && (fabs(SUB(w, other.w)) < epsilon);
    }

    // Access components by index (const)
    MATH_PLATFORM float operator[](int k) const {
        switch (k) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        }
        return 0.0;
    }

    // Access components by index (mutable)
    MATH_PLATFORM float &operator[](int k) {
        switch (k) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        case 3:
            return w;
        }
        return x;
    }

    // Equality operators
    MATH_PLATFORM bool operator==(const Vec4 &other) const {
        return epsilonEquals(other);
    }

    MATH_PLATFORM bool operator!=(const Vec4 &other) const {
        return !(*this == other);
    }

    // Min and max
    MATH_PLATFORM static Vec4 min(const Vec4 &a, const Vec4 &b) {
        return Vec4(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z,
                    a.w < b.w ? a.w : b.w);
    }

    MATH_PLATFORM static Vec4 max(const Vec4 &a, const Vec4 &b) {
        return Vec4(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z,
                    a.w > b.w ? a.w : b.w);
    }

    // Min and max components
    MATH_PLATFORM float minComponent() const {
        return fmin(fmin(x, y), fmin(z, w));
    }

    MATH_PLATFORM float maxComponent() const {
        return fmax(fmax(x, y), fmax(z, w));
    }

    static MATH_PLATFORM Vec4 toHomogeneous(const Vec3 &x) {
        return Vec4(x.x, x.y, x.z, 1.0);
    }

    MATH_PLATFORM Vec3 dehomogenise() {
        assert(w != 0 && "Point not defined in 3d");
        return Vec3(x, y, z) / w;
    }
};

// Inline operator implementations
inline MATH_PLATFORM void operator+=(Vec4 &lhs, const Vec4 &rhs) {
    lhs = lhs + rhs;
}

inline MATH_PLATFORM void operator-=(Vec4 &lhs, const Vec4 &rhs) {
    lhs = lhs - rhs;
}

inline MATH_PLATFORM void operator*=(Vec4 &lhs, const Vec4 &rhs) {
    lhs = lhs * rhs;
}

inline MATH_PLATFORM void operator*=(Vec4 &lhs, float scalar) {
    lhs = lhs * scalar;
}

inline MATH_PLATFORM void operator/=(Vec4 &lhs, float scalar) {
    lhs = lhs / scalar;
}

inline MATH_PLATFORM Vec4 operator*(float scalar, Vec4 v) {
    return v * scalar;
}

struct Mat4 {
    float m[4][4];

    // Constructor to initialize the matrix with individual values
    MATH_PLATFORM Mat4(float m00, float m01, float m02, float m03, float m10, float m11,
                       float m12, float m13, float m20, float m21, float m22, float m23,
                       float m30, float m31, float m32, float m33) {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[0][3] = m03;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[1][3] = m13;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
        m[2][3] = m23;
        m[3][0] = m30;
        m[3][1] = m31;
        m[3][2] = m32;
        m[3][3] = m33;
    }

    static MATH_PLATFORM Mat4 Identity() {
        return Mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    }

    // Matrix-vector multiplication
    MATH_PLATFORM Vec4 operator*(const Vec4 &v) const {
        return Vec4(ADD(ADD(ADD(MUL(m[0][0], v.x), MUL(m[0][1], v.y)), MUL(m[0][2], v.z)),
                        MUL(m[0][3], v.w)),
                    ADD(ADD(ADD(MUL(m[1][0], v.x), MUL(m[1][1], v.y)), MUL(m[1][2], v.z)),
                        MUL(m[1][3], v.w)),
                    ADD(ADD(ADD(MUL(m[2][0], v.x), MUL(m[2][1], v.y)), MUL(m[2][2], v.z)),
                        MUL(m[2][3], v.w)),
                    ADD(ADD(ADD(MUL(m[3][0], v.x), MUL(m[3][1], v.y)), MUL(m[3][2], v.z)),
                        MUL(m[3][3], v.w)));
    }

    // Matrix-matrix multiplication
    MATH_PLATFORM Mat4 operator*(const Mat4 &rhs) const {
        Mat4 result(0, 0, 0, 0, // Initializing to zeros
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.m[i][j] =
                    ADD(ADD(ADD(MUL(m[i][0], rhs.m[0][j]), MUL(m[i][1], rhs.m[1][j])),
                            MUL(m[i][2], rhs.m[2][j])),
                        MUL(m[i][3], rhs.m[3][j]));
            }
        }
        return result;
    }

    MATH_PLATFORM Mat3 getLinearPart() const {
        // assumes rotation/translation/scale matrix
        assert(m[3][0] == 0 && m[3][1] == 0 && m[3][2] == 0);

        return Mat3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1],
                    m[2][2]);
    }
};

struct AABB {
    Vec3 min = Vec3::Const(1e30);
    Vec3 max = Vec3::Const(-1e30);

    MATH_PLATFORM void reset() {
        min = Vec3::Const(1e30);
        max = Vec3::Const(-1e30);
    }

    MATH_PLATFORM void grow(const Vec3 &p) {
        min = Vec3::min(min, p);
        max = Vec3::max(max, p);
    }

    MATH_PLATFORM void grow(const AABB &other) {
        min = Vec3::min(min, other.min);
        max = Vec3::max(max, other.max);
    }

    MATH_PLATFORM float area() const {
        Vec3 e = max - min;
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }

    MATH_PLATFORM bool contains(const Vec3 &x) const {
        return (Vec3::min(min, x) == min) && (Vec3::max(max, x) == max);
    }

    __host__ void print() {
        printf("[ (%.2f,%.2f,%.2f), (%.2f,%.2f,%.2f) ]\n", min.x, min.y, min.z, max.x, max.y,
               max.z);
    }
};

struct Ray {
    Vec3 o, r;
};

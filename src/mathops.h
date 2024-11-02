#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include "config.h"
#include <stdio.h>
#include <ostream>

// #ifdef USE_CPU_RENDER
#define ADD(x, y) ((x) + (y))
#define SUB(x, y) ((x) - (y))
#define MUL(x, y) ((x) * (y))
#define DIV(x, y) ((x) / (y))
#define FMADD(x, y, z) ((x) * (y) + (z))  // Fused multiply-add
#define SQRT(x) (sqrtf(x))
// #else
// #define ADD(x, y) (__fadd_rn((x), (y)))
// #define SUB(x, y) (__fsub_rn((x), (y)))
// #define MUL(x, y) (__fmul_rn((x), (y)))
// #define DIV(x, y) (__frcp_rn((y)) * (x))  // Fast reciprocal multiply
// #define FMADD(x, y, z) (__fmaf_rn((x), (y), (z)))
// #define SQRT(x) (sqrtf(x))
// #endif

#define VEC3_PLATFORM __host__ __device__

struct Vec3 {
    float x, y, z;

    VEC3_PLATFORM Vec3(float x = 0.0f, float y = 0.0f, float z = 0.0f)
        : x(x), y(y), z(z) {}

    VEC3_PLATFORM void set(float a) {
        x = y = z = a;
    }

    void print() {
        printf("(%.2f, %.2f, %.2f)\n", x, y, z);
    }

    VEC3_PLATFORM Vec3 operator+(const Vec3& other) const {
        return Vec3(ADD(x, other.x), ADD(y, other.y), ADD(z, other.z));
    }

    VEC3_PLATFORM Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    VEC3_PLATFORM Vec3 operator-(const Vec3& other) const {
        return Vec3(SUB(x, other.x), SUB(y, other.y), SUB(z, other.z));
    }

    VEC3_PLATFORM Vec3 operator*(const Vec3& other) const {
        return Vec3(MUL(x, other.x), MUL(y, other.y), MUL(z, other.z));
    }

    VEC3_PLATFORM Vec3 operator*(float scalar) const {
        return Vec3(MUL(x, scalar), MUL(y, scalar), MUL(z, scalar));
    }

    VEC3_PLATFORM Vec3 operator/(const Vec3& other) const {
        return Vec3(DIV(x, other.x), DIV(y, other.y), DIV(z, other.z));
    }

    VEC3_PLATFORM Vec3 operator/(float scalar) const {
        float t = DIV(1.0, scalar);
        return Vec3(MUL(x, t), MUL(y, t), MUL(z, t));
    }

    VEC3_PLATFORM float dot(const Vec3& other) const {
        return FMADD(x, other.x, FMADD(y, other.y, MUL(z, other.z)));
    }

    VEC3_PLATFORM Vec3 cross(const Vec3& other) const {
        return Vec3(
            SUB(MUL(y, other.z), MUL(z, other.y)),
            SUB(MUL(z, other.x), MUL(x, other.z)),
            SUB(MUL(x, other.y), MUL(y, other.x)));
    }

    VEC3_PLATFORM float magnitude() const {
        return SQRT(dot(*this));
    }

    VEC3_PLATFORM Vec3 normalized() const {
        float t = DIV(1.0, magnitude());
        return Vec3(MUL(x, t), MUL(y, t), MUL(z, t));
    }

    VEC3_PLATFORM void normalize() {
        float t = DIV(1.0, magnitude());
        *this = Vec3(MUL(x, t), MUL(y, t), MUL(z, t));
    }

    VEC3_PLATFORM bool epsilonEquals(const Vec3& other, float epsilon = 1e-5f) const {
        return (fabs(SUB(x, other.x)) < epsilon) &&
               (fabs(SUB(y, other.y)) < epsilon) &&
               (fabs(SUB(z, other.z)) < epsilon);
    }

    VEC3_PLATFORM float operator[](int k) const {
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

    VEC3_PLATFORM float& operator[](int k) {
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

    VEC3_PLATFORM bool operator==(const Vec3& other) const {
        return epsilonEquals(other);
    }

    VEC3_PLATFORM bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }

    VEC3_PLATFORM static Vec3 min(const Vec3& a, const Vec3& b) {
        return Vec3(
            a.x < b.x ? a.x : b.x,
            a.y < b.y ? a.y : b.y,
            a.z < b.z ? a.z : b.z);
    }

    VEC3_PLATFORM static Vec3 max(const Vec3& a, const Vec3& b) {
        return Vec3(
            a.x > b.x ? a.x : b.x,
            a.y > b.y ? a.y : b.y,
            a.z > b.z ? a.z : b.z);
    }

    VEC3_PLATFORM float minComponent() const {
        return (x < y && x < z) ? x : (y < z ? y : z);
    }

    VEC3_PLATFORM float maxComponent() const {
        return (x > y && x > z) ? x : (y > z ? y : z);
    }
};

inline VEC3_PLATFORM void operator+=(Vec3& lhs, const Vec3& rhs) {
    lhs = lhs + rhs;
}

inline VEC3_PLATFORM void operator-=(Vec3& lhs, const Vec3& rhs) {
    lhs = lhs - rhs;
}

inline VEC3_PLATFORM void operator*=(Vec3& lhs, const Vec3& rhs) {
    lhs = lhs * rhs;
}

inline VEC3_PLATFORM void operator*=(Vec3& lhs, float a) {
    lhs = lhs * a;
}

inline VEC3_PLATFORM void operator/=(Vec3& lhs, float a) {
    lhs = lhs / a;
}

inline VEC3_PLATFORM Vec3 operator*(float a, Vec3 v) {
    return v * a;
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& a) {
    char buf[128];
    snprintf(buf, 128, "(%.2f, %.2f, %.2f)\n", a.x, a.y, a.z);
    os << buf;
    return os;
}

struct Mat3 {
    float m[3][3];

    VEC3_PLATFORM Mat3(float m00, float m01, float m02,
                   float m10, float m11, float m12,
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

    VEC3_PLATFORM Vec3 operator*(const Vec3& v) const {
        return Vec3(
            ADD(ADD(MUL(m[0][0], v.x), MUL(m[0][1], v.y)), MUL(m[0][2], v.z)),
            ADD(ADD(MUL(m[1][0], v.x), MUL(m[1][1], v.y)), MUL(m[1][2], v.z)),
            ADD(ADD(MUL(m[2][0], v.x), MUL(m[2][1], v.y)), MUL(m[2][2], v.z)));
    }
};

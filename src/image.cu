#include <stdio.h>

#include <cstdint>

#include "image.h"
#include "logger.h"

#define STBI_ONLY_PNG
#include "stb_image_write.h"

uint8_t clamp_256(float c) {
    if (c < 0.0)
        c = 0.0;
    if (c >= 1.0)
        c = 1.0;
    return (uint8_t)(255 * c);
}

// https://64.github.io/tonemapping/
float luminance(Vec3 v) {
    return v.dot(Vec3(0.2126f, 0.7152f, 0.0722f));
}

Vec3 change_luminance(Vec3 c_in, float l_out) {
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

Vec3 reinhard_extended_luminance(Vec3 v, float max_white_l) {
    float l_old = luminance(v);
    float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    float l_new = numerator / (1.0f + l_old);
    return change_luminance(v, l_new);
}

Vec3 reinhard_simple(Vec3 c) {
    return c / (1 + c);
}

float linear_to_srgb_gamma(float c) {
    // source: GPT
    if (c <= 0.0031308) {
        return 12.92 * c;
    } else {
        return 1.055 * powf(c, 1 / 2.4) - 0.055;
    }
}

const char *get_file_extension(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot || dot == path) {
        return NULL;
    }
    return dot + 1;
}

void check_sanity(Vec3 v, int x, int y) {
    if (!isfinite(v.x)) {
        log_error("pixel.x = %f is not finite at (%d,%d)\n", v.x, x, y);
        exit(EXIT_FAILURE);
    }
    if (!isfinite(v.y)) {
        log_error("pixel.y = %f is not finite at (%d,%d)\n", v.y, x, y);
        exit(EXIT_FAILURE);
    }
    if (!isfinite(v.z)) {
        log_error("pixel.z = %f is not finite at (%d,%d)\n", v.z, x, y);
        exit(EXIT_FAILURE);
    }
}

void write_image(Vec3 *linearpixels, int width, int height, const char *filename) {
    uint8_t *buf = (uint8_t *)malloc(width * height * 3 * sizeof(uint8_t));
    assert(buf && "malloc");
    // apply tonemapping, hdr, etc...

    float whitePointLuminance = luminance({1, 1, 1});

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const Vec3 &c = linearpixels[y * width + x];
            check_sanity(c, x, y);
            whitePointLuminance = fmaxf(whitePointLuminance, luminance(c));
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3 c = linearpixels[y * width + x];

            c = reinhard_extended_luminance(c, whitePointLuminance);
            c = c.map(linear_to_srgb_gamma);

            int i = (height - 1 - y) * width + x;
            buf[3 * i + 0] = clamp_256(c.x);
            buf[3 * i + 1] = clamp_256(c.y);
            buf[3 * i + 2] = clamp_256(c.z);
        }
    }

    const char *extension = get_file_extension(filename);

    if (!strcmp(extension, "png")) {
        stbi_write_png(filename, width, height, 3, (void *)buf, 3 * width);
    } else {
        log_error("Unsupported output image type: %s\n", extension);
    }

    free(buf);
    buf = NULL;
}

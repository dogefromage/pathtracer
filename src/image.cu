#include <stdio.h>

#include "image.h"
#include <cstdint>

uint8_t clamp_channel(float c) {
    if (c < 0.0) c = 0.0;
    if (c > 1.0) c = 1.0;
    return (uint8_t)(255 * c);
}

Vec3 linear_to_gamma(Vec3 c) {
    return Vec3(
        sqrtf(c.x),
        sqrtf(c.y),
        sqrtf(c.z)
    );
}

Vec3 color_correct(Vec3 c) {
    return linear_to_gamma(c);
}

void write_bmp(Vec3* pixels, int width, int height, const char* filename) {
    unsigned int header[14];
    int i, j;
    FILE* fp = fopen(filename, "wb");
    uint8_t pad[3] = {0, 0, 0};

    header[0] = 0x4d420000;
    header[1] = 54 + 3 * height * width;
    header[2] = 0;
    header[3] = 54;
    header[4] = 40;
    header[5] = width;
    header[6] = height;
    header[7] = 0x00180001;
    header[8] = 0;
    header[9] = 3 * width * height;
    header[10] = header[11] = header[12] = header[13] = 0;

    fwrite((uint8_t*)header + 2, 1, 54, fp);
    fflush(fp);

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            int pixel = i * width + j;
            Vec3 c = pixels[pixel];
            Vec3 c_corr = color_correct(c);
            uint8_t R = clamp_channel(c_corr.x);
            uint8_t G = clamp_channel(c_corr.y);
            uint8_t B = clamp_channel(c_corr.z);
            fwrite(&B, 1, 1, fp);
            fwrite(&G, 1, 1, fp);
            fwrite(&R, 1, 1, fp);
        }
        fwrite(pad, width % 4, 1, fp);
    }

    fclose(fp);
}
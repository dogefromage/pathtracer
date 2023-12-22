#include "image.h"

#include <stdio.h>

static char clamp_channel(float c) {
    if (c < 0.0) c = 0.0;
    if (c > 1.0) c = 1.0;
    return (char)(255 * c);
}

// https://stackoverflow.com/questions/27613601/rendering-an-image-using-c
void write_bmp(vec3_t* pixels, int width, int height, const char* filename) {
    unsigned int header[14];
    int i, j;
    FILE* fp = fopen(filename, "wb");
    unsigned char pad[3] = {0, 0, 0};

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

    fwrite((char*)header + 2, 1, 54, fp);
    fflush(fp);

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            int pixel = i * width + j;
            unsigned char R = clamp_channel(pixels[pixel].x);
            unsigned char G = clamp_channel(pixels[pixel].y);
            unsigned char B = clamp_channel(pixels[pixel].z);
            fwrite(&B, 1, 1, fp);
            fwrite(&G, 1, 1, fp);
            fwrite(&R, 1, 1, fp);
        }
        fwrite(pad, width % 4, 1, fp);
    }

    fclose(fp);
}
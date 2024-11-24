#pragma once
#include <functional>

#include <string>
#include "mathops.h"

typedef struct {
    struct {
        int width, height;
    } output;

    struct {
        int seed, samples, samples_per_round;
    } sampling;

    struct {
        Vec3 clear_color;
    } world;

} settings_t;

void settings_parse_yaml(settings_t& settings, const std::string& filename);

void settings_print(settings_t& settings);
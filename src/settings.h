#pragma once
#include <functional>

#include <string>
#include "mathops.h"

typedef struct {
    struct {
        int width = 600, height = 600;
    } output;

    struct {
        int seed = 42, samples = 100, samples_per_round = 10;
    } sampling;

    struct {
        Vec3 clear_color;
    } world;

} settings_t;

void settings_parse_yaml(settings_t& settings, const std::string& filename);

void settings_print(const settings_t& settings);
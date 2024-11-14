#pragma once
#include <functional>

#include <string>

typedef struct {
    struct {
        int width = 600, height = 600;
    } output;

    struct {
        int seed = 42, samples = 100, samples_per_round = 10;
    } sampling;
        
    struct {
        float sensor_height = 0.2, focal_length = 0.4;
    } camera;

    struct {
        bool verbose = false;
    } debug;

} settings_t;

void settings_parse_yaml(settings_t& settings, const std::string& filename);

void settings_print(const settings_t& settings);
#include "config.h"

static int tok_vec3(void *field, char *input) {
    // printf("tok_vec3:  %s\n", input);
    Vec3 &v = *(Vec3 *)field;
    if (3 != sscanf(input, "%f %f %f", &v.x, &v.y, &v.z)) {
        fprintf(stderr, "unable to parse vec3 arg input=%s\n", input);
        return 1;
    }
    return 0;
}

static int tok_int(void *field, char *input) {
    // printf("tok_int:  %s\n", input);
    if (1 != sscanf(input, "%d", (int *)field)) {
        fprintf(stderr, "unable to parse int arg input=%s\n", input);
        return 1;
    }
    return 0;
}

static int tok_path(void *field, char *input) {
    // printf("tok_path: %s\n", input);
    strncpy(*(path_t *)field, input, sizeof(path_t));
    return 0;
}

static void parse(int argc, char *argv[], void *field, const char *name,
                  int tokenizer(void *, char *)) {
    for (int i = 1; i < argc - 1; i++) {
        // printf("TEST %s    %s\n", argv[i], name);
        if (!strcmp(argv[i], name)) {
            // ignore the error of tokenizer because i can
            tokenizer(field, argv[i + 1]);
            return;
        }
    }
    fprintf(stderr, "unable to find config item '%s'. all items are required.\n", name);
}

int load_config(config_t *cfg, int argc, char *argv[]) {
    // fprintf(stderr, "unable to open logfile\n");

    parse(argc, argv, &cfg->world_clear_color, "--world-clear-color", tok_vec3);

    parse(argc, argv, &cfg->seed, "--sampling-seed", tok_int);
    parse(argc, argv, &cfg->samples, "--sampling-samples", tok_int);
    parse(argc, argv, &cfg->samples_every_update, "--sampling-samples-every-update", tok_int);

    parse(argc, argv, &cfg->resolution_x, "--output-resolution-x", tok_int);
    parse(argc, argv, &cfg->resolution_y, "--output-resolution-y", tok_int);

    parse(argc, argv, &cfg->log_level, "--logger-log-level", tok_int);
    parse(argc, argv, &cfg->log_stdout, "--logger-log-stdout", tok_int);

    parse(argc, argv, &cfg->path_gltf, "--path-gltf", tok_path);
    parse(argc, argv, &cfg->dir_output, "--dir-output", tok_path);

    return 0;
}

// int resolution_x, resolution_y;
// int samples, seed, samples_every_update;

// Vec3 world_clear_color;

// char* scene_gltf;

// int log_level, log_stdout;
// char* log_file;

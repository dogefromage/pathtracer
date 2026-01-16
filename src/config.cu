#include "config.h"

static int tok_vec3(void *field, char *input) {
    // printf("tok_vec3:  %s\n", input);
    Vec3 &v = *(Vec3 *)field;
    if (3 != sscanf(input, "%f %f %f", &v.x, &v.y, &v.z)) {
        fprintf(stderr, "unable to parse vec3 arg input=%s\n", input);
        return 1;
    }
    // v.print();
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

static int tok_float(void *field, char *input) {
    // printf("tok_float:  %s\n", input);
    if (1 != sscanf(input, "%f", (float *)field)) {
        fprintf(stderr, "unable to parse float arg input=%s\n", input);
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
                  int tokenizer(void *, char *), bool optional = false) {

    for (int i = 1; i < argc - 1; i++) {
        // printf("TEST %s    %s\n", argv[i], name);
        if (!strcmp(argv[i], name)) {
            // ignore the error of tokenizer because i can
            tokenizer(field, argv[i + 1]);
            return;
        }
    }

    if (!optional) {
        fprintf(stderr, "ERROR unable to find required config item '%s'.\n", name);
    }
}

int load_config(config_t *cfg, int argc, char *argv[]) {

    printf("Reading config from args: ");
    for (int i = 0; i < argc; i++) {
        printf("%s ", argv[i]);
    }
    printf("\n\n");

    parse(argc, argv, &cfg->world_clear_color, "--world-clear-color", tok_vec3, true);
    parse(argc, argv, &cfg->world_clear_color_texture, "--world-clear-color-texture", tok_path);

    parse(argc, argv, &cfg->seed, "--sampling-seed", tok_int);
    parse(argc, argv, &cfg->samples, "--sampling-samples", tok_int);
    parse(argc, argv, &cfg->samples_every_update, "--sampling-samples-every-update", tok_int);

    parse(argc, argv, &cfg->resolution_x, "--output-resolution-x", tok_int);
    parse(argc, argv, &cfg->resolution_y, "--output-resolution-y", tok_int);
    parse(argc, argv, &cfg->output_exposure, "--output-exposure", tok_float);

    parse(argc, argv, &cfg->log_level, "--logger-log-level", tok_int);
    parse(argc, argv, &cfg->log_stdout, "--logger-log-stdout", tok_int);

    parse(argc, argv, &cfg->path_gltf, "--path-gltf", tok_path);
    parse(argc, argv, &cfg->dir_output, "--dir-output", tok_path);

    parse(argc, argv, &cfg->default_camera_position, "--default-camera-position", tok_vec3);
    parse(argc, argv, &cfg->default_camera_target, "--default-camera-target", tok_vec3);
    parse(argc, argv, &cfg->default_camera_updir, "--default-camera-updir", tok_vec3);
    parse(argc, argv, &cfg->default_camera_yfov, "--default-camera-yfov", tok_float);

    return 0;
}

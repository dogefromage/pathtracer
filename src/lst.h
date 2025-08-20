#pragma once
#include <cstdint>

#include "mathops.h"
#include "random.h"
#include "scene.h"

enum lst_source_type {
    LST_SOURCE_FACE,
    LST_SOURCE_LIGHT,
};

typedef struct {
    lst_source_type type;
    uint32_t index;
} lst_node_t;

typedef struct {
    fixed_array<lst_node_t> nodes;
} lst_t;

void lst_build(lst_t &lst, const scene_t &scene);
void lst_free_host(lst_t &lst);
void lst_copy_device(lst_t **d_lst, const lst_t *h_lst);
void lst_free_device(lst_t *d_lst);

// typedef struct {
//     Vec3 sourcePosition;
//     Vec3 radiance;
//     float p;
// } light_sample_t;

// PLATFORM void
// lst_sample(light_sample_t& light, const lst_t* lst, const scene_t* scene, rand_state_t&
// rstate);

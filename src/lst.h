#pragma once
#include <cstdint>
#include "mathops.h"
#include "scene.h"
#include "random.h"

typedef struct {
    uint32_t face;
} lst_node_t;

typedef struct {
    lst_node_t* lsnodes;
    uint32_t nodeCount;
} lst_t;

void lst_build(lst_t& lst, const obj_scene_data& scene);
void lst_free_host(lst_t& lst);
__host__ int lst_copy_device(lst_t** d_lst, const lst_t* h_lst);
__host__ int lst_free_device(lst_t* d_lst);

typedef struct {
    Vec3 position;
    Vec3 light;
    float p;
} light_sample_t;

PLATFORM void
lst_sample(light_sample_t& sample, const lst_t* lst, const obj_scene_data* scene, rand_state_t& rstate);

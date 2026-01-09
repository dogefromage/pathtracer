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

class LST {
    CudaLocation _location = CudaLocation::Host;

  public:
    fixed_array<lst_node_t> nodes;

    void build(const Scene &scene);
    void device_from_host(const LST &h_lst);
    void _free();
};

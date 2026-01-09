#include "logger.h"
#include "lst.h"
#include "utils.h"
#include <vector>

void LST::build(const Scene &scene) {
    log_info("Building LST... \n");

    std::vector<lst_node_t> lss;

    for (uint32_t i = 0; i < scene.faces.count; i++) {
        const face_t &face = scene.faces[i];
        // check if emissive
        const material_t &mat = scene.materials[face.material];
        if (mat.emissive.maxComponent() > 0) {
            // face is emissive
            lss.push_back({type : LST_SOURCE_FACE, index : i});
        }
    }

    for (uint32_t i = 0; i < scene.lights.count; i++) {
        lss.push_back({type : LST_SOURCE_LIGHT, index : i});
    }

    nodes.count = lss.size();
    nodes.items = (lst_node_t *)malloc(sizeof(lst_node_t) * nodes.count);
    std::copy(lss.begin(), lss.end(), nodes.items);

    log_info("Done building LST\n");

    if (nodes.count == 0) {
        log_warning("No lights found in scene.\n");
    }
}

void LST::device_from_host(const LST &h_lst) {
    log_info("Copying lst_t to device... \n");

    *this = h_lst;
    size_t totalSize = 0;
    totalSize += copy_device_fixed_array(&this->nodes, &h_lst.nodes);

    char buf[64];
    human_readable_size(buf, totalSize);
    log_info("Done [%s]\n", buf);
}

void LST::_free() {
    if (_location == CudaLocation::Host) {
        free(nodes.items);
        nodes.items = NULL;
    }
    if (_location == CudaLocation::Device) {
        device_free(nodes.items);
    }
}

#include "logger.h"
#include "lst.h"
#include "utils.h"
#include <vector>

void lst_build(lst_t &lst, const Scene &scene) {
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
        // const light_t& light = scene.lights[i];
        lss.push_back({type : LST_SOURCE_LIGHT, index : i});
    }

    // fixed_array_from_vector(lst.nodes, lss);
    lst.nodes.count = lss.size();
    lst.nodes.items = (lst_node_t *)malloc(sizeof(lst_node_t) * lst.nodes.count);
    std::copy(lss.begin(), lss.end(), lst.nodes.items);

    log_info("Done building LST\n");

    if (lst.nodes.count == 0) {
        log_warning("No lights found in scene.\n");
    }
}

void lst_free_host(lst_t &lst) {
    free(lst.nodes.items);
    lst.nodes.items = NULL;
}

void lst_copy_device(lst_t **d_lst, const lst_t *h_lst) {
    log_info("Copying lst_t to device... \n");

    lst_t placeholder = *h_lst;
    size_t totalSize = 0;

    totalSize += copy_device_fixed_array(&placeholder.nodes, &h_lst->nodes);
    totalSize += copy_device_struct(d_lst, &placeholder);

    char buf[64];
    human_readable_size(buf, totalSize);
    log_info("Done [%s]\n", buf);
}

void lst_free_device(lst_t *d_lst) {
    lst_t placeholder;
    copy_host_struct(&placeholder, d_lst);

    device_free(placeholder.nodes.items);
    device_free(d_lst);
}

// __device__ void
// lst_sample_point_light(light_sample_t& sample, const light_t& light, const scene_t* scene,
// rand_state_t& rstate) {
//     sample.p = 1;
//     sample.position = light.position;
//     sample.light = light.color * light.intensity;
// }

// __device__ void
// lst_sample_directional_light(light_sample_t& sample, const light_t& light, const scene_t*
// scene, rand_state_t& rstate) {
//     assert(false);
// }

// __device__ void
// lst_sample(light_sample_t& sample, const lst_t* lst, const scene_t* scene, rand_state_t&
// rstate) {
//     // if (!lst->nodes.count) {
//     //     sample.p = 0;
//     //     return;
//     // }

//     // // pick light node uniformly randoms
//     // int nodeIndex = (int)(lst->nodes.count * random_uniform(rstate));
//     // const lst_node_t& node = lst->nodes[nodeIndex];

//     // if (node.type == LST_SOURCE_FACE) {
//     //     assert(false);
//     // }
//     // if (node.type == LST_SOURCE_LIGHT) {
//     //     const light_t& light = scene->lights[node.index];

//     //     switch (light.type) {
//     //         case LIGHT_POINT:
//     //             lst_sample_point_light(sample, light, scene, rstate);
//     //             break;
//     //         case LIGHT_DIRECTIONAL:
//     //             lst_sample_directional_light(sample, light, scene, rstate);
//     //             break;
//     //         default:
//     //             assert(false);
//     //             break;
//     //     }

//     // }
// }

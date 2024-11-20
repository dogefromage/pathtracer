#include "lst.h"
#include <vector>
#include "utils.h"

void lst_build(lst_t& lst, const obj_scene_data& scene) {
    printf("Building lst_t...  ");

    std::vector<lst_node_t> lss;
    for (uint32_t i = 0; i < scene.face_count; i++) {
        const obj_face& face = scene.face_list[i];
        // check if emissive
        const obj_material& mat = scene.material_list[face.material_index];
        bool isEmissive = mat.emit.maxComponent() > 0;
        if (!isEmissive) {
            continue;
        }
        // face is emissive
        lss.push_back({ i });
    }

    lst.nodeCount = lss.size();
    size_t bufsize = lst.nodeCount * sizeof(lst_node_t);
    lst.lsnodes = (lst_node_t*)malloc(bufsize);
    memcpy(lst.lsnodes, lss.data(), bufsize);

    printf("Done\n");

    if (lst.nodeCount == 0) {
        printf("Warning! No lights found in scene.\n");
    }
}

void lst_free_host(lst_t& lst) {
    free(lst.lsnodes);
    lst.lsnodes = NULL;
}

__host__ int
lst_copy_device(lst_t** d_lst, const lst_t* h_lst) {
    printf("Copying lst_t to device... ");

    lst_t m_lst = *h_lst;

    cudaError_t err;
    size_t curr_bytes, total_bytes;
    total_bytes = 0;

    curr_bytes = sizeof(lst_node_t) * m_lst.nodeCount;
    total_bytes += curr_bytes;
    err = cudaMalloc(&m_lst.lsnodes, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(m_lst.lsnodes, h_lst->lsnodes,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;
    
    printf("Done [%ldkB]\n", total_bytes / 1000);

    return 0;
}

__host__ int
lst_free_device(lst_t* d_lst) {
    cudaError_t err;
    lst_t m_lst;
    err = cudaMemcpy(&m_lst, d_lst, sizeof(lst_t), cudaMemcpyDeviceToHost);
    if (check_cuda_err(err)) return err;
    err = cudaFree(d_lst);
    if (check_cuda_err(err)) return err;
    err = cudaFree(m_lst.lsnodes);
    if (check_cuda_err(err)) return err;
    return 0;

}

// generated by gpt may not work
static PLATFORM Vec3 
sample_triangle_uniform(rand_state_t& rstate, const Vec3 &a, const Vec3 &b, const Vec3 &c) {
    float u1 = random_uniform(rstate);
    float u2 = random_uniform(rstate);
    if (u1 + u2 > 1) {
        u1 = 1 - u1;
        u2 = 1 - u2;
    }
    return a * u1 + b * u2 + c * (1 - u1 + u2);
}

PLATFORM void
lst_sample(light_sample_t& sample, const lst_t* lst, const obj_scene_data* scene, rand_state_t& rstate) {
    if (!lst->nodeCount) {
        return;
    }

    // find element
    int randomNode = random_uniform(rstate) * lst->nodeCount;

    const lst_node_t& node = lst->lsnodes[randomNode];

    const obj_face& face = scene->face_list[node.face];

    // TODO...
}
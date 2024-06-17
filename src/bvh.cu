#include <stdio.h>

#include "assert.h"
#include "bvh.h"
#include "utils.h"

static void
calculate_face_centroid(const obj_scene_data* scene, obj_face* face, mfloat_t* centroid) {
    mfloat_t totalArea = 0;
    vec3_zero(centroid);
    for (uint32_t i = 2; i < face->vertex_count; i++) {
        mfloat_t* A = scene->vertex_list[face->vertices[0].position].v;
        mfloat_t* B = scene->vertex_list[face->vertices[i - 1].position].v;
        mfloat_t* C = scene->vertex_list[face->vertices[i].position].v;
        mfloat_t edge1[3], edge2[3], cross[3], triCentroid[3];

        vec3_subtract(edge1, B, A);
        vec3_subtract(edge2, C, A);
        vec3_cross(cross, edge1, edge2);
        mfloat_t triArea = 0.5 * vec3_length(cross);

        // calculate centroid on triangle
        vec3_add(triCentroid, A, B);
        vec3_add(triCentroid, triCentroid, C);
        vec3_multiply_f(triCentroid, triCentroid, 0.33333333);

        // weighted sum of centroids
        vec3_multiply_f(triCentroid, triCentroid, triArea);
        vec3_add(centroid, centroid, triCentroid);
        totalArea += triArea;
    }
    assert(totalArea);
    vec3_multiply_f(centroid, centroid, 1.0 / totalArea);

    // for (int i = 0; i < face->vertex_count; i++) {
    //     mfloat_t* A = scene->vertex_list[face->vertices[i].position].v;
    //     printf("(%.3f, %.3f, %.3f), ", A[0], A[1], A[2]);
    // }
    // printf("centroid = (%.4f, %.4f, %.4f)\n", centroid[0], centroid[1], centroid[2]);
}

static void aabb_init(aabb_t* aabb) {
    vec3_const(aabb->min, 1e30);
    vec3_const(aabb->max, -1e30);
}

static mfloat_t aabb_area(aabb_t* aabb) {
    mfloat_t e[3];
    vec3_subtract(e, aabb->max, aabb->min);
    // assert(e[0] >= 0 && e[1] >= 0 && e[2] >= 0);
    return e[0] * e[1] + e[1] * e[2] + e[2] * e[0];
}

static void aabb_grow_face(const obj_scene_data* scene, aabb_t* aabb, uint32_t faceIndex) {
    obj_face* face = &scene->face_list[faceIndex];
    for (uint32_t j = 0; j < face->vertex_count; j++) {
        mfloat_t* v = scene->vertex_list[face->vertices[j].position].v;
        vec3_min(aabb->min, aabb->min, v);
        vec3_max(aabb->max, aabb->max, v);
    }
}

static void update_node_bounds(const obj_scene_data* scene, bvh_t* bvh, uint32_t nodeIndex) {
    bvh_node_t* node = &bvh->nodes[nodeIndex];
    aabb_init(&node->bounds);
    for (uint32_t i = node->start; i < node->end; i++) {
        aabb_grow_face(scene, &node->bounds, bvh->indices[i]);
    }
}

// static void update_node_bounds(const obj_scene_data* scene, bvh_t* bvh, uint32_t nodeIndex) {
//     bvh_node_t* node = &bvh->nodes[nodeIndex];
//     vec3_const(node->bounds.min, 1e30);
//     vec3_const(node->bounds.max, -1e30);
//     for (uint32_t i = node->start; i < node->end; i++) {
//         obj_face* face = &scene->face_list[bvh->indices[i]];
//         for (uint32_t j = 0; j < face->vertex_count; j++) {
//             mfloat_t* v = scene->vertex_list[face->vertices[j].position].v;
//             vec3_min(node->bounds.min, node->bounds.min, v);
//             vec3_max(node->bounds.max, node->bounds.max, v);
//         }
//     }
// }

void swap(uint32_t* a, uint32_t* b) {
    uint32_t t = *a;
    *a = *b;
    *b = t;
}

// static mfloat_t
// quick_select_primitive(bvh_t* bvh, int axis, int left, int right, int k) {
//     if (left == right) {
//         return bvh->centroids[bvh->indices[left]].v[axis];
//     }
//     // partition
//     int i = left;
//     int j = right - 1;
//     while (i < j) {
//         if (bvh->centroids[bvh->indices[i]].v[axis] < bvh->centroids[bvh->indices[right]].v[axis]) {
//             i++;
//         } else {
//             swap(&bvh->indices[i], &bvh->indices[j--]);
//         }
//     }
//     int p = i;
//     if (bvh->centroids[bvh->indices[p]].v[axis] < bvh->centroids[bvh->indices[right]].v[axis]) {
//         p += 1;
//     }
//     swap(&bvh->indices[p], &bvh->indices[right]);
//     // recurse
//     if (k == p) {
//         return bvh->centroids[bvh->indices[left]].v[axis];
//     } else if (k < p) {
//         return quick_select_primitive(bvh, axis, left, p - 1, k);
//     } else {
//         return quick_select_primitive(bvh, axis, p + 1, right, k);
//     }
// }

bvh_t* comparison_bvh;
int comparison_axis = 0;

int bvh_comparator(const void* a, const void* b) {
    assert(comparison_bvh);
    mfloat_t a_val = comparison_bvh->centroids[*(uint32_t*)a].v[comparison_axis];
    mfloat_t b_val = comparison_bvh->centroids[*(uint32_t*)b].v[comparison_axis];
    if (a_val == b_val) return 0;
    if (a_val < b_val)
        return -1;
    else
        return 1;
}

uint32_t split_by_median(bvh_t* bvh, bvh_node_t* node, int count, bvh_stats_t* stats) {
    mfloat_t extent[3];
    vec3_subtract(extent, node->bounds.max, node->bounds.min);
    assert(extent[0] > 0 && extent[1] > 0 && extent[2] > 0);

    int axis = 0;
    if (extent[1] > extent[axis]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;

    comparison_axis = axis;
    comparison_bvh = bvh;
    qsort(&bvh->indices[node->start], count, sizeof(uint32_t), bvh_comparator);
    return node->start + count / 2;
}

// uint32_t split_by_average(bvh_t* bvh, bvh_node_t* node, int count, bvh_stats_t* stats) {
//     mfloat_t extent[3];
//     vec3_subtract(extent, node->bounds.max, node->bounds.min);
//     assert(extent[0] > 0 && extent[1] > 0 && extent[2] > 0);

//     int axis = 0;
//     if (extent[1] > extent[axis]) axis = 1;
//     if (extent[2] > extent[axis]) axis = 2;

//     // split pos as average pos
//     mfloat_t splitPos = 0;
//     for (int i = node->start; i < node->end; i++) {
//         splitPos += bvh->centroids[bvh->indices[i]].v[axis];
//     }
//     splitPos /= count;
//     // quicksort-esque splitting
//     int i = node->start;
//     int j = node->end - 1;
//     while (i < j) {
//         if (bvh->centroids[bvh->indices[i]].v[axis] < splitPos) {
//             i++;
//         } else {
//             swap(&bvh->indices[i], &bvh->indices[j--]);
//         }
//     }

//     int leftCount = i - node->start;
//     if (leftCount == 0 || leftCount == count) {
//         stats->totalSkippedFaces += count;
//         return 0;  // cannot split because overlapping or something
//     }
//     return i;
// }


static mfloat_t
evaluate_sah(const obj_scene_data* scene,
             bvh_t* bvh, bvh_node_t* node, int axis, mfloat_t splitPos) {
    // determine triangle counts and bounds for this split candidate
    aabb_t leftBox, rightBox;
    aabb_init(&leftBox);
    aabb_init(&rightBox);
    uint32_t leftCount = 0, rightCount = 0;
    for (uint32_t i = node->start; i < node->end; i++) {
        mfloat_t centroid = bvh->centroids[bvh->indices[i]].v[axis];
        if (centroid < splitPos) {
            leftCount++;
            aabb_grow_face(scene, &leftBox, bvh->indices[i]);
        } else {
            rightCount++;
            aabb_grow_face(scene, &rightBox, bvh->indices[i]);
        }
    }

    mfloat_t cost = leftCount * aabb_area(&leftBox) + rightCount * aabb_area(&rightBox);
    return cost > 0 ? cost : 1e30;
}

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
static void
subdivide(const obj_scene_data* scene, bvh_t* bvh, uint32_t nodeIndex, bvh_stats_t* stats) {
    bvh_node_t* node = &bvh->nodes[nodeIndex];
    int count = node->end - node->start;
    if (count <= 2) {
        return;  // stop criterion
    }

    // uint32_t i = split_by_average(bvh, node, count, stats);
    // uint32_t i = split_by_median(bvh, node, count, stats);

    int bestAxis = -1;
    mfloat_t bestPos = 0, bestCost = 1e30;
    for (int axis = 0; axis < 3; axis++) {
        for (uint32_t i = node->start; i < node->end; i++) {
            mfloat_t pos = bvh->centroids[bvh->indices[i]].v[axis];
            mfloat_t cost = evaluate_sah(scene, bvh, node, axis, pos);
            if (cost < bestCost) {
                bestAxis = axis;
                bestCost = cost;
                bestPos = pos;
            }
        }
    }
    assert(bestAxis >= 0);

    // partitioning
    int i = node->start;
    int j = node->end - 1;
    while (i < j) {
        if (bvh->centroids[bvh->indices[i]].v[bestAxis] < bestPos) {
            i++;
        } else {
            swap(&bvh->indices[i], &bvh->indices[j--]);
        }
    }

    int leftCount = i - node->start;
    if (leftCount == 0 || leftCount == count) {
        printf("Skipped split\n");
        stats->totalSkippedFaces += count;
        return;
    }

    uint32_t leftIndex = bvh->nodeCount++;
    uint32_t rightIndex = bvh->nodeCount++;
    bvh_node_t* left = &bvh->nodes[leftIndex];
    bvh_node_t* right = &bvh->nodes[rightIndex];
    left->start = node->start;
    left->end = right->start = i;
    right->end = node->end;

    printf("Split: %u => %u / %u\n", left->start, left->end - left->start, right->end - right->start);
    
    node->leftChild = leftIndex;
    node->rightChild = rightIndex;
    node->start = node->end = 0;  // make non-leaf
    update_node_bounds(scene, bvh, leftIndex);
    update_node_bounds(scene, bvh, rightIndex);
    subdivide(scene, bvh, leftIndex, stats);
    subdivide(scene, bvh, rightIndex, stats);
}

static void
calculate_stats(const obj_scene_data* scene, bvh_t* bvh, uint32_t nodeIndex, bvh_stats_t* stats) {
    stats->numberLeaves = 0;
    stats->averageLeafSize = 0;
    for (uint32_t i = 0; i < bvh->nodeCount; i++) {
        bvh_node_t* node = &bvh->nodes[i];
        uint32_t faces = node->end - node->start;
        if (faces > 0) {
            stats->numberLeaves++;
            stats->averageLeafSize += faces;
        }
    }
    stats->averageLeafSize /= stats->numberLeaves;
}

static void
print_stats(const obj_scene_data* scene, bvh_t* bvh, uint32_t nodeIndex, bvh_stats_t* stats) {
    printf("bvh_t stats:\n");
    printf("  node count = %u\n", bvh->nodeCount);
    printf("  optimal node count = %u\n", bvh->maxNodeCount);
    printf("  number leafs = %u\n", stats->numberLeaves);
    printf("  skipped faces = %u / %u = %.3f\n",
           stats->totalSkippedFaces, bvh->primitiveCount,
           stats->totalSkippedFaces / (double)bvh->primitiveCount);
    printf("  average leaf size = %.2f\n", stats->averageLeafSize);
}

__host__ void
bvh_build(bvh_t* bvh, const obj_scene_data* scene) {
    printf("Building bvh_t... \n");

    bvh_stats_t stats;
    memset(&stats, 0, sizeof(bvh_stats_t));

    bvh->primitiveCount = scene->face_count;
    bvh->maxNodeCount = 2 * bvh->primitiveCount - 1;

    // bvh->nodeCount = 0;
    bvh->nodes = (bvh_node_t*)malloc(sizeof(bvh_node_t) * bvh->maxNodeCount);

    // mutable primitive list for sorting faces
    bvh->indices = (uint32_t*)malloc(sizeof(uint32_t) * bvh->primitiveCount);
    for (uint32_t i = 0; i < bvh->primitiveCount; i++) {
        bvh->indices[i] = i;
    }
    // calculate centroids, accesses work with scene face indices
    bvh->centroids = (struct vec3*)malloc(sizeof(struct vec3) * bvh->primitiveCount);
    for (uint32_t i = 0; i < bvh->primitiveCount; i++) {
        calculate_face_centroid(scene, &scene->face_list[i], bvh->centroids[i].v);
    }

    uint32_t rootIndex = /* BVH_ROOT_NODE ;*/ bvh->nodeCount++;
    bvh_node_t* root = &bvh->nodes[rootIndex];
    root->start = 0;
    root->end = bvh->primitiveCount;
    update_node_bounds(scene, bvh, rootIndex);
    subdivide(scene, bvh, rootIndex, &stats);

    calculate_stats(scene, bvh, rootIndex, &stats);
    print_stats(scene, bvh, rootIndex, &stats);

    // not needed anymore
    free(bvh->centroids);
    bvh->centroids = NULL;
}

__host__ void
bvh_free_host(bvh_t* h_bvh) {
    free(h_bvh->nodes);
    h_bvh->nodes = NULL;
    free(h_bvh->indices);
    h_bvh->indices = NULL;
}

__host__ int
bvh_copy_device(bvh_t** d_bvh, const bvh_t* h_bvh) {
    printf("Copying bvh_t to device... ");

    bvh_t m_bvh = *h_bvh;

    cudaError_t err;
    size_t curr_bytes, total_bytes;
    total_bytes = 0;

    curr_bytes = sizeof(bvh_node_t) * m_bvh.maxNodeCount;
    total_bytes += curr_bytes;
    err = cudaMalloc(&m_bvh.nodes, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(m_bvh.nodes, h_bvh->nodes,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    curr_bytes = sizeof(uint32_t) * m_bvh.primitiveCount;
    total_bytes += curr_bytes;
    err = cudaMalloc(&m_bvh.indices, curr_bytes);
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(m_bvh.indices, h_bvh->indices,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    total_bytes += sizeof(bvh_t);
    err = cudaMalloc(d_bvh, sizeof(bvh_t));
    if (check_cuda_err(err)) return err;
    err = cudaMemcpy(*d_bvh, &m_bvh,
                     sizeof(bvh_t), cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) return err;

    printf("Done [%ldkB]\n", total_bytes / 1000);

    return 0;
}

__host__ int
bvh_free_device(bvh_t* d_bvh) {
    cudaError_t err;
    bvh_t m_bvh;
    err = cudaMemcpy(&m_bvh, d_bvh, sizeof(bvh_t), cudaMemcpyDeviceToHost);
    if (check_cuda_err(err)) return err;
    err = cudaFree(d_bvh);
    if (check_cuda_err(err)) return err;
    err = cudaFree(m_bvh.nodes);
    if (check_cuda_err(err)) return err;
    err = cudaFree(m_bvh.indices);
    if (check_cuda_err(err)) return err;
    return 0;
}

static __host__ __device__ int
intersect_aabb(const aabb_t* aabb, const Ray* ray, mfloat_t min_t) {
    mfloat_t t1[VEC3_SIZE], t2[VEC3_SIZE], tminv[VEC3_SIZE], tmaxv[VEC3_SIZE];
    vec3_subtract(t1, aabb->min, ray->o);
    vec3_divide(t1, t1, ray->r);
    vec3_subtract(t2, aabb->max, ray->o);
    vec3_divide(t2, t2, ray->r);

    vec3_min(tminv, t1, t2);
    vec3_max(tmaxv, t1, t2);

    mfloat_t tmin = MFMAX(tminv[0], MFMAX(tminv[1], tminv[2]));
    mfloat_t tmax = MFMIN(tmaxv[0], MFMIN(tmaxv[1], tmaxv[2]));

    return tmax >= tmin && tmin < min_t && tmax > 0;
}

__host__ __device__ void
bvh_intersect(const __restrict__ bvh_t* bvh, uint32_t nodeIndex,
              const __restrict__ obj_scene_data* scene, const Ray* ray, Intersection* hit) {
    bvh_node_t* node = &bvh->nodes[nodeIndex];
    if (!intersect_aabb(&node->bounds, ray, hit->distance)) {
        return;
    }
    if (node->end - node->start > 0) {
        // is leaf
        for (uint32_t i = node->start; i < node->end; i++) {
            intersect_face(scene, ray, hit, bvh->indices[i]);
        }
    } else {
        bvh_intersect(bvh, node->leftChild, scene, ray, hit);
        bvh_intersect(bvh, node->rightChild, scene, ray, hit);
    }
}

// stack size must be below bvh tree height + 1,
// 100 should probably be enough for all eternity
#define TRAVERSAL_STACK_SIZE 100

__host__ __device__ void
bvh_intersect_iterative(const __restrict__ bvh_t* bvh,
                        const __restrict__ obj_scene_data* scene, const Ray* ray, Intersection* hit) {
    // for (size_t i = 0; i < scene->face_count; i++) {
    // for (size_t i = 0; i < bvh->primitiveCount; i++) {
    //     intersect_face(scene, ray, hit, i);
    // }

    int stack[TRAVERSAL_STACK_SIZE];
    int depth = 0;
    stack[0] = BVH_ROOT_NODE;

    while (depth >= 0) {
        int u = stack[depth];
        bvh_node_t* node = &bvh->nodes[u];
        depth--;

        if (intersect_aabb(&node->bounds, ray, hit->distance)) {
            if (node->end - node->start > 0) {
                for (uint32_t i = node->start; i < node->end; i++) {
                    intersect_face(scene, ray, hit, bvh->indices[i]);
                }
                // is leaf
            } else {
                // no leaf, recurse
                stack[++depth] = node->rightChild;
                stack[++depth] = node->leftChild;
            }
        }
    }
}
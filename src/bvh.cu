#include <stdio.h>

#include "assert.h"
#include "bvh.h"
#include "utils.h"
#include <iostream>
#include <limits>

static Vec3
calculate_face_centroid(const scene_t& scene, const face_t& face) {
    float totalArea = 0;
    Vec3 centroid = { 0, 0, 0 };

    for (uint32_t i = 2; i < face.vertexCount; i++) {
        const Vec3& A = scene.vertices[face.vertices[0]].position;
        const Vec3& B = scene.vertices[face.vertices[i - 1]].position;
        const Vec3& C = scene.vertices[face.vertices[i]].position;
        
        Vec3 edge1 = B.cross(A);
        Vec3 edge2 = C.cross(A);
        Vec3 cross = edge1.cross(edge2);
        float triArea = 0.5 * cross.magnitude();
        
        Vec3 triCentroid = (1 / 3.0f) * (A + B + C);

        // weighted sum of centroids
        centroid += triCentroid * triArea;
        totalArea += triArea;
    }

    if (totalArea == 0) {
        return scene.vertices[face.vertices[0]].position;
    }
    centroid /= totalArea;

    return centroid;
}

static void aabb_init(aabb_t& aabb) {
    aabb.min.set(1e30);
    aabb.max.set(-1e30);
}

static float aabb_area(const aabb_t& aabb) {
    Vec3 e = aabb.max - aabb.min;
    return e.x * e.y + e.y * e.z + e.z * e.x;
}

static void aabb_grow_point(aabb_t& aabb, const Vec3& p) {
    aabb.min = Vec3::min(aabb.min, p);
    aabb.max = Vec3::max(aabb.max, p);
}

static void aabb_grow_face(const scene_t& scene, aabb_t& aabb, uint32_t faceIndex) {
    const face_t &face = scene.faces[faceIndex];
    for (uint32_t j = 0; j < face.vertexCount; j++) {
        const Vec3& v = scene.vertices[face.vertices[j]].position;
        aabb_grow_point(aabb, v);
    }
}

static void update_node_bounds(const scene_t& scene, bvh_t& bvh, uint32_t nodeIndex) {
    bvh_node_t& node = bvh.nodes[nodeIndex];
    aabb_init(node.bounds);
    for (uint32_t i = node.start; i < node.end; i++) {
        aabb_grow_face(scene, node.bounds, bvh.indices[i]);
    }
}

void swap(uint32_t* a, uint32_t* b) {
    uint32_t t = *a;
    *a = *b;
    *b = t;
}

static float
evaluate_sah(const scene_t& scene,
             bvh_t& bvh, bvh_node_t& node, int axis, float splitPos) {
    // determine triangle counts and bounds for this split candidate
    aabb_t leftBox, rightBox;
    aabb_init(leftBox);
    aabb_init(rightBox);
    uint32_t leftCount = 0, rightCount = 0;
    for (uint32_t i = node.start; i < node.end; i++) {
        float centroid = bvh.centroids[bvh.indices[i]][axis];
        if (centroid < splitPos) {
            leftCount++;
            aabb_grow_face(scene, leftBox, bvh.indices[i]);
        } else {
            rightCount++;
            aabb_grow_face(scene, rightBox, bvh.indices[i]);
        }
    }

    float cost = leftCount * aabb_area(leftBox) + rightCount * aabb_area(rightBox);
    return cost > 0 ? cost : 1e30;
}

// static float
// find_split_plane_centroids(const scene_t& scene, bvh_t& bvh, bvh_node_t& node,
//                            int* bestAxis, float* bestSplit) {
//     float bestCost = 1e30;
//     for (int a = 0; a < 3; a++) {
//         for (uint32_t i = node.start; i < node.end; i++) {
//             float candidate = bvh.centroids[bvh.indices[i]][a];
//             float cost = evaluate_sah(scene, bvh, node, a, candidate);
//             if (cost < bestCost) {
//                 *bestAxis = a;
//                 *bestSplit = candidate;
//                 bestCost = cost;
//             }
//         }
//     }
//     assert(*bestAxis >= 0);
//     return bestCost;
// }

static float
find_best_split_plane_bands(const scene_t& scene, bvh_t& bvh, bvh_node_t& node,
                      int* bestAxis, float* bestSplit, int numBands) {

    AABB centroidBounds;
    aabb_init(centroidBounds);
    for (size_t i = node.start; i < node.end; i++) {
        const Vec3& c = bvh.centroids[bvh.indices[i]];
        centroidBounds = centroidBounds.including(c);
    }

    float bestCost = 1e30;
    for (int a = 0; a < 3; a++) {
        float boundsMin = centroidBounds.min[a];
        float boundsMax = centroidBounds.max[a];
        if (std::abs(boundsMin - boundsMax) < std::numeric_limits<float>::epsilon()) {
            continue;
        }
        float scale = (boundsMax - boundsMin) / numBands;
        for (int i = 1; i < numBands; i++) {
            float candidate = boundsMin + i * scale;
            float cost = evaluate_sah(scene, bvh, node, a, candidate);
            if (cost < bestCost) {
                *bestAxis = a;
                *bestSplit = candidate;
                bestCost = cost;
            }
        }
    }
    assert(*bestAxis >= 0);
    return bestCost;
}

#define N_BANDS 8

static float
find_best_split_plane(const scene_t& scene, bvh_t& bvh, bvh_node_t& node,
                      int* bestAxis, float* bestSplit) {
    // int count = node.end - node.start;
    // if (count < N_BANDS) {
    //     return find_split_plane_centroids(scene, bvh, node, bestAxis, bestSplit);
    // } else {
    return find_best_split_plane_bands(scene, bvh, node, bestAxis, bestSplit, N_BANDS);
    // }
}

static void
subdivide(const scene_t& scene, bvh_t& bvh, uint32_t nodeIndex, bvh_stats_t& stats, int depth) {

    stats.maxDepth = std::max(stats.maxDepth, depth);

    bvh_node_t& node = bvh.nodes[nodeIndex];
    int count = node.end - node.start;
    if (count <= 2) {
        return;  // stop criterion
    }
    // uint32_t i = split_by_average(bvh, node, count, stats);
    // uint32_t i = split_by_median(bvh, node, count, stats);

    int bestAxis = -1;
    float bestPos = 0;
    float bestCost = find_best_split_plane(scene, bvh, node, &bestAxis, &bestPos);

    // partitioning
    int i = node.start;
    int j = node.end - 1;
    while (i <= j) {
        if (bvh.centroids[bvh.indices[i]][bestAxis] < bestPos) {
            i++;
        } else {
            swap(&bvh.indices[i], &bvh.indices[j--]);
        }
    }

    int leftCount = i - node.start;
    if (leftCount == 0 || leftCount == count) {
        // printf("Skipped split\n");
        stats.totalSkippedFaces += count;
        return;
    }

    uint32_t leftIndex = bvh.nodeCount++;
    uint32_t rightIndex = bvh.nodeCount++;
    bvh_node_t& left = bvh.nodes[leftIndex];
    bvh_node_t& right = bvh.nodes[rightIndex];
    left.start = node.start;
    left.end = right.start = i;
    right.end = node.end;

    // printf("Split: %u => %u / %u\n", left->start, left->end - left->start, right->end - right->start);

    time_t currTime;
    time(&currTime);
    double elapsed_seconds = difftime(currTime, stats.lastInfo);
    if (elapsed_seconds > 1.0) {
        printf("created %u bvh nodes (of at most %d)\n", bvh.nodeCount, bvh.maxNodeCount);
        stats.lastInfo = currTime;
    }

    node.leftChild = leftIndex;
    node.rightChild = rightIndex;
    node.start = node.end = 0;  // make non-leaf
    update_node_bounds(scene, bvh, leftIndex);
    update_node_bounds(scene, bvh, rightIndex);
    subdivide(scene, bvh, leftIndex, stats, depth + 1);
    subdivide(scene, bvh, rightIndex, stats, depth + 1);
}

static void
calculate_stats(const scene_t& scene, bvh_t& bvh, uint32_t nodeIndex, bvh_stats_t& stats) {
    stats.numberLeaves = 0;
    stats.averageLeafSize = 0;
    for (uint32_t i = 0; i < bvh.nodeCount; i++) {
        bvh_node_t& node = bvh.nodes[i];
        uint32_t faces = node.end - node.start;
        if (faces > 0) {
            stats.numberLeaves++;
            stats.averageLeafSize += faces;
        }
    }
    stats.averageLeafSize /= stats.numberLeaves;
}

static void
print_stats(const scene_t& scene, bvh_t& bvh, uint32_t nodeIndex, bvh_stats_t& stats) {
    if (!doVerbosePrinting) {
        return;
    }

    printf("\nbvh_t stats:\n");
    printf("  node count = %u\n", bvh.nodeCount);
    printf("  optimal node count = %u\n", bvh.maxNodeCount);
    printf("  number leafs = %u\n", stats.numberLeaves);
    printf("  skipped faces = %u / %u = %.3f\n",
           stats.totalSkippedFaces, bvh.primitiveCount,
           stats.totalSkippedFaces / (float)bvh.primitiveCount);
    printf("  average leaf size = %.2f\n", stats.averageLeafSize);
    printf("  max tree height =  %d\n", stats.maxDepth);
}

void
bvh_build(bvh_t& bvh, const scene_t& scene) {
    printf("Building bvh_t...  \n");
    fflush(stdout);

    bvh_stats_t stats;
    memset(&stats, 0, sizeof(bvh_stats_t));
    time(&stats.lastInfo);

    bvh.primitiveCount = scene.faces.count;
    bvh.maxNodeCount = 2 * bvh.primitiveCount - 1;

    bvh.nodeCount = 0;
    bvh.nodes = (bvh_node_t*)malloc(sizeof(bvh_node_t) * bvh.maxNodeCount);
    assert(bvh.nodes && "malloc");

    // mutable primitive list for sorting faces
    bvh.indices = (uint32_t*)malloc(sizeof(uint32_t) * bvh.primitiveCount);
    assert(bvh.indices && "malloc");

    for (uint32_t i = 0; i < bvh.primitiveCount; i++) {
        bvh.indices[i] = i;
    }
    // calculate centroids, accesses work with scene face indices
    bvh.centroids = (Vec3*)malloc(sizeof(Vec3) * bvh.primitiveCount);
    assert(bvh.centroids && "malloc");

    for (uint32_t i = 0; i < bvh.primitiveCount; i++) {
        bvh.centroids[i] = calculate_face_centroid(scene, scene.faces[i]);
    }

    uint32_t rootIndex = bvh.nodeCount++;
    bvh_node_t& root = bvh.nodes[rootIndex];
    root.start = 0;
    root.end = bvh.primitiveCount;
    update_node_bounds(scene, bvh, rootIndex);
    subdivide(scene, bvh, rootIndex, stats, 1);

    calculate_stats(scene, bvh, rootIndex, stats);
    print_stats(scene, bvh, rootIndex, stats);

    // not needed anymore
    free(bvh.centroids);
    bvh.centroids = NULL;

    if (stats.maxDepth + 1 > BVH_TRAVERSAL_STACK_SIZE) {
        printf("ERROR bvh max height (%d) is too large, increase BVH_TRAVERSAL_STACK_SIZE (%d)\n", 
            stats.maxDepth, BVH_TRAVERSAL_STACK_SIZE);
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
}

void bvh_free_host(bvh_t& h_bvh) {
    free(h_bvh.nodes);
    h_bvh.nodes = NULL;
    free(h_bvh.indices);
    h_bvh.indices = NULL;
}

void bvh_copy_device(bvh_t** d_bvh, const bvh_t* h_bvh) {
    printf("Copying bvh_t to device... ");

    bvh_t m_bvh = *h_bvh;

    cudaError_t err;
    size_t curr_bytes, total_bytes;
    total_bytes = 0;

    curr_bytes = sizeof(bvh_node_t) * m_bvh.maxNodeCount;
    total_bytes += curr_bytes;
    err = cudaMalloc(&m_bvh.nodes, curr_bytes);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);
    err = cudaMemcpy(m_bvh.nodes, h_bvh->nodes,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);

    curr_bytes = sizeof(uint32_t) * m_bvh.primitiveCount;
    total_bytes += curr_bytes;
    err = cudaMalloc(&m_bvh.indices, curr_bytes);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);
    err = cudaMemcpy(m_bvh.indices, h_bvh->indices,
                     curr_bytes, cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);

    total_bytes += sizeof(bvh_t);
    err = cudaMalloc(d_bvh, sizeof(bvh_t));
    if (check_cuda_err(err)) exit(EXIT_FAILURE);
    err = cudaMemcpy(*d_bvh, &m_bvh,
                     sizeof(bvh_t), cudaMemcpyHostToDevice);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);

    printf("Done [%ldkB]\n", total_bytes / 1000);
}

void bvh_free_device(bvh_t* d_bvh) {
    cudaError_t err;
    bvh_t m_bvh;
    err = cudaMemcpy(&m_bvh, d_bvh, sizeof(bvh_t), cudaMemcpyDeviceToHost);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);
    err = cudaFree(d_bvh);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);
    err = cudaFree(m_bvh.nodes);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);
    err = cudaFree(m_bvh.indices);
    if (check_cuda_err(err)) exit(EXIT_FAILURE);
}

#define RAY_NO_HIT 1e30f

static PLATFORM float
intersect_aabb(const aabb_t& aabb, const Ray& ray, float min_t) {
    Vec3 t1 = (aabb.min - ray.o) / ray.r;
    Vec3 t2 = (aabb.max - ray.o) / ray.r;
    Vec3 tminv = Vec3::min(t1, t2);
    Vec3 tmaxv = Vec3::max(t1, t2);

    float tmin = tminv.maxComponent();
    float tmax = tmaxv.minComponent();

    if (tmax >= tmin && tmin < min_t && tmax > 0) {
        return tmin;
    } else {
        return RAY_NO_HIT;
    }
}

PLATFORM void
bvh_intersect(const __restrict__ bvh_t* bvh, uint32_t nodeIndex,
              const __restrict__ scene_t* scene, const Ray& ray, intersection_t& hit) {
    bvh_node_t &node = bvh->nodes[nodeIndex];
    if (!intersect_aabb(node.bounds, ray, hit.distance)) {
        return;
    }
    if (node.end - node.start > 0) {
        // is leaf
        for (uint32_t i = node.start; i < node.end; i++) {
            intersect_face(scene, ray, hit, bvh->indices[i]);
        }
    } else {
        bvh_intersect(bvh, node.leftChild, scene, ray, hit);
        bvh_intersect(bvh, node.rightChild, scene, ray, hit);
    }
}

PLATFORM void
bvh_intersect_iterative(const __restrict__ bvh_t* bvh,
                        const __restrict__ scene_t* scene, const Ray& ray, intersection_t& hit) {

    bvh_node_t* stack[BVH_TRAVERSAL_STACK_SIZE];
    int depth = 0;
    stack[0] = &bvh->nodes[0];

    while (depth >= 0) {
        bvh_node_t* node = stack[depth--];

        while (node != NULL) {

            if (node->end - node->start > 0) {
                // is leaf
                for (uint32_t i = node->start; i < node->end; i++) {
                    intersect_face(scene, ray, hit, bvh->indices[i]);
                }
                node = NULL;
            } else {

                bvh_node_t* child1 = &bvh->nodes[node->leftChild];
                bvh_node_t* child2 = &bvh->nodes[node->rightChild];

                float t1 = intersect_aabb(child1->bounds, ray, hit.distance);
                float t2 = intersect_aabb(child2->bounds, ray, hit.distance);

                if (t2 < t1) {
                    bvh_node_t* tempc = child1;
                    child1 = child2;
                    child2 = tempc;
                    float tempt = t1;
                    t1 = t2;
                    t2 = tempt;
                }

                if (t1 == RAY_NO_HIT) {
                    node = NULL;
                } else {
                    node = child1;
                    if (t2 != RAY_NO_HIT) {
                        stack[++depth] = child2;
                    }
                }
            }
        }
    }
}


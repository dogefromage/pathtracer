#include <stdio.h>

#include <iostream>
#include <limits>

#include "assert.h"
#include "bvh.h"
#include "logger.h"
#include "utils.h"

#define BVH_TRAVERSAL_STACK_SIZE 64

static Vec3 calculate_face_centroid(const Scene &scene, const face_t &face) {
    float totalArea = 0;
    Vec3 centroid = {0, 0, 0};

    for (uint32_t i = 2; i < face.vertexCount; i++) {
        const Vec3 &A = scene.vertices[face.vertices[0]].position;
        const Vec3 &B = scene.vertices[face.vertices[i - 1]].position;
        const Vec3 &C = scene.vertices[face.vertices[i]].position;

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

static void aabb_grow_face(const Scene &scene, AABB &aabb, uint32_t faceIndex) {
    const face_t &face = scene.faces[faceIndex];
    for (uint32_t j = 0; j < face.vertexCount; j++) {
        const Vec3 &v = scene.vertices[face.vertices[j]].position;
        aabb.grow(v);
    }
}

static void update_node_bounds(const Scene &scene, bvh_t &bvh, uint32_t nodeIndex) {
    bvh_node_t &node = bvh.nodes[nodeIndex];
    node.bounds.reset();
    for (uint32_t i = node.start; i < node.end; i++) {
        aabb_grow_face(scene, node.bounds, bvh.indices[i]);
    }
}

void swap(uint32_t *a, uint32_t *b) {
    uint32_t t = *a;
    *a = *b;
    *b = t;
}

#define NUM_BINS 16

struct Bin {
    AABB bounds;
    int primitiveCount = 0;
};

// https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/
static float find_best_split_plane(const Scene &scene, bvh_t &bvh, bvh_node_t &node, int *bestAxis, float *bestSplit) {
    AABB centroidBounds;
    for (size_t i = node.start; i < node.end; i++) {
        const Vec3 &c = bvh.centroids[bvh.indices[i]];
        centroidBounds.grow(c);
    }

    float bestCost = 1e30f;

    for (int a = 0; a < 3; a++) {
        float boundsMin = centroidBounds.min[a];
        float boundsMax = centroidBounds.max[a];

        if (std::abs(boundsMin - boundsMax) < std::numeric_limits<float>::epsilon()) {
            continue;
        }

        Bin bins[NUM_BINS];

        float scale = (boundsMax - boundsMin) / NUM_BINS;
        float inv_scale = 1 / scale;

        for (uint32_t i = node.start; i < node.end; i++) {
            float centroid = bvh.centroids[bvh.indices[i]][a];

            int binIndex = min(NUM_BINS - 1, (int)((centroid - boundsMin) * inv_scale));
            Bin &bin = bins[binIndex];
            aabb_grow_face(scene, bin.bounds, bvh.indices[i]);
            bin.primitiveCount++;
        }

        float leftArea[NUM_BINS - 1], rightArea[NUM_BINS - 1];
        int leftCount[NUM_BINS - 1], rightCount[NUM_BINS - 1];

        AABB leftBox, rightBox;
        int leftSum = 0, rightSum = 0;
        for (int i = 0; i < NUM_BINS - 1; i++) {
            // left
            leftSum += bins[i].primitiveCount;
            leftCount[i] = leftSum;
            leftBox.grow(bins[i].bounds);
            leftArea[i] = leftBox.area();
            // right
            rightSum += bins[NUM_BINS - 1 - i].primitiveCount;
            rightCount[NUM_BINS - 2 - i] = rightSum;
            rightBox.grow(bins[NUM_BINS - 1 - i].bounds);
            rightArea[NUM_BINS - 2 - i] = rightBox.area();
        }

        for (int i = 0; i < NUM_BINS - 1; i++) {
            float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
            if (planeCost < bestCost) {
                *bestAxis = a;
                *bestSplit = boundsMin + scale * (i + 1);
                bestCost = planeCost;
            }
        }
    }
    assert(*bestAxis >= 0);

    return bestCost;
}

static void subdivide(const Scene &scene, bvh_t &bvh, uint32_t nodeIndex, bvh_stats_t &stats, int depth) {
    stats.maxDepth = std::max(stats.maxDepth, depth);

    bvh_node_t &node = bvh.nodes[nodeIndex];
    int count = node.end - node.start;
    if (count <= 2) {
        return; // stop criterion
    }

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
    assert(0 < leftCount || leftCount < count);

    uint32_t leftIndex = bvh.nodeCount++;
    uint32_t rightIndex = bvh.nodeCount++;
    bvh_node_t &left = bvh.nodes[leftIndex];
    bvh_node_t &right = bvh.nodes[rightIndex];
    left.start = node.start;
    left.end = right.start = i;
    right.end = node.end;

    time_t currTime;
    time(&currTime);
    double elapsed_seconds = difftime(currTime, stats.lastInfo);
    if (elapsed_seconds > 1.0) {
        log_trace("created %u bvh nodes (of at most %d)\n", bvh.nodeCount, bvh.nodes.count);
        stats.lastInfo = currTime;
    }

    node.leftChild = leftIndex;
    node.rightChild = rightIndex;
    node.start = node.end = 0; // make non-leaf
    update_node_bounds(scene, bvh, leftIndex);
    update_node_bounds(scene, bvh, rightIndex);
    subdivide(scene, bvh, leftIndex, stats, depth + 1);
    subdivide(scene, bvh, rightIndex, stats, depth + 1);
}

static void calculate_stats(const Scene &scene, bvh_t &bvh, uint32_t nodeIndex, bvh_stats_t &stats) {
    stats.numberLeaves = 0;
    stats.averageLeafSize = 0;
    for (uint32_t i = 0; i < bvh.nodeCount; i++) {
        bvh_node_t &node = bvh.nodes[i];
        uint32_t faces = node.end - node.start;
        if (faces > 0) {
            stats.numberLeaves++;
            stats.averageLeafSize += faces;
        }
    }
    stats.averageLeafSize /= stats.numberLeaves;
}

static void print_stats(const Scene &scene, bvh_t &bvh, uint32_t nodeIndex, bvh_stats_t &stats) {
    log_trace("\nbvh_t stats:\n");
    log_trace("  node count = %u\n", bvh.nodeCount);
    log_trace("  optimal node count = %u\n", bvh.nodes.count);
    log_trace("  number leafs = %u\n", stats.numberLeaves);
    log_trace("  skipped faces = %u / %u = %.3f\n", stats.totalSkippedFaces, bvh.indices.count,
              stats.totalSkippedFaces / (float)bvh.indices.count);
    log_trace("  average leaf size = %.2f\n", stats.averageLeafSize);
    log_trace("  max tree height =  %d\n", stats.maxDepth);
}

void bvh_build(bvh_t &bvh, const Scene &scene) {
    log_info("Building BVH...  \n");
    fflush(stdout);

    bvh_stats_t stats;
    memset(&stats, 0, sizeof(bvh_stats_t));
    time(&stats.lastInfo);

    uint32_t primitiveCount = scene.faces.count;
    uint32_t maxNodeCount = 2 * primitiveCount - 1;
    // bvh.primitiveCount = scene.faces.count;
    // bvh.maxNodeCount = 2 * bvh.primitiveCount - 1;

    bvh.nodeCount = 0;
    bvh.nodes.items = (bvh_node_t *)malloc(sizeof(bvh_node_t) * maxNodeCount);
    assert(bvh.nodes.items && "malloc");
    bvh.nodes.count = maxNodeCount;

    // mutable primitive list for sorting faces
    bvh.indices.items = (uint32_t *)malloc(sizeof(uint32_t) * primitiveCount);
    assert(bvh.indices.items && "malloc");
    bvh.indices.count = primitiveCount;

    for (uint32_t i = 0; i < primitiveCount; i++) {
        bvh.indices[i] = i;
    }
    // calculate centroids, accesses work with scene face indices
    bvh.centroids.items = (Vec3 *)malloc(sizeof(Vec3) * primitiveCount);
    assert(bvh.centroids.items && "malloc");
    bvh.centroids.count = primitiveCount;

    for (uint32_t i = 0; i < primitiveCount; i++) {
        bvh.centroids[i] = calculate_face_centroid(scene, scene.faces[i]);
    }

    uint32_t rootIndex = bvh.nodeCount++;
    bvh_node_t &root = bvh.nodes[rootIndex];
    root.start = 0;
    root.end = primitiveCount;
    update_node_bounds(scene, bvh, rootIndex);
    subdivide(scene, bvh, rootIndex, stats, 1);

    calculate_stats(scene, bvh, rootIndex, stats);
    print_stats(scene, bvh, rootIndex, stats);

    // not needed anymore
    free(bvh.centroids.items);
    bvh.centroids.items = NULL;

    if (stats.maxDepth + 1 > BVH_TRAVERSAL_STACK_SIZE) {
        log_error("bvh max height (%d) is too large, increase BVH_TRAVERSAL_STACK_SIZE (%d)\n", stats.maxDepth,
                  BVH_TRAVERSAL_STACK_SIZE);
        exit(EXIT_FAILURE);
    }

    log_info("Done building BVH\n");
}

void bvh_free_host(bvh_t &h_bvh) {
    free(h_bvh.nodes.items);
    h_bvh.nodes.items = NULL;
    free(h_bvh.indices.items);
    h_bvh.indices.items = NULL;
}

void bvh_copy_device(bvh_t **d_bvh, const bvh_t *h_bvh) {
    log_info("Copying bvh_t to device... \n");

    bvh_t placeholder = *h_bvh;

    size_t totalSize = 0;

    totalSize += copy_device_fixed_array(&placeholder.nodes, &h_bvh->nodes);
    totalSize += copy_device_fixed_array(&placeholder.indices, &h_bvh->indices);

    totalSize += copy_device_struct(d_bvh, &placeholder);

    char buf[64];
    human_readable_size(buf, totalSize);
    log_info("Done [%s]\n", buf);
}

void bvh_free_device(bvh_t *d_bvh) {
    bvh_t placeholder;
    copy_host_struct(&placeholder, d_bvh);

    device_free(placeholder.nodes.items);
    device_free(placeholder.indices.items);

    device_free(d_bvh);
}

#define RAY_NO_HIT 1e30f

static __device__ float intersect_aabb(const AABB &aabb, const Ray &ray, const Vec3 &r_inv, float min_t) {
    Vec3 t1 = (aabb.min - ray.o) * r_inv;
    Vec3 t2 = (aabb.max - ray.o) * r_inv;
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

__device__ void bvh_intersect_iterative(const __restrict__ bvh_t *bvh, const Scene &scene, const Ray &ray,
                                        intersection_t &hit) {
    Vec3 r_inv = 1.0 / ray.r;

    bvh_node_t *stack[BVH_TRAVERSAL_STACK_SIZE];
    int depth = 0;
    stack[0] = &bvh->nodes.items[0];

    while (depth >= 0) {
        bvh_node_t *node = stack[depth--];

        while (node != NULL) {
            if (node->end - node->start > 0) {
                // is leaf
                for (uint32_t i = node->start; i < node->end; i++) {
                    intersect_face(scene, ray, hit, bvh->indices[i]);
                }
                node = NULL;
            } else {
                bvh_node_t *child1 = &bvh->nodes.items[node->leftChild];
                bvh_node_t *child2 = &bvh->nodes.items[node->rightChild];

                float t1 = intersect_aabb(child1->bounds, ray, r_inv, hit.distance);
                float t2 = intersect_aabb(child2->bounds, ray, r_inv, hit.distance);

                if (t2 < t1) {
                    bvh_node_t *tempc = child1;
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
                        depth++;
                        assert(depth < BVH_TRAVERSAL_STACK_SIZE);
                        stack[depth] = child2;
                    }
                }
            }
        }
    }
}

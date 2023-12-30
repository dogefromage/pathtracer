#include "bvh.h"
#include "assert.h"

void calculate_face_centroid(obj_scene_data* scene, obj_face* face, mfloat_t* centroid) {
    mfloat_t totalArea = 0;
    vec3_zero(centroid);
    for (uint32_t i = 2; i < face->vertex_count; i++) {
        mfloat_t* A = scene->vertex_list[face->vertices[0].position].v;
        mfloat_t* B = scene->vertex_list[face->vertices[i - 1].position].v;
        mfloat_t* C = scene->vertex_list[face->vertices[i].position].v;
        mfloat_t edge1[VEC3_SIZE], edge2[VEC3_SIZE], cross[VEC3_SIZE], triCentroid[VEC3_SIZE];

        vec3_subtract(edge1, B, A);
        vec3_subtract(edge2, C, A);
        vec3_cross(cross, edge1, edge2);
        mfloat_t triArea = 0.5 * vec3_length(cross);

        // calculate centroid on triangle
        vec3_assign(triCentroid, A);
        vec3_add(triCentroid, triCentroid, B);
        vec3_add(triCentroid, triCentroid, C);
        vec3_multiply_f(triCentroid, triCentroid, 0.33333333);

        // weighted sum of centroids
        vec3_multiply_f(triCentroid, triCentroid, triArea);
        vec3_add(centroid, centroid, triCentroid);
        totalArea += triArea;
    }
    assert(totalArea);
    vec3_multiply_f(centroid, centroid, 1.0 / totalArea);
}

void update_node_bounds(obj_scene_data* scene, BVH* bvh, uint32_t nodeIndex) {
    BVHNode* node = &bvh->nodes[nodeIndex];
    vec3_const(node->bounds.min, 1e30);
    vec3_const(node->bounds.max, -1e30);
    for (uint32_t i = node->start; i < node->end; i++) {
        obj_face* face = &scene->face_list[bvh->indices[i]];
        for (uint32_t j = 0; j < face->vertex_count; j++) {
            mfloat_t* v = scene->vertex_list[face->vertices[j].position].v;
            vec3_min(node->bounds.min, node->bounds.min, v);
            vec3_max(node->bounds.max, node->bounds.max, v);
        }
    }
}

void swap(uint32_t* a, uint32_t* b) {
    uint32_t t = *a;
    *a = *b;
    *b = t;
}

int intersect_aabb(AABB* aabb, Ray* ray, mfloat_t min_t) {

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

void subdivide(obj_scene_data* scene, BVH* bvh, uint32_t nodeIndex) {
    BVHNode* node = &bvh->nodes[nodeIndex];
    // https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
    int count = node->end - node->start;
    if (count <= 2) {
        return;  // stop criterion
    }
    mfloat_t extent[VEC3_SIZE];
    vec3_subtract(extent, node->bounds.max, node->bounds.min);
    int axis = 0;
    if (extent[1] > extent[axis]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;
    mfloat_t splitPos = node->bounds.min[axis] + extent[axis] * 0.5;

    // quicksort-esque splitting
    int i = node->start;
    int j = node->end - 1;
    while (i <= j) {
        if (bvh->centroids[bvh->indices[i]].v[axis] < splitPos) {
            i++;
        } else {
            swap(&bvh->indices[i], &bvh->indices[j--]);
        }
    }

    int leftCount = i - node->start;
    if (leftCount == 0 || leftCount == count) {
        return;  // cannot split because overlapping or something
    }

    uint32_t leftIndex = bvh->nodeCount++;
    uint32_t rightIndex = bvh->nodeCount++;
    BVHNode* left = &bvh->nodes[leftIndex];
    BVHNode* right = &bvh->nodes[rightIndex];
    left->start = node->start;
    left->end = right->start = i;
    right->end = node->end;
    node->leftChild = leftIndex;
    node->rightChild = rightIndex;
    node->start = node->end = 0;  // make non-leaf
    update_node_bounds(scene, bvh, leftIndex);
    update_node_bounds(scene, bvh, rightIndex);
    subdivide(scene, bvh, leftIndex);
    subdivide(scene, bvh, rightIndex);
}

void bvh_build(BVH* bvh, obj_scene_data* scene) {
    size_t primitiveCount = scene->face_count;
    size_t maxNodeCount = 2 * primitiveCount - 1;

    bvh->nodeCount = 0;
    bvh->nodes = (BVHNode*)malloc(sizeof(BVHNode) * maxNodeCount);

    // mutable primitive list for sorting faces
    bvh->indices = (uint32_t*)malloc(sizeof(uint32_t) * primitiveCount);
    for (uint32_t i = 0; i < primitiveCount; i++) {
        bvh->indices[i] = i;
    }
    // calculate centroids, accesses work with scene face indices
    bvh->centroids = (struct vec3*)malloc(sizeof(struct vec3) * primitiveCount);
    for (uint32_t i = 0; i < primitiveCount; i++) {
        calculate_face_centroid(scene, &scene->face_list[i], bvh->centroids[i].v);
    }

    uint32_t rootIndex = bvh->nodeCount++;
    BVHNode* root = &bvh->nodes[rootIndex];
    root->start = 0;
    root->end = primitiveCount;
    update_node_bounds(scene, bvh, rootIndex);
    subdivide(scene, bvh, rootIndex);

    // not needed anymore
    free(bvh->centroids);
    bvh->centroids = NULL;
}

void bvh_intersect(BVH* bvh, uint32_t nodeIndex, obj_scene_data* scene, Ray* ray, Intersection* hit) {
    BVHNode* node = &bvh->nodes[nodeIndex];
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

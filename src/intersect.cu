#include "intersect.h"

// void Ray_at(mfloat_t* out, Ray* ray, mfloat_t t) {
//     vec3_multiply_f(out, ray->r, t);
//     vec3_add(out, ray->o, out);
// }

static __device__ void
barycentric_lincom(
    mfloat_t* out,
    const mfloat_t* A, const mfloat_t* B, const mfloat_t* C,
    mfloat_t t, mfloat_t u, mfloat_t v) {
    mfloat_t x[3];
    vec3_multiply_f(out, A, t);
    vec3_multiply_f(x, B, u);
    vec3_add(out, out, x);
    vec3_multiply_f(x, C, v);
    vec3_add(out, out, x);
}

#define TRIANGLE_DETERMINANT_EPS 1e-12
#define TRIANGLE_MARGIN_EPS 1e-12
// super fast Möller Trumbore ray-triangle intersection
// https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
// https://www.youtube.com/watch?v=fK1RPmF_zjQ
__device__ int moeller_trumbore_intersect(
    const mfloat_t* orig, const mfloat_t* dir,
    const mfloat_t* vert0, const mfloat_t* vert1, const mfloat_t* vert2,
    mfloat_t* t, mfloat_t* u, mfloat_t* v) {
    mfloat_t edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
    mfloat_t det, inv_det;

    // Find vectors for two edges sharing vert0
    vec3_subtract(edge1, vert1, vert0);
    vec3_subtract(edge2, vert2, vert0);

    // Begin calculating determinant - also used to calculate U parameter
    vec3_cross(pvec, dir, edge2);
    // if determinant is near zero, ray lies in plane of triangle
    det = vec3_dot(edge1, pvec);

#ifdef TEST_CULL
    // Define TEST_CULL if culling is desired
    if (det < TRIANGLE_DETERMINANT_EPS)
        return 0;

    // Calculate distance from vert0 to ray origin
    vec3_subtract(tvec, orig, vert0);

    // Calculate U parameter and test bounds
    *u = vec3_dot(tvec, pvec);
    if (*u < 0.0 || *u > det)
        return 0;

    // Prepare to test V parameter
    vec3_cross(qvec, tvec, edge1);

    // Calculate V parameter and test bounds
    *v = vec3_dot(dir, qvec);
    if (*v < 0.0 || *u + *v > det)
        return 0;

    // Calculate t, scale parameters, ray intersects triangle
    *t = vec3_dot(edge2, qvec);
    inv_det = 1.0 / det;
    *t *= inv_det;
    *u *= inv_det;
    *v *= inv_det;
#else
    // The non-culling branch
    if (det > -TRIANGLE_DETERMINANT_EPS && det < TRIANGLE_DETERMINANT_EPS)
        return 0;
    inv_det = 1.0 / det;

    // Calculate distance from vert0 to ray origin
    vec3_subtract(tvec, orig, vert0);

    // Calculate U parameter and test bounds
    *u = vec3_dot(tvec, pvec) * inv_det;
    if (*u < -TRIANGLE_MARGIN_EPS || *u > (1.0 + TRIANGLE_MARGIN_EPS))
        return 0;

    // Prepare to test V parameter
    vec3_cross(qvec, tvec, edge1);

    // Calculate V parameter and test bounds
    *v = vec3_dot(dir, qvec) * inv_det;
    if (*v < -TRIANGLE_MARGIN_EPS || *u + *v > (1.0 + TRIANGLE_MARGIN_EPS))
        return 0;

    // Calculate t, ray intersects triangle
    *t = vec3_dot(edge2, qvec) * inv_det;
#endif
    return 1;
}

__device__ void 
intersect_face(const __restrict__ obj_scene_data* scene,
                               const Ray* ray, Intersection* hit, int faceIndex) {
    obj_face* face = &scene->face_list[faceIndex];

    // loop over n-gon triangles fan-style
    for (size_t i = 2; i < face->vertex_count; i++) {
        obj_face_vertex* a = &face->vertices[0];
        obj_face_vertex* b = &face->vertices[i - 1];
        obj_face_vertex* c = &face->vertices[i];
        mfloat_t* A = scene->vertex_list[a->position].v;
        mfloat_t* B = scene->vertex_list[b->position].v;
        mfloat_t* C = scene->vertex_list[c->position].v;

        mfloat_t t, u, v;
        int has_hit = moeller_trumbore_intersect(
            ray->o, ray->r, A, B, C, &t, &u, &v);

        if (has_hit && t >= 0 && t < hit->distance) {
            hit->has_hit = 1;
            hit->distance = t;
            if (face->material_index >= 0 && face->material_index < (int)scene->material_count) {
                hit->mat = &scene->material_list[face->material_index];
            } else {
                hit->mat = NULL;
            }

            mfloat_t t = 1.0 - u - v;
            barycentric_lincom(hit->position, A, B, C, t, u, v);
            barycentric_lincom(hit->texture_coord,
                               scene->vertex_texture_list[a->texture].v,
                               scene->vertex_texture_list[b->texture].v,
                               scene->vertex_texture_list[c->texture].v,
                               t, u, v);
            barycentric_lincom(hit->normal,
                               scene->vertex_normal_list[a->normal].v,
                               scene->vertex_normal_list[b->normal].v,
                               scene->vertex_normal_list[c->normal].v,
                               t, u, v);
            vec3_normalize(hit->normal, hit->normal);
            if (vec3_dot(hit->normal, ray->r) > 0) {
                // hit backside of face, mirror normal
                vec3_multiply_f(hit->normal, hit->normal, -1);
            }
        }
    }
}

__device__ void 
intersect_crude(const __restrict__ obj_scene_data* scene, const Ray* ray, Intersection* hit) {
    for (size_t i = 0; i < scene->face_count; i++) {
        intersect_face(scene, ray, hit, i);
    }
}

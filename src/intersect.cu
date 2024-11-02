#include "intersect.h"

// void Ray_at(mfloat_t* out, Ray* ray, mfloat_t t) {
//     vec3_multiply_f(out, ray->r, t);
//     vec3_add(out, ray->o, out);
// }

static PLATFORM Vec3 barycentric_lincom(
    const Vec3& A, const Vec3& B, const Vec3& C,
    float t, float u, float v) {
    return t * A + u * B + v * C;
}

#define TRIANGLE_DETERMINANT_EPS 1e-12
#define TRIANGLE_MARGIN_EPS 1e-12
static PLATFORM int
moeller_trumbore_intersect(const Ray& ray,
                           const Vec3& vert0, const Vec3& vert1, const Vec3& vert2, float* t, float* u, float* v) {
    Vec3 edge1, edge2, tvec, pvec, qvec;
    float det, inv_det;

    // Find vectors for two edges sharing vert0
    edge1 = vert1 - vert0;
    edge2 = vert2 - vert0;

    // Begin calculating determinant - also used to calculate U parameter
    pvec = ray.r.cross(edge2);
    // if determinant is near zero, ray lies in plane of triangle
    det = edge1.dot(pvec);

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
    tvec = ray.o - vert0;

    // Calculate U parameter and test bounds
    *u = tvec.dot(pvec) * inv_det;
    if (*u < -TRIANGLE_MARGIN_EPS || *u > (1.0 + TRIANGLE_MARGIN_EPS))
        return 0;

    // Prepare to test V parameter
    qvec = tvec.cross(edge1);

    // Calculate V parameter and test bounds
    *v = ray.r.dot(qvec) * inv_det;
    if (*v < -TRIANGLE_MARGIN_EPS || *u + *v > (1.0 + TRIANGLE_MARGIN_EPS)) {
        return 0;
    }

    // Calculate t, ray intersects triangle
    *t = edge2.dot(qvec) * inv_det;
#endif
    return 1;
}

PLATFORM void
intersect_face(const __restrict__ obj_scene_data* scene,
               const Ray& ray, intersection_t& hit, int faceIndex) {
    obj_face& face = scene->face_list[faceIndex];

    // loop over n-gon triangles fan-style
    for (size_t i = 2; i < face.vertex_count; i++) {
        obj_face_vertex& a = face.vertices[0];
        obj_face_vertex& b = face.vertices[i - 1];
        obj_face_vertex& c = face.vertices[i];
        Vec3 A = scene->vertex_list[a.position];
        Vec3 B = scene->vertex_list[b.position];
        Vec3 C = scene->vertex_list[c.position];

        float t, u, v;
        int has_hit = moeller_trumbore_intersect(ray, A, B, C, &t, &u, &v);

        if (has_hit && t >= 0 && t < hit.distance) {
            hit.has_hit = 1;
            hit.distance = t;
            if (face.material_index >= 0 && face.material_index < (int)scene->material_count) {
                hit.mat = &scene->material_list[face.material_index];
            } else {
                hit.mat = NULL;
            }

            float t = 1.0 - u - v;
            hit.position = barycentric_lincom(A, B, C, t, u, v);
            hit.texture_coord = barycentric_lincom(
                scene->vertex_texture_list[a.texture],
                scene->vertex_texture_list[b.texture],
                scene->vertex_texture_list[c.texture],
                t, u, v);

            hit.normal = barycentric_lincom(
                scene->vertex_normal_list[a.normal],
                scene->vertex_normal_list[b.normal],
                scene->vertex_normal_list[c.normal],
                t, u, v);
            hit.normal.normalize();

            if (hit.normal.dot(ray.r) > 0) {
                // hit backside of face, mirror normal
                hit.normal = -hit.normal;
            }
        }
    }
}

PLATFORM void
intersect_crude(const __restrict__ obj_scene_data* scene, const Ray& ray, intersection_t& hit) {
    for (size_t i = 0; i < scene->face_count; i++) {
        intersect_face(scene, ray, hit, i);
    }
}

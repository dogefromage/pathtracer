#include <assert.h>

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
intersect_face(const __restrict__ scene_t* scene,
               const Ray& ray, intersection_t& hit, int faceIndex) {
    const face_t& face = scene->faces[faceIndex];

    // loop over n-gon triangles fan-style
    for (size_t i = 2; i < face.vertexCount; i++) {
        const uint32_t& a = face.vertices[0];
        const uint32_t& b = face.vertices[i - 1];
        const uint32_t& c = face.vertices[i];
        const Vec3& A = scene->vertices[a].position;
        const Vec3& B = scene->vertices[b].position;
        const Vec3& C = scene->vertices[c].position;

        float t, u, v;
        int has_hit = moeller_trumbore_intersect(ray, A, B, C, &t, &u, &v);

        if (has_hit && t >= 0 && t < hit.distance) {
            hit.has_hit = 1;
            hit.distance = t;
            assert(face.material < (int)scene->materials.count);

            hit.mat = &scene->materials[face.material];

            float t = 1.0 - u - v;
            hit.position = barycentric_lincom(A, B, C, t, u, v);

            hit.texture_coord.set(0);  // TODO
            // hit.texture_coord = barycentric_lincom(
            //     scene->vertex_texture_list[a.texture],
            //     scene->vertex_texture_list[b.texture],
            //     scene->vertex_texture_list[c.texture],
            //     t, u, v);

            switch (face.shading) {
                case FLAT_SHADING:
                    hit.trueNormal = face.faceNormal;
                    break;
                case BARY_SHADING:
                    hit.trueNormal = barycentric_lincom(
                        scene->vertices[a].normal,
                        scene->vertices[b].normal,
                        scene->vertices[c].normal,
                        t, u, v);
                    hit.trueNormal.normalize();
                    break;
            }

            if (hit.trueNormal.dot(ray.r) > 0) {
                // hit backside of face, mirror normal
                hit.lightingNormal = -hit.trueNormal;
            } else {
                hit.lightingNormal = hit.trueNormal;
            }
        }
    }
}

PLATFORM void
intersect_crude(const __restrict__ scene_t* scene, const Ray& ray, intersection_t& hit) {
    for (size_t i = 0; i < scene->faces.count; i++) {
        intersect_face(scene, ray, hit, i);
    }
}

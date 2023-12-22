#include "renderer.h"

#include "math.h"
#include "obj_parser.h"

// super fast Möller Trumbore ray-triangle intersection
// https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
// https://www.youtube.com/watch?v=fK1RPmF_zjQ

#define EPSILON 1e-6

#define CROSS(dest, v1, v2)                  \
    dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
    dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
    dest[2] = v1[0] * v2[1] - v1[1] * v2[0];

#define DOT(v1, v2) \
    (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])

#define SUB(dest, v1, v2)    \
    dest[0] = v1[0] - v2[0]; \
    dest[1] = v1[1] - v2[1]; \
    dest[2] = v1[2] - v2[2];

#define BARY(x, A, B, C, t, u, v)          \
    x[0] = t * A[0] + u * B[0] + v * C[0]; \
    x[1] = t * A[1] + u * B[1] + v * C[1]; \
    x[2] = t * A[2] + u * B[2] + v * C[2]

int moeller_trumbore_intersect(double orig[3], double dir[3], double vert0[3], double vert1[3], double vert2[3],
                               double* t, double* u, double* v) {
    double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
    double det, inv_det;

    // Find vectors for two edges sharing vert0
    SUB(edge1, vert1, vert0);
    SUB(edge2, vert2, vert0);

    // Begin calculating determinant - also used to calculate U parameter
    CROSS(pvec, dir, edge2);
    // if determinant is near zero, ray lies in plane of triangle
    det = DOT(edge1, pvec);

#ifdef TEST_CULL
    // Define TEST_CULL if culling is desired
    if (det < EPSILON)
        return 0;

    // Calculate distance from vert0 to ray origin
    SUB(tvec, orig, vert0);

    // Calculate U parameter and test bounds
    *u = DOT(tvec, pvec);
    if (*u < 0.0 || *u > det)
        return 0;

    // Prepare to test V parameter
    CROSS(qvec, tvec, edge1);

    // Calculate V parameter and test bounds
    *v = DOT(dir, qvec);
    if (*v < 0.0 || *u + *v > det)
        return 0;

    // Calculate t, scale parameters, ray intersects triangle
    *t = DOT(edge2, qvec);
    inv_det = 1.0 / det;
    *t *= inv_det;
    *u *= inv_det;
    *v *= inv_det;
#else
    // The non-culling branch
    if (det > -EPSILON && det < EPSILON)
        return 0;
    inv_det = 1.0 / det;

    // Calculate distance from vert0 to ray origin
    SUB(tvec, orig, vert0);

    // Calculate U parameter and test bounds
    *u = DOT(tvec, pvec) * inv_det;
    if (*u < 0.0 || *u > 1.0)
        return 0;

    // Prepare to test V parameter
    CROSS(qvec, tvec, edge1);

    // Calculate V parameter and test bounds
    *v = DOT(dir, qvec) * inv_det;
    if (*v < 0.0 || *u + *v > 1.0)
        return 0;

    // Calculate t, ray intersects triangle
    *t = DOT(edge2, qvec) * inv_det;
#endif

    return 1;
}

static void intersect_face(obj_scene_data* scene, ray_t* ray, Intersection* hit, int faceIndex) {
    obj_face* face = scene->face_list[faceIndex];
    double O[3], D[3];
    O[0] = ray->o.x;
    O[1] = ray->o.y;
    O[2] = ray->o.z;
    D[0] = ray->r.x;
    D[1] = ray->r.y;
    D[2] = ray->r.z;

    // loop over n-gon triangles fan-style
    for (int i = 2; i < face->vertex_count; i++) {
        int a = 0;
        int b = i - 1;
        int c = i;
        int vert_a = face->vertex_index[a];
        int vert_b = face->vertex_index[b];
        int vert_c = face->vertex_index[c];
        double* A = scene->vertex_list[vert_a]->e;
        double* B = scene->vertex_list[vert_b]->e;
        double* C = scene->vertex_list[vert_c]->e;
        double t, u, v;
        int has_hit = moeller_trumbore_intersect(
            O, D, A, B, C, &t, &u, &v);

        if (has_hit && t > hit->distance) {
            hit->distance = t;
            hit->material = face->material_index;
            double t = 1.0 - u - v;
            BARY(hit->position, A, B, C, t, u, v);
            BARY(hit->texture_coord,
                 scene->vertex_texture_list[face->texture_index[a]]->e,
                 scene->vertex_texture_list[face->texture_index[b]]->e,
                 scene->vertex_texture_list[face->texture_index[c]]->e,
                 t, u, v);
            BARY(hit->normal,
                 scene->vertex_normal_list[face->normal_index[a]]->e,
                 scene->vertex_normal_list[face->normal_index[b]]->e,
                 scene->vertex_normal_list[face->normal_index[c]]->e,
                 t, u, v);

            // normalize normal
            double invNorm = 1.0 / sqrt(DOT(hit->normal, hit->normal));
            for (int i = 0; i < 3; i++) {
                hit->normal[i] *= invNorm;
            }
        }
    }
}

// static void intersect_sphere(obj_scene_data* scene, ray_t* ray, Intersection* hit, int sphere) {
//     // intersect some sphere at
//     scene->sphere_list[sphere]
//     vec3_t sphere_origin =
//     float sphere_radius = 1.0;

//     // length to plane
//     float t = v3_dot(ray->r, v3_sub(sphere_origin, ray->o));

//     if (t < 0) {
//         return 0;  // behind camera
//     }

//     // intersection point at parallel plane
//     vec3_t x = ray_at(ray, t);

//     // find distance from x to origin
//     float dXO = v3_length(v3_sub(sphere_origin, x));

//     float dXISqr = sphere_radius * sphere_radius - dXO * dXO;

//     if (dXISqr < 0) {
//         return 0;  // ray misses sphere
//     }

//     return 1;  // ray hits sphere, doesn't matter where for now

//     // float dXI = sqrtf(dXISqr);
//     // ...
// }

// static void intersect_plane(obj_scene_data* scene, ray_t* ray, Intersection* hit, int plane) {

// }

static void intersect(obj_scene_data* scene, ray_t* ray, Intersection* hit) {
    hit->distance = -1;
    for (int i = 0; i < scene->face_count; i++) {
        intersect_face(scene, ray, hit, i);
    }
}

void render(vec3_t* pixel, obj_scene_data* scene, float u, float v) {
    // printf("Render (%f, %f)", u, v);

    float sensor_height = 0.2;
    float focal_length = 0.6;

    vec3_t cam_pos = vec3(0, 0, -5);
    vec3_t cam_dir = v3_norm(vec3(
        sensor_height * u,
        sensor_height * v,
        1 * focal_length));
    ray_t camera_ray = { .o=cam_pos, .r=cam_dir };

    Intersection hit;

    intersect(scene, &camera_ray, &hit);

    // NORMAL
    pixel->x = hit.normal[0];
    pixel->y = hit.normal[1];
    pixel->z = hit.normal[2];

    // // DEPTH
    // float far = 8;
    // float near = 2;
    // float b = (hit.distance - near) / (far - near);
    // *pixel = (vec3_t){b, b, b};

    // // HAS HIT
    // if (hit.distance >= 0) {
    //     *pixel = (vec3_t){0, 0, 0};
    // } else {
    //     *pixel = (vec3_t){0, 0, 0};
    // }
}
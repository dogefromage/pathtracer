#include "renderer.h"

#include <assert.h>
#include <stdio.h>

#include "math.h"
#include "obj_parser.h"

void Ray_at(mfloat_t* out, Ray* ray, mfloat_t t) {
    vec3_multiply_f(out, ray->r, t);
    vec3_add(out, ray->o, out);
}

mfloat_t random_frac() {
    return rand() / ((mfloat_t)RAND_MAX);
}

void sphere_sample_uniform(mfloat_t* out) {
    mfloat_t x, y, z, d;
    do {
        x = 2 * random_frac() - 1;
        y = 2 * random_frac() - 1;
        z = 2 * random_frac() - 1;
        d = x * x + y * y + z * z;
    } while (d > 1);
    mfloat_t inv_l = 1.0 / MSQRT(d);
    out[0] = x * inv_l;
    out[1] = y * inv_l;
    out[2] = z * inv_l;
}

void hemi_sample_uniform(mfloat_t* out, mfloat_t* normal_unit) {
    sphere_sample_uniform(out);
    double dotp = vec3_dot(out, normal_unit);
    if (dotp < 0) {
        // mirror out by normal
        mfloat_t corr[VEC3_SIZE];
        vec3_multiply_f(corr, normal_unit, -2 * dotp);
        vec3_add(out, out, corr);
    }
}

void barycentric_lincom(
    mfloat_t* out,
    mfloat_t* A, mfloat_t* B, mfloat_t* C,
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
int moeller_trumbore_intersect(
    mfloat_t* orig, mfloat_t* dir,
    mfloat_t* vert0, mfloat_t* vert1, mfloat_t* vert2,
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

void intersect_face(obj_scene_data* scene, Ray* ray, Intersection* hit, int faceIndex) {
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
            assert(face->material_index >= 0 && face->material_index < (int)scene->material_count);
            hit->mat = scene->material_list[face->material_index];
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

void intersect(obj_scene_data* scene, Ray* ray, Intersection* hit) {
    hit->has_hit = 0;
    hit->distance = CLEAR_DISTANCE;
    for (size_t i = 0; i < scene->face_count; i++) {
        intersect_face(scene, ray, hit, i);
    }
}

void get_camera_ray(Ray* ray, obj_scene_data* scene, mfloat_t u, mfloat_t v) {
    assert(scene->camera);

    // camera settings
    mfloat_t sensor_height = 0.2,
             focal_length = 0.25;

    struct vec3 U, V, W,
        *P = &scene->vertex_list[scene->camera->position],
        *T = &scene->vertex_list[scene->camera->target],
        *Up = &scene->vertex_normal_list[scene->camera->updir];

    vec3_subtract(W.v, T->v, P->v);
    vec3_cross(U.v, W.v, Up->v);
    vec3_cross(V.v, U.v, W.v);

    vec3_normalize(U.v, U.v);
    vec3_normalize(V.v, V.v);
    vec3_normalize(W.v, W.v);

    vec3_multiply_f(U.v, U.v, 0.5 * sensor_height);
    vec3_multiply_f(V.v, V.v, 0.5 * sensor_height);
    vec3_multiply_f(W.v, W.v, focal_length);

    // U, V, W build orthogonal basis for camera ray direction
    // D = u*U + v*V + 1*W

    // build change of basis matrix A
    struct mat3 A;
    vec3_assign(&A.m11, U.v);
    vec3_assign(&A.m12, V.v);
    vec3_assign(&A.m13, W.v);

    // camera coordinate vector for pixel
    struct vec3 x = {.x = u, .y = v, .z = 1};

    vec3_assign(ray->o, P->v);
    vec3_multiply_mat3(ray->r, x.v, A.v);
    vec3_normalize(ray->r, ray->r);
}

// offset new ray slightly from triangle in normal dir
#define SAVE_RAY_EPS 1e-6
void initialize_safe_ray(Ray* ray, mfloat_t* origin, mfloat_t *dir, mfloat_t* normal) {
    vec3_multiply_f(ray->o, normal, SAVE_RAY_EPS);
    vec3_add(ray->o, ray->o, origin);
    vec3_assign(ray->r, dir);
}

void get_brdf(obj_material* mat, mfloat_t* out) {
    mfloat_t distribution = 1 / MPI; // basic diffuse
    vec3_assign(out, mat->diff);
    vec3_multiply_f(out, out, distribution);
}

#define SAMPLES 500
// #define BOUNCES 5
#define RR_PROB_MAX 0.99

void integrate_Li(mfloat_t* light, obj_scene_data* scene, Ray* v_inv, int depth, mfloat_t* throughput) {
    Intersection hit;
    intersect(scene, v_inv, &hit);
    if (!hit.has_hit) {
        return;
    }

    vec3_add(light, light, hit.mat->emit);

    mfloat_t rr_prob = MFMIN(vec3_max_component(throughput), RR_PROB_MAX);
    if (random_frac() >= rr_prob) {
        return; // ray dies
    }

    mfloat_t omega_i[VEC3_SIZE];
    mfloat_t prob_i = 1 / (2 * MPI); // because even distribution
    hemi_sample_uniform(omega_i, hit.normal);

    Ray bounce_ray;
    initialize_safe_ray(&bounce_ray, hit.position, omega_i, hit.normal);

    mfloat_t brdf[VEC3_SIZE], next_throughput[VEC3_SIZE], bounce_light[VEC3_SIZE];

    mfloat_t cosTheta = vec3_dot(hit.normal, omega_i);
    get_brdf(hit.mat, brdf);

    vec3_multiply(next_throughput, throughput, brdf);
    vec3_multiply_f(next_throughput, next_throughput, cosTheta / (prob_i * rr_prob));

    // recurse
    vec3_zero(bounce_light);
    integrate_Li(bounce_light, scene, &bounce_ray, depth + 1, next_throughput);

    vec3_multiply(bounce_light, bounce_light, brdf);
    vec3_multiply_f(bounce_light, bounce_light, cosTheta / (prob_i * rr_prob));
    vec3_add(light, light, bounce_light);
}

void render(mfloat_t* color, obj_scene_data* scene, mfloat_t u, mfloat_t v, int samples) {
    vec3_zero(color);
    Ray camera_ray;
    get_camera_ray(&camera_ray, scene, u, v);
    mfloat_t initial_throughput[VEC3_SIZE];
    vec3_one(initial_throughput);

    for (int i = 0; i < samples; i++) {
        integrate_Li(color, scene, &camera_ray, 0, initial_throughput);
    }
    vec3_divide_f(color, color, (mfloat_t)samples);
}
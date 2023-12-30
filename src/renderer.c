#include "renderer.h"

#include <assert.h>
#include <stdio.h>

#include "math.h"
#include "obj_parser.h"

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
void initialize_safe_ray(Ray* ray, mfloat_t* origin, mfloat_t* dir, mfloat_t* normal) {
    vec3_multiply_f(ray->o, normal, SAVE_RAY_EPS);
    vec3_add(ray->o, ray->o, origin);
    vec3_assign(ray->r, dir);
}

void get_brdf(obj_material* mat, mfloat_t* out) {
    mfloat_t distribution = 1 / MPI;  // basic diffuse
    vec3_assign(out, mat->diff);
    vec3_multiply_f(out, out, distribution);
}

// IT WORKS!!!
void intersect(BVH* bvh, obj_scene_data* scene, Ray* ray, Intersection* hit) {
    hit->has_hit = 0;
    hit->distance = CLEAR_DISTANCE;
#if 0
    intersect_crude(scene, ray, hit);
#else
    bvh_intersect(bvh, 0, scene, ray, hit);
#endif
}

#define RR_PROB_MAX 0.99

void integrate_Li(mfloat_t* light, BVH* bvh, obj_scene_data* scene, Ray* v_inv, int depth, mfloat_t* throughput) {
    Intersection hit;
    intersect(bvh, scene, v_inv, &hit);
    if (!hit.has_hit) {
        return;
    }

    vec3_add(light, light, hit.mat->emit);

    mfloat_t rr_prob = MFMIN(vec3_max_component(throughput), RR_PROB_MAX);
    if (random_frac() >= rr_prob) {
        return;  // ray dies
    }

    mfloat_t omega_i[VEC3_SIZE];
    mfloat_t prob_i = 1 / (2 * MPI);  // because even distribution
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
    integrate_Li(bounce_light, bvh, scene, &bounce_ray, depth + 1, next_throughput);

    vec3_multiply(bounce_light, bounce_light, brdf);
    vec3_multiply_f(bounce_light, bounce_light, cosTheta / (prob_i * rr_prob));
    vec3_add(light, light, bounce_light);
}

void render(mfloat_t* color, BVH* bvh, obj_scene_data* scene, mfloat_t u, mfloat_t v, int samples) {
    vec3_zero(color);
    Ray camera_ray;
    get_camera_ray(&camera_ray, scene, u, v);
    mfloat_t initial_throughput[VEC3_SIZE];
    vec3_one(initial_throughput);

    for (int i = 0; i < samples; i++) {
        integrate_Li(color, bvh, scene, &camera_ray, 0, initial_throughput);
    }
    vec3_divide_f(color, color, (mfloat_t)samples);
}
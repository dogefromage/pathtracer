#include "brdf.h"

static PLATFORM void 
sphere_sample_uniform(mfloat_t* out, rand_state_t* rstate) {
    mfloat_t x, y, z, d;
    do {
        x = 2 * random_uniform(rstate) - 1;
        y = 2 * random_uniform(rstate) - 1;
        z = 2 * random_uniform(rstate) - 1;
        d = x * x + y * y + z * z;
    } while (d > 1);
    mfloat_t inv_l = 1.0 / MSQRT(d);
    out[0] = x * inv_l;
    out[1] = y * inv_l;
    out[2] = z * inv_l;
}

static PLATFORM void 
hemi_sample_uniform(mfloat_t* out, mfloat_t* normal_unit, rand_state_t* rstate) {
    sphere_sample_uniform(out, rstate);
    double dotp = vec3_dot(out, normal_unit);
    if (dotp < 0) {
        // mirror out by normal
        mfloat_t corr[VEC3_SIZE];
        vec3_multiply_f(corr, normal_unit, -2 * dotp);
        vec3_add(out, out, corr);
    }
}

PLATFORM void 
sample_brdf(brdf_t* out, mfloat_t* v_inv, Intersection* hit, rand_state_t* rstate) {

    hemi_sample_uniform(out->omega_i, hit->normal, rstate);
    
    bool isDiffuse = vec3_max_component(hit->mat->spec) < 0.99;
    if (isDiffuse) {
        hemi_sample_uniform(out->omega_i, hit->normal, rstate);
        out->prob_i = 1 / (2 * MPI);  /* probability of chosen direction */
    
        mfloat_t distribution = 1 / MPI;  // basic diffuse
        vec3_assign(out->brdf, hit->mat->diff);
        vec3_multiply_f(out->brdf, out->brdf, distribution);
    } else {
        // is specular
        // mirror ray TODO

        mfloat_t v[3];
        vec3_negative(v, v_inv);

        hemi_sample_uniform(out->omega_i, hit->normal, rstate);
        mfloat_t t = vec3_dot(v, hit->normal) / vec3_dot(hit->normal, hit->normal);

        mfloat_t rv[3];
        vec3_multiply_f(rv, hit->normal, -2.0 * t);
        vec3_add(rv, rv, v);
        vec3_negative(rv, rv);
        vec3_assign(out->omega_i, rv);

        out->prob_i = 1.0;

        mfloat_t cos_theta = -vec3_dot(v_inv, hit->normal);
        vec3_const(out->brdf, 1.0 / cos_theta);
    }
}
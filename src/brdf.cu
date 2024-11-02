#include "brdf.h"

static PLATFORM Vec3
sphere_sample_uniform(rand_state_t& rstate) {
    Vec3 r;
    do {
        r.x = 2 * random_uniform(rstate) - 1;
        r.y = 2 * random_uniform(rstate) - 1;
        r.z = 2 * random_uniform(rstate) - 1;
    } while (r.dot(r) > 1);
    
    return r.normalized();
}

static PLATFORM Vec3
hemi_sample_uniform(const Vec3& normal_unit, rand_state_t& rstate) {
    Vec3 r = sphere_sample_uniform(rstate);
    float dotp = r.dot(normal_unit);
    if (dotp < 0) {
        // mirror out by normal
        return r - 2 * dotp * normal_unit;
    } else {
        return r;
    }
}

PLATFORM void 
sample_brdf(brdf_t& out, const Vec3& v_inv, const intersection_t& hit, rand_state_t& rstate) {

    bool isDiffuse = hit.mat->spec.maxComponent() < 0.99;
    if (isDiffuse) {
        out.omega_i = hemi_sample_uniform(hit.normal, rstate);
        out.prob_i = 1 / (2 * M_PIf);  /* probability of chosen direction */
        out.brdf = hit.mat->diff / M_PIf;
    } else {
        // is specular
        Vec3 v = -v_inv;
        float cos_theta = v.dot(hit.normal);

        Vec3 rv = 2 * cos_theta * hit.normal - v;
        out.omega_i = rv;
        out.prob_i = 1.0;

        out.brdf.set(1.0 / cos_theta);
    }
}
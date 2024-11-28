#include "bsdf.h"

typedef struct {
    Vec3 v;
    float p;
} sample_t;

static PLATFORM Vec3
world_of_local_dir(const Vec3& normal, const Vec3& v) {
    // let normal be new z coordinate
    Vec3 z = normal;

    Vec3 e = { 1, 0, 0 };
    if (std::abs(e.dot(normal)) >= 0.9) {
        // choose other e since very close
        e = { 0, 1, 0 };
    }

    Vec3 x = z.cross(e);
    x.normalize();
    Vec3 y = z.cross(x);

    return x * v.x + y * v.y + z * v.z;
}

// static PLATFORM Vec3
// sphere_sample_uniform(rand_state_t& rstate) {
//     Vec3 r;
//     do {
//         r.x = 2 * random_uniform(rstate) - 1;
//         r.y = 2 * random_uniform(rstate) - 1;
//         r.z = 2 * random_uniform(rstate) - 1;
//     } while (r.dot(r) > 1);
    
//     return r.normalized();
// }

// static PLATFORM Vec3
// hemi_sample_uniform(const Vec3& normal_unit, rand_state_t& rstate) {
//     Vec3 r = sphere_sample_uniform(rstate);
//     r.z = std::abs(r.z);
//     return world_of_local_dir(normal_unit, r);
// }

static PLATFORM sample_t
hemi_sample_cosine(const Vec3 normal_unit, rand_state_t& rstate) {
    // weird trick: sample a circle and just "push" upwards into hemisphere
    // https://www.youtube.com/watch?v=c6NvZ74LAhE

    float x, y, d = 0;
    do {
        x = 2 * random_uniform(rstate) - 1;
        y = 2 * random_uniform(rstate) - 1;
        d = x * x + y * y;
    } while (d >= 1);

    float z = SQRT(1 - d);
    Vec3 local = Vec3(x, y, z);

    float cos_theta = z;
    float p = cos_theta / M_PI;

    Vec3 world = world_of_local_dir(normal_unit, local);
    return { world, p };
}

// https://blog.demofox.org/2017/01/09/raytracing-reflection-refraction-fresnel-total-internal-reflection-and-beers-law/
static PLATFORM float 
fresnel_reflect_amount(float n1, float n2, Vec3 normal, Vec3 v_inv) {
        // Schlick aproximation
        float r0 = (n1-n2) / (n1+n2);
        r0 *= r0;
        float cosX = -normal.dot(v_inv);
        if (n1 > n2)
        {
            float n = n1/n2;
            float sinT2 = n*n*(1.0-cosX*cosX);
            // Total internal reflection
            if (sinT2 > 1.0)
                return 1.0;
            cosX = SQRT(1.0-sinT2);
        }
        float x = 1.0-cosX;
        float ret = r0+(1.0-r0)*x*x*x*x*x;
        return ret;
}

static PLATFORM Vec3 
reflect(Vec3 normal, Vec3 v_inv) {
    return v_inv - 2 * normal.dot(v_inv) * normal;
}

static PLATFORM Vec3 
refract(float n1, float n2, Vec3 normal, Vec3 v_inv) {
    Vec3 p1 = v_inv - v_inv.dot(normal) * normal;
    Vec3 p2 = (n1 / n2) * p1;
    float sin2out = p2.dot(p2);
    if (sin2out >= 1.0) {
        // total internal reflection
        return reflect(normal, v_inv);
    } 
    float cosOut = SQRT(1.0 - sin2out);
    
    return p2 - normal * cosOut; // horizontal and vertical component
}

static PLATFORM void 
sample_bsdf_mirror(bsdf_sample_t& out, const Vec3& v_inv, const intersection_t& hit, rand_state_t& rstate) {
    out.omega_i = reflect(hit.lightingNormal, v_inv);
    out.prob_i = 1.0;
    float cos_theta = out.omega_i.dot(hit.lightingNormal);
    out.bsdf.set(1.0 / cos_theta);
}

static PLATFORM void 
sample_bsdf_glass(bsdf_sample_t& out, const Vec3& v_inv, const intersection_t& hit, rand_state_t& rstate) {
    bool isBackface = v_inv.dot(hit.trueNormal) > 0;

    float n1 = 1.0;
    float n2 = hit.mat->ior;
    if (isBackface) {
        float temp = n1;
        n1 = n2;
        n2 = temp;
    }

    float R = fresnel_reflect_amount(n1, n2, hit.lightingNormal, v_inv);

    bool isReflecting = random_uniform(rstate) < R;
    if (isReflecting) {

        sample_bsdf_mirror(out, v_inv, hit, rstate);
        out.prob_i *= R;
        out.bsdf *= R;
        return;

    } else {
        // REFRACTION
        out.omega_i = refract(n1, n2, hit.lightingNormal, v_inv);
        out.prob_i = 1 - R;
        float cos_theta = std::abs(out.omega_i.dot(hit.lightingNormal));
        out.bsdf.set((1 - R) / cos_theta);
    }
}

PLATFORM void 
sample_bsdf(bsdf_sample_t& out, const Vec3& v_inv, const intersection_t& hit, rand_state_t& rstate) {

    bool isGlass = false;
    // bool isGlass = hit.mat->diffuse <= 0.01;
    if (isGlass) {
        sample_bsdf_glass(out, v_inv, hit, rstate);
        return;
    }

    bool isMirror = false;
    // bool isMirror = hit.mat->spec.maxComponent() >= 0.99;
    if (isMirror) {
        sample_bsdf_mirror(out, v_inv, hit, rstate);
        return;
    }

    // diffuse
    out.bsdf = hit.mat->color / M_PIf;
    sample_t sample = hemi_sample_cosine(hit.lightingNormal, rstate);
    out.omega_i = sample.v;
    out.prob_i = sample.p;

    // UNIFORM
    // out.omega_i = hemi_sample_uniform(hit.normal, rstate);
    // out.prob_i = 1 / (2 * M_PIf);  /* probability of chosen direction */

    return;
}

PLATFORM Vec3 
evaluate_bsdf(const Vec3& v_inv, const Vec3& w, const intersection_t& hit, rand_state_t& rstate) {
    // assume diffuse
    return hit.mat->color / M_PIf;
}

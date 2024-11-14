#include "bsdf.h"

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

// https://blog.demofox.org/2017/01/09/raytracing-reflection-refraction-fresnel-total-internal-reflection-and-beers-law/
PLATFORM float fresnel_reflect_amount(float n1, float n2, Vec3 normal, Vec3 v_inv) {
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

PLATFORM Vec3 reflect(Vec3 normal, Vec3 v_inv) {
    return v_inv - 2 * normal.dot(v_inv) * normal;
}

PLATFORM Vec3 refract(float n1, float n2, Vec3 normal, Vec3 v_inv) {
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

PLATFORM void 
sample_bsdf(bsdf_t& out, const Vec3& v_inv, const intersection_t& hit, rand_state_t& rstate) {

    bool isBackface = v_inv.dot(hit.normal) > 0;

    bool isGlass = hit.mat->dissolve < 0.1;
    bool isMirror = hit.mat->spec.maxComponent() >= 0.99;

    if (isGlass) {
        
        float n1 = 1.0;
        float n2 = hit.mat->refract_index;
        if (isBackface) {
            float temp = n1;
            n1 = n2;
            n2 = temp;
        }

        float R = fresnel_reflect_amount(n1, n2, hit.normal, v_inv);
        // printf("%.2f\n", R);

        bool isReflecting = random_uniform(rstate) < R;
        if (isReflecting) {

            out.omega_i = reflect(hit.normal, v_inv);
            out.prob_i = R;

            float cos_theta = out.omega_i.dot(hit.normal);
            out.bsdf.set(R / cos_theta);

        } else {

            // out.omega_i.set(0);
            // out.prob_i = 1 - R;
            // out.bsdf.set(0);
            
            out.omega_i = refract(n1, n2, hit.normal, v_inv);
            out.prob_i = 1 - R;

            float cos_theta = -out.omega_i.dot(hit.normal);
            out.bsdf.set((1 - R) / cos_theta);
        }

        return;
    }

    if (isMirror) {
        out.omega_i = reflect(hit.normal, v_inv);
        out.prob_i = 1.0;

        float cos_theta = out.omega_i.dot(hit.normal);

        out.bsdf.set(1.0 / cos_theta);
        return;
    }

    // diffuse
    out.omega_i = hemi_sample_uniform(hit.normal, rstate);
    out.prob_i = 1 / (2 * M_PIf);  /* probability of chosen direction */
    out.bsdf = hit.mat->diff / M_PIf;
    return;
}
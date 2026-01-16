#include "brdf.h"

// // //
// //
// https://blog.demofox.org/2017/01/09/raytracing-reflection-refraction-fresnel-total-internal-reflection-and-beers-law/
// // static __device__ float fresnel_reflect_amount(float n1, float n2, Vec3 normal, Vec3
// v_inv) {
// //     // Schlick aproximation
// //     float r0 = (n1 - n2) / (n1 + n2);
// //     r0 *= r0;
// //     float cosX = -normal.dot(v_inv);
// //     if (n1 > n2) {
// //         float n = n1 / n2;
// //         float sinT2 = n * n * (1.0 - cosX * cosX);
// //         // Total internal reflection
// //         if (sinT2 > 1.0)
// //             return 1.0;
// //         cosX = SQRT(1.0 - sinT2);
// //     }
// //     float x = 1.0 - cosX;
// //     float ret = r0 + (1.0 - r0) * x * x * x * x * x;
// //     return ret;
// // }

// static __device__ Vec3 reflect(Vec3 normal, Vec3 v_inv) {
//     return v_inv - 2 * normal.dot(v_inv) * normal;
// }

// // static __device__ Vec3 refract(float n1, float n2, Vec3 normal, Vec3 v_inv) {
// //     Vec3 p1 = v_inv - v_inv.dot(normal) * normal;
// //     Vec3 p2 = (n1 / n2) * p1;
// //     float sin2out = p2.dot(p2);
// //     if (sin2out >= 1.0) {
// //         // total internal reflection
// //         return reflect(normal, v_inv);
// //     }
// //     float cosOut = SQRT(1.0 - sin2out);

// //     return p2 - normal * cosOut; // horizontal and vertical component
// // }

// static __device__ void sample_bsdf_mirror(bsdf_sample_t &out, const Vec3 &v_inv,
//                                           const intersection_t &hit, rand_state_t &rstate) {
//     out.omega_i = reflect(hit.shaded_normal, v_inv);
//     out.prob_i = 1.0;
//     float cos_theta = out.omega_i.dot(hit.shaded_normal);
//     out.bsdf = Spectrum::fromRGB(hit.color) / cos_theta;
// }

// // static __device__ void sample_bsdf_glass(bsdf_sample_t &out, const Vec3 &v_inv, const
// // intersection_t &hit,
// //                                          rand_state_t &rstate) {
// //     bool isBackface = v_inv.dot(hit.true_normal) > 0;

// //     float n1 = 1.0;
// //     float n2 = hit.mat->ior;
// //     if (isBackface) {
// //         float temp = n1;
// //         n1 = n2;
// //         n2 = temp;
// //     }

// //     float R = fresnel_reflect_amount(n1, n2, hit.incident_normal, v_inv);

// //     bool isReflecting = random_uniform(rstate) < R;
// //     if (isReflecting) {
// //         sample_bsdf_mirror(out, v_inv, hit, rstate);
// //         out.prob_i *= R;
// //         out.bsdf *= R;
// //         return;

// //     } else {
// //         // REFRACTION
// //         out.omega_i = refract(n1, n2, hit.incident_normal, v_inv);
// //         out.prob_i = 1 - R;
// //         float cos_theta = std::abs(out.omega_i.dot(hit.incident_normal));
// //         out.bsdf = (1 - R) / cos_theta * Spectrum::Itentity();
// //     }
// // }

// __device__ void sample_bsdf(bsdf_sample_t &out, const Vec3 &v_inv, const intersection_t &hit,
//                             rand_state_t &rstate) {

//     // // bool isGlass = false;
//     // bool isGlass = hit.mat->transmission >= 0.9;
//     // if (isGlass) {
//     //     sample_bsdf_glass(out, v_inv, hit, rstate);
//     //     return;
//     // }

//     // // bool isMirror = false;
//     // bool isMirror = hit.mat->metallic >= 0.9;
//     // if (isMirror) {
//     //     sample_bsdf_mirror(out, v_inv, hit, rstate);
//     //     return;
//     // }

//     // diffuse
//     out.omega_i = hemi_sample_uniform(hit.incident_normal, rstate);
//     out.bsdf = Spectrum::fromRGB(hit.color) / M_PIf;
//     out.prob_i = 1.0 / (2 * M_PIf);

//     return;
// }

// __device__ void evaluate_bsdf(bsdf_sample_t &out, const Vec3 &v_inv, const Vec3 &w,
//                               const intersection_t &hit, rand_state_t &rstate) {

//     out.omega_i = w;

//     // // bool isGlass = false;
//     // bool isGlass = hit.mat->transmission >= 0.9;
//     // if (isGlass) {
//     //     out.bsdf = Spectrum::Zero();
//     //     out.prob_i = 0;
//     // }

//     // bool isMirror = hit.mat->metallic >= 0.9;
//     // if (isMirror) {
//     //     // probability is zero since mirror does not allow direct light
//     //     out.bsdf = Spectrum::Zero();
//     //     out.prob_i = 0;
//     //     return;
//     // }

//     // diffuse
//     out.bsdf = Spectrum::fromRGB(hit.color) / M_PIf;
//     out.prob_i = 1.0 / (2 * M_PIf);
// }

__device__ Spectrum BRDF::eval(const Vec3 &wo, const Vec3 &wi) const {
    // diffuse
    float cos_theta_wi = fmaxf(wi.z, 0);
    return cos_theta_wi / M_PIf * Spectrum::fromRGB(base_color);
}

__device__ float BRDF::pdf(const Vec3 &wo, const Vec3 &wi) const {
    // diffuse
    return 1.0 / (2 * M_PIf);
}

__device__ brdf_sample_t BRDF::sample(const Vec3 &wo, rand_state_t &rstate) const {
    // // diffuse
    Vec3 wi = sphere_sample_uniform(rstate);
    wi.z = std::abs(wi.z);

    brdf_sample_t out = {
        .wi = wi,
        .f_cos_theta = eval(wo, wi),
        .pdf = pdf(wo, wi),
        .is_delta = false,
    };
    return out;
}

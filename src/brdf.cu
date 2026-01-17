#include "brdf.h"

__device__ Spectrum fresnel_schlick(Spectrum F0, Vec3 v, Vec3 h) {
    // https://www.youtube.com/watch?v=gya7x9H3mV0
    float x = (1 - v.dot(h));
    return F0 + (Spectrum::Identity() - F0) * x * x * x * x * x;
}

__device__ float normal_factor_ggx(Vec3 h, float roughness) {
    // https://www.youtube.com/watch?v=gya7x9H3mV0
    float n_dot_h = h.z;
    float alpha = roughness * roughness;
    float b = n_dot_h * n_dot_h * (alpha * alpha - 1.0f) + 1.0f;
    float D_ggx = alpha * alpha / (M_PIf * b * b);
    return D_ggx;
}

__device__ float geometry_g1_schlick_ggx(Vec3 v, float roughness) {
    // https://www.youtube.com/watch?v=gya7x9H3mV0
    float alpha = roughness * roughness;
    float k = 0.5 * alpha;
    float n_dot_v = v.z;
    float g1_schlick_ggx = n_dot_v / (n_dot_v * (1.0f - k) + k);
    return g1_schlick_ggx;
}

__device__ float geometry_schlick_ggx(Vec3 l, Vec3 v, float roughness) {
    float G_l = geometry_g1_schlick_ggx(l, roughness);
    float G_v = geometry_g1_schlick_ggx(v, roughness);
    return G_l * G_v;
}

__device__ Spectrum brdf_microfacet(const Vec3 &l, const Vec3 &v, float metallic,
                                    float roughness, const Spectrum &baseColor,
                                    float specular) {
    Vec3 h = (l + v).normalized();

    Spectrum F0_diel = (0.16 * specular * specular) * Spectrum::Identity();
    // metallic specularity is dependent on color, dielectric is neutral and max 16% of light
    Spectrum F0 = metallic * baseColor + (1 - metallic) * F0_diel;
    Spectrum F = fresnel_schlick(F0, v, h);
    float D = normal_factor_ggx(h, roughness);
    float G = geometry_schlick_ggx(l, v, roughness);

    float n_dot_v = fmaxf(v.z, 0.001f); // prevent div/0
    float n_dot_l = fmaxf(l.z, 0.001f); // prevent div/0

    Spectrum f_spec = (F * D * G) / (4.0f * n_dot_v * n_dot_l);

    float one_over_pi = M_1_PIf;
    Spectrum f_diff = baseColor * (Spectrum::Identity() - F) * (1.0 - metallic) * one_over_pi;

    Spectrum f = f_diff + f_spec;

    return f;
}

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
// //         out.bsdf = (1 - R) / cos_theta * Spectrum::Identity();
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
    float cos_theta_wi = fmaxf(wi.z, 0);

    // // diffuse
    // Spectrum f_diffuse_only = Spectrum::fromRGB(base_color) / M_PIf;
    // return cos_theta_wi * f_diffuse_only;

    Spectrum f_microfacet = brdf_microfacet(wi, wo, metallic, roughness, baseColor, specular);
    return f_microfacet * cos_theta_wi;
}

__device__ float BRDF::pdf(const Vec3 &wo, const Vec3 &wi) const {
    // uniform sampling
    return 1.0 / (2 * M_PIf);
}

__device__ brdf_sample_t BRDF::sample(const Vec3 &wo, rand_state_t &rstate) const {
    // uniform sampling
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

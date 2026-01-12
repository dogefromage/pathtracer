#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <iostream>

#include "bsdf.h"
#include "headers.h"
#include "lst.h"
#include "mathops.h"
#include "random.h"
#include "renderer.h"
#include "scene.h"

struct context_t {
    const Scene &scene;
    const BVH &bvh;
    const LST &lst;
    const config_t cfg;
    rand_state_t rstate;

    __device__ context_t(const Scene &_scene, const BVH &_bvh, const LST &_lst, config_t _cfg)
        : scene(_scene), bvh(_bvh), lst(_lst), cfg(_cfg) {
    }
};

static __device__ void get_camera_ray(Ray &ray, const Scene &scene, float u, float v) {
    const Vec3 &P = scene.camera.position;
    const Vec3 &T = scene.camera.target;
    const Vec3 &Up = scene.camera.updir;

    Vec3 W = T - P;
    Vec3 U = W.cross(Up);
    Vec3 V = U.cross(W);

    float focal_length = 0.1; // doesn't really matter at this point
    float yheight = atanf(0.5 * scene.camera.yfov) * focal_length;

    U = U.normalized() * yheight;
    V = V.normalized() * yheight;
    W = W.normalized() * focal_length;

    // U, V, W build orthogonal basis for camera ray direction
    // D = u*U + v*V + 1*W
    // build change of basis matrix A
    Mat3 A(U.x, V.x, W.x, U.y, V.y, W.y, U.z, V.z, W.z);

    // camera coordinate vector for pixel
    Vec3 x = {u, v, 1};

    ray.o = P;
    ray.r = (A * x).normalized();
}

// offset new ray slightly from triangle in normal dir
#define SAVE_RAY_EPS 1e-6

static __device__ void initialize_safe_ray(Ray &ray, const Vec3 &origin, const Vec3 &dir,
                                           const Vec3 &normal) {
    bool transmit = dir.dot(normal) < 0;

    ray.o = origin + SAVE_RAY_EPS * (transmit ? -normal : normal);
    ray.r = dir;
}

static __device__ void intersect(const BVH &bvh, const Scene &scene, const Ray &ray,
                                 intersection_t &hit) {
#ifdef USE_INTERSECT_CRUDE
    intersect_crude(scene, ray, hit);
#else
    bvh_intersect_iterative(bvh, scene, ray, hit);
#endif
}

static __device__ Vec3 sample_triangle_uniform(rand_state_t &rstate, const Vec3 &a,
                                               const Vec3 &b, const Vec3 &c) {
    float u1, u2;
    do {
        u1 = random_uniform(rstate);
        u2 = random_uniform(rstate);
    } while (u1 + u2 > 1);
    return a + u1 * (b - a) + u2 * (c - a);
}

struct area_light_sample_t {
    float p_als; // other fields are valid if p_als > 0
    Vec3 normal, dir_hit_to_light;
    float distance, area;
};
static __device__ void sample_area_light(area_light_sample_t &out, context_t &c,
                                         const Vec3 &shadow_pos, const Vec3 &shadow_normal,
                                         int face_index, bool direction_given) {
    const face_t &face = c.scene.faces[face_index];
    const material_t &mat = c.scene.materials[face.material];
    assert(face.vertexCount == 3);

    const Vec3 &A = c.scene.vertices[face.vertices[0]].position;
    const Vec3 &B = c.scene.vertices[face.vertices[1]].position;
    const Vec3 &C = c.scene.vertices[face.vertices[2]].position;

    Vec3 tri_cross = (C - B).cross(A - B);
    float tri_cross_length = tri_cross.magnitude();
    out.normal = tri_cross / tri_cross_length;
    out.area = 0.5 * tri_cross_length;

    if (!direction_given) {
        Vec3 light_pos = sample_triangle_uniform(c.rstate, A, B, C);
        out.dir_hit_to_light = (light_pos - shadow_pos).normalized();
    }

    Ray shadow_ray;
    initialize_safe_ray(shadow_ray, shadow_pos, out.dir_hit_to_light, shadow_normal);

    // check visibility
    intersection_t light_hit;
    intersect(c.bvh, c.scene, shadow_ray, light_hit);

    bool visible =
        light_hit.has_hit && light_hit.distance > 0 && light_hit.faceIndex == face_index;
    if (!visible) {
        out.p_als = 0;
        return;
    }

    // ????
    // // check if light points towards face
    // intersection_t pdf_hit;
    // intersect_face(c.scene, shadow_ray, pdf_hit, face_index);

    out.distance = light_hit.distance;
    float cos_theta_y = std::abs(out.normal.dot(out.dir_hit_to_light));
    out.p_als = out.distance * out.distance / (out.area * cos_theta_y);
}

struct light_source_sample_t {
    Spectrum incoming_radiance;
    Vec3 dir_hit_to_light;
    float p_lss;
};

// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md
// intensity parameter can be:
// luminous intensity (point light) or illuminance (area light) or luminance (light ray)
// color will be normalized and scalar will be multiplied with intensity
static __device__ Spectrum rgb_to_radiometric(Vec3 color, float intensity) {
    float color_magnitude = color.magnitude();
    Vec3 color_normalized = Vec3::Const(1);
    if (color_magnitude > 1e-12) {
        // for if color is black
        color_normalized = color / color_magnitude;
    }
    Spectrum spectrum = Spectrum::fromRGB(color_normalized);
    float photometric_intensity =
        intensity * color_magnitude; // for simplicity multiply with color norm

    // stems from relationship between
    float alpha = photometric_intensity / (683.0 * spectrum.luminance());
    Spectrum radiometric_spectrum = spectrum * alpha;
    // radiant intensity (point light) or irradiance (area light) or radiance (light ray)
    return radiometric_spectrum;
}

static __device__ void sample_light_source(light_source_sample_t &out, context_t &c,
                                           const Ray &ray, const intersection_t &hit) {
    if (c.lst.nodes.count == 0) {
        out.p_lss = 0;
        return;
    }

    // pick light node uniformly random
    int node_index = (int)(c.lst.nodes.count * random_uniform(c.rstate));
    const lst_node_t &node = c.lst.nodes[node_index % c.lst.nodes.count];

    out.p_lss = 1.0f / (float)c.lst.nodes.count;

    if (node.type == LST_SOURCE_LIGHT) {
        const light_t &light = c.scene.lights[node.index];

        if (light.type == LIGHT_POINT) {
            // TODO check maths to add radius
            Vec3 light_pos = light.position;
            // float light_rad = 0.1;
            // + light_rad * sphere_sample_uniform(c.rstate);

            Vec3 dir_hit_to_light = light_pos - hit.position;
            float distance = dir_hit_to_light.magnitude();
            out.dir_hit_to_light = dir_hit_to_light / distance;

            Ray shadow_ray;
            initialize_safe_ray(shadow_ray, hit.position, out.dir_hit_to_light,
                                hit.true_normal);

            intersection_t light_hit;
            intersect(c.bvh, c.scene, shadow_ray, light_hit);
            bool visible = distance < light_hit.distance;
            if (!visible) {
                out.p_lss = 0;
                return;
            }

            Spectrum radiant_intensity = rgb_to_radiometric(light.color, light.intensity);
            // for some reason, weird maths
            Spectrum radiance = radiant_intensity / (distance * distance);

            out.incoming_radiance = radiance;
            out.p_lss *= 1;
            return;

        } else {
            // directional light

            out.dir_hit_to_light = -light.direction;
            out.dir_hit_to_light.normalize();

            Ray shadow_ray;
            initialize_safe_ray(shadow_ray, hit.position, out.dir_hit_to_light,
                                hit.incident_normal);
            intersection_t shadow_hit;
            intersect(c.bvh, c.scene, shadow_ray, shadow_hit);
            if (shadow_hit.has_hit) {
                // not sun ray
                out.p_lss = 0;
                return;
            }

            Spectrum radiance = rgb_to_radiometric(light.color, light.intensity);
            out.incoming_radiance = radiance;
            out.p_lss *= 1;
            return;
        }

    } else {
        // emitting face:

        area_light_sample_t als;
        sample_area_light(als, c, hit.position, hit.true_normal, node.index, false);

        out.dir_hit_to_light = als.dir_hit_to_light;

        const face_t &face = c.scene.faces[node.index];
        const material_t &mat = c.scene.materials[face.material];

        Spectrum irradiance = rgb_to_radiometric(mat.emissive, 1);

        // cos(theta)/r^2 cancels with dA, therefore radiance == radiosity here
        out.incoming_radiance = irradiance;
        out.p_lss *= als.p_als;
        return;
    }
}

static __device__ float evaluate_direct_p(context_t &c, const Ray &ray,
                                          const intersection_t &hit) {

    uint32_t num_nodes = c.lst.nodes.count;
    if (num_nodes == 0) {
        return 0;
    }

    float p_total = 0;

    for (int i = 0; i < num_nodes; i++) {
        float p_node;

        const lst_node_t &node = c.lst.nodes[i];

        if (node.type == LST_SOURCE_LIGHT) {
            // at the moment, all light sources are dirac-delta, thus direct_p is 0 for non
            // light-sampled rays
            p_node = 0;

        } else {
            // face
            area_light_sample_t la;
            la.dir_hit_to_light = ray.r;
            sample_area_light(la, c, hit.position, hit.true_normal, node.index, true);
            p_node = la.p_als;
        }

        p_total += p_node;
    }

    p_total /= (float)num_nodes;
    return p_total;
}

#define RR_PROB_MAX 0.99

static __device__ Spectrum integrate_Li(context_t &c, Ray ray) {
    Spectrum light = Spectrum::Zero();
    Spectrum throughput = Spectrum::Itentity();

    for (int depth = 0;; depth++) {
        intersection_t hit;
        intersect(c.bvh, c.scene, ray, hit);
        if (!hit.has_hit) {
            Vec3 equi_uv = projectEquirectangular(ray.r);
            float4 clear_lookup =
                tex2D<float4>(c.scene.textures[c.scene.clearTexture], equi_uv.x, equi_uv.y);
            Vec3 clear_color = Vec3(clear_lookup.x, clear_lookup.y, clear_lookup.z);
            clear_color *= c.cfg.world_clear_color;
            light += throughput * Spectrum::fromRGB(clear_color);
            break;
        }

        // EMISSIVE LIGHT
        Spectrum Le = Spectrum::fromRGB(hit.mat->emissive) * throughput;
        light += Le;

        float rr_prob = fminf(throughput.luminance(), RR_PROB_MAX);
        if (random_uniform(c.rstate) >= rr_prob) {
            break; // ray dies
        }

        // DIRECT LIGHT
        light_source_sample_t lss;
        sample_light_source(lss, c, ray, hit);

        if (lss.p_lss > 0) {

            bsdf_sample_t lss_bsdf;
            evaluate_bsdf(lss_bsdf, ray.r, lss.dir_hit_to_light, hit, c.rstate);

            float cos_theta_x = std::abs(hit.true_normal.dot(lss.dir_hit_to_light));
            Spectrum direct_outgoing_radiance =
                lss_bsdf.bsdf * lss.incoming_radiance * cos_theta_x;

            // balance heuristic on part of direct light
            float weight = lss.p_lss / (lss.p_lss + lss_bsdf.prob_i);
            // printf("%.2f, %.2f\n", lss.p_direct, lss.light_dir_bsdf.prob_i);
            light += (weight / lss.p_lss) * throughput * direct_outgoing_radiance;
        }

        // INDIRECT LIGHT
        bsdf_sample_t indirect_bsdf;
        sample_bsdf(indirect_bsdf, ray.r, hit, c.rstate);
        // if (!(indirect_bsdf.prob_i > 0)) {
        //     printf("indirect_bsdf.prob_i = %.3f\n", indirect_bsdf.prob_i);
        //     // indirect_bsdf.prob_i = 0;
        // }
        assert(indirect_bsdf.prob_i > 0 && "sample with 0 probability");

        // set next ray
        initialize_safe_ray(ray, hit.position, indirect_bsdf.omega_i, hit.true_normal);
        float p_direct = evaluate_direct_p(c, ray, hit);
        float weight = indirect_bsdf.prob_i / (p_direct + indirect_bsdf.prob_i);

        float cos_theta = std::abs(hit.true_normal.dot(indirect_bsdf.omega_i));
        // find next throughput
        throughput *=
            indirect_bsdf.bsdf * (weight * cos_theta / (indirect_bsdf.prob_i * rr_prob));
    }

    return light;
}

__global__ void render_kernel(Vec3 *img, const BVH bvh, const Scene scene, const LST lst,
                              config_t cfg, int previous_samples, int current_samples) {
    int pixel_x = threadIdx.x + blockDim.x * blockIdx.x;
    int pixel_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (pixel_x >= cfg.resolution_x || pixel_y >= cfg.resolution_y) {
        return; // out of image
    }

    uint64_t curand_tid = pixel_y * cfg.resolution_x + pixel_x;

    context_t c(scene, bvh, lst, cfg);

    random_init(c.rstate, cfg.seed, curand_tid);

    Spectrum total_light = Spectrum::Zero();

    for (int i = 0; i < current_samples; i++) {
        float sensor_variance = 0.33;
        float sensor_x = (float)pixel_x + sensor_variance * random_normal(c.rstate);
        float sensor_y = (float)pixel_y + sensor_variance * random_normal(c.rstate);

        float u = (2 * sensor_x - cfg.resolution_x) / (float)cfg.resolution_y;
        float v = (2 * sensor_y - cfg.resolution_y) / (float)cfg.resolution_y;

        Ray camera_ray;
        get_camera_ray(camera_ray, scene, u, v);

        Spectrum current_light = integrate_Li(c, camera_ray);
        total_light += current_light;
    }

    total_light *= cfg.output_exposure;

    total_light /= (float)current_samples;
    Vec3 pixel_color = total_light.toRGB();

    int total_samples = previous_samples + current_samples;
    Vec3 last_pixel = img[pixel_y * cfg.resolution_x + pixel_x];

    Vec3 next_pixel = last_pixel * (previous_samples / (float)total_samples) +
                      pixel_color * (current_samples / (float)total_samples);

    img[pixel_y * cfg.resolution_x + pixel_x] = next_pixel;
}

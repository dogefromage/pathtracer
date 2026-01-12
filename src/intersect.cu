#include <assert.h>

#include "intersect.h"

static __device__ Vec3 barycentric_lincom(const Vec3 &A, const Vec3 &B, const Vec3 &C, float t, float u, float v) {
    return t * A + u * B + v * C;
}

#define TRIANGLE_DETERMINANT_EPS 1e-12
#define TRIANGLE_MARGIN_EPS 1e-12
static __device__ int moeller_trumbore_intersect(const Ray &ray, const Vec3 &vert0, const Vec3 &vert1, const Vec3 &vert2,
                                                 float *t, float *u, float *v) {
    Vec3 edge1, edge2, tvec, pvec, qvec;
    float det, inv_det;

    // Find vectors for two edges sharing vert0
    edge1 = vert1 - vert0;
    edge2 = vert2 - vert0;

    // Begin calculating determinant - also used to calculate U parameter
    pvec = ray.r.cross(edge2);
    // if determinant is near zero, ray lies in plane of triangle
    det = edge1.dot(pvec);

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
    tvec = ray.o - vert0;

    // Calculate U parameter and test bounds
    *u = tvec.dot(pvec) * inv_det;
    if (*u < -TRIANGLE_MARGIN_EPS || *u > (1.0 + TRIANGLE_MARGIN_EPS))
        return 0;

    // Prepare to test V parameter
    qvec = tvec.cross(edge1);

    // Calculate V parameter and test bounds
    *v = ray.r.dot(qvec) * inv_det;
    if (*v < -TRIANGLE_MARGIN_EPS || *u + *v > (1.0 + TRIANGLE_MARGIN_EPS)) {
        return 0;
    }

    // Calculate t, ray intersects triangle
    *t = edge2.dot(qvec) * inv_det;
#endif
    return 1;
}

__device__ float linearize_from_srgb_scalar(float c) {
    // Standard sRGB -> linear conversion
    if (c <= 0.04045f)
        return c / 12.92f;
    else
        return powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ static Vec3 linearize_from_srgb(const Vec3 srgb) {
    return Vec3(linearize_from_srgb_scalar(srgb.x), linearize_from_srgb_scalar(srgb.y), linearize_from_srgb_scalar(srgb.z));
}

__device__ void intersect_face(const Scene &scene, const Ray &ray, intersection_t &hit, int faceIndex) {
    const face_t &face = scene.faces[faceIndex];

    // loop over n-gon triangles fan-style
    for (size_t i = 2; i < face.vertexCount; i++) {
        const uint32_t &b = face.vertices[i - 1];
        const uint32_t &a = face.vertices[0];
        const uint32_t &c = face.vertices[i];
        const vertex_t &A = scene.vertices[a];
        const vertex_t &B = scene.vertices[b];
        const vertex_t &C = scene.vertices[c];

        float t, u, v;
        int has_hit = moeller_trumbore_intersect(ray, A.position, B.position, C.position, &t, &u, &v);

        if (has_hit && t >= 0 && t < hit.distance) {
            hit.faceIndex = faceIndex;
            hit.has_hit = true;
            hit.distance = t;
            assert(face.material < (int)scene.materials.count);

            hit.mat = &scene.materials[face.material];

            float t = 1.0 - u - v;
            hit.position = barycentric_lincom(A.position, B.position, C.position, t, u, v);
            hit.texcoord0 = barycentric_lincom(A.texcoord0, B.texcoord0, C.texcoord0, t, u, v);

            // get color based on https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
            Vec3 color = Vec3::Const(1);
            float alpha = 1;
            if (hit.mat->textureColor >= 0) {
                float4 texLookup = tex2D<float4>(scene.textures[hit.mat->textureColor], hit.texcoord0.x, hit.texcoord0.y);
                Vec3 color_srgb = Vec3(texLookup.x, texLookup.y, texLookup.z);
                color = linearize_from_srgb(color_srgb);
                alpha = texLookup.w;
            }
            color *= hit.mat->baseColorFactor.xyz();
            alpha *= hit.mat->baseColorFactor.w;

            hit.color = color;

            // TODO interpret alpha for some kind of transparency
            // switch (hit.mat->alphaMode) {
            // case ALPHA_OPAQUE:
            //     hit.color = texColor;
            //     break;
            // case ALPHA_MASK:
            //     if (texLookup.w > hit.mat->alphaCutoff) {
            //         hit.color = texColor;
            //     }
            // case ALPHA_BLEND:
            //     hit.color += lookup.w * (lookup_color - hit.color);
            // }

            switch (face.shading) {
            case FLAT_SHADING:
                hit.true_normal = face.faceNormal;
                break;
            case BARY_SHADING:
                hit.true_normal =
                    barycentric_lincom(scene.vertices[a].normal, scene.vertices[b].normal, scene.vertices[c].normal, t, u, v);
                hit.true_normal.normalize();
                break;
            }

            if (hit.true_normal.dot(ray.r) > 0) {
                // hit backside of face, mirror normal
                hit.incident_normal = -hit.true_normal;
            } else {
                hit.incident_normal = hit.true_normal;
            }

            CHECK_VEC(hit.true_normal);
            CHECK_VEC(hit.incident_normal);
            assert(hit.true_normal != Vec3::Zero());
            assert(hit.incident_normal != Vec3::Zero());
        }
    }
}

__device__ void intersect_crude(const Scene &scene, const Ray &ray, intersection_t &hit) {
    for (size_t i = 0; i < scene.faces.count; i++) {
        intersect_face(scene, ray, hit, i);
    }
}

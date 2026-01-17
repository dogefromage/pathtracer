#include <assert.h>

#include "intersect.h"

static __device__ Vec3 barycentric_lincom(const Vec3 &A, const Vec3 &B, const Vec3 &C, float t,
                                          float u, float v) {
    return t * A + u * B + v * C;
}

#define TRIANGLE_DETERMINANT_EPS 1e-12
#define TRIANGLE_MARGIN_EPS 1e-12
static __device__ int moeller_trumbore_intersect(const Ray &ray, const Vec3 &vert0,
                                                 const Vec3 &vert1, const Vec3 &vert2, float *t,
                                                 float *u, float *v) {
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

__device__ void intersect_face(const Scene &scene, const Ray &ray, intersection_t &hit,
                               int faceIndex) {
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
        int has_hit =
            moeller_trumbore_intersect(ray, A.position, B.position, C.position, &t, &u, &v);

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
            Vec3 baseColor = hit.mat->baseColorFactor.xyz();

            hit.alpha = hit.mat->baseColorFactor.w;
            if (hit.mat->baseColorTexture >= 0) {
                const texture_t &tex = scene.textures[hit.mat->baseColorTexture];
                Vec4 color_lookup = sample_texture(tex, hit.texcoord0.x, hit.texcoord0.y, true);
                baseColor *= color_lookup.xyz();
                hit.alpha *= color_lookup.w;
            }

            hit.brdf.baseColor = Spectrum::fromRGB(baseColor);
            hit.brdf.specular = hit.mat->specular;
            hit.brdf.metallic = hit.mat->metallicFactor;
            hit.brdf.roughness = hit.mat->roughnessFactor;

            if (hit.mat->metallicRoughnessTexture >= 0) {
                const texture_t &tex = scene.textures[hit.mat->metallicRoughnessTexture];
                Vec4 mr = sample_texture(tex, hit.texcoord0.x, hit.texcoord0.y, false);
                hit.brdf.roughness *= mr.y; // green channel is roughness
                hit.brdf.metallic *= mr.z;  // blue channel is metallic
            }

            switch (hit.mat->alphaMode) {
            case ALPHA_OPAQUE:
                hit.alpha = 1.0;
                break;
            case ALPHA_MASK:
                hit.alpha = hit.alpha > hit.mat->alphaCutoff ? 1.0 : 0.0;
                break;
            case ALPHA_BLEND:
                break; // keep
            }

            Vec3 normal, tangent;
            float tangent_handedness = 1;

            switch (face.shading) {
            case FLAT_SHADING:
                normal = face.normal;
                tangent = face.tangent.xyz();
                tangent_handedness = face.tangent.w;
                break;
            case BARY_SHADING:
                normal = barycentric_lincom(A.normal, B.normal, C.normal, t, u, v);
                tangent = barycentric_lincom(A.tangent.xyz(), B.tangent.xyz(), C.tangent.xyz(),
                                             t, u, v);
                tangent_handedness = A.tangent.w;
                // do not normalize basis now. mikktspace says its more accurate to do so later
                break;
            }

            if (tangent_handedness < 0) {
                // ensure is in { -1, 1 }
                tangent_handedness = -1;
            } else {
                tangent_handedness = 1;
            }

            hit.true_normal = normal; // keeps pointing in correct dir even if backface hit

            // left handed if tangent_handedness == -1
            Vec3 bitangent = tangent_handedness * normal.cross(tangent);
            tangent.normalize();
            bitangent.normalize();
            normal.normalize();

            if (normal.dot(ray.r) > 0) {
                // hit backside of face, flip everything that must be flipped
                tangent *= -1;
                bitangent *= -1;
                normal *= -1;
            }

            hit.incident_normal = normal;
            hit.tangentBasis = mat3FromColumns(tangent, bitangent, normal);

            if (hit.mat->normalTexture >= 0) {
                // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html - 3.9.3.
                const texture_t &tex = scene.textures[hit.mat->normalTexture];
                Vec4 coords = sample_texture(tex, hit.texcoord0.x, hit.texcoord0.y, false);
                coords = 2 * coords - Vec4::Const(1);
                hit.shaded_normal = hit.tangentBasis * coords.xyz();
                hit.shaded_normal.normalize();
            } else {
                hit.shaded_normal = normal;
            }

            CHECK_VEC(tangent);
            CHECK_VEC(bitangent);
            CHECK_VEC(hit.true_normal);
            CHECK_VEC(hit.incident_normal);
            CHECK_VEC(hit.shaded_normal);
            assert(tangent != Vec3::Zero());
            assert(bitangent != Vec3::Zero());
            assert(hit.true_normal != Vec3::Zero());
            assert(hit.incident_normal != Vec3::Zero());
            assert(hit.shaded_normal != Vec3::Zero());
        }
    }
}

__device__ void intersect_crude(const Scene &scene, const Ray &ray, intersection_t &hit) {
    for (size_t i = 0; i < scene.faces.count; i++) {
        intersect_face(scene, ray, hit, i);
    }
}

#include "Renderer.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>

#include <stdio.h>

inline __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __device__ float4 operator*(float4 s, float4 a)
{
    return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}

inline __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/******************************************
*******************************************
*******************************************/

// Ignore the 4th dimension here!
// float4 is processed faster than float3

inline __device__ float dot3(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float4 normalize3(float4 v)
{
    float invLen = __frsqrt_rn(dot3(v, v));
    return v * invLen;
}

/******************************************
*******************************************
*******************************************/

inline __device__ float interpolate(float a, float b, float coeff)
{
    return a + (b - a) * coeff;
}

// TODO: Check if the "if"s can be removed and does it effect performance!
inline __device__ float intersect_with_ray(float4 rayOrigin, float4 rayDirection,
    float4 center, float radiusSquare) {
    // p is ray origin
    // c is sphere origin
    // d is projection point of vec(pc) on ray.
    // cd is perpendicular to pd
    // theta is the angle between vec(pc) and vec(pd)
    float4 pc = center - rayOrigin;

    // vec(pc) . vec(raydir) = || pc || . || raydir || . cos(theta)
    // || raydir || = 1
    // vec(pc) . vec(raydir) = || pc || . cos(theta)
    //if it's negative it means theta > 90. So sphere is behind.
    float pdDistance = dot3(pc, rayDirection);

    if (pdDistance < 0.0f) {
        // Sphere is behind
        // TODO: Check for if the camera is inside of the sphere!
        return -1.0f;
    }

    // || cd || ^ 2 = || pc || ^ 2 - || pd || ^ 2
    float cdDistanceSquare = dot3(pc, pc) - pdDistance * pdDistance;
    if (cdDistanceSquare > radiusSquare) {
        // Ray passes from too far!
        return -1.0f;
    }

    // T0 & T1 is intersection point.
    // || td || ^ 2 + || cd || ^ 2 = || R || ^ 2
    float td = sqrtf(radiusSquare - cdDistanceSquare);

    // || pt0 || = || pd || - || td ||
    // || pt1 || = || pd || + || td ||
    return (pdDistance - td) < 0.0f ? pdDistance + td : pdDistance - td;
}

#define CUDART_INF_F __int_as_float(0x7f800000)

inline __device__ float4 Trace(Sphere* spheres, int numberOfSpheres,
    float4 rayOrigin, float4 rayDirection, float4 bgColor, int currentDepth, int maxDepth)
{
        float tnear = CUDART_INF_F;
        int sphereIndex = -1;

        // TODO: Check dynamic parallelism! GT 755M does not support it :(
        for (int i = 0; i < numberOfSpheres; i++) {
            // find intersection of this ray with the sphere in the scene
            float t = intersect_with_ray(rayOrigin, rayDirection,
                spheres[i].center, spheres[i].radius * spheres[i].radius);

            if (t >= 0.0f && t < tnear) {
                tnear = t;
                sphereIndex = i;
            }
        }

        if( sphereIndex == -1) return make_float4(bgColor.x, bgColor.y, bgColor.z, 1.0f);

        float4 pixelColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 hitPoint = rayOrigin + tnear * rayDirection;
        float4 hitNormal = normalize3(hitPoint - spheres[sphereIndex].center);

        float bias = 0.0001f;
        bool inside = false;

        // if normal and the ray does not face to each other,
        // then the intersection is inside of the sphere!

        float cosRayNormal = dot3(rayDirection, hitNormal);
        if ( cosRayNormal > 0.0f) {
            hitNormal = -1.0f * hitNormal;
            cosRayNormal = dot3(rayDirection, hitNormal);
            inside = true;
        }

        float transparency = spheres[sphereIndex].material.transparency;
        float reflection = spheres[sphereIndex].material.reflection;

        if ((transparency > 0.0f || reflection > 0.0f) && currentDepth < maxDepth) {
            float facingRatio = -cosRayNormal;
            // TODO: Learn this!
            float fresnelEffect = interpolate(powf(1.0f - facingRatio, 3), 1.0f, 0.1f);

            // Trace the reflection;
            // TODO: Learn this!
            float4 reflectionDirection = rayDirection - 2.0f * cosRayNormal * hitNormal;
            reflectionDirection = normalize3(reflectionDirection);

            // TODO: Remove recursion!
            float4 reflectionColor = Trace(spheres, numberOfSpheres, hitPoint + hitNormal * bias,
                reflectionDirection, bgColor,  currentDepth + 1, maxDepth);

            float4 refractionColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (transparency > 0.0f) {
                float indexOfRefraction = 1.1f;
                float eta = (inside) ? indexOfRefraction : 1.0f / indexOfRefraction;
                float cosi = -cosRayNormal;
                float k = 1 - eta * eta * (1.0f - cosRayNormal * cosRayNormal);

                // Trace the refraction
                float4 refractionDirection = rayDirection * eta + hitNormal * (eta *  cosi - sqrtf(k));
                refractionDirection = normalize3(refractionDirection);
                refractionColor = Trace(spheres, numberOfSpheres, hitPoint - hitNormal * bias,
                    refractionDirection, bgColor, currentDepth + 1, maxDepth);
            }

            // TODO: Learn this!
            float4 tracedColor = reflectionColor * fresnelEffect +
                refractionColor * transparency * (1.0f - fresnelEffect);

            pixelColor = tracedColor * spheres[sphereIndex].material.color;
        }
        else {
            // Here is either the max depth or diffuse object
            // No need to ray trace!

            float transmission = 1.0f;

            // Hack now! the last element is light
            float4 lightDirection = spheres[numberOfSpheres - 1].center- hitPoint;
            float4 lightColor = spheres[numberOfSpheres - 1].material.color;

            lightDirection = normalize3(lightDirection);

            // If there is an object in the middle, there is no lighting!
            for (int i = 0; i < numberOfSpheres; ++i) {
                float dist = intersect_with_ray(hitPoint + hitNormal * bias, lightDirection,
                    spheres[i].center, spheres[i].radius);

                if (dist >= 0.0f) {
                    transmission -= spheres[i].material.transparency;
                    if (transmission < 0.0f)
                        break;
                }
            }

            if (transmission < 0.0f) transmission = 0.0f;

            pixelColor = pixelColor + spheres[sphereIndex].material.color * transmission *
                fmaxf(0.0f, dot3(hitNormal, lightDirection)) * lightColor;
        }

        pixelColor.x = fminf(1.0f, pixelColor.x);
        pixelColor.y = fminf(1.0f, pixelColor.y);
        pixelColor.z = fminf(1.0f, pixelColor.z);
        pixelColor.w = fminf(1.0f, pixelColor.w);

        return pixelColor;
}

__global__ void RenderKernel(uchar4* renderTarget,
    Camera* camera, Viewport* viewport, Sphere* spheres, int numberOfSpheres, int maxDepth) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const float degreeToRadian = 3.14159265359 / 180.0f;
    float halfFov = __tanf(0.5f * camera->fov * degreeToRadian);

    float xDirection = (2.0f * (x + 0.5f) * (1.0f / viewport->width) - 1.0f) * halfFov * viewport->aspectRatio;
    float yDirection = (1.0f - 2.0f * (y + 0.5f) * (1.0f / viewport->height)) * halfFov;

    float4 rayDirection = make_float4(xDirection, yDirection, -1.0f, 1.0f);
    rayDirection = normalize3(rayDirection);

    float4 rayOrigin = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 color = Trace(spheres, numberOfSpheres, rayOrigin, rayDirection, camera->bgColor, 0, maxDepth);

    //Convert RGBA to BRGA
    renderTarget[y * viewport->width + x].x = color.z * 255;
    renderTarget[y * viewport->width + x].y = color.y * 255;
    renderTarget[y * viewport->width + x].z = color.x * 255;
    renderTarget[y * viewport->width + x].w = 255;
}

void CallRenderKernel(RenderCallPayload* renderCallPayload) {
    RenderKernel<<<renderCallPayload->numberOfBlocks, renderCallPayload->threadsPerBlock>>>(
        renderCallPayload->renderTarget,
        renderCallPayload->camera,
        renderCallPayload->viewport,
        renderCallPayload->spheres,
        renderCallPayload->numberOfSpheres,
        renderCallPayload->maxDepth);
}

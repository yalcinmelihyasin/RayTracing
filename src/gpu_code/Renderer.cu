#include "Renderer.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_functions.h>

#include <stdio.h>

inline __device__ float4 operator*(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __device__ float4 operator*(float s, float4 a) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __device__ float4 operator*(float4 s, float4 a) {
    return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}

inline __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float4& operator +=(float4& a, const float4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

inline __device__ float4& operator *=(float4& a, const float4& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

inline __device__ float4& operator *=(float4& a, float c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    a.w *= c;
    return a;
}


inline __device__ float4 saturate_vector(float4 v) {
    return make_float4(
        __saturatef(v.x),
        __saturatef(v.y),
        __saturatef(v.z),
        __saturatef(v.w)
    );
}

/******************************************
*******************************************
*******************************************/

// Ignore the 4th dimension here!
// float4 is processed faster than float3

inline __device__ float dot3(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float length3(float4 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

inline __device__ float4 normalize3(float4 v) {
    float invLen = __frsqrt_rn(length3(v));
    return v * invLen;
}

/******************************************
*******************************************
*******************************************/

inline __device__ float interpolate(float a, float b, float coeff) {
    return a + (b - a) * coeff;
}

struct TraceInformation {
    float4 cont;
    float4 origin;
    float4 direction;
};

struct PixelInformation {
    float4 cont;
    float4 color;
    int depth;

    float4 currentRayOrigin;
    float4 currentRayDirection;

    bool isContributing;

    int numberOfNextRays;
    TraceInformation nextRays[2];
};

// TODO: Check if the "if"s can be removed and does it effect performance!
inline __device__ float intersect_with_ray(float4 rayOrigin, float4 rayDirection,
    float4 center, float radius) {
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

    float radiusSquare = radius * radius;

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

inline __device__ void FindRayIntersect(Sphere const* spheres, int numberOfSpheres,
    float4 rayOrigin, float4 rayDirection, float* tnear, int* sphereIndex) {
    // TODO: Check dynamic parallelism! GT 755M does not support it :(
    for (int i = 0; i < numberOfSpheres; i++) {
        // find intersection of this ray with the sphere in the scene
        float t = intersect_with_ray(rayOrigin, rayDirection,
            spheres[i].center, spheres[i].radius);

        if (t >= 0.0f && t < *tnear) {
            *tnear = t;
            *sphereIndex = i;
        }
    }
}

__device__ const float bias = 1e-1f;

inline __device__ float4 CalculateLightColor(Sphere const* spheres, int numberOfSpheres, int sphereIndex,
    PixelInformation* pixelInfo, float4 hitPoint, float4 hitNormal) {
    // Here is either the max depth or diffuse object
    // No need to ray trace!

    // Hack now! the last element is light
    float4 lightDirection = spheres[numberOfSpheres - 1].center - hitPoint;
    float4 lightColor = spheres[numberOfSpheres - 1].material.color;
    float transmission = 1.0f;
    float emission = spheres[numberOfSpheres - 1].material.emission;

    lightDirection = normalize3(lightDirection);

    // If there is an object in the middle, there is no lighting!
    for (int i = 0; i < numberOfSpheres - 1; ++i) {
        float dist = intersect_with_ray(hitPoint + hitNormal * bias, lightDirection,
            spheres[i].center, spheres[i].radius);

        if (dist >= 0.0f) {
            transmission -= 1.0f - spheres[i].material.transparency;
            lightColor *= spheres[i].material.color;
            if (transmission < 0.0f) {
                transmission = 0.0f;
                break;
            }
        }
    }

    return transmission * emission * fmaxf(0.0f, dot3(hitNormal, lightDirection)) * lightColor;
}

inline __device__ void PrepareTrace(Material const* sphereMaterial , PixelInformation* pixelInfo,
    float cosRayNormal, float4 hitPoint, float4 hitNormal, bool inside) {

    float fresnelEffect = interpolate(powf(1.0f + cosRayNormal, 3), 1.0f, 0.1f);

    const float treshhold = 1e-5f;

    if (sphereMaterial->reflection > 0.0f) {
        float4 reflectionDirection = pixelInfo->currentRayDirection - 2.0f * cosRayNormal * hitNormal;
        float4 contribution = pixelInfo->cont* sphereMaterial->color *
            fresnelEffect;

        if (length3(contribution) > treshhold) {
            // Trace the reflection;
            pixelInfo->nextRays[pixelInfo->numberOfNextRays].direction = normalize3(reflectionDirection);
            pixelInfo->nextRays[pixelInfo->numberOfNextRays].origin = hitPoint + hitNormal * bias;
            pixelInfo->nextRays[pixelInfo->numberOfNextRays].cont = contribution;
            pixelInfo->numberOfNextRays++;
            pixelInfo->isContributing = false;
        }
    }

    if (sphereMaterial->transparency > 0.0f) {
        float indexOfRefraction = 1.1f;
        float eta = (inside) ? indexOfRefraction : 1.0f / indexOfRefraction;
        float cosi = -cosRayNormal;
        float k = 1 - eta * eta * (1.0f - cosRayNormal * cosRayNormal);
        float4 refractionDirection = pixelInfo->currentRayDirection * eta + hitNormal * (eta *  cosi - sqrtf(k));

        float4 contribution = pixelInfo->cont * sphereMaterial->color *
            sphereMaterial->transparency * (1.0f - fresnelEffect);

        if (length3(contribution) > treshhold) {
            // Trace the refraction
            pixelInfo->nextRays[pixelInfo->numberOfNextRays].direction = normalize3(refractionDirection);
            pixelInfo->nextRays[pixelInfo->numberOfNextRays].origin = hitPoint - hitNormal * bias;
            pixelInfo->nextRays[pixelInfo->numberOfNextRays].cont = contribution;
            pixelInfo->numberOfNextRays++;
            pixelInfo->isContributing = false;
        }
    }
}

inline __device__ void Trace(Sphere const* spheres, int numberOfSpheres,
    PixelInformation* pixelInfo, float4 bgColor, int maxDepth) {
    pixelInfo->isContributing = true;
    int sphereIndex = -1;
    float tnear = CUDART_INF_F;

    FindRayIntersect(spheres, numberOfSpheres, pixelInfo->currentRayOrigin, pixelInfo->currentRayDirection,
        &tnear, &sphereIndex);

    if (sphereIndex == -1) {
        pixelInfo->color = pixelInfo->cont * bgColor;
        return;
    }

    bool inside = false;

    float4 hitPoint = pixelInfo->currentRayOrigin + tnear * pixelInfo->currentRayDirection;
    float4 hitNormal = normalize3(hitPoint - spheres[sphereIndex].center);

    // if normal and the ray does not face to each other,
    // then the intersection is inside of the sphere!
    float cosRayNormal = dot3(pixelInfo->currentRayDirection, hitNormal);
    if ( cosRayNormal > 0.0f) {
        hitNormal = -1.0f * hitNormal;
        cosRayNormal = dot3(pixelInfo->currentRayDirection, hitNormal);
        inside = true;
    }

    if (pixelInfo->depth < maxDepth) {
        PrepareTrace(&spheres[sphereIndex].material, pixelInfo, cosRayNormal, hitPoint, hitNormal, inside);
    }

    if (pixelInfo->isContributing) {
        pixelInfo->color = CalculateLightColor(spheres, numberOfSpheres, sphereIndex, pixelInfo, hitPoint, hitNormal);
    }

    pixelInfo->color *= pixelInfo->cont * spheres[sphereIndex].material.color;
    pixelInfo->color += spheres[sphereIndex].material.emission * spheres[sphereIndex].material.color;
}

inline __device__ void CalculateCameraRay(Camera const* camera, Viewport const* viewport, PixelInformation* pixelInfo) {
    pixelInfo->cont = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    pixelInfo->depth = 0;
    pixelInfo->isContributing = true;

    const float degreeToRadian = 3.14159265359 / 180.0f;
    float halfFov = __tanf(0.5f * camera->fov * degreeToRadian);

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float xDirection = (2.0f * (x + 0.5f) * (1.0f / viewport->width) - 1.0f) * halfFov * viewport->aspectRatio;
    float yDirection = (1.0f - 2.0f * (y + 0.5f) * (1.0f / viewport->height)) * halfFov;

    pixelInfo->numberOfNextRays = 0;

    // For now camera is at zero
    pixelInfo->currentRayOrigin = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    pixelInfo->currentRayDirection = make_float4(xDirection, yDirection, -1.0f, 1.0f);
    pixelInfo->currentRayDirection = normalize3(pixelInfo->currentRayDirection);
}

__global__ void RenderKernel(uchar4* renderTarget,
    Camera const* camera, Viewport const* viewport, Sphere const* spheres, int numberOfSpheres, int maxDepth) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int capacity = 20;
    PixelInformation pixelInfoQueue[capacity];
    int currentRayIndex = 0;
    int maxRayIndex = 1;
    CalculateCameraRay(camera, viewport, &pixelInfoQueue[currentRayIndex]);

    float4 finalColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    while (currentRayIndex < maxRayIndex) {
        Trace(spheres, numberOfSpheres, &pixelInfoQueue[currentRayIndex], camera->bgColor, maxDepth);
        if (pixelInfoQueue[currentRayIndex].isContributing) {
            finalColor += pixelInfoQueue[currentRayIndex].color;
        }

        for (int i = 0; i < pixelInfoQueue[currentRayIndex].numberOfNextRays; i++) {
            if (maxRayIndex >= capacity) {
                printf("Capacity full!\n");
                break;
            }

            pixelInfoQueue[maxRayIndex].depth = pixelInfoQueue[currentRayIndex].depth + 1;
            pixelInfoQueue[maxRayIndex].numberOfNextRays = 0;
            pixelInfoQueue[maxRayIndex].currentRayDirection = pixelInfoQueue[currentRayIndex].nextRays[i].direction;
            pixelInfoQueue[maxRayIndex].currentRayOrigin = pixelInfoQueue[currentRayIndex].nextRays[i].origin;
            pixelInfoQueue[maxRayIndex].cont = pixelInfoQueue[currentRayIndex].nextRays[i].cont;
            maxRayIndex++;
        }
        currentRayIndex++;
    }

    finalColor = saturate_vector(finalColor);

    //Convert RGBA to BRGA
    renderTarget[y * viewport->width + x].x = finalColor.z * 255;
    renderTarget[y * viewport->width + x].y = finalColor.y * 255;
    renderTarget[y * viewport->width + x].z = finalColor.x * 255;
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

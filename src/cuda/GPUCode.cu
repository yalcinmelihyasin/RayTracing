#include "GPUCode.cuh"

// TODO: Find out why it's required for cuda_gl_interop
#include <GL/glew.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <vector_functions.h>

#include <string.h>
#include <stdio.h>
#include <cassert>

struct GPUContext
{
    bool init;
    cudaGraphicsResource_t pixelBuffer;
};

inline __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 normalize(float3 v)
{
    float invLen = __frsqrt_rn(dot(v, v));
    return v * invLen;
}

// TODO: Check if the "if"s can be removed and does it effect performance!
inline __device__ float intersect_with_ray(float3 rayOrigin, float3 rayDirection,
    float3 center, float radiusSquare) {
    // p is ray origin
    // c is sphere origin
    // d is projection point of vec(pc) on ray.
    // cd is perpendicular to pd
    // theta is the angle between vec(pc) and vec(pd)
    float3 pc = center - rayOrigin;

    // vec(pc) . vec(raydir) = || pc || . || raydir || . cos(theta)
    // || raydir || = 1
    // vec(pc) . vec(raydir) = || pc || . cos(theta)
    //if it's negative it means theta > 90. So sphere is behind.
    float pdDistance = dot(pc, rayDirection);

    if (pdDistance < 0.0f) {
        // Sphere is behind
        // TODO: Check for if the camera is inside of the sphere!
        return -1.0f;
    }

    // || cd || ^ 2 = || pc || ^ 2 - || pd || ^ 2
    float cdDistanceSquare = dot(pc, pc) - pdDistance * pdDistance;
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

inline __device__ float4 Trace(SphereGPU* spheres, int numberOfSpheres,
    float3 rayOrigin, float3 rayDirection, float3 bgColor, int currentDepth)
{
        float tnear = CUDART_INF_F;
        int sphereIndex = -1;

        // TODO: Check dynamic parallelism! GT 755M does not support it :(
        for (int i = 0; i < numberOfSpheres; i++) {
            // find intersection of this ray with the sphere in the scene
            float3 center = *(float3*)spheres[i].center;

            float t = intersect_with_ray(rayOrigin, rayDirection,
                center, spheres[i].radius * spheres[i].radius);

            if (t >= 0.0f && t < tnear) {
                tnear = t;
                sphereIndex = i;
            }
        }

        if( sphereIndex == -1) return make_float4(bgColor.x, bgColor.y, bgColor.z, 1.0f);

        float4 pixelColor = make_float4(spheres[sphereIndex].color[0], spheres[sphereIndex].color[1],
            spheres[sphereIndex].color[2], 1.0f);
        return pixelColor;
}

__global__ void RenderKernel(uchar4* renderTarget,
    CameraGPU* camera, SphereGPU* spheres, int numberOfSpheres) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const float degreeToRadian = 3.14159265359 / 180.0f;
    float halfFov = __tanf(0.5f * camera->fov * degreeToRadian);

    float xDirection = (2.0f * (x + 0.5f) * (1.0f / camera->width) - 1.0f) * halfFov * camera->aspectRatio;
    float yDirection = (1.0f - 2.0f * (y + 0.5f) * (1.0f / camera->height)) * halfFov;

    float3 ray = make_float3(xDirection, yDirection, -1.0f);
    ray = normalize(ray);

    float3 bgColor = *(float3*)camera->bgColor;
    float4 color = Trace(spheres, numberOfSpheres, make_float3(0.0f, 0.0f, 0.0f), ray, bgColor, 0);

    //Convert RGBA to BRGA
    renderTarget[y * camera->width + x] = make_uchar4(color.z * 255, color.y * 255, color.x * 255, 255);
}


void CreateGPUContext(GPUContext** context) {
    *context = (GPUContext*) malloc(sizeof(GPUContext));
    memset(*context, 0, sizeof(GPUContext));
}

void FreeGPUContext(GPUContext* context) {
    free(context);
}

void InitGPURendering(GPUContext* context) {
    cudaSetDevice(0);
    context->init = true;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
}

void RegisterPixelBuffer(GPUContext* context, GLuint buffer) {
    cudaError_t error = cudaGraphicsGLRegisterBuffer(&context->pixelBuffer, buffer,
        cudaGraphicsMapFlagsWriteDiscard);
    assert(error == cudaSuccess);
}

void UnregisterPixelBuffer(GPUContext* context) {
    cudaGraphicsUnregisterResource(context->pixelBuffer);
}

void RenderOnGPU(GPUContext* context, SphereGPU const* spheres, size_t numberOfSpheres, CameraGPU* camera) {
    cudaError_t error = cudaSuccess;
    (void)error;

    SphereGPU* gpuSpheres = nullptr;
    error = cudaMalloc((void**)&gpuSpheres, sizeof(SphereGPU) * numberOfSpheres);
    error = cudaMemcpy(gpuSpheres, spheres, sizeof(SphereGPU) * numberOfSpheres, cudaMemcpyHostToDevice);

    CameraGPU* gpuCamera = nullptr;
    error = cudaMalloc((void**)&gpuCamera, sizeof(CameraGPU));
    error = cudaMemcpy(gpuCamera, camera, sizeof(CameraGPU), cudaMemcpyHostToDevice);

    uchar4* renderTarget = nullptr;
    size_t num_bytes = 0;
    error = cudaGraphicsMapResources(1, &context->pixelBuffer, 0);
    error = cudaGraphicsResourceGetMappedPointer((void**)&renderTarget, &num_bytes, context->pixelBuffer);

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks(camera->width / threadsPerBlock.x, camera->height / threadsPerBlock.y);
    RenderKernel<<<numBlocks, threadsPerBlock>>>(renderTarget, gpuCamera, gpuSpheres, numberOfSpheres);

    error = cudaGraphicsUnmapResources(1, &context->pixelBuffer, 0);
    error = cudaFree(gpuCamera);
    error = cudaFree(gpuSpheres);
}
#include "GPUCode.cuh"

#include <cpu/Vec3.h>
#include <cpu/Sphere.h>
#include <vector>

#include <cuda.h>

__global__ void RenderKernel(float3* pixels, unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 

    //if (x >= width || y >= height) return;

    float r = 1.0f;
    float g = 0.0f;
    float b = 0.0f;

    pixels[y * width + x] = make_float3(r, g, b);
}

__global__ void SetArray(float* a)
{
    a[threadIdx.x] = 1.0f;
}

void RenderOnGPU(std::vector<Sphere> spheres, int width, int height, float cameraPosition[3], int depth, float* pixels)
{
    //float degreeToRadian = M_PI / 180.0f;
    //float halfFov = tanf(0.5f * fov * degreeToRadian);

    //for (int y = 0; y < height; ++y) {
    //    for (int x = 0; x < width; ++x) {
    //        float xDirection = (2.0f * (x + 0.5f) * inverseWidth - 1.0f) * halfFov * aspectRatio;
    //        float yDirection = (1.0f - 2.0f * (y + 0.5f) * inverseHeight) * halfFov;
    //        Vec3f ray(xDirection, yDirection, -1.0f);
    //        ray.Normalize();
    //        frame[x + y* width] = Trace(Vec3f(0, 0, 0), ray, 0);
    //    }
    //}

    float3* cudaPixels;
    int cudaPixelSize = width * height * 3 * sizeof(float);
    cudaMalloc(&cudaPixels, cudaPixelSize);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    RenderKernel<<<numBlocks, threadsPerBlock>>>(cudaPixels, width, height);

    cudaMemcpy(pixels, cudaPixels, cudaPixelSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaPixels);
}

void InitGPURendering()
{
    cudaSetDevice(0);
}

void CopyImageToGPU(float* pixel, int width, int height)
{
    float3* 
}
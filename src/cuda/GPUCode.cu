#include "GPUCode.cuh"

#include <cpu/Vec3.h>
#include <cpu/Sphere.h>
#include <vector>

// TODO: Find out why it's required for cuda_gl_interop
#include <GL/glew.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cassert>

struct GPUContext
{
    bool init;
    cudaGraphicsResource_t pixelBuffer;
};

//BGRA texture format
__global__ void RenderKernel(uchar4* renderTarget, int width) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
    renderTarget[y * width + x] = make_uchar4(x, y, x + y, 255);
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

void CopyImageToGPU(GPUContext* context, uint8_t* pixels, int width, int height) {
    cudaError_t error = cudaGraphicsMapResources(1, &context->pixelBuffer, 0);
    assert(error == cudaSuccess);

    // Map buffer object
    uchar4* renderTarget = 0;
    size_t num_bytes;
    error = cudaGraphicsResourceGetMappedPointer((void**)&renderTarget, &num_bytes, context->pixelBuffer);
    assert(renderTarget);

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    RenderKernel<<<numBlocks, threadsPerBlock>>>(renderTarget, width);

    // Unmap buffer object
    error = cudaGraphicsUnmapResources(1, &context->pixelBuffer, 0);
    assert(error == cudaSuccess);
}

void RegisterPixelBuffer(GPUContext* context, GLuint buffer) {
    cudaError_t error = cudaGraphicsGLRegisterBuffer(&context->pixelBuffer, buffer,
        cudaGraphicsMapFlagsWriteDiscard);
    assert(error == cudaSuccess);
}

void UnregisterPixelBuffer(GPUContext* context) {
    cudaGraphicsUnregisterResource(context->pixelBuffer);
}
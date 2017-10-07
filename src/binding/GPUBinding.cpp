#include "GPUBinding.h"

#include <gpu_code/Renderer.cuh>

// TODO: Find out why it's required for cuda_gl_interop
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cassert>
#include <memory.h>

struct GPUContext
{
    cudaGraphicsResource_t pixelBuffer;
};

void CreateGPUContext(GPUContext** context) {
    *context = (GPUContext*)malloc(sizeof(GPUContext));
    memset(*context, 0, sizeof(GPUContext));
}

void DestroyGPUContext(GPUContext* context) {
    free(context);
}

void InitGPURendering(GPUContext* context) {
    cudaSetDevice(0);
}

void RegisterPixelBuffer(GPUContext* context, GLuint buffer) {
    cudaError_t error = cudaGraphicsGLRegisterBuffer(&context->pixelBuffer, buffer,
        cudaGraphicsMapFlagsWriteDiscard);
    assert(error == cudaSuccess);
}

void UnregisterPixelBuffer(GPUContext* context) {
    cudaGraphicsUnregisterResource(context->pixelBuffer);
}

void RenderOnGPU(GPUContext* context, Sphere const* spheres, int numberOfSpheres,
    Camera const* camera, Viewport* viewport, int maxDepth) {
    RenderCallPayload payload;

    if (!context || !spheres || !camera) return;

    cudaError_t error = cudaSuccess;
    (void)error;

    payload.numberOfSpheres = numberOfSpheres;
    payload.maxDepth = maxDepth;

    error = cudaMalloc((void**)&payload.spheres, sizeof(Sphere) * numberOfSpheres);
    error = cudaMemcpy(payload.spheres, spheres, sizeof(Sphere) * numberOfSpheres, cudaMemcpyHostToDevice);

    error = cudaMalloc((void**)&payload.camera, sizeof(Camera));
    error = cudaMemcpy(payload.camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);

    error = cudaMalloc((void**)&payload.viewport, sizeof(Viewport));
    error = cudaMemcpy(payload.viewport, viewport, sizeof(Viewport), cudaMemcpyHostToDevice);

    size_t num_bytes = 0;
    error = cudaGraphicsMapResources(1, &context->pixelBuffer, 0);
    error = cudaGraphicsResourceGetMappedPointer((void**)&payload.renderTarget, &num_bytes, context->pixelBuffer);

    payload.threadsPerBlock = dim3(32, 32, 1);
    payload.numberOfBlocks = dim3(viewport->width / payload.threadsPerBlock.x,
        viewport->height / payload.threadsPerBlock.y);

    CallRenderKernel(&payload);

    error = cudaGraphicsUnmapResources(1, &context->pixelBuffer, 0);
    error = cudaFree(payload.camera);
    error = cudaFree(payload.spheres);
    error = cudaFree(payload.viewport);
}
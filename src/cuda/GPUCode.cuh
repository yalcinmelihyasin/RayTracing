#pragma once

#include <stdint.h>

typedef struct GPUContext GPUContext;

#ifndef GLuint
typedef unsigned int GLuint;
#endif

struct CameraGPU {
    int width;
    int height;
    int maxDepth;

    float fov;
    float aspectRatio;

    float bgColor[3];

    float position[3];
    float rotation[3];
};

struct SphereGPU {
    float center[3];
    float radius;
    float color[3];
};

void CreateGPUContext(GPUContext** context);
void FreeGPUContext(GPUContext* context);
void InitGPURendering(GPUContext* context);

void RegisterPixelBuffer(GPUContext* context, GLuint bufferId);
void UnregisterPixelBuffer(GPUContext* context);

void RenderOnGPU(GPUContext* context, SphereGPU const* spheres, size_t numberOfSpheres, CameraGPU* camera);

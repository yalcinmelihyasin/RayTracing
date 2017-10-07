#pragma once

#include "GPUTypes.h"

typedef struct GPUContext GPUContext;

#ifndef GLuint
typedef unsigned int GLuint;
#endif

void CreateGPUContext(GPUContext** context);
void DestroyGPUContext(GPUContext* context);

void InitGPURendering(GPUContext* context);

void RegisterPixelBuffer(GPUContext* context, GLuint bufferId);
void UnregisterPixelBuffer(GPUContext* context);

void RenderOnGPU(GPUContext* context, Sphere const* spheres, int numberOfSpheres,
    Camera const* camera, Viewport* viewport, int maxDepth);

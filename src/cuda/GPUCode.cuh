#pragma once

#include "cpu/Sphere.h"
#include <stdint.h>

typedef struct GPUContext GPUContext;

#ifndef GLuint
typedef unsigned int GLuint;
#endif

struct SphereGPU {
    float center[3];
    float radius;
};

void CreateGPUContext(GPUContext** context);
void FreeGPUContext(GPUContext* context);

void InitGPURendering(GPUContext* context);
void CopyImageToGPU(GPUContext* context, uint8_t* pixels, int width, int height);
void RegisterPixelBuffer(GPUContext* context, GLuint bufferId);
void UnregisterPixelBuffer(GPUContext* context);

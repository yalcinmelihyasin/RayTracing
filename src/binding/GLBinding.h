#pragma once

#include "GPUBinding.h"

typedef struct GLContext GLContext;
typedef struct GLFWwindow GLFWwindow;

void CreateGLContext(GLContext** context);
void DestroyGLContext(GLContext* context);

void InitGLRendering(GLContext* glContext, GPUContext* gpuContext, int width, int height, GLFWwindow* window);
void TerminateGLRendering(GLContext* glContext, GPUContext* gpuContext);

void RenderGLContext(GLContext* context);

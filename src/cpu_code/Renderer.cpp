#include "Renderer.h"

#include <binding/GPUBinding.h>
#include <binding/GLBinding.h>

#include <vector>

#include <stdlib.h>
#include <memory.h>


struct Frame {
    std::vector<Sphere> spheres;
    int maxDepth;
};

struct Renderer {
    GPUContext* gpuContext;
    GLContext* glContext;

    Frame currentFrame;
    Viewport viewport;
    Camera camera;
};

void CreateRenderer(Renderer** renderer) {
    Renderer* newRenderer = (Renderer*)malloc(sizeof(Renderer));
    memset(newRenderer, 0, sizeof(Renderer));

    CreateGPUContext(&newRenderer->gpuContext);
    if (!newRenderer->gpuContext) return;

    CreateGLContext(&newRenderer->glContext);

    *renderer = newRenderer;
}

void DestroyRenderer(Renderer* renderer) {
    if (!renderer) return;
    DestroyGPUContext(renderer->gpuContext);
    DestroyGLContext(renderer->glContext);

    free(renderer);
}

void InitRenderer(Renderer* renderer, GLFWwindow* window, int width, int height) {
    renderer->viewport.width = width;
    renderer->viewport.height = height;
    renderer->viewport.aspectRatio = ((float)width) / height;

    InitGPURendering(renderer->gpuContext);
    InitGLRendering(renderer->glContext, renderer->gpuContext, width, height, window);
}

void TerminateRenderer(Renderer* renderer) {
    TerminateGLRendering(renderer->glContext, renderer->gpuContext);
}

inline static void Float3ArrayToFloat4(float4* out, float in[3], float w) {
    out->x = in[0];
    out->y = in[1];
    out->z = in[2];
    out->w = w;
}

void SetRendererCamera(Renderer* renderer, float position[3], float rotation[3], float fov, float backgroundColor[3]) {
    Camera* camera = &renderer->camera;
    Float3ArrayToFloat4(&camera->bgColor, backgroundColor, 1.0f);
    Float3ArrayToFloat4(&camera->position, position, 1.0f);
    Float3ArrayToFloat4(&camera->rotation, rotation, 1.0f);
    camera->fov = fov;
}

void AddSpheresToRenderer(Renderer* renderer, RTSphere* spheres, int numberOfSpheres) {
    for (int i = 0; i < numberOfSpheres; i++) {
        renderer->currentFrame.spheres.emplace_back(&spheres[i]);
    }
}

void ClearRendererFrame(Renderer* renderer) 
{
    renderer->currentFrame.spheres.clear();
    renderer->currentFrame.maxDepth = 0;
}

void RenderFrame(Renderer* renderer, int depth) {
    renderer->currentFrame.maxDepth = depth;
    RenderOnGPU(renderer->gpuContext, &renderer->currentFrame.spheres[0],
        (int)renderer->currentFrame.spheres.size(), &renderer->camera, &renderer->viewport, depth);

    RenderGLContext(renderer->glContext);
}

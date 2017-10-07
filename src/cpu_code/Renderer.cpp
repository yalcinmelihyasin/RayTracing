#include "Renderer.h"

#include <binding/GPUBinding.h>
#include <binding/GLBinding.h>

#include <vector>

#include <stdlib.h>
#include <memory.h>


static_assert(sizeof(Sphere) == sizeof(RTSphere), "Size of internal types should be equal to public types!");
static_assert(sizeof(Material) == sizeof(RTMaterial), "Size of internal types should be equal to public types!");

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

void SetRendererCamera(Renderer* renderer, float position[3], float rotation[3], float fov, float backgroundColor[3]) {
    Camera* camera = &renderer->camera;

    memcpy(&camera->bgColor, backgroundColor, sizeof(float) * 3);
    camera->bgColor.w = 1.0f;

    memcpy(&camera->position, position, sizeof(float) * 3);
    camera->position.w = 1.0f;

    memcpy(&camera->rotation, rotation, sizeof(float) * 3);
    camera->rotation.w = 1.0f;

    camera->fov = fov;
}

void AddSpheresToRenderer(Renderer* renderer, RTSphere* spheres, int numberOfSpheres) {
    Sphere* gpuSpheres = (Sphere*)spheres;

    for (int i = 0; i < numberOfSpheres; i++) {
        renderer->currentFrame.spheres.emplace_back(gpuSpheres[i]);
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

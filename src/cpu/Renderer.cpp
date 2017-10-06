#include "Renderer.h"

#include <cmath>
#include <algorithm>

#include <cuda/GPUCode.cuh>

#ifndef  M_PI
#define M_PI 3.14159265358979323846f
#endif // ! M_PI


static inline float interpolate(const float &a, const float &b, const float &coeff)
{
    return a + (b - a) * coeff;
}

Renderer::Renderer(int width, int height, float fov, int maxDepth, float bgColor[3]) {
    camera.width = width;
    camera.height = height;
    camera.maxDepth = maxDepth;

    camera.fov = fov;
    camera.aspectRatio = ((float)width) / height;

    memcpy(camera.bgColor, bgColor, sizeof(camera.bgColor));
    memset(camera.position, 0, sizeof(camera.position));
    memset(camera.rotation, 0, sizeof(camera.rotation));
}


void Renderer::AddSphere(SphereGPU* sphere) {
    spheres.push_back(*sphere);
}


void Renderer::Render(GPUContext* cudaContext) {
    RenderOnGPU(cudaContext, &spheres[0], spheres.size(), &camera);
}

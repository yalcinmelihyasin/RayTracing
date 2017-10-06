#pragma once

#include "cuda/GPUCode.cuh"

#include <vector>

typedef struct GPUContext GPUContext;

class Renderer
{
public:
    Renderer(int width, int height, float fov, int maxDepth, float bgColor[3]);

    void AddSphere(SphereGPU* sphere);
    void Render(GPUContext* cudaContext);
private:
    //Renderable object collection
    std::vector<SphereGPU> spheres;

    //camera variables
    CameraGPU camera;

    Renderer() = delete;
    Renderer(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer& operator=(Renderer&&) = delete;
};


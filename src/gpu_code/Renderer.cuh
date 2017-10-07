#pragma once

#include <binding/GPUTypes.h>

struct RenderCallPayload {
    dim3 numberOfBlocks;
    dim3 threadsPerBlock;

    uchar4* renderTarget;
    Viewport* viewport;
    Camera* camera;
    Sphere* spheres;
    int numberOfSpheres;
    int maxDepth;
};

void CallRenderKernel(RenderCallPayload* renderCallPayload);
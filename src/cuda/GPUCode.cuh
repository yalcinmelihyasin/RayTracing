#ifndef _MY_INCLUDE_
#define _MY_INCLUDE_

#include "cpu/Sphere.h"
#include <vector>

struct Sphere_GPU
{
    float center[3];
    float radius;
};

void InitGPURendering();
void RenderOnGPU(std::vector<Sphere> spheres, int width, int height, float cameraPosition[3], int depth, float* pixels);

#endif

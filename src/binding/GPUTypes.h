#pragma once

#include "cpu_code/CPUTypes.h"
#include <vector_types.h>

struct Viewport {
    int width;
    int height;
    float aspectRatio;
};

// TODO: turn this into a real camera model
struct Camera {
    float4 position;
    float4 rotation;
    float fov;
    float4 bgColor;
};

struct Material {
    float4 color;
    float reflection;
    float transparency;
    float emission;
};

struct Sphere {
    float4 center;
    float radius;
    Material material;

    Sphere(RTSphere* sphere);
};

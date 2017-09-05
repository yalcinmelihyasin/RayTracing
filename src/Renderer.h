#pragma once

#include "Sphere.h"
#include "Light.h"

#include <vector>

class Renderer
{
public:
    Renderer(int renderWidth, int renderHeight, float renderFOV, int renderMaxDepth,
        const Vec3f& renderBackgroundColor);
    ~Renderer();

    void AddSphere(const Sphere& sphere);
    void Render();
    const Vec3f* GetFrame();

    void SetLight(const Light& rendererLight) { light = rendererLight; }

    int GetWidth() { return width; }
    int GetHeight() { return height; }
private:
    //Renderable object collection
    std::vector<Sphere> spheres;
    Light light;

    //Output & output settings
    Vec3f* frame;
    Vec3f backgroundColor;

    //camera variables
    int width;
    int height;
    float inverseWidth;
    float inverseHeight;
    float fov;
    float aspectRatio;

    // Ray tracing settings
    int maxDepth;

    Vec3f Trace(const Vec3f& origin, const Vec3f direction, const int depth);

    Renderer() = delete;
    Renderer(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer& operator=(Renderer&&) = delete;
};


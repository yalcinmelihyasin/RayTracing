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

Renderer::Renderer(int renderWidth, int renderHeight, float renderFOV, int renderMaxDepth,
    const Vec3f& renderBackgroundColor)
    : width(renderWidth)
    , height(renderHeight)
    , fov(renderFOV)
    , maxDepth(renderMaxDepth)
    , backgroundColor(renderBackgroundColor) {

    inverseWidth = 1.0f / width;
    inverseHeight = 1.0f / height;
    aspectRatio = (float)width / (float)height;

    frame = new uint8_t[width * height * 4];
}

Renderer::~Renderer() {
    delete[] frame;
}

void Renderer::AddSphere(const Sphere& sphere) {
    spheres.push_back(sphere);
}

Vec3f Renderer::Trace(const Vec3f& rayOrigin, const Vec3f rayDirection, const int depth)
{
    float tnear = INFINITY;
    const Sphere* sphere = nullptr;

    // find intersection of this ray with the sphere in the scene
    for (size_t i = 0; i < spheres.size(); ++i) {
        float t = spheres[i].IntersectWithRay(rayOrigin, rayDirection);

        if (t >= 0.0f && t < tnear) {
            tnear = t;
            sphere = &spheres[i];
        }
    }

    //If there is no object, then it's the background!
    if (!sphere) return backgroundColor;

    Vec3f pixelColor = Vec3f(0.0f, 0.0f, 0.0f);

    Vec3f hitPoint = rayOrigin + tnear * rayDirection;
    Vec3f hitNormal = (hitPoint - sphere->GetCenter()).Normalize();

    float bias = 0.0001f; // TODO: Check why ?
    bool inside = false;

    // if normal and the ray does not face to each other,
    // then the intersection is inside of the sphere!
    if (rayDirection.dot(hitNormal) > 0.0f) {
        hitNormal = -hitNormal;
        inside = true;
    }

    const Material& sphereMaterial = sphere->GetMaterial();
    if ((sphereMaterial.GetTransparency() > 0.0f || sphereMaterial.GetReflection() > 0.0f) &&
        depth < maxDepth) {
        float cosRayNormalAngle = rayDirection.dot(hitNormal);
        float facingRatio = -cosRayNormalAngle;
        // TODO: Learn this!
        float fresnelEffect = interpolate(pow(1.0f - facingRatio, 3), 1.0f, 0.1f);

        // Trace the reflection;
        // TODO: Learn this!
        Vec3f reflectionDirection = rayDirection - 2.0f * cosRayNormalAngle * hitNormal;
        reflectionDirection.Normalize();
        Vec3f reflectionColor = Trace(hitPoint + hitNormal * bias, reflectionDirection, depth + 1);

        // If the sphere is transparent compute refraction
        // TODO: Learn this!
        Vec3f refractionColor(0.0f, 0.0f, 0.0f);
        if (sphereMaterial.GetTransparency() > 0.0f) {
            float indexOfRefraction = 1.1f;
            float eta = (inside) ? indexOfRefraction : 1.0f / indexOfRefraction;
            float cosi = -cosRayNormalAngle;
            float k = 1 - eta * eta * (1.0f - cosRayNormalAngle * cosRayNormalAngle);

            // Trace the refraction
            Vec3f refractionDirection = rayDirection * eta + hitNormal * (eta *  cosi - sqrt(k));
            refractionDirection.Normalize();
            refractionColor = Trace(hitPoint - hitNormal * bias, refractionDirection, depth + 1);
        }

        // TODO: Learn this!
        Vec3f tracedColor = reflectionColor * fresnelEffect +
            refractionColor * sphereMaterial.GetTransparency() * (1.0f - fresnelEffect);

        pixelColor = tracedColor * sphereMaterial.GetColor();
    }
    else {
        // Here is either the max depth or diffuse object
        // No need to ray trace!

        float transmission = 1.0f;
        Vec3f lightDirection = light.position- hitPoint;
        lightDirection.Normalize();

        // If there is an object in the middle, there is no lighting!
        for (size_t i = 0; i < spheres.size(); ++i) {
            if (spheres[i].IntersectWithRay(hitPoint + hitNormal * bias, lightDirection ) >= 0.0f) {
                transmission -= spheres[i].GetMaterial().GetTransparency();
                if (transmission < 0.0f)
                    break;
            }
        }

        if (transmission < 0.0f) transmission = 0.0f;

        pixelColor += sphereMaterial.GetColor() * transmission *
            std::max(0.0f, hitNormal.dot(lightDirection)) * light.color;
    }

    return pixelColor;
}

void Renderer::Render(GPUContext* cudaContext) {
    //float degreeToRadian = M_PI / 180.0f;
    //float halfFov = tanf(0.5f * fov * degreeToRadian);

    //for (int y = 0; y < height; ++y) {
    //    for (int x = 0; x < width; ++x) {
    //        float xDirection = (2.0f * (x + 0.5f) * inverseWidth - 1.0f) * halfFov * aspectRatio;
    //        float yDirection = (1.0f - 2.0f * (y + 0.5f) * inverseHeight) * halfFov;
    //        Vec3f ray(xDirection, yDirection, -1.0f);
    //        ray.Normalize();
    //        //Vec3f const& currentFrame = Trace(Vec3f(0, 0, 0), ray, 0);
    //        int index = (x + y * width) * 4;
    //        frame[index] = 255;
    //        frame[index + 1] = 0;
    //        frame[index + 2] = 0;
    //        frame[index + 3] = 0;
    //    }
    //}

    CopyImageToGPU(cudaContext, frame, width, height);
    //float camera[3] = {};
    //return RenderOnGPU(spheres, width, height, camera, 5, frame);
}

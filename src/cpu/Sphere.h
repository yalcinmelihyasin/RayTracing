#pragma once

#include "Material.h"

class Sphere
{
public:
    Sphere(const Vec3f& sphereCenter, float sphereRadius, const Material& sphereMaterial)
        : center(sphereCenter)
        , radius(sphereRadius)
        , material(sphereMaterial)
    {
        radiusSquare = radius * radius;
    }

    ~Sphere();

    ///Find the nearest intersection
    float IntersectWithRay(const Vec3f &rayOrigin, const Vec3f &rayDirection) const;

    const Vec3f& GetCenter() const { return center; }
    const Material& GetMaterial() const { return material; }
private:
    Vec3f center;
    float radius;
    float radiusSquare;
    Material material;
};


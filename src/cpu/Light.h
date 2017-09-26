#pragma once

#include "Vec3.h"

class Light
{
public:
    Vec3f position;
    Vec3f color;
    Light();
    Light(const Vec3f& lightPosition, const Vec3f& lightColor) : position(lightPosition), color(lightColor) {}
    ~Light();
};


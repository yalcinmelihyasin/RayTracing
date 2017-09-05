#pragma once

#include "Vec3.h"

class Material
{
public:
    Material(const Vec3f& materialColor, float materialReflection, float materialTransparency)
        : color(materialColor)
        , reflection(materialReflection)
        , transparency(materialTransparency)
    {}

    ~Material();

    float GetTransparency() const { return transparency; }
    float GetReflection() const { return reflection; }
    const Vec3f& GetColor() const { return color; }
private:
    Vec3f color;
    float reflection;
    float transparency;
};


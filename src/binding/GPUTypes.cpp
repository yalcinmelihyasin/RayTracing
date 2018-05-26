#include "GPUTypes.h"

inline static void Float4ArrayToFloat4(float4* out, float in[4]) {
    out->x = in[0];
    out->y = in[1];
    out->z = in[2];
    out->w = in[3];
}


inline static void CopyMaterial(Material* material, RTMaterial* rtMaterial) {
    Float4ArrayToFloat4(&material->color, rtMaterial->color);
    material->emission = rtMaterial->emission;
    material->reflection = rtMaterial->reflection;
    material->transparency = rtMaterial->transparency;
}

Sphere::Sphere(RTSphere* sphere) {
    this->radius = sphere->radius;
    Float4ArrayToFloat4(&this->center, sphere->position);
    CopyMaterial(&this->material, &sphere->material);

}
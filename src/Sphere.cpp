#include "Sphere.h"

#include <cmath>


float Sphere::IntersectWithRay(const Vec3f &rayOrigin, const Vec3f &rayDirection) const
{
    // p is ray origin
    // c is sphere origin
    // d is projection point of vec(pc) on ray.
    // cd is perpendicular to pd
    // theta is the angle between vec(pc) and vec(pd)
    Vec3f pc = center - rayOrigin;

    // vec(pc) . vec(raydir) = || pc || . || raydir || . cos(theta)
    // || raydir || = 1
    // vec(pc) . vec(raydir) = || pc || . cos(theta)
    //if it's negative it means theta > 90. So sphere is behind.
    float pdDistance = pc.dot(rayDirection);
    if (pdDistance < 0.0f) {
        // Sphere is behind
        // TODO: Check for if the camera is inside of the sphere!
        return -1.0f;
    }

    // || cd || ^ 2 = || pc || ^ 2 - || pd || ^ 2
    float cdDistanceSquare = pc.dot(pc) - pdDistance * pdDistance;
    if (cdDistanceSquare > radiusSquare) {
        // Ray passes from too far!
        return -1.0f;
    }

    // T0 & T1 is intersection point.
    // || td || ^ 2 + || cd || ^ 2 = || R || ^ 2
    float td = sqrtf(radiusSquare - cdDistanceSquare);

    // || pt0 || = || pd || - || td ||
    // || pt1 || = || pd || + || td ||
    return (pdDistance - td) < 0.0f ? pdDistance + td : pdDistance - td;
}

Sphere::~Sphere()
{
}

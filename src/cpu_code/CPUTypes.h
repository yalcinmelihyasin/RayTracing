#pragma once

struct RTMaterial {
    float color[4];
    float reflection;
    float transparency;
    float emission;
};

struct RTSphere {
    float position[4];
    float radius;
    RTMaterial material;
};
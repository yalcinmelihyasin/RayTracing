#pragma once

#include "CPUTypes.h"

typedef struct Renderer Renderer;
typedef struct GLFWwindow GLFWwindow;

void CreateRenderer(Renderer** renderer);
void DestroyRenderer(Renderer* renderer);

void InitRenderer(Renderer* renderer, GLFWwindow* window, int width, int height);
void TerminateRenderer(Renderer* renderer);

void SetRendererCamera(Renderer* renderer, float position[3], float rotation[3], float fov, float backgroundColor[3]);

void AddSpheresToRenderer(Renderer* renderer, RTSphere* spheres, int numberOfSpheres);

void ClearRendererFrame(Renderer* renderer);
void RenderFrame(Renderer* renderer, int depth);


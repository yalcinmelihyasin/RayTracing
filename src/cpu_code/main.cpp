#include "Renderer.h"

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <glfw3.h>

// Include GLM
#include <glm/glm.hpp>

// Include standard headers
#include <stdio.h>
#include <stdlib.h>

#include <Windows.h>

using namespace glm;

bool InitWindow(GLFWwindow** window, int width, int height) {
    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return false;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    GLFWwindow* newWindow  = glfwCreateWindow(width, height, "Ray Tracing", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window\n");
        getchar();
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(newWindow);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(newWindow, GLFW_STICKY_KEYS, GL_TRUE);

    // Disable VSYNC : Just to test do not burn the GPU :D
    glfwSwapInterval(0);

    *window = newWindow;
    return true;
}

void TerminateWindow(GLFWwindow* window) {
    glfwDestroyWindow(window);
    // Close OpenGL window and terminate GLFW
    glfwTerminate();
}

void AddPrimitives(Renderer* renderer) {

    static float z = 0.0f;
    static float movement = 0.1f;

    z += movement;
    if (z > 30.0f || z < -30.0f ) movement *= -1;

    RTSphere spheres[] = {
        { { 0.0f, -10004.0f, -20.0f, 1.0f },  10000.0f, { { 0.20f, 0.20f, 0.20f, 1.0f }, 0.0f, 1.0f, 0.0f } },
        { { 0.0f, 0.0f, -20.0f, 1.0f },       4.0f,     { { 1.00f, 0.32f, 0.36f, 1.0f }, 1.0f, 0.5f, 0.0f } },
        { { 5.0f, -1.0f, -15.0f, 1.0f },      2.0f,     { { 0.90f, 0.76f, 0.46f, 1.0f }, 1.0f, 0.0f, 0.0f } },
        { { 5.0f, 0.0f, -25.0f, 1.0f },       3.0f,     { { 0.65f, 0.77f, 0.97f, 1.0f }, 1.0f, 0.0f, 0.0f } },
        { { -5.5f, 0.0f, -15.0f, 1.0f },      3.0f,     { { 0.90f, 0.90f, 0.90f, 1.0f }, 1.0f, 0.0f, 0.0f } },
        { { 0.0f, 20.0f, z, 1.0f },           3.0f,     { { 3.3f,  3.4f,  3.5f,  1.0f }, 0.0f, 0.0f, 1.0f } }
    };

    AddSpheresToRenderer(renderer, spheres, 6);
}

int main( void ) {
    int width = 1024;
    int height = 768;

    GLFWwindow* window = nullptr;
    if (!InitWindow(&window, width, height)) return -1;

    Renderer* renderer;

    CreateRenderer(&renderer);
    InitRenderer(renderer, window, width, height);

    float cameraPosition[3] = { 0 };
    float cameraRotation[3] = { 0 };
    float bgColor[3] = { 1.0f, 1.0f, 1.0f };
    SetRendererCamera(renderer, cameraPosition, cameraRotation, 45, bgColor);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double pc_frequency = freq.QuadPart / 1000.0;

    char fps_string[10];
    int fps = 0;

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    double begin = counter.QuadPart / pc_frequency;

    do{
        ClearRendererFrame(renderer);
        AddPrimitives(renderer);
        RenderFrame(renderer, 1);

        QueryPerformanceCounter(&counter);
        double end = counter.QuadPart / pc_frequency;

        if (end - begin < 1000.0f) {
            ++fps;
        }
        else {
            memset(fps_string, 0, 10);
            sprintf(fps_string, "%d\n", fps);
            OutputDebugStringA(fps_string);
            fps = 0;
            begin = end;
        }
    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    TerminateRenderer(renderer);
    TerminateWindow(window);

    return 0;
}

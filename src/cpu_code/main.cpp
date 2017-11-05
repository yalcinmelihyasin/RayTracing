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

void AddPrimitives(Renderer* renderer, float deltaTime) {

    static float z = 0.0f;
    static float movement = 0.06f;
    static float distance = 60.0f;

    z += movement * deltaTime;
    if (z > distance || z < -distance ) movement *= -1;

    RTSphere spheres[] = {
        { { 0.0f, -10004.0f, -20.0f, 1.0f },  10000.0f, { { 0.20f, 0.20f, 0.20f, 1.0f }, 1.0f, 0.0f, 0.0f } },
        { { 0.0f, 0.0f, -20.0f, 1.0f },       4.0f,     { { 1.00f, 0.32f, 0.36f, 1.0f }, 1.0f, 1.0f, 0.0f } },
        { { 5.0f, -1.0f, -15.0f, 1.0f },      2.0f,     { { 0.90f, 0.76f, 0.46f, 1.0f }, 1.0f, 0.0f, 0.0f } },
        { { 5.0f, 0.0f, -25.0f, 1.0f },       3.0f,     { { 0.65f, 0.77f, 0.97f, 1.0f }, 1.0f, 0.0f, 0.0f } },
        { { -5.5f, 0.0f, -15.0f, 1.0f },      3.0f,     { { 0.00f, 1.00f, 0.00f, 1.0f }, 1.0f, 0.0f, 0.0f } },
        { { -20.0f, 20.0f, z, 1.0f },         3.0f,     { { 0.9f,  0.95f, 1.0f,  1.0f }, 0.0f, 0.0f, 3.3f } }
    };

    AddSpheresToRenderer(renderer, spheres, 6);
}

static struct {
    double pc_freq;
    double second_start;
    double previousTime;
    float deltaTime;
    LARGE_INTEGER counter;
    char fps_string[10];
    int fps = 0;
} timeProperties;

void HandleTime() {
    QueryPerformanceCounter(&timeProperties.counter);
    double end = timeProperties.counter.QuadPart / timeProperties.pc_freq;

    if (end - timeProperties.second_start < 1000.0f) {
        ++timeProperties.fps;
    }
    else {
        memset(timeProperties.fps_string, 0, 10);
        sprintf(timeProperties.fps_string, "%d\n", timeProperties.fps);
        OutputDebugStringA(timeProperties.fps_string);
        timeProperties.fps = 0;
        timeProperties.second_start = end;
    }

    timeProperties.deltaTime = end - timeProperties.previousTime;
    timeProperties.previousTime = end;
}

static struct {
    int previousIncreaseState;
    int previousDecreaseState;
} inputProperties;

int HandleInput(GLFWwindow* window, int depth) {
    int currentState = glfwGetKey(window, GLFW_KEY_KP_ADD);

    if (currentState == GLFW_PRESS && inputProperties.previousIncreaseState == GLFW_RELEASE) {
        depth = ((depth + 1) % 10 + 10) % 10;
        printf("Depth increased! %d\n", depth);
    }

    inputProperties.previousIncreaseState = currentState;
    currentState = glfwGetKey(window, GLFW_KEY_KP_SUBTRACT);

    if (currentState == GLFW_PRESS && inputProperties.previousDecreaseState == GLFW_RELEASE) {
        depth = ((depth - 1) % 10 + 10) % 10;
        printf("Depth decreased! %d\n", depth);
    }

    inputProperties.previousDecreaseState = currentState;
    return depth;
}

int main( void ) {
    int width = 1024;
    int height = 768;
    int depth = 0;

    GLFWwindow* window = nullptr;
    if (!InitWindow(&window, width, height)) return -1;

    Renderer* renderer;

    CreateRenderer(&renderer);
    InitRenderer(renderer, window, width, height);

    float cameraPosition[3] = { 0 };
    float cameraRotation[3] = { 0 };
    float bgColor[3] = { 0.53f, 0.81f, 0.98f };
    SetRendererCamera(renderer, cameraPosition, cameraRotation, 45, bgColor);

    QueryPerformanceFrequency(&timeProperties.counter);
    timeProperties.pc_freq = timeProperties.counter.QuadPart / 1000.0;

    QueryPerformanceCounter(&timeProperties.counter);
    timeProperties.second_start = timeProperties.counter.QuadPart / timeProperties.pc_freq;
    timeProperties.previousTime = timeProperties.second_start;

    inputProperties.previousDecreaseState = GLFW_RELEASE;
    inputProperties.previousIncreaseState = GLFW_RELEASE;

    do{
        ClearRendererFrame(renderer);
        AddPrimitives(renderer, timeProperties.deltaTime);
        RenderFrame(renderer, depth);

        HandleTime();
        depth = HandleInput(window, depth);
    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    TerminateRenderer(renderer);
    TerminateWindow(window);

    return 0;
}

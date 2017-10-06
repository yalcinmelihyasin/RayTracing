#include "Renderer.h"
#include "cuda/GPUCode.cuh"

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>

#include <vector>

// Include standard headers
#include <stdio.h>
#include <stdlib.h>

#include <Windows.h>

using namespace glm;

const char* vertexShader =
"#version 330 core\n"
"layout(location = 0) in vec3 vertexPosition;\n"
"layout(location = 1) in vec2 vertexUV;\n"

"out vec2 UV;\n"

"void main() {\n"
"    gl_Position = vec4(vertexPosition, 1);\n"
"    UV = vertexUV;\n"
"}\n";

const char* fragmentShader =
"#version 330 core\n"

"out vec3 color;\n"
"in vec2 UV;\n"
"uniform sampler2D myTextureSampler;\n"

"void main() {\n"
"    color = texture( myTextureSampler, UV ).rgb;\n"
"}\n";

static struct {
    GLuint texture;

    GLuint vertexArray;
    GLuint vertexBuffer;
    GLuint uvBuffer;

    GLuint shaderProgram;
    GLuint gpuBuffer;

    GPUContext* gpuContext;

    int width;
    int height;

}windowState = { 0 };

bool InitWindow() {
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

    windowState.width = 1024;
    windowState.height = 768;

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(windowState.width, windowState.height, "Ray Tracing", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window\n");
        getchar();
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Disable VSYNC : Just to test do not burn the GPU :D
    glfwSwapInterval(0);

    return true;
}

void TerminateWindow() {
    glfwDestroyWindow(window);
    // Close OpenGL window and terminate GLFW
    glfwTerminate();
}

GLuint LoadShaders(const char * vertexShaderCode, const char * fragmentShaderCode) {

    // Create the shaders
    GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

    // Compile Vertex Shader
    glShaderSource(VertexShaderID, 1, &vertexShaderCode, NULL);
    glCompileShader(VertexShaderID);

    GLint result;
    int infoLogLength;

    // Check Vertex Shader
    glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &result);
    glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0) {
        std::vector<char> VertexShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(VertexShaderID, infoLogLength, NULL, &VertexShaderErrorMessage[0]);
        printf("%s\n", &VertexShaderErrorMessage[0]);
    }

    // Compile Fragment Shader
    glShaderSource(FragmentShaderID, 1, &fragmentShaderCode, NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0) {
        std::vector<char> FragmentShaderErrorMessage(infoLogLength + 1);
        glGetShaderInfoLog(FragmentShaderID, infoLogLength, NULL, &FragmentShaderErrorMessage[0]);
        printf("%s\n", &FragmentShaderErrorMessage[0]);
    }

    // Link the program
    GLuint ProgramID = glCreateProgram();
    glAttachShader(ProgramID, VertexShaderID);
    glAttachShader(ProgramID, FragmentShaderID);
    glLinkProgram(ProgramID);

    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (infoLogLength > 0) {
        std::vector<char> ProgramErrorMessage(infoLogLength + 1);
        glGetProgramInfoLog(ProgramID, infoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }

    glDetachShader(ProgramID, VertexShaderID);
    glDetachShader(ProgramID, FragmentShaderID);

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}

void InitGLRenderer()
{
    // Create the quad spans the screen
    windowState.vertexArray = 0;
    glGenVertexArrays(1, &windowState.vertexArray);
    glBindVertexArray(windowState.vertexArray);

    GLfloat vertexBufferData[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,
    };

    // Bind quad to vertex buffer
    windowState.vertexBuffer = 0;
    glGenBuffers(1, &windowState.vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, windowState.vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexBufferData), vertexBufferData, GL_STATIC_DRAW);

    GLfloat uvBufferData[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,

        0.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 0.0f
    };

    glGenBuffers(1, &windowState.uvBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, windowState.uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvBufferData), uvBufferData, GL_STATIC_DRAW);

    size_t bufferSize = windowState.width * windowState.height * 4;

    glGenBuffers(1, &windowState.gpuBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, windowState.gpuBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, nullptr, GL_DYNAMIC_COPY);
    RegisterPixelBuffer(windowState.gpuContext, windowState.gpuBuffer);

    // Create opengl texture
    windowState.texture = 0;
    glGenTextures(1, &windowState.texture);
    glBindTexture(GL_TEXTURE_2D, windowState.texture);

    // Set image to texture
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, windowState.width, windowState.height, 0, GL_BGRA,
    GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    windowState.shaderProgram = LoadShaders(vertexShader, fragmentShader);
}

void TerminateGLRenderer() {
    UnregisterPixelBuffer(windowState.gpuContext);
    glDeleteTextures(1, &windowState.texture);
    glDeleteBuffers(1, &windowState.vertexBuffer);
    glDeleteBuffers(1, &windowState.uvBuffer);
    glDeleteBuffers(1, &windowState.gpuBuffer);

    glDeleteVertexArrays(1, &windowState.vertexArray);
    glDeleteProgram(windowState.shaderProgram);
}

void RenderToTexture(Renderer& renderer) {
    renderer.Render(windowState.gpuContext);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowState.width, windowState.height,
        GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
}

void RenderGL() {
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(windowState.shaderProgram);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, windowState.vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, windowState.uvBuffer);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void InitRayTracing(Renderer& renderer) {
    SphereGPU sphere1 = { { 0.0f, -10004.0f, -20.0f },  10000.0f,   0.20f, 0.20f, 0.20f };
    SphereGPU sphere2 = { { 0.0f, 0.0f, -20.0f },       4.0f,       1.00f, 0.32f, 0.36f };
    SphereGPU sphere3 = { { 5.0f, -1.0f, -15.0f },      2.0f,       0.90f, 0.76f, 0.46f };
    SphereGPU sphere4 = { { 5.0f, 0.0f, -25.0f },       3.0f,       0.65f, 0.77f, 0.97f };
    SphereGPU sphere5 = { { -5.5f, 0.0f, -15.0f },      3.0f,       0.90f, 0.90f, 0.90f };

    renderer.AddSphere(&sphere1);
    renderer.AddSphere(&sphere2);
    renderer.AddSphere(&sphere3);
    renderer.AddSphere(&sphere4);
    renderer.AddSphere(&sphere5);

}

int main( void ) {
    if (!InitWindow()) return -1;
    float bgColor[3] = {1.0f, 1.0f, 1.0f};

    Renderer renderer(windowState.width, windowState.height, 30.0f, 5, bgColor);
    InitRayTracing(renderer);

    CreateGPUContext(&windowState.gpuContext);
    InitGPURendering(windowState.gpuContext);

    InitGLRenderer();

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double pc_frequency = freq.QuadPart / 1000.0;

    char fps_string[10];
    int fps = 0;

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    double begin = counter.QuadPart / pc_frequency;

    do{
        RenderToTexture(renderer);
        RenderGL();

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

    TerminateGLRenderer();
    FreeGPUContext(windowState.gpuContext);
    TerminateWindow();

    return 0;
}

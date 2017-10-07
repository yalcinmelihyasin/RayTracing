#include "GLBinding.h"

#include <GL/glew.h>
#include <glfw3.h>

#include "GPUBinding.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

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

struct GLContext {
    GLFWwindow* window;

    GLuint vertexArray;
    GLuint vertexBuffer;
    GLuint uvBuffer;

    GLuint gpuBuffer;
    GLuint texture;

    GLuint shaderProgram;

    int width;
    int height;
};

void CreateGLContext(GLContext** context) {
    GLContext* newContext = (GLContext*)malloc(sizeof(GLContext));
    memset(newContext, 0, sizeof(GLContext));
    *context = newContext;
}

void DestroyGLContext(GLContext* context) {
    free(context);
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
        char infoMessage[255];
        glGetShaderInfoLog(VertexShaderID, infoLogLength, NULL, infoMessage);
        printf("%s\n", infoMessage);
    }

    // Compile Fragment Shader
    glShaderSource(FragmentShaderID, 1, &fragmentShaderCode, NULL);
    glCompileShader(FragmentShaderID);

    // Check Fragment Shader
    glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &result);
    glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);

    if (infoLogLength > 0) {
        char infoMessage[255];
        glGetShaderInfoLog(FragmentShaderID, infoLogLength, NULL, infoMessage);
        printf("%s\n", infoMessage);
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
        char infoMessage[255];
        glGetProgramInfoLog(ProgramID, infoLogLength, NULL, infoMessage);
        printf("%s\n", infoMessage);
    }

    glDetachShader(ProgramID, VertexShaderID);
    glDetachShader(ProgramID, FragmentShaderID);

    glDeleteShader(VertexShaderID);
    glDeleteShader(FragmentShaderID);

    return ProgramID;
}

void InitGLRendering(GLContext* glContext, GPUContext* gpuContext, int width, int height, GLFWwindow* window)
{
    glContext->width = width;
    glContext->height = height;
    glContext->window = window;

    // Create the quad spans the screen
    glContext->vertexArray = 0;
    glGenVertexArrays(1, &glContext->vertexArray);
    glBindVertexArray(glContext->vertexArray);

    GLfloat vertexBufferData[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f,  1.0f, 0.0f,
    };

    // Bind quad to vertex buffer
    glContext->vertexBuffer = 0;
    glGenBuffers(1, &glContext->vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, glContext->vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexBufferData), vertexBufferData, GL_STATIC_DRAW);

    GLfloat uvBufferData[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,

        0.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 0.0f
    };

    glGenBuffers(1, &glContext->uvBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, glContext->uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvBufferData), uvBufferData, GL_STATIC_DRAW);

    size_t bufferSize = width * height * 4;

    glGenBuffers(1, &glContext->gpuBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glContext->gpuBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, nullptr, GL_DYNAMIC_COPY);
    RegisterPixelBuffer(gpuContext, glContext->gpuBuffer);

    // Create opengl texture
    glContext->texture = 0;
    glGenTextures(1, &glContext->texture);
    glBindTexture(GL_TEXTURE_2D, glContext->texture);

    // Set image to texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glContext->shaderProgram = LoadShaders(vertexShader, fragmentShader);
}

void TerminateGLRendering(GLContext* glContext, GPUContext* gpuContext) {
    UnregisterPixelBuffer(gpuContext);
    glDeleteTextures(1, &glContext->texture);
    glDeleteBuffers(1, &glContext->vertexBuffer);
    glDeleteBuffers(1, &glContext->uvBuffer);
    glDeleteBuffers(1, &glContext->gpuBuffer);

    glDeleteVertexArrays(1, &glContext->vertexArray);
    glDeleteProgram(glContext->shaderProgram);
}

void RenderGLContext(GLContext* context) {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, context->width, context->height,
        GL_BGRA, GL_UNSIGNED_BYTE, nullptr);

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(context->shaderProgram);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, context->vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, context->uvBuffer);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    // Swap buffers
    glfwSwapBuffers(context->window);
    glfwPollEvents();
}

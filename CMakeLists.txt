cmake_minimum_required (VERSION 2.8)
project (RayTracing)

#OpenGL is required
find_package(OpenGL REQUIRED)

#CUDA is required
find_package(CUDA REQUIRED)

set(CUDA_PATH $ENV{CUDA_PATH})

include_directories(
    #OPENGL related
    external/glfw-3.2.1/include/GLFW/
    external/glm-0.9.8.4/
    external/glew-2.0.0/include/

    #CUDA
    ${CUDA_PATH}/include/

    #Internal
    src/
)

link_directories(
    ${CMAKE_SOURCE_DIR}/external/glfw-3.2.1/lib-vc2015
    ${CMAKE_SOURCE_DIR}/external/glew-2.0.0/lib/Release/x64
)

set(ALL_LIBS
    ${OPENGL_LIBRARY}
    glfw3.lib
    glew32s.lib
)

add_definitions(
    -DNO_DEFAULT_LIB
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
    -D_CRT_SECURE_NO_WARNINGS
)

#CUDA definitions
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_30,code=sm_30
)

source_group(cuda src/cuda/*.*)

set(CUDA_SRCS
    src/cuda/GPUCode.cu
    src/cuda/GPUCode.cuh
)

set(CPU_SRCS
    src/cpu/main.cpp
    src/cpu/Renderer.h
    src/cpu/Renderer.cpp
)


# Create Binary
cuda_add_executable(RayTracing ${CUDA_SRCS} ${CPU_SRCS})
target_link_libraries(RayTracing ${ALL_LIBS})

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

source_group(gpu_code src/gpu_code/*.*)
source_group(cpu_code src/cpu_code/*.*)
source_group(binding src/binding/*.*)

set(GPU_SRCS
    src/gpu_code/Renderer.cu
    src/gpu_code/Renderer.cuh
)

set(CPU_SRCS
    src/cpu_code/main.cpp
    src/cpu_code/Renderer.h
    src/cpu_code/Renderer.cpp
)

set(BINDING_SRCS
    src/binding/GPUTypes.h
    src/binding/GPUBinding.h
    src/binding/GPUBinding.cpp
    src/binding/GLBinding.h
    src/binding/GLBinding.cpp
)

# Create Binary
cuda_add_executable(RayTracing ${GPU_SRCS} ${CPU_SRCS} ${BINDING_SRCS})
target_link_libraries(RayTracing ${ALL_LIBS})

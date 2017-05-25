cmake_minimum_required (VERSION 2.6)
project (RayTracing)

find_package(OpenGL REQUIRED)

if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
    message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
    message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()

#add_subdirectory(external)

# On Visual 2005 and above, this module can set the debug working directory
cmake_policy(SET CMP0026 OLD)

include_directories(
    external/glfw-3.2.1/include/GLFW/
    external/glm-0.9.8.4/
    external/glew-2.0.0/include/
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
    -DTW_STATIC
    -DNO_DEFAULT_LIB
    -DTW_NO_LIB_PRAGMA
    -DTW_NO_DIRECT3D
    -DGLEW_STATIC
    -D_CRT_SECURE_NO_WARNINGS
)

# Create Binary
add_executable(RayTracing RayTracing/RayTracing.cpp)
target_link_libraries(RayTracing ${ALL_LIBS})

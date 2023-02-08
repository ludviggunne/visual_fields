
cmake_minimum_required(VERSION 3.5)

project(visual_fields)

add_executable(${PROJECT_NAME}

    src/main.c
)

add_subdirectory(extern/glfw)
add_subdirectory(extern/glad-cmake-3.3)
add_subdirectory(extern/cglm)

target_link_libraries(${PROJECT_NAME} 

PUBLIC
    glfw
    glad
    cglm
    
)
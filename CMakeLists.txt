cmake_minimum_required(VERSION 3.20)

project(icp VERSION 2025.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glm           CONFIG REQUIRED)
find_package(GLEW                 REQUIRED)
find_package(glfw3         CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(imgui                REQUIRED) 
find_package(Stb                  REQUIRED)

if(WIN32)
    find_package(OpenCV CONFIG REQUIRED PATHS "C:/opencv/build")
elseif(UNIX)
    find_package(OpenCV CONFIG REQUIRED)
endif()

file(GLOB_RECURSE PROJECT_SOURCES
    src/*.cpp
    src/*.h
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/build/debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/build/release)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/resources
)


target_link_libraries(${PROJECT_NAME} PRIVATE 
    GLEW::GLEW
    glfw
    glm::glm
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    imgui::imgui
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${Stb_INCLUDE_DIR}
)

configure_file("app_settings.json" "app_settings.json" COPYONLY)

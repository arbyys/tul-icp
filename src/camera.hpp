#pragma once

#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

class Camera
{
public:

    // Camera Attributes
    glm::vec3 position{};
    glm::vec3 front{};
    glm::vec3 right{};
    glm::vec3 up{}; // camera local UP vector

    GLfloat yaw = -90.0f;
    GLfloat pitch = 0.0f;;
    GLfloat roll = 0.0f;

    // Camera options
    GLfloat movement_speed = 1.0f;
    GLfloat mouse_sensitivity = 0.25f;

    Camera() {
        // Default constructor initializes camera's position and orientation
        this->update_camera_vectors();
    }

    Camera(glm::vec3 position) :position(position)
    {
        this->up = glm::vec3(0.0f, 1.0f, 0.0f);
        // initialization of the camera reference system
        this->update_camera_vectors();
    }

    glm::mat4 get_view_matrix()
    {
        return glm::lookAt(this->position, this->position + this->front, this->up);
    }

    glm::vec3 process_input(GLFWwindow* window, GLfloat deltaTime)
    {
        glm::vec3 direction{ 0 };

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            direction += front; // add unit vector to final direction  

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            direction -= front;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            direction -= right;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            direction += right;


        glm::vec3 movement = direction == glm::vec3(0) ? glm::vec3(0) : glm::normalize(direction)* movement_speed* deltaTime;
        return movement;
    }

    void process_mouse_movement(GLfloat xoffset, GLfloat yoffset, GLboolean constraintPitch = GL_TRUE)
    {
        xoffset *= this->mouse_sensitivity;
        yoffset *= this->mouse_sensitivity;

        this->yaw += xoffset;
        this->pitch += yoffset;

        if (constraintPitch)
        {
            if (this->pitch > 89.0f)
                this->pitch = 89.0f;
            if (this->pitch < -89.0f)
                this->pitch = -89.0f;
        }

        this->update_camera_vectors();
    }

private:
    void update_camera_vectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(this->yaw)) * cos(glm::radians(this->pitch));
        front.y = sin(glm::radians(this->pitch));
        front.z = sin(glm::radians(this->yaw)) * cos(glm::radians(this->pitch));

        this->front = glm::normalize(front);
        this->right = glm::normalize(glm::cross(this->front, glm::vec3(0.0f, 1.0f, 0.0f)));
        this->up = glm::normalize(glm::cross(this->right, this->front));
    }
};
#pragma once

#include <GLFW/glfw3.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

class Camera
{
public:
    enum class FlightSpeedTier {
        Slow,
        Normal,
        Fast
    };

    // Camera Attributes
    glm::vec3 position{};
    glm::vec3 front{};
    glm::vec3 right{};
    glm::vec3 up{}; // camera local UP vector

    GLfloat yaw = -90.0f;
    GLfloat pitch = 0.0f;;
    GLfloat roll = 0.0f;

    // Camera options
    GLfloat movement_speed = 8.0f;
    GLfloat mouse_sensitivity = 0.25f;
    FlightSpeedTier flight_speed_tier = FlightSpeedTier::Normal;

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
        const glm::vec3 world_up(0.0f, 1.0f, 0.0f);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            direction += front; // add unit vector to final direction  

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            direction -= front;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            direction -= right;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            direction += right;

        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            direction += world_up;

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
            direction -= world_up;

        GLfloat sprint_multiplier = 1.0f;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS)
            sprint_multiplier = 1.75f;

        glm::vec3 movement = direction == glm::vec3(0)
            ? glm::vec3(0)
            : glm::normalize(direction) * movement_speed * get_flight_speed_multiplier() * sprint_multiplier * deltaTime;
        return movement;
    }

    void cycle_flight_speed_tier()
    {
        if (flight_speed_tier == FlightSpeedTier::Slow) {
            flight_speed_tier = FlightSpeedTier::Normal;
        }
        else if (flight_speed_tier == FlightSpeedTier::Normal) {
            flight_speed_tier = FlightSpeedTier::Fast;
        }
        else {
            flight_speed_tier = FlightSpeedTier::Slow;
        }
    }

    const char* get_flight_speed_tier_label() const
    {
        switch (flight_speed_tier) {
        case FlightSpeedTier::Slow:
            return "SLOW";
        case FlightSpeedTier::Fast:
            return "FAST";
        case FlightSpeedTier::Normal:
        default:
            return "NORMAL";
        }
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
    GLfloat get_flight_speed_multiplier() const
    {
        switch (flight_speed_tier) {
        case FlightSpeedTier::Slow:
            return 0.75f;
        case FlightSpeedTier::Fast:
            return 5.0f;
        case FlightSpeedTier::Normal:
        default:
            return 2.0f;
        }
    }

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
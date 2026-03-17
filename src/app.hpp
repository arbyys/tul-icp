// icp.cpp 
// author: JJ

#pragma once
#include <vector>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "dequeue.hpp"
#include "assets.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>  
#include "shaders/ShaderProgram.hpp"
#include "shaders/Mesh.hpp"
#include "shaders/Model.hpp"
#include "camera.hpp"

class App {
public:
    App();

    bool init(void);
    int run(void);
    void add_console_log(const char* msg);

    ~App();
protected:
    // projection related variables    
    int width{ 0 }, height{ 0 };
    float fov = 60.0f;
    // store projection matrix here, update only on callbacks
    glm::mat4 projection_matrix = glm::identity<glm::mat4>();
    void update_projection_matrix(void);
private:
    cv::VideoCapture capture;
    void draw_cross_normalized(cv::Mat& img, cv::Point2f center_normalized, int size);
    // object detection
    cv::CascadeClassifier face_cascade = cv::CascadeClassifier("resources/haarcascade_frontalface_default.xml");
    cv::Point2f find_face(cv::Mat& frame);
    std::vector<cv::Point2f> find_faces(cv::Mat& frame);

    // tracer thread loop and communication variables
    void tracker_thread(cv::VideoCapture& capture);
    synced_deque<cv::Mat> frame_buffer{ 3 };
    std::mutex buffer_mutex;
    std::vector<cv::Point2f> detections;
    std::atomic<bool> terminate;
    std::atomic<bool> frames_available;

    // image compression
    float target_coefficient = 0.5f;
    int num_threads = 6;

    //new GL stuff
    GLFWwindow* window = nullptr;
    void destroy(void);

    GLuint shader_prog_ID{ 0 };
    GLuint VBO_ID{ 0 };
    GLuint VAO_ID{ 0 };
    GLuint FBO_ID{ 0 };
    GLuint RBO_ID{ 0 };
    GLuint texture_id{ 0 };

    // vsync enabled
    bool is_vsync_on{ true };

    // should be ImGUI window displayed?
    bool show_info{ true };

    // init stuff
    void init_opencv();
    void init_glew(void);
    void init_glfw(void);
    void init_gl_debug();
    void init_assets(void);
    void init_imgui(void);

    // framebuffer stuff
    int fb_width = 1920;
    int fb_height = 1080;
    void init_framebuffer(void);
    void rescale_framebuffer(float width, float height);

    // camera related 
    Camera camera;
    // remember last cursor position, move relative to that in the next frame
    double cursor_last_x{ 0 };
    double cursor_last_y{ 0 };
    
    // in-window console stuff
    std::vector<char*> console_lines;
    bool scroll_to_bottom = false;
    static std::string SanitizeUTF8(const char* msg);

    // info prints
    void print_opencv_info();
    void print_glfw_info(void);
    void print_glm_info();
    void print_gl_info();

    // callbacks
    static void glfw_error_callback(int error, const char* description);
    static void glfw_framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void glfw_cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
    static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

    // 3D scene stuff
    // shared library of shaders for all models, automatic resource management 
    std::unordered_map<std::string, std::shared_ptr<ShaderProgram>> shader_library;

    // shared library of meshes for all models, automatic resource management 
    std::unordered_map<std::string, std::shared_ptr<Mesh>> mesh_library;

    // all objects of the scene addressable by name
    std::unordered_map<std::string, Model> scene;
};


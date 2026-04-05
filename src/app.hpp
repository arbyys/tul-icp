// icp.cpp 
// author: JJ

#pragma once
#include <vector>
#include <atomic>
#include <filesystem>
#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "dequeue.hpp"
#include "assets.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>  
#include "shaders/ShaderProgram.hpp"
#include "shaders/Mesh.hpp"
#include "shaders/Model.hpp"
#include "camera.hpp"
#include "shaders/Texture.hpp"
#include "audio_manager.hpp"

class App {
public:
    struct PlanetParams {
        std::string name;
        std::filesystem::path model_path;
        std::filesystem::path default_texture_path;
        std::unordered_map<std::string, std::filesystem::path> material_textures;
        std::filesystem::path audio_path;
        std::string audio_key;
        glm::vec3 start_position{ 0.0f, 0.0f, 0.0f };
        glm::vec3 start_rotation{ 0.0f, 0.0f, 0.0f };
        glm::vec3 start_scale{ 1.0f, 1.0f, 1.0f };
        float normalized_radius = 1.2f;
        float collision_radius = 1.2f;
        float teleport_distance = 12.0f;
        glm::vec3 orbit_center{ 0.0f, 0.0f, 0.0f };
        float orbit_radius = 0.0f;
        float orbit_speed_deg = 0.0f;
        float orbit_angle_deg = 0.0f;
        float self_rotation_speed_deg = 0.0f;
        float audio_min_distance = 5.0f;
        float audio_max_distance = 1000.0f;
        float audio_volume = 1.0f;
    };

    App();

    bool init(void);
    int run(void);
    void add_console_log(const char* msg);

    void set_planet_transform(const std::string& name, const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale);
    void set_planet_orbit_speed(const std::string& name, float orbit_speed_deg);
    glm::vec3 get_planet_position(const std::string& name) const;
    const std::vector<PlanetParams>& get_planet_params(void) const;

    ~App();
protected:
    // projection related variables    
    int width{ 0 }, height{ 0 };
    const float default_fov = 60.0f;
    float fov = 60.0f;
    // store projection matrix here, update only on callbacks
    glm::mat4 projection_matrix = glm::identity<glm::mat4>();
    void update_projection_matrix(void);
private:

    cv::VideoCapture capture;
    cv::CascadeClassifier face_cascade = cv::CascadeClassifier("resources/haarcascade_frontalface_default.xml");


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

    // openGL IDs
    GLuint shader_prog_ID{ 0 };
    GLuint VBO_ID{ 0 };
    GLuint VAO_ID{ 0 };
    GLuint FBO_ID{ 0 };
    GLuint RBO_ID{ 0 };
    GLuint texture_id{ 0 };

    // vsync enabled
    bool is_vsync_on{ true };

    // fullscreen stuff
    bool fullscreen = false;
    void ToggleFullscreen(GLFWwindow* window);
    int windowed_width = 0;
    int windowed_height = 0;
    int window_x = 0;
    int window_y = 0;

    // screenshot stuff
    std::string screenshot_folder = "screenshots";
    std::string ensure_dir_and_get_filename(const std::string& folder, const std::string& filename);
    std::string get_timestamp_filename(const std::string& prefix = "screenshot", const std::string& ext = "png");
    void take_screenshot_fbo(GLuint fbo, int width, int height, std::string filename);

    // antialiasing stuff
    bool antialiasing_on = false;
    int antialiasing_level = 2;
    void toggle_aliasing(void);

    // focus toggle
    bool scene_in_focus = false;
    bool imgui_initialized = false;

    // placeholder controls for future orbital logic
    int orbit_speed_placeholder = 0;
    std::size_t next_teleport_index = 0;

    // init stuff
    void init_opencv();
    void init_glew(void);
    void init_glfw(void);
    void init_gl_debug(void);
    void init_assets(void);
    void init_imgui(void);

    // framebuffer stuff
    int fb_width = 1920;
    int fb_height = 1080;
    void init_framebuffer(void);
    void rescale_framebuffer(int width, int height);

    // scene camera related 
    Camera camera;
    // remember last cursor position, move relative to that in the next frame
    double cursor_last_x{ 0 };
    double cursor_last_y{ 0 };
    
    // webcam related
    GLuint webcam_tex;
    void init_webcam_tex(int rows, int cols);

    // in-window console stuff
    std::vector<std::string> console_lines;
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

    std::vector<PlanetParams> planets_params;

    AudioManager audio_manager;
    void update_spatial_audio();
    float player_collision_padding = 0.8f;
    void resolve_player_planet_collisions();
    void teleport_to_next_planet();
    void update_planets(float delta_t);

    PlanetParams* find_planet_params(const std::string& name);
    const PlanetParams* find_planet_params(const std::string& name) const;
    std::shared_ptr<Texture> get_or_load_texture(const std::filesystem::path& path);
    std::shared_ptr<Texture> pick_planet_texture(const PlanetParams& params, const std::string& material_name);

    // 3D scene stuff
    // shared library of shaders for all models, automatic resource management 
    std::unordered_map<std::string, std::shared_ptr<ShaderProgram>> shader_library;

    // shared library of meshes for all models, automatic resource management 
    std::unordered_map<std::string, std::shared_ptr<Mesh>> mesh_library;

    // all objects of the scene addressable by name
    std::unordered_map<std::string, Model> scene;

    // shared library of textures for all models, automatic resource management 
    std::unordered_map<std::string, std::shared_ptr<Texture>> texture_library;
};


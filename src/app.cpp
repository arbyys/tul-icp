// icp.cpp 
// Author: JJ

// C++ 
#include <iostream>
#include <chrono>
#include <stack>
#include <random>
#include <numeric>
#include <vector>
#include <thread>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <stdexcept>

// OpenCV 
#include <opencv2\opencv.hpp>

// 3D libs
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

// json
#include <nlohmann/json.hpp>

// imgui
#include <imgui.h>               // main ImGUI header
#include <imgui_impl_glfw.h>     // GLFW bindings
#include <imgui_impl_opengl3.h>  // OpenGL bindings

// stb (for screenshots)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// local stuff
#include "app.hpp"
#include "utils.hpp"
#include "fpsmeter.hpp"
#include "dequeue.hpp"
#include "shaders/OBJLoader.hpp"

namespace {
std::vector<App::PlanetParams> create_default_planets_params()
{
    constexpr float planet_height = 3.0f;
    constexpr float planet_spacing = 50.0f;

    auto make_planet = [](
        const std::string& name,
        const std::filesystem::path& model_path,
        const std::filesystem::path& texture_path,
        const std::filesystem::path& audio_path,
        const std::string& audio_key,
        float audio_min_distance,
        float audio_max_distance,
        float x_position,
        float rotation_x_deg,
        float orbit_speed_deg) {
            App::PlanetParams params;
            params.name = name;
            params.model_path = model_path;
            params.default_texture_path = texture_path;
            params.audio_path = audio_path;
            params.audio_key = audio_key;
            params.audio_min_distance = audio_min_distance;
            params.audio_max_distance = audio_max_distance;
            params.start_position = glm::vec3(x_position, planet_height, 0.0f);
            params.start_rotation = glm::vec3(rotation_x_deg, 0.0f, 0.0f);
            params.start_scale = glm::vec3(1.0f, 1.0f, 1.0f);
            
            params.orbit_center = glm::vec3(0.0f, planet_height, 0.0f);
            params.orbit_radius = x_position;
            params.orbit_speed_deg = orbit_speed_deg;
            
            params.normalized_radius = 2.4f;
            params.collision_radius = 2.4f;
            return params;
    };

    std::vector<App::PlanetParams> planets = {
        make_planet("sun", "resources/models/planets/sun.obj", "resources/textures/sun.jpeg", "resources/audio/planets/sun.mp3", "snd_planet_sun", 5.0f, 1200.0f, 0.0f * planet_spacing, 0.0f, 0.0f),
        make_planet("mercury", "resources/models/planets/mercury.obj", "resources/textures/mercury.jpg", "resources/audio/planets/mercury.mp3", "snd_planet_mercury", 5.0f, 700.0f, 1.0f * planet_spacing, 90.0f, 47.3f),
        make_planet("venus", "resources/models/planets/venus.obj", "resources/textures/venus_1.png", "resources/audio/planets/venus.mp3", "snd_planet_venus", 5.0f, 700.0f, 2.0f * planet_spacing, 0.0f, 35.0f),
        make_planet("earth", "resources/models/planets/earth.obj", "resources/textures/earth.jpg", "resources/audio/planets/earth.mp3", "snd_planet_earth", 5.0f, 700.0f, 3.0f * planet_spacing, 90.0f, 29.7f),
        make_planet("mars", "resources/models/planets/mars.obj", "resources/textures/mars.jpg", "resources/audio/planets/mars.mp3", "snd_planet_mars", 5.0f, 700.0f, 4.0f * planet_spacing, 90.0f, 24.1f),
        make_planet("jupiter", "resources/models/planets/jupiter.obj", "resources/textures/jupiter.jpeg", "resources/audio/planets/jupiter.mp3", "snd_planet_jupiter", 8.0f, 900.0f, 5.0f * planet_spacing, 0.0f, 13.0f),
        make_planet("saturn", "resources/models/planets/saturn.obj", "resources/textures/saturn.png", "resources/audio/planets/saturn.mp3", "snd_planet_saturn", 8.0f, 900.0f, 6.0f * planet_spacing, 0.0f, 9.6f),
        make_planet("uranus", "resources/models/planets/uranus.obj", "resources/textures/uranus.jpeg", "resources/audio/planets/uranus.mp3", "snd_planet_uranus", 8.0f, 900.0f, 7.0f * planet_spacing, 0.0f, 6.8f),
        make_planet("neptune", "resources/models/planets/neptune.obj", "resources/textures/neptune_base.jpg", "resources/audio/planets/neptune.mp3", "snd_planet_neptune", 8.0f, 900.0f, 8.0f * planet_spacing, 90.0f, 5.4f)
    };

    planets[6].material_textures = {
        { "mat0", "resources/textures/saturn_moons.jpeg" },
        { "mat2", "resources/textures/saturn.png" },
        { "mat3", "resources/textures/saturn.png" }
    };

    planets[8].material_textures = {
        { "mat2", "resources/textures/neptune_atmosfera_1.jpg" },
        { "mat3", "resources/textures/neptune_atmosfera_2.jpg" },
        { "mat4", "resources/textures/neptune_atmosfera_3.jpg" },
        { "mat5", "resources/textures/neptune_atmosfera_3.jpg" },
        { "mat6", "resources/textures/neptune_atmosfera_3.jpg" },
        { "mat7", "resources/textures/neptune_atmosfera_3.jpg" },
        { "mat8", "resources/textures/neptune_atmosfera_3.jpg" },
        { "mat9", "resources/textures/neptune_atmosfera_3.jpg" },
        { "mat10", "resources/textures/neptune_atmosfera_3.jpg" }
    };

    return planets;
}
}

void App::draw_cross_normalized(cv::Mat& img, cv::Point2f center_normalized, int size)
{
    center_normalized.x = std::clamp(center_normalized.x, 0.0f, 1.0f);
    center_normalized.y = std::clamp(center_normalized.y, 0.0f, 1.0f);
    size = std::clamp(size, 1, std::min(img.cols, img.rows));

    cv::Point2f center_absolute(center_normalized.x * img.cols, center_normalized.y * img.rows);

    cv::Point2f p1(center_absolute.x - size / 2, center_absolute.y);
    cv::Point2f p2(center_absolute.x + size / 2, center_absolute.y);
    cv::Point2f p3(center_absolute.x, center_absolute.y - size / 2);
    cv::Point2f p4(center_absolute.x, center_absolute.y + size / 2);

    cv::line(img, p1, p2, CV_RGB(255, 0, 0), 3);
    cv::line(img, p3, p4, CV_RGB(255, 0, 0), 3);
}

cv::Point2f App::find_face(cv::Mat& frame)
{
    cv::Point2f center(0.0f, 0.0f);

    cv::Mat scene_grey;
    cv::cvtColor(frame, scene_grey, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(scene_grey, faces);

    if (faces.size() > 0)
    {

        // compute "center" as normalized coordinates of the face
        cv::Rect rect = faces[0];
        center.x = (rect.x + rect.width / 2.0f) / frame.cols;
        center.y = (rect.y + rect.height / 2.0f) / frame.rows;
    }

    std::cout << "found normalized center: " << center << std::endl;

    return center;
}

std::vector<cv::Point2f> App::find_faces(cv::Mat& frame)
{
    std::vector<cv::Point2f> centers;

    cv::Mat scene_grey;
    cv::cvtColor(frame, scene_grey, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(scene_grey, faces);


    if (faces.size() > 0)
    {
        for (const cv::Rect& rect : faces)
        {
            // compute "center" as normalized coordinates of the face
            cv::Point2f center(0.0f, 0.0f);
            center.x = (rect.x + rect.width / 2.0f) / frame.cols;
            center.y = (rect.y + rect.height / 2.0f) / frame.rows;
            centers.push_back(center);
        }
    }

    return centers;
}

App::App()
    : planets_params(create_default_planets_params())
{
    // default constructor
    // nothing to do here (so far...)
}

int App::run(void)
{

    try {
        double now = glfwGetTime();
        // FPS related
        double fps_last_displayed = now;
        int fps_counter_frames = 0;
        double fps = 0.0;

        // animation related
        double frame_begin_timepoint = now;
        double delta_t = 0.1;

        // Clear color saved to OpenGL state machine: no need to set repeatedly in game loop
        glClearColor(0, 0, 0, 0);

        GLfloat r, g, b, a;
        r = g = b = a = 1.0f; //white color

        // random coloring setup
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, 0.01);


        // Activate shader program. There is only one program, so activation can be out of the loop. 
        // In more realistic scenarios, you will activate different shaders for different 3D objects.

        // Get uniform location in GPU program. This will not change, so it can be moved out of the game loop.

        // get first position of mouse cursor
        glfwGetCursorPos(window, &cursor_last_x, &cursor_last_y);

        camera.position = glm::vec3(0, 0, 40);

        width = fb_width;
        height = fb_height;
        update_projection_matrix();

        while (!glfwWindowShouldClose(window))
        {
            // draw into framebuffer
            // activate framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, FBO_ID);

            //rescale_framebuffer(fb_width, fb_height);
            // set viewport to framebuffer's size
            glViewport(0, 0, fb_width, fb_height);

            // clear existing
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // set shader uniforms once (if all models use same shader)
            auto current_shader = shader_library.at("simple_shader");
            //set transformations
            update_planets(static_cast<float>(delta_t));
            if (scene_in_focus) {
                glm::vec3 movement = camera.process_input(window, delta_t);
                camera.position += movement;
                resolve_player_planet_collisions();
            }
            update_spatial_audio();
            //std::cout << movement << std::endl;

            // set uniforms for shader - common for all objects (do not set for each object individually, they use same shader anyway)
            current_shader->setUniform("uV_m", camera.get_view_matrix());
            current_shader->setUniform("uP_m", projection_matrix);

            for (auto& [name, model] : scene) {
                model.draw();
            }

            // unbind framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            auto& io = ImGui::GetIO();

            if (scene_in_focus) {
                io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
            }
            else {
                io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGuiViewport* viewport = ImGui::GetMainViewport();

            ImGui::SetNextWindowPos(viewport->Pos);
            ImGui::SetNextWindowSize(viewport->Size);
            //ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse;

            // ImGui prepare render (only if required)
            ImGui::Begin("Main", 0, flags);

            const float window_width = std::max(1.0f, ImGui::GetContentRegionAvail().x);
            const float window_height = std::max(1.0f, ImGui::GetContentRegionAvail().y);

            const float console_height = 0.2f;
            const float scene_height = window_height * (1.0f - console_height);
            const float scene_width = 0.7f * window_width;

            // scene part of the main window
            ImGui::BeginChild("ScenePanel", ImVec2(scene_width, scene_height), true, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
            ImGui::Text("Scene");
            ImGui::Image(
                texture_id,
                ImVec2(window_width, window_height),
                ImVec2(0, 1),
                ImVec2(1, 0)
            );
            ImGui::EndChild();
            
            float info_width = 0.4f;

            // info panel 
            ImGui::SameLine();
            ImGui::BeginChild("InfoPanel", ImVec2((window_width - scene_width) * info_width, scene_height * 0.5), true);
            GLint major = 0, minor = 0;
            glGetIntegerv(GL_MAJOR_VERSION, &major);
            glGetIntegerv(GL_MINOR_VERSION, &minor);
            ImGui::Text("OpenGL version: %d.%d", major, minor);
            GLint profile = 0;
            glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile);
            std::string profile_string = "";
            if (profile & GL_CONTEXT_CORE_PROFILE_BIT)
                profile_string += "core ";

            if (profile & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT)
                profile_string += "compatibility";
            ImGui::Text("OpenGL profile: %s", profile_string.c_str());
            ImGui::Text("           FPS: %.1f", fps);
            ImGui::Text("         Vsync: %s", is_vsync_on ? "ON" : "OFF");
            ImGui::Text("            AA: %s", antialiasing_on ? "ON" : "OFF");
            ImGui::EndChild();

            // controls panel
            ImGui::SameLine();
            ImGui::BeginChild("ControlsPanel", ImVec2((window_width - scene_width) * (1 - info_width) - 16, scene_height * 0.5), true);
            ImGui::Text("Mouse movement controls camera");
            ImGui::Text("WASD to move");
            ImGui::Text("Hold CTRL to sprint");
            ImGui::Text("SPACE / SHIFT to move up / down");
            ImGui::Text("F to toggle fullscreen");
            ImGui::Text("V to toggle vsync");
            ImGui::Text("X to toggle scene focus");
            ImGui::Text("F1 to take screenshot");
            ImGui::Text("F2 to toggle antialiasing");
            ImGui::Text("F3 to cycle flight speed: %s", camera.get_flight_speed_tier_label());
            ImGui::Separator();
            ImGui::Text("Loaded planets: %d", static_cast<int>(scene.size()));
            if (scene.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No planet assets loaded (check resources paths).\n");
            }
            ImGui::Text("MMB to reset FOV (current: %.1f)", fov);
            ImGui::Text("T to teleport to the next planet");
            ImGui::Text("UP / DOWN orbit speed placeholder: %d", orbit_speed_placeholder);
            ImGui::EndChild();

            // camera part of the main window
            int camera_offset_x = 15;
            int camera_offset_y = 30;
            ImGui::SetCursorPosX(scene_width + camera_offset_x);
            ImGui::SetCursorPosY(scene_height * 0.5 + camera_offset_y);
            ImGui::BeginChild("CameraPanel", ImVec2(window_width - scene_width - 8, scene_height * 0.5), true);
            ImGui::Text("Camera");
            //ImGui::Image(cameraTex, ImVec2(cameraWidth, topHeight - 20));
            ImGui::EndChild();
            
            // console part of the main window
            ImGui::BeginChild("ConsolePanel", ImVec2(window_width, window_height * console_height), true, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
            // Child window for scrolling region
            ImGui::BeginChild("ConsoleScrollRegion", ImVec2(window_width, windowed_height * console_height - 20), true);

            for (int i = 0; i < console_lines.size(); i++)
            {
                ImGui::TextUnformatted(console_lines[i]);
            }
            

            //if (scrollToBottom)
            //    ImGui::SetScrollHereY(1.0f);

            //scrollToBottom = false;
            ImGui::EndChild();

            // Input field
            static char InputBuf[256];
            if (ImGui::InputText("Input", InputBuf, IM_ARRAYSIZE(InputBuf),
                ImGuiInputTextFlags_EnterReturnsTrue))
            {
                console_lines.push_back(_strdup(InputBuf));
                scroll_to_bottom = true;
                InputBuf[0] = 0;
            }

            ImGui::EndChild();

            ImGui::End();
            // render the whole thing
            ImGui::Render();

            // set viewport back to window's size
            glViewport(0, 0, window_width, window_height);
            // clear existing window contents
            glClear(GL_COLOR_BUFFER_BIT);
            // render scene into window
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            // swap buffers
            glfwSwapBuffers(window);
            // poll events
            glfwPollEvents();


            // triange color logc
            r += dist(gen);
            g += dist(gen);
            b += dist(gen);

            r = std::clamp(r, 0.0f, 1.0f);
            g = std::clamp(g, 0.0f, 1.0f);
            b = std::clamp(b, 0.0f, 1.0f);


            //if (show_info) {
            //    ImGui::SetNextWindowPos(ImVec2(10, 10));
            //    ImGui::SetNextWindowSize(ImVec2(250, 100));
            //
            //    ImGui::Begin("Info", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
            //    ImGui::Text("V-Sync: %s", is_vsync_on ? "ON" : "OFF");
            //    ImGui::Text("FPS: %.1f", FPS);
            //    ImGui::Text("(press RMB to release mouse)");
            //    ImGui::Text("(hit D to show/hide info)");
            //    ImGui::End();
            //}
            // Time/FPS measurement
            now = glfwGetTime();
            delta_t = now - frame_begin_timepoint; //compute delta_t
            frame_begin_timepoint = now; // set new start

            fps_counter_frames++;
            if (now - fps_last_displayed >= 1) {
                fps = fps_counter_frames / (now - fps_last_displayed);
                fps_last_displayed = now;
                fps_counter_frames = 0;
                std::cout << "\r[FPS]" << fps << "     "; // Compare: FPS with/without ImGUI
            }
            //glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    }
    catch (std::exception const& e) {
        std::cerr << "App failed : " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

App::~App()
{   
    destroy();
    std::cout << "Bye...\n";

}

void App::destroy(void)
{
    if (imgui_initialized) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        imgui_initialized = false;
    }

    // clean up OpenCV
    cv::destroyAllWindows();

    // clean up audio
    audio_manager.shutdown();

    // clean-up GLFW
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

bool App::init(void)
{
    try {
        std::cout << "Current working directory: " << std::filesystem::current_path().generic_string() << '\n';

        if (!std::filesystem::exists("resources"))
            throw std::runtime_error("Directory 'resources' not found. Various media files are expected to be there.");

        init_opencv();
        init_glfw();
        init_glew();
        init_gl_debug();
        init_framebuffer();
        audio_manager.init();

        //print_opencv_info();
        //print_glfw_info();
        //print_gl_info();
        //print_glm_info();

        glfwSwapInterval(is_vsync_on ? 1 : 0); // vsync

        // init assets (models, sounds, textures, level map, ...)
        // (this may take a while, app window is hidden in the meantime)...
        init_assets();

        // Initialize ImGUI (see https://github.com/ocornut/imgui/wiki/Getting-Started)
        init_imgui();

        // When all is loaded, show the window.
        glfwShowWindow(window);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // Planet rings use alpha and need to be visible from both sides.
        glDisable(GL_CULL_FACE);

    }
    catch (std::exception const& e) {
        std::cerr << "Init failed : " << e.what() << std::endl;
        throw;
    }

    return true;
}

void App::init_assets(void) {
    scene.clear();
    shader_library.clear();
    texture_library.clear();

    shader_library.emplace(
        "simple_shader",
        std::make_shared<ShaderProgram>(
            std::filesystem::path("resources/shaders/tex.vert"),
            std::filesystem::path("resources/shaders/tex.frag"))
    );

    Texture::init_chkboard();

    for (const auto& params : planets_params) {
        get_or_load_texture(params.default_texture_path);
        for (const auto& [_, texture_path] : params.material_textures) {
            get_or_load_texture(texture_path);
        }
    }

    for (const auto& params : planets_params) {
        std::vector<OBJMeshPart> mesh_parts;
        std::vector<std::filesystem::path> referenced_mtl_files;
        if (!loadOBJWithMaterials(params.model_path, mesh_parts, referenced_mtl_files)) {
            throw std::runtime_error("Failed to load model: " + params.model_path.string());
        }
        (void)referenced_mtl_files;

        glm::vec3 bounds_min(std::numeric_limits<float>::max());
        glm::vec3 bounds_max(std::numeric_limits<float>::lowest());
        for (const auto& part : mesh_parts) {
            for (const auto& vertex : part.vertices) {
                bounds_min = glm::min(bounds_min, vertex.position);
                bounds_max = glm::max(bounds_max, vertex.position);
            }
        }

        const glm::vec3 model_center = 0.5f * (bounds_min + bounds_max);
        float max_radius = 0.0f;
        for (const auto& part : mesh_parts) {
            for (const auto& vertex : part.vertices) {
                max_radius = std::max(max_radius, glm::length(vertex.position - model_center));
            }
        }

        if (max_radius <= 0.0f) {
            throw std::runtime_error("Model has invalid radius: " + params.model_path.string());
        }

        const float scale_to_radius = params.normalized_radius / max_radius;

        Model planet;
        for (const auto& part : mesh_parts) {
            std::vector<Vertex> normalized_vertices = part.vertices;
            for (auto& vertex : normalized_vertices) {
                vertex.position = (vertex.position - model_center) * scale_to_radius;
            }

            auto mesh = std::make_shared<Mesh>(normalized_vertices, part.indices, GL_TRIANGLES);
            auto texture = pick_planet_texture(params, part.material_name);
            planet.addMesh(mesh, shader_library.at("simple_shader"), texture);
        }

        planet.setPosition(params.start_position);
        planet.setEulerAngles(params.start_rotation);
        planet.setScale(params.start_scale);
        scene[params.name] = std::move(planet);
    }

    for (const auto& params : planets_params) {
        if (!audio_manager.load3D(
            params.audio_key,
            params.audio_path,
            params.audio_min_distance,
            params.audio_max_distance,
            params.audio_volume)) {
            throw std::runtime_error("Failed to load audio: " + params.audio_path.string());
        }
    }
}

std::shared_ptr<Texture> App::get_or_load_texture(const std::filesystem::path& path)
{
    const std::string key = path.generic_string();
    auto it = texture_library.find(key);
    if (it != texture_library.end()) {
        return it->second;
    }

    auto texture = std::make_shared<Texture>(path);
    texture_library.emplace(key, texture);
    return texture;
}

std::shared_ptr<Texture> App::pick_planet_texture(const PlanetParams& params, const std::string& material_name)
{
    auto it = params.material_textures.find(material_name);
    if (it != params.material_textures.end()) {
        return get_or_load_texture(it->second);
    }
    return get_or_load_texture(params.default_texture_path);
}

void App::init_glfw(void)
{

    /* Initialize the library */
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        throw std::runtime_error("GLFW can not be initialized.");
    }

    // try to open OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // open window, but hidden - it will be enabled later, after asset initialization
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(800, 600, "ICP", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("GLFW window can not be created.");
    }

    glfwSetWindowUserPointer(window, this);

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    // disable mouse cursor
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // GLFW callbacks registration
    glfwSetFramebufferSizeCallback(window, glfw_framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetScrollCallback(window, glfw_scroll_callback);
    glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);

    // antialiasing
    glfwWindowHint(GLFW_SAMPLES, antialiasing_level);
    toggle_aliasing();
}

void App::init_gl_debug(void) {
    if (GLEW_ARB_debug_output)
    {
        glDebugMessageCallback(MessageCallback, this);
        glEnable(GL_DEBUG_OUTPUT);

        //default is asynchronous debug output, use this to simulate glGetError() functionality
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageControl(
            GL_DONT_CARE,                // source
            GL_DONT_CARE,                // type
            GL_DEBUG_SEVERITY_NOTIFICATION, // severity
            0,                           // count (number of IDs)
            nullptr,                     // array of IDs
            GL_FALSE                     // disable those messages
        );

        std::cout << "GL_DEBUG enabled." << std::endl;
    }
    else
        std::cout << "GL_DEBUG NOT SUPPORTED!" << std::endl;
}

void App::init_glew(void) {
    // init glew
    // http://glew.sourceforge.net/basic.html
    // TODO: add error checking!
    GLenum glew = glewInit();
    GLenum wglew = wglewInit();

    if (glew != GLEW_OK || wglew != GLEW_OK) {
        std::cout << "glew init failed" << std::endl;
        return;
    }

    if (!GLEW_ARB_direct_state_access)
        throw std::runtime_error("No DSA :-(");
}

void App::init_opencv(void) {
   //open first available camera, using any API available (autodetect) 
   capture = cv::VideoCapture(0, cv::CAP_ANY);

   //open video file
   //capture = cv::VideoCapture("resources/video.mkv");

   if (!capture.isOpened())
   {
       std::cerr << "no source?" << std::endl;
       return;
   }
   else
   {
       std::cout << "Source: " <<
           ": width=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) <<
           ", height=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << '\n';
   }
}

void App::init_imgui()
{
    // see https://github.com/ocornut/imgui/wiki/Getting-Started

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    imgui_initialized = true;
    std::cout << "ImGUI version: " << ImGui::GetVersion() << "\n";
}

void App::tracker_thread(cv::VideoCapture& capture) {
    cv::Mat frame;
    cv::Mat compressed_frame;
    while (1) {
        if (terminate) {
            return;
        }

        bool new_frame = capture.read(frame);
        if (!new_frame) {
            //std::cout << "missing capture" << std::endl;
            continue;
        }


        frame_buffer.push_back(frame);
        frames_available = true;

        // image compression for detection
        auto bytes = lossy_quality_limit(frame, target_coefficient);

        std::cout << "Target: " << target_coefficient << "\n";
        // if a suitable compression was found
        if (bytes.size() > 0)
        {
            //compressed_frame = cv::imdecode(bytes, cv::IMREAD_ANYCOLOR);

            auto size_compressed = bytes.size();
            auto size_uncompressed = frame.elemSize() * frame.total();
            auto size_compressed_limit = size_uncompressed * target_coefficient;

            std::cout << "Size: uncompressed = " << size_uncompressed << ", compressed = " << size_compressed << ", = " << size_compressed / (size_uncompressed / 100.0) << "% \n";
            // find faces and pass the data to main thread
            //std::vector<cv::Point2f> centers = find_faces(compressed_frame);
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
            //    detections = centers;
            }
        }
    }
}

void App::init_framebuffer() {

    // Create framebuffer
    glCreateFramebuffers(1, &FBO_ID);

    // --- Color texture ---
    glCreateTextures(GL_TEXTURE_2D, 1, &texture_id);
    glTextureStorage2D(texture_id, 1, GL_RGBA8, fb_width, fb_height);

    glTextureParameteri(texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Attach texture to framebuffer
    glNamedFramebufferTexture(FBO_ID, GL_COLOR_ATTACHMENT0, texture_id, 0);

    // --- Depth/stencil renderbuffer ---
    glCreateRenderbuffers(1, &RBO_ID);
    glNamedRenderbufferStorage(RBO_ID, GL_DEPTH24_STENCIL8, fb_width, fb_height);

    // Attach renderbuffer
    glNamedFramebufferRenderbuffer(
        FBO_ID,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_RENDERBUFFER,
        RBO_ID
    );

    // Specify draw buffers (important in DSA!)
    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
    glNamedFramebufferDrawBuffers(FBO_ID, 1, drawBuffers);

    // Check completeness
    if (glCheckNamedFramebufferStatus(FBO_ID, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("FBO not complete!\n");
}

void App::rescale_framebuffer(int width, int height) {
    // Resize color texture
    glTextureStorage2D(texture_id, 1, GL_RGB8, width, height);  // immutable storage

    // If you want to reset filtering (optional if it hasn't changed)
    glTextureParameteri(texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Reattach texture to framebuffer (optional, usually stays attached)
    glNamedFramebufferTexture(FBO_ID, GL_COLOR_ATTACHMENT0, texture_id, 0);

    // Resize depth/stencil renderbuffer
    glNamedRenderbufferStorage(RBO_ID, GL_DEPTH24_STENCIL8, width, height);

    // Reattach renderbuffer (optional, usually stays attached)
    glNamedFramebufferRenderbuffer(FBO_ID, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO_ID);

    // Optional: ensure draw buffers are still set
    GLenum drawBuffers[] = { GL_COLOR_ATTACHMENT0 };
    glNamedFramebufferDrawBuffers(FBO_ID, 1, drawBuffers);
}

void App::add_console_log(const char* msg) {
    console_lines.push_back(const_cast<char*>(msg));
    scroll_to_bottom = true;
}

void App::glfw_error_callback(int error, const char* description)
{
    std::cerr << "GLFW error: " << description << std::endl;
}

void App::glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto this_inst = static_cast<App*>(glfwGetWindowUserPointer(window));
    if ((action == GLFW_PRESS) || (action == GLFW_REPEAT)) {
        GLint stat;
        switch (key) {
        case GLFW_KEY_ESCAPE:
            // Exit The App
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;

        case GLFW_KEY_V:
            // Vsync on/off
            this_inst->is_vsync_on = !this_inst->is_vsync_on;
            glfwSwapInterval(this_inst->is_vsync_on);
            break;

        case GLFW_KEY_F:
            // fullscreen toggle
            this_inst->fullscreen = !this_inst->fullscreen;
            this_inst->ToggleFullscreen(window);
            break;
        
        case GLFW_KEY_F1:
            // take screenshot
            this_inst->take_screenshot_fbo(0, this_inst->width, this_inst->height, this_inst->get_timestamp_filename("bruhapp"));
            break;

        case GLFW_KEY_F2:
            // toggle antialiasing
            this_inst->antialiasing_on = !this_inst->antialiasing_on;
            this_inst->toggle_aliasing();
            
            break;

        case GLFW_KEY_F3:
            this_inst->camera.cycle_flight_speed_tier();
            break;

        case GLFW_KEY_UP:
            // Placeholder for future orbit-speed based movement.
            this_inst->orbit_speed_placeholder = std::clamp(this_inst->orbit_speed_placeholder + 1, -10, 10);
            break;

        case GLFW_KEY_DOWN:
            // Placeholder for future orbit-speed based movement.
            this_inst->orbit_speed_placeholder = std::clamp(this_inst->orbit_speed_placeholder - 1, -10, 10);
            break;

        case GLFW_KEY_T:
            this_inst->teleport_to_next_planet();
            break;

        case GLFW_KEY_X:
            // focus toggle
            // 3 things:
            // toggle ImGui inputs
            // toggle cursor visibility
            // toggle camera movement
            // optionally disable time passing in scene?
            this_inst->scene_in_focus = !this_inst->scene_in_focus;

            if (this_inst->scene_in_focus) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }

        default:
            break;
        }
    }
}

void App::glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // get App instance
    auto this_inst = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (!this_inst->scene_in_focus) return;

    this_inst->fov -= 10 * yoffset; // yoffset is mostly +1 or -1; one degree difference in fov is not visible
    this_inst->fov = std::clamp(this_inst->fov, 20.0f, 170.0f); // limit FOV to reasonable values...
    this_inst->update_projection_matrix();
}

void App::glfw_cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    auto app = static_cast<App*>(glfwGetWindowUserPointer(window));

    if (app->scene_in_focus)
        app->camera.process_mouse_movement(xpos - app->cursor_last_x, (ypos - app->cursor_last_y) * -1.0);
    app->cursor_last_x = xpos;
    app->cursor_last_y = ypos;
}

void App::glfw_framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    auto this_inst = static_cast<App*>(glfwGetWindowUserPointer(window));
    this_inst->width = width;
    this_inst->height = height;

    // set viewport
    glViewport(0, 0, width, height);
    //now your canvas has [0,0] in bottom left corner, and its size is [width x height] 

    this_inst->update_projection_matrix();
}

void App::update_projection_matrix(void)
{
    if (height < 1)
        height = 1;   // avoid division by 0

    float ratio = static_cast<float>(width) / height;

    projection_matrix = glm::perspective(
        glm::radians(fov),   // The vertical Field of View, in radians: the amount of "zoom". Think "camera lens". Usually between 90� (extra wide) and 30� (quite zoomed in)
        ratio,               // Aspect Ratio. Depends on the size of your window.
        0.1f,                // Near clipping plane. Keep as big as possible, or you'll get precision issues.
        20000.0f             // Far clipping plane. Keep as little as possible.
    );
}

void App::glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    auto this_inst = static_cast<App*>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS) {
        switch (button) {
        //case GLFW_MOUSE_BUTTON_LEFT: {
        //    int mode = glfwGetInputMode(window, GLFW_CURSOR);
        //    if (mode == GLFW_CURSOR_NORMAL) {
        //        // we are outside of application, catch the cursor
        //        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        //    }
        //    else {
        //        // we are already inside our game: shoot, click, etc.
        //        std::cout << "Bang!\n";
        //    }
        //    break;
        //}
        case GLFW_MOUSE_BUTTON_RIGHT:
            // release the cursor
            //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            break;

        case GLFW_MOUSE_BUTTON_MIDDLE:
            this_inst->fov = this_inst->default_fov;
            this_inst->update_projection_matrix();
            break;
        default:
            break;
        }
    }
}

const std::vector<App::PlanetParams>& App::get_planet_params(void) const
{
    return planets_params;
}

App::PlanetParams* App::find_planet_params(const std::string& name)
{
    for (auto& params : planets_params) {
        if (params.name == name) {
            return &params;
        }
    }
    return nullptr;
}

const App::PlanetParams* App::find_planet_params(const std::string& name) const
{
    for (const auto& params : planets_params) {
        if (params.name == name) {
            return &params;
        }
    }
    return nullptr;
}

void App::set_planet_transform(const std::string& name, const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale)
{
    PlanetParams* params = find_planet_params(name);
    if (!params) {
        return;
    }

    params->start_position = position;
    params->start_rotation = rotation;
    params->start_scale = scale;

    auto it = scene.find(name);
    if (it != scene.end()) {
        it->second.setPosition(position);
        it->second.setEulerAngles(rotation);
        it->second.setScale(scale);
    }
}

void App::set_planet_orbit_speed(const std::string& name, float orbit_speed_deg)
{
    PlanetParams* params = find_planet_params(name);
    if (!params) {
        return;
    }
    params->orbit_speed_deg = orbit_speed_deg;
}

glm::vec3 App::get_planet_position(const std::string& name) const
{
    auto it = scene.find(name);
    if (it != scene.end()) {
        return it->second.getPosition();
    }
    return glm::vec3(0.0f, 0.0f, 0.0f);
}

void App::update_planets(float delta_t)
{
    const float orbit_speed_scale = std::max(0.0f, 1.0f + static_cast<float>(orbit_speed_placeholder) * 0.1f);

    for (auto& params : planets_params) {
        auto it = scene.find(params.name);
        if (it == scene.end()) {
            continue;
        }

        if (params.orbit_radius > 0.0f) {
            params.orbit_angle_deg = std::fmod(params.orbit_angle_deg + params.orbit_speed_deg * orbit_speed_scale * delta_t, 360.0f);
            const float angle_rad = glm::radians(params.orbit_angle_deg);
            params.start_position = params.orbit_center + glm::vec3(
                std::cos(angle_rad) * params.orbit_radius,
                params.start_position.y,
                std::sin(angle_rad) * params.orbit_radius
            );
        }

        if (params.self_rotation_speed_deg != 0.0f) {
            params.start_rotation.y = std::fmod(params.start_rotation.y + params.self_rotation_speed_deg * delta_t, 360.0f);
        }

        it->second.setPosition(params.start_position);
        it->second.setEulerAngles(params.start_rotation);
        it->second.setScale(params.start_scale);
    }
}

void App::update_spatial_audio()
{
    audio_manager.set_listener_position(
        camera.position.x,
        camera.position.y,
        camera.position.z,
        camera.front.x,
        camera.front.y,
        camera.front.z
    );

    for (const auto& params : planets_params) {
        auto it = scene.find(params.name);
        if (it == scene.end()) {
            audio_manager.stop3DLoop(params.name);
            continue;
        }

        const glm::vec3& pos = it->second.getPosition();
        audio_manager.ensure3DLoop(params.name, params.audio_key, pos.x, pos.y, pos.z);
    }

    audio_manager.clean_finished_sounds();
}

void App::resolve_player_planet_collisions()
{
    constexpr float min_distance_epsilon = 0.001f;

    for (const auto& params : planets_params) {
        auto it = scene.find(params.name);
        if (it == scene.end()) {
            continue;
        }

        const glm::vec3& center = it->second.getPosition();
        glm::vec3 offset = camera.position - center;
        float distance = glm::length(offset);
        float min_allowed_distance = params.collision_radius + player_collision_padding;

        if (distance >= min_allowed_distance) {
            continue;
        }

        if (distance < min_distance_epsilon) {
            offset = glm::vec3(0.0f, 1.0f, 0.0f);
            distance = 1.0f;
        }

        camera.position = center + (offset / distance) * min_allowed_distance;
    }
}

void App::teleport_to_next_planet()
{
    if (scene.empty() || planets_params.empty()) {
        return;
    }

    for (std::size_t attempt = 0; attempt < planets_params.size(); ++attempt) {
        const PlanetParams& params = planets_params[next_teleport_index % planets_params.size()];
        next_teleport_index = (next_teleport_index + 1) % planets_params.size();

        auto it = scene.find(params.name);
        if (it == scene.end()) {
            continue;
        }

        const glm::vec3& center = it->second.getPosition();
        const float stand_off_distance = params.collision_radius + player_collision_padding + params.teleport_distance;
        camera.position = center + glm::vec3(0.0f, 0.0f, stand_off_distance);

        const glm::vec3 world_up(0.0f, 1.0f, 0.0f);
        glm::vec3 view_dir = glm::normalize(center - camera.position);
        camera.front = view_dir;

        glm::vec3 computed_right = glm::cross(camera.front, world_up);
        if (glm::length(computed_right) < 0.0001f) {
            computed_right = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        camera.right = glm::normalize(computed_right);
        camera.up = glm::normalize(glm::cross(camera.right, camera.front));

        camera.yaw = glm::degrees(std::atan2(view_dir.z, view_dir.x));
        camera.pitch = glm::degrees(std::asin(std::clamp(view_dir.y, -1.0f, 1.0f)));

        resolve_player_planet_collisions();
        return;
    }
}

void GLAPIENTRY App::MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
    auto const src_str = [source]() {
        switch (source)
        {
        case GL_DEBUG_SOURCE_API: return "API";
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "WINDOW SYSTEM";
        case GL_DEBUG_SOURCE_SHADER_COMPILER: return "SHADER COMPILER";
        case GL_DEBUG_SOURCE_THIRD_PARTY: return "THIRD PARTY";
        case GL_DEBUG_SOURCE_APPLICATION: return "APPLICATION";
        case GL_DEBUG_SOURCE_OTHER: return "OTHER";
        default: return "Unknown";
        }
        }();

    auto const type_str = [type]() {
        switch (type)
        {
        case GL_DEBUG_TYPE_ERROR: return "ERROR";
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "DEPRECATED_BEHAVIOR";
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "UNDEFINED_BEHAVIOR";
        case GL_DEBUG_TYPE_PORTABILITY: return "PORTABILITY";
        case GL_DEBUG_TYPE_PERFORMANCE: return "PERFORMANCE";
        case GL_DEBUG_TYPE_MARKER: return "MARKER";
        case GL_DEBUG_TYPE_OTHER: return "OTHER";
        default: return "Unknown";
        }
        }();

    auto const severity_str = [severity]() {
        switch (severity) {
        case GL_DEBUG_SEVERITY_NOTIFICATION: return "NOTIFICATION";
        case GL_DEBUG_SEVERITY_LOW: return "LOW";
        case GL_DEBUG_SEVERITY_MEDIUM: return "MEDIUM";
        case GL_DEBUG_SEVERITY_HIGH: return "HIGH";
        default: return "Unknown";
        }
        }();

    std::ostringstream oss;
    oss << "[GL CALLBACK]: "
        << "source = " << src_str
        << ", type = " << type_str
        << ", severity = " << severity_str
        << ", ID = '" << id << "'"
        << ", message = '" << message << "'"
        << "\n";

    std::cout << "printing log" << std::endl;
    std::cout << oss.str() << std::endl;
    App* app = (App*)userParam;
    app->add_console_log(SanitizeUTF8(oss.str().c_str()).c_str());
}

std::string App::SanitizeUTF8(const char* msg)
{
    std::string out;
    const unsigned char* p = (const unsigned char*)msg;

    while (*p)
    {
        if (*p < 0x80) {
            out.push_back(*p);
        }
        else {
            out.push_back('?');  // replace invalid UTF-8
        }
        p++;
    }
    return out;
}

void App::ToggleFullscreen(GLFWwindow* window)
{
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    if (fullscreen) {
        // Save the current window position and size
        glfwGetWindowPos(window, &window_x, &window_y);
        glfwGetWindowSize(window, &windowed_width, &windowed_height);

        // Switch to fullscreen
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
    }
    else {
        // Restore the previous window position and size
        glfwSetWindowMonitor(window, NULL, window_x, window_y, windowed_width, windowed_height, GLFW_DONT_CARE);
    }
}

void App::take_screenshot_fbo(GLuint fbo, int width, int height, std::string filename)
{
    // Bind FBO as read framebuffer temporarily
    GLint prevReadFBO;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFBO);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);

    // Allocate buffer for RGBA8 pixels
    std::vector<unsigned char> pixels(width * height * 4);

    // Read pixels from the framebuffer
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // Restore previous read framebuffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFBO);

    // Flip vertically (OpenGL origin is bottom-left)
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width * 4; ++x) {
            std::swap(pixels[y * width * 4 + x], pixels[(height - 1 - y) * width * 4 + x]);
        }
    }

    // Save to PNG
    std::cout << "saved image at " << filename << std::endl;
    // check if screenshots folder exists
    // if not, create it
    std::string path = ensure_dir_and_get_filename(screenshot_folder, filename);

    stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4);
}

std::string App::get_timestamp_filename(const std::string& prefix, const std::string& ext)
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    // Convert to local time
    std::tm tm;
    localtime_s(&tm, &t);

    // Format as YYYYMMDD_HHMMSS
    std::ostringstream oss;
    oss << prefix << "_"
        << std::put_time(&tm, "%Y%m%d_%H%M%S")
        << "." << ext;

    return oss.str();
}

std::string App::ensure_dir_and_get_filename(const std::string& folder, const std::string& filename)
{
    std::filesystem::path dir(folder);
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir); // Creates folder and any missing parent folders
    }
    return (dir / filename).string();
}

void App::toggle_aliasing(void) {
    if (antialiasing_on) {
        glEnable(GL_MULTISAMPLE);
    }
    else {
        glDisable(GL_MULTISAMPLE);
    }
}
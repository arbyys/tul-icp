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


// local stuff
#include "app.hpp"
#include "utils.hpp"
#include "fpsmeter.hpp"
#include "dequeue.hpp"
#include "shaders/OBJLoader.hpp"

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

        camera.position = glm::vec3(0, 0, 10);

        width = fb_width;
        height = fb_height;

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
            glm::vec3 movement = camera.process_input(window, delta_t);
            camera.position += movement;
            //std::cout << movement << std::endl;

            // set uniforms for shader - common for all objects (do not set for each object individually, they use same shader anyway)
            current_shader->setUniform("uV_m", camera.get_view_matrix());
            current_shader->setUniform("uP_m", projection_matrix);

            
            std::cout << "view: " << glm::to_string(camera.get_view_matrix()) << std::endl;
            std::cout << "proj: " << glm::to_string(projection_matrix) << std::endl;

            for (auto& [name, model] : scene) {
                model.draw();
            }

            // unbind framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

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

            const float window_width = ImGui::GetContentRegionAvail().x;
            const float window_height = ImGui::GetContentRegionAvail().y;

            const float console_height = 0.2f;
            const float scene_height = window_height * (1.0f - console_height);
            const float scene_width = 0.5f * window_width;

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
            
            ImGui::SameLine();

            // camera part of the main window
            ImGui::BeginChild("CameraPanel", ImVec2(window_width - scene_width, scene_height), true);
            ImGui::Text("Camera");
            //ImGui::Image(cameraTex, ImVec2(cameraWidth, topHeight - 20));
            ImGui::EndChild();
            
            // console part of the main window
            ImGui::BeginChild("ConsolePanel", ImVec2(window_width, window_height * console_height), true);
            // Child window for scrolling region
            ImGui::BeginChild("ConsoleScrollRegion", ImVec2(0, 0), true);

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
    // clean up ImGUI
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // clean up OpenCV
    cv::destroyAllWindows();

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
        // assume ALL objects are non-transparent 
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);

    }
    catch (std::exception const& e) {
        std::cerr << "Init failed : " << e.what() << std::endl;
        throw;
    }

    return true;
}

void App::init_assets(void) {
    // all shaders: load, compile, link, initialize params, place to library
    shader_library.emplace("simple_shader", std::make_shared<ShaderProgram>(std::filesystem::path("resources/shaders/tex.vert"), std::filesystem::path("resources/shaders/tex.frag")));

    // mesh library: meshes, that can be shared by multiple models
    //mesh_library.emplace("teapot", std::make_shared<Mesh>(generateCube()));

    // create default texture
    Texture::init_chkboard();

    // load textures
    texture_library.emplace("wood_box", std::make_shared<Texture>("resources/textures/box_rgb888.png"));
    std::unordered_map<std::string, std::filesystem::path> filepaths = {
        { "teapot", "resources/models/teapot_tri_vnt.obj" },
        { "sphere", "resources/models/sphere_tri_vnt.obj" }
    };

    for (const auto& [name, path] : filepaths) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("File does not exist: " + path.string());
        }

        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;
        if (!loadOBJ(path, vertices, indices)) {
            throw std::runtime_error("Loading failed: " + path.string());
        }

        mesh_library.emplace(name, std::make_shared<Mesh>(vertices, indices, GL_TRIANGLES));
    }

    Model m;
    m.addMesh(mesh_library.at("sphere"), shader_library.at("simple_shader"), texture_library.at("wood_box"));
    scene.emplace("teapot", m);
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
    glfwWindowHint(GLFW_SAMPLES, 4);
    glEnable(GL_MULTISAMPLE);
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

    glGenFramebuffers(1, &FBO_ID);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO_ID);

    // Color texture
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fb_width, fb_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_id, 0);

    // Depth buffer
    glGenRenderbuffers(1, &RBO_ID);
    glBindRenderbuffer(GL_RENDERBUFFER, RBO_ID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, fb_width, fb_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO_ID);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        printf("FBO not complete!\n");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void App::rescale_framebuffer(float width, float height)
{
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_id, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, RBO_ID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO_ID);
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
        switch (key) {
        case GLFW_KEY_ESCAPE:
            // Exit The App
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_V:
            // Vsync on/off
            this_inst->is_vsync_on = !this_inst->is_vsync_on;
            glfwSwapInterval(this_inst->is_vsync_on);
            std::cout << "VSync: " << this_inst->is_vsync_on << "\n";
            break;
        default:
            break;
        }
    }
}

void App::glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    // get App instance
    auto this_inst = static_cast<App*>(glfwGetWindowUserPointer(window));
    this_inst->fov -= 10 * yoffset; // yoffset is mostly +1 or -1; one degree difference in fov is not visible
    this_inst->fov = std::clamp(this_inst->fov, 20.0f, 170.0f); // limit FOV to reasonable values...
    this_inst->update_projection_matrix();
}

void App::glfw_cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    auto app = static_cast<App*>(glfwGetWindowUserPointer(window));

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
    std::cout << "fov: " << fov << std::endl;
    std::cout << "ratio: " << ratio << std::endl;

    projection_matrix = glm::perspective(
        glm::radians(fov),   // The vertical Field of View, in radians: the amount of "zoom". Think "camera lens". Usually between 90° (extra wide) and 30° (quite zoomed in)
        ratio,               // Aspect Ratio. Depends on the size of your window.
        0.1f,                // Near clipping plane. Keep as big as possible, or you'll get precision issues.
        20000.0f             // Far clipping plane. Keep as little as possible.
    );
}

void App::glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
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
        default:
            break;
        }
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
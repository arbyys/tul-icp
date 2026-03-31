#pragma once 

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <GL/glew.h> 
#include <glm/glm.hpp>
#include "noncopyable.hpp"

class Texture : private NonCopyable
{
public:
    enum class Interpolation {
        nearest,
        linear,
        linear_mipmap_linear,
    };

    Texture() = default;
    Texture(const cv::Mat& image, Interpolation interpolation = Interpolation::linear_mipmap_linear); // default = best texture filtering
    Texture(const glm::vec3& vec); // synthetic single-color RGB texture
    Texture(const glm::vec4& vec); // synthetic single-color RGBA texture
    Texture(const std::filesystem::path& path, Interpolation interpolation = Interpolation::linear_mipmap_linear);

    ~Texture();

    void bind(void);
    GLuint get_name() const;
    int get_height(void);
    int get_width(void);
    void set_interpolation(Interpolation interpolation);
    void replace_image(const cv::Mat& image);
    static void init_chkboard(void);
private:
    cv::Mat load_image(const std::filesystem::path& path);
    static GLuint gen_ckboard(void);  // create default texture
    static inline GLuint ckboard_; // class-shared ckboard variable. gen_ckboard() is called just once.
    GLuint name_{ ckboard_ }; // set default-constructed texture to ckboard pattern
};


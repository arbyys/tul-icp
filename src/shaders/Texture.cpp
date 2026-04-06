#include "Texture.hpp"


GLuint Texture::gen_ckboard(void) {
    if (glIsTexture(ckboard_) != GL_TRUE) { // default checker-board texture yet not valid texture
        glCreateTextures(GL_TEXTURE_2D, 1, &ckboard_);

        cv::Vec3b black{ 0, 0, 0 };
        cv::Vec3b white{ 255, 255, 255 };
        cv::Mat ckb = cv::Mat(2, 2, CV_8UC3, black);  // 2x2 RGB pixels, default pixel color = black
        ckb.at<cv::Vec3b>(0, 0) = white;
        ckb.at<cv::Vec3b>(1, 1) = white;

        glTextureStorage2D(ckboard_, 1, GL_RGB8, ckb.cols, ckb.rows);
        glTextureSubImage2D(ckboard_, 0, 0, 0, ckb.cols, ckb.rows, GL_BGR, GL_UNSIGNED_BYTE, ckb.data);
        glTextureParameteri(ckboard_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(ckboard_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(ckboard_, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(ckboard_, GL_TEXTURE_WRAP_T, GL_REPEAT);

    }
    return ckboard_;
}

void Texture::init_chkboard() {
    ckboard_ = gen_ckboard();
}

cv::Mat Texture::load_image(const std::filesystem::path& path) {
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED); // Read with (potential) alpha, do not rotate by EXIF.

    // check! cv::imread does NOT throw exception, if the image is not found.
    if (image.empty()) {
        throw std::runtime_error{ std::string("no texture in file: ").append(path.string()) };
    }
    return image;
}

std::vector<cv::Mat> Texture::load_images(const std::vector<std::filesystem::path>& paths) {
    std::vector<cv::Mat> images;
    for (unsigned int i = 0; i < paths.size(); i++)
    {
        cv::Mat image = cv::imread(paths[i].string(), cv::IMREAD_UNCHANGED);
        if (image.empty())
        {
            std::cerr << "no texture: " << paths[i] << std::endl;
            exit(1);
        }
        images.push_back(image);
    }

    return images;
}


Texture::Texture(const std::filesystem::path& path, Interpolation interpolation) : Texture{ load_image(path), interpolation } {}

Texture::Texture(const std::vector<std::filesystem::path>& paths, Interpolation interpolation) {
    std::vector<cv::Mat> faces = load_images(paths);

    int width = faces[0].cols;
    int height = faces[0].rows;
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &name_);
    glTextureStorage2D(name_, 1, GL_RGB8, width, height);

    for (unsigned int i = 0; i < faces.size(); i++)
    {
        cv::Mat image = faces[i];
        if (!image.empty())
        {
            unsigned char* data = image.data;

            glTextureSubImage3D(
                name_,
                0,
                0, 0, i,
                width, height, 1,
                GL_BGRA,
                GL_UNSIGNED_BYTE,
                data
            );
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
        }
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTextureParameteri(name_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(name_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(name_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(name_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(name_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

Texture::Texture(const glm::vec3& vec) : Texture{ cv::Mat{1, 1, CV_8UC3, cv::Scalar{vec.b, vec.g, vec.r}}, Interpolation::nearest } {}

Texture::Texture(const glm::vec4& vec) : Texture{ cv::Mat{1, 1, CV_8UC4, cv::Scalar{vec.b, vec.g, vec.r, vec.a}}, Interpolation::nearest } {}

Texture::Texture(cv::Mat const& image, Interpolation interpolation)
{
    if (image.empty()) {
        throw std::runtime_error{ "the input image is empty" };
    }

    cv::Mat upload_image = image;
    if (!upload_image.isContinuous()) {
        upload_image = upload_image.clone();
    }

    cv::flip(upload_image, upload_image, 0);  // OpenGL vs. Window coordinates...

    glCreateTextures(GL_TEXTURE_2D, 1, &name_);

    GLint previous_unpack_alignment = 4;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &previous_unpack_alignment);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    switch (upload_image.type()) {
    case CV_8UC1: // single channel image - greyscale
        // upload only one channel
        glTextureStorage2D(name_, 1, GL_R8, upload_image.cols, upload_image.rows);
        glTextureSubImage2D(name_, 0, 0, 0, upload_image.cols, upload_image.rows, GL_RED, GL_UNSIGNED_BYTE, upload_image.data);
        // use data also for other channels
        glTextureParameteri(name_, GL_TEXTURE_SWIZZLE_G, GL_RED);
        glTextureParameteri(name_, GL_TEXTURE_SWIZZLE_B, GL_RED);
        break;
    case CV_8UC3:  // RGB
        glTextureStorage2D(name_, 1, GL_RGB8, upload_image.cols, upload_image.rows);
        glTextureSubImage2D(name_, 0, 0, 0, upload_image.cols, upload_image.rows, GL_BGR, GL_UNSIGNED_BYTE, upload_image.data);
        break;
    case CV_8UC4:  // RGBA
        glTextureStorage2D(name_, 1, GL_RGBA8, upload_image.cols, upload_image.rows);
        glTextureSubImage2D(name_, 0, 0, 0, upload_image.cols, upload_image.rows, GL_BGRA, GL_UNSIGNED_BYTE, upload_image.data);
        break;
    default:
        glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);
        throw std::runtime_error{ "unsupported number of channels or channel depth in texture" };
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);

    set_interpolation(interpolation);

    // Configures the way the texture repeats
    glTextureParameteri(name_, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(name_, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

Texture::~Texture() {
    glDeleteTextures(1, &name_);
}

GLuint Texture::get_name() const {
    return name_;
}

void Texture::bind(int unit) {
    glBindTextureUnit(unit, name_); // bind to some texturing unit, e.g. 0
}

void Texture::set_interpolation(Interpolation interpolation) {
    // Select texture filering method 
    switch (interpolation) {
    case Interpolation::nearest:
        // nearest neighbor - ugly & fast 
        glTextureParameteri(name_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(name_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        break;
    case Interpolation::linear:
        // bilinear - nicer & slower
        glTextureParameteri(name_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(name_, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        break;
    case Interpolation::linear_mipmap_linear:
        // Trilinear: MIPMAP filtering + automatic MIPMAP generation - nicest, needs more memory. Notice: MIPMAP is only for image minifying.
        glTextureParameteri(name_, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // bilinear magnifying
        glTextureParameteri(name_, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // trilinear minifying
        glGenerateTextureMipmap(name_);  // Generate mipmaps now.
        break;
    }
}

int Texture::get_height(void) {
    int tex_height = 0;
    int basemiplevel = 0; // base image
    glGetTextureLevelParameteriv(name_, basemiplevel, GL_TEXTURE_HEIGHT, &tex_height);

    return tex_height;
}

int Texture::get_width(void) {
    int tex_width = 0;
    int basemiplevel = 0; // base image
    glGetTextureLevelParameteriv(name_, basemiplevel, GL_TEXTURE_WIDTH, &tex_width);

    return tex_width;
}

void Texture::replace_image(const cv::Mat& image) {
    // immutable texture format used: only content can be changed (size and data format MUST match)

    cv::Mat upload_image = image;
    if (!upload_image.isContinuous()) {
        upload_image = upload_image.clone();
    }

    // check size
    if ((upload_image.rows != get_height()) || (upload_image.cols != get_width()))
        throw std::runtime_error("improper image replacement size");

    // check channels and format
    int tex_format = 0;
    int basemiplevel = 0; // base image
    glGetTextureLevelParameteriv(name_, basemiplevel, GL_TEXTURE_INTERNAL_FORMAT, &tex_format);

    GLint previous_unpack_alignment = 4;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &previous_unpack_alignment);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    switch (upload_image.type()) {
    case CV_8UC1: // single channel image - greyscale
        if (tex_format != GL_R8) {
            glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);
            throw std::runtime_error("improper image replacement channel data, GL_R8 was the original");
        }
        glTextureSubImage2D(name_, 0, 0, 0, upload_image.cols, upload_image.rows, GL_RED, GL_UNSIGNED_BYTE, upload_image.data);
        break;
    case CV_8UC3:  // RGB
        if (tex_format != GL_RGB8) {
            glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);
            throw std::runtime_error("improper image replacement channel data, GL_RGB8 was the original");
        }
        glTextureSubImage2D(name_, 0, 0, 0, upload_image.cols, upload_image.rows, GL_BGR, GL_UNSIGNED_BYTE, upload_image.data);
        break;
    case CV_8UC4:  // RGBA
        if (tex_format != GL_RGBA8) {
            glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);
            throw std::runtime_error("improper image replacement channel data, GL_RGBA8 was the original");
        }
        glTextureSubImage2D(name_, 0, 0, 0, upload_image.cols, upload_image.rows, GL_BGRA, GL_UNSIGNED_BYTE, upload_image.data);
        break;
    default:
        glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);
        throw std::runtime_error{ "unsupported number of channels or channel depth in texture" };
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, previous_unpack_alignment);
}
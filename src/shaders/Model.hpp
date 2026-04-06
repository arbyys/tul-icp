#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <memory> 

#include <GL/glew.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "Vertex.hpp"
#include "Mesh.hpp"
#include "ShaderProgram.hpp"
#include "Texture.hpp"

class Model 
{
private:
    // origin point of whole model
    glm::vec3 pivot_position{}; // [0,0,0] of the object
    glm::vec3 eulerAngles{};    // pitch, yaw, roll
    glm::vec3 scale{ 1.0f };

    glm::mat4 local_model_matrix{ 1.0 }; //cache, and for complex transformations (default = identity) 

    glm::mat4 createMM(const glm::vec3& origin, const glm::vec3& eAng, const glm::vec3& scale) {
        // keep angles in proper range
        glm::vec3 eA{ wrap_angle(eAng.x), wrap_angle(eAng.y), wrap_angle(eAng.z) };

        glm::mat4 t = glm::translate(glm::mat4(1.0f), origin);
        glm::mat4 rotm = glm::yawPitchRoll(glm::radians(eA.y), glm::radians(eA.x), glm::radians(eA.z)); //yaw, pitch, roll
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * rotm * s;
    }

    float wrap_angle(float angle) { // wrap any float to [0, 360)
        angle = std::fmod(angle, 360.0f);
        if (angle < 0.0f) {
            angle += 360.0f;
        }
        return angle;
    }
public:

    // mesh related data
    struct mesh_package {
        std::shared_ptr<Mesh> mesh;         // geometry & topology, vertex attributes
        std::shared_ptr<ShaderProgram> shader;     // which shader to use to draw this part of the model
        std::shared_ptr<Texture> texture;

        glm::vec3 origin;                   // mesh origin relative to origin of the whole model
        glm::vec3 eulerAngles;              // mesh rotation relative to orientation of the whole model
        glm::vec3 scale;                    // mesh scale relative to scale of the whole model
    };

    std::vector<mesh_package> meshes;

    Model() = default;

    void addMesh(
        std::shared_ptr<Mesh> mesh,
        std::shared_ptr<ShaderProgram> shader,
        std::shared_ptr<Texture> text = std::make_shared<Texture>(),
        glm::vec3 origin = glm::vec3(0.0f),      // dafault value
        glm::vec3 eulerAngles = glm::vec3(0.0f), // dafault value
        glm::vec3 scale = glm::vec3(1.0f)       // dafault value
    ) 
    {
        meshes.emplace_back(mesh, shader, text, origin, eulerAngles, scale);
    }

    void setPosition(const glm::vec3& new_position) {
        pivot_position = new_position;
        local_model_matrix = createMM(pivot_position, eulerAngles, scale);
    }

    const glm::vec3& getPosition() const {
        return pivot_position;
    }

    void setEulerAngles(const glm::vec3& new_eulerAngles) {
        eulerAngles = new_eulerAngles;
        local_model_matrix = createMM(pivot_position, eulerAngles, scale);
    }

    void setScale(const glm::vec3& new_scale) {
        scale = new_scale;
        local_model_matrix = createMM(pivot_position, eulerAngles, scale);
    }

    // for complex (externally provided) transformations 
    void setModelMatrix(const glm::mat4& modelm) {
        local_model_matrix = modelm;
    }

    void translate(const glm::vec3& offset) {
        pivot_position += offset;
        local_model_matrix = createMM(pivot_position, eulerAngles, scale);
    }

    void rotate(const glm::vec3& pitch_yaw_roll_offs) {
        eulerAngles += pitch_yaw_roll_offs;
        local_model_matrix = createMM(pivot_position, eulerAngles, scale);
    }

    void change_scale(const glm::vec3& scale_offs) {
        scale *= scale_offs;
        local_model_matrix = createMM(pivot_position, eulerAngles, scale);
    }

    void draw() {
        // call draw() on mesh (all meshes)
        for (auto const& mesh_pkg : meshes) {
            mesh_pkg.shader->use(); // select proper shader

            glm::mat4 mesh_model_matrix = createMM(mesh_pkg.origin, mesh_pkg.eulerAngles, mesh_pkg.scale);
            mesh_pkg.shader->setUniform("uM_m", mesh_model_matrix * local_model_matrix);
            mesh_pkg.texture->bind();               // NEW: use proper texture
            mesh_pkg.shader->setUniform("tex0", 0); // NEW: send active texture unit to shader
            mesh_pkg.mesh->draw();   // draw mesh
        }
    }
};

#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <GL/glew.h>

#include "Vertex.hpp"

struct OBJMeshPart {
	std::string material_name;
	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;
};

bool loadOBJ(const std::filesystem::path& filename,
	std::vector <Vertex>& vertices,
	std::vector <GLuint>& indices);

bool loadOBJWithMaterials(
	const std::filesystem::path& filename,
	std::vector<OBJMeshPart>& mesh_parts,
	std::vector<std::filesystem::path>& referenced_mtl_files);

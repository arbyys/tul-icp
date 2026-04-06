#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "OBJLoader.hpp"

namespace {
struct FaceToken {
	int vertex_index = 0;
	int uv_index = 0;
	int normal_index = 0;
};

int resolve_obj_index(int raw_index, int size) {
	if (raw_index > 0) {
		return raw_index - 1;
	}
	if (raw_index < 0) {
		return size + raw_index;
	}
	return -1;
}

bool parse_face_token(const std::string& token, FaceToken& out) {
	if (token.empty()) {
		return false;
	}

	out = FaceToken{};
	std::size_t first_slash = token.find('/');
	if (first_slash == std::string::npos) {
		out.vertex_index = std::stoi(token);
		return true;
	}

	std::size_t second_slash = token.find('/', first_slash + 1);
	const std::string v = token.substr(0, first_slash);
	if (!v.empty()) {
		out.vertex_index = std::stoi(v);
	}

	if (second_slash == std::string::npos) {
		const std::string vt = token.substr(first_slash + 1);
		if (!vt.empty()) {
			out.uv_index = std::stoi(vt);
		}
		return true;
	}

	const std::string vt = token.substr(first_slash + 1, second_slash - first_slash - 1);
	const std::string vn = token.substr(second_slash + 1);
	if (!vt.empty()) {
		out.uv_index = std::stoi(vt);
	}
	if (!vn.empty()) {
		out.normal_index = std::stoi(vn);
	}

	return true;
}

Vertex make_vertex(
	const FaceToken& token,
	const std::vector<glm::vec3>& positions,
	const std::vector<glm::vec2>& uvs,
	const std::vector<glm::vec3>& normals)
{
	Vertex vertex{};

	const int p = resolve_obj_index(token.vertex_index, static_cast<int>(positions.size()));
	const int t = resolve_obj_index(token.uv_index, static_cast<int>(uvs.size()));
	const int n = resolve_obj_index(token.normal_index, static_cast<int>(normals.size()));

	if (p >= 0 && p < static_cast<int>(positions.size())) {
		vertex.position = positions[static_cast<std::size_t>(p)];
	}
	if (t >= 0 && t < static_cast<int>(uvs.size())) {
		vertex.tex_coords = uvs[static_cast<std::size_t>(t)];
	}
	if (n >= 0 && n < static_cast<int>(normals.size())) {
		vertex.normal = normals[static_cast<std::size_t>(n)];
	}

	return vertex;
}

std::string trim(const std::string& value) {
	const std::size_t first = value.find_first_not_of(" \t\r\n");
	if (first == std::string::npos) {
		return {};
	}
	const std::size_t last = value.find_last_not_of(" \t\r\n");
	return value.substr(first, last - first + 1);
}
}

bool loadOBJWithMaterials(
	const std::filesystem::path& filename,
	std::vector<OBJMeshPart>& mesh_parts,
	std::vector<std::filesystem::path>& referenced_mtl_files)
{
	mesh_parts.clear();
	referenced_mtl_files.clear();

	std::ifstream file_reader(filename);
	if (!file_reader.is_open()) {
		std::cerr << "[OBJ] Failed to open: " << filename.string() << "\n";
		return false;
	}

	std::cout << "Loading model: " << filename.string() << std::endl;

	std::vector<glm::vec3> positions;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;
	std::unordered_map<std::string, std::size_t> part_index_by_material;

	auto get_part_for_material = [&](const std::string& material_name) -> OBJMeshPart& {
		auto it = part_index_by_material.find(material_name);
		if (it != part_index_by_material.end()) {
			return mesh_parts[it->second];
		}

		mesh_parts.push_back(OBJMeshPart{});
		OBJMeshPart& new_part = mesh_parts.back();
		new_part.material_name = material_name;
		part_index_by_material.emplace(material_name, mesh_parts.size() - 1);
		return new_part;
	};

	std::string current_material = "default";
	std::string line;
	while (std::getline(file_reader, line)) {
		line = trim(line);
		if (line.empty() || line[0] == '#') {
			continue;
		}

		if (line.rfind("v ", 0) == 0) {
			std::istringstream iss(line);
			std::string tag;
			glm::vec3 position{};
			iss >> tag >> position.x >> position.y >> position.z;
			positions.push_back(position);
			continue;
		}

		if (line.rfind("vt ", 0) == 0) {
			std::istringstream iss(line);
			std::string tag;
			glm::vec2 uv{};
			iss >> tag >> uv.y >> uv.x;
			uvs.push_back(uv);
			continue;
		}

		if (line.rfind("vn ", 0) == 0) {
			std::istringstream iss(line);
			std::string tag;
			glm::vec3 normal{};
			iss >> tag >> normal.x >> normal.y >> normal.z;
			normals.push_back(normal);
			continue;
		}

		if (line.rfind("mtllib ", 0) == 0) {
			std::string mtl_file = trim(line.substr(7));
			if (!mtl_file.empty()) {
				referenced_mtl_files.push_back(mtl_file);
			}
			continue;
		}

		if (line.rfind("usemtl ", 0) == 0) {
			const std::string parsed = trim(line.substr(7));
			current_material = parsed.empty() ? "default" : parsed;
			continue;
		}

		if (line.rfind("f ", 0) != 0) {
			continue;
		}

		std::istringstream iss(line.substr(2));
		std::vector<FaceToken> polygon_tokens;
		std::string token;
		while (iss >> token) {
			FaceToken face_token{};
			if (parse_face_token(token, face_token) && face_token.vertex_index != 0) {
				polygon_tokens.push_back(face_token);
			}
		}

		if (polygon_tokens.size() < 3) {
			continue;
		}

		OBJMeshPart& part = get_part_for_material(current_material);

		for (std::size_t i = 1; i + 1 < polygon_tokens.size(); ++i) {
			const Vertex a = make_vertex(polygon_tokens[0], positions, uvs, normals);
			const Vertex b = make_vertex(polygon_tokens[i], positions, uvs, normals);
			const Vertex c = make_vertex(polygon_tokens[i + 1], positions, uvs, normals);

			const GLuint base = static_cast<GLuint>(part.vertices.size());
			part.vertices.push_back(a);
			part.vertices.push_back(b);
			part.vertices.push_back(c);
			part.indices.push_back(base);
			part.indices.push_back(base + 1);
			part.indices.push_back(base + 2);
		}
	}

	mesh_parts.erase(
		std::remove_if(mesh_parts.begin(), mesh_parts.end(), [](const OBJMeshPart& part) {
			return part.vertices.empty() || part.indices.empty();
		}),
		mesh_parts.end());

	if (mesh_parts.empty()) {
		std::cerr << "[OBJ] No drawable geometry in: " << filename.string() << "\n";
		return false;
	}

	std::cout << "Model loaded: " << filename.string() << " (parts: " << mesh_parts.size() << ")" << std::endl;
	return true;
}

bool loadOBJ(const std::filesystem::path& filename, std::vector<Vertex>& vertices, std::vector<GLuint>& indices)
{
	vertices.clear();
	indices.clear();

	std::vector<OBJMeshPart> mesh_parts;
	std::vector<std::filesystem::path> referenced_mtl_files;
	if (!loadOBJWithMaterials(filename, mesh_parts, referenced_mtl_files)) {
		return false;
	}

	for (const OBJMeshPart& part : mesh_parts) {
		const GLuint offset = static_cast<GLuint>(vertices.size());
		vertices.insert(vertices.end(), part.vertices.begin(), part.vertices.end());
		for (GLuint idx : part.indices) {
			indices.push_back(offset + idx);
		}
	}

	return !vertices.empty() && !indices.empty();
}

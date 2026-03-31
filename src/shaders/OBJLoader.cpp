#include <string>
#include <algorithm>
#include <GL/glew.h> 
#include <glm/glm.hpp>
#include <iostream>
#include <fstream>
#include <istream>

#include "OBJloader.hpp"

#define MAX_LINE_SIZE 1024

bool read_all_lines(const std::filesystem::path& filename, std::vector<std::string>& buffer)
{
	// clear buffer before reading file content into it
	buffer.clear();
	std::string line;
	std::ifstream file_reader(filename);
	while (getline(file_reader, line)) {
		buffer.push_back(line);
	}
	file_reader.close();

	return true;
}

bool loadOBJ(const std::filesystem::path& filename, std::vector<Vertex>& vertices, std::vector<GLuint>& indices)
{
	std::cout << "Loading model: " << filename.string() << std::endl;

	std::vector<unsigned int> vertex_indices, uv_indices, normal_indices;

	// temporary buffers
	std::vector<glm::vec3> temp_vertices;
	std::vector<glm::vec2> temp_uvs;
	std::vector<glm::vec3> temp_normals;

	// clear vertices and indices buffers before writing to them
	vertices.clear();
	indices.clear();

	// read file data into buffer
	std::vector<std::string> file_contents;
	read_all_lines(filename, file_contents);

	for (const std::string& line : file_contents)
	{
		// skip empty lines
		if (line.empty())
		{
			continue;
		}

		// start of the line identifying the record type
		std::string recordId = line.substr(0, 2);
		// v - vertex record
		if (recordId == "v ") {
			glm::vec3 vertex{};
			sscanf_s(line.c_str(), "v %f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			temp_vertices.push_back(vertex);
		}
		// vt - vertex texture (uv) coordinate record
		else if (recordId == "vt") {
			glm::vec2 uv{};
			sscanf_s(line.c_str(), "vt %f %f\n", &uv.y, &uv.x);
			temp_uvs.push_back(uv);
		}
		// vn - vertex normal vector record
		else if (recordId == "vn") {
			glm::vec3 normal{};
			sscanf_s(line.c_str(), "vn %f %f %f\n", &normal.x, &normal.y, &normal.z);
			temp_normals.push_back(normal);
		}
		// f - the way [v, [vt, vn]] are connected into triangles
		else if (recordId == "f ") {
			// split into cases - dependent on format of the lines
			auto containsDoubleSlash = line.find("//") != std::string::npos;
			auto slashCount = std::count(line.begin(), line.end(), '/');
			//auto doubleSlashCount = std::count(line.begin(), line.end(), '//');
			// order of indexing - vertexes, uvs, normals
			// [v v v v uv uv uv uv un un un un]
			unsigned int indices[12]{};

			// case 1
			// vertex and normal vector
			// f %d//%d %d//%d %d//%d
			// f  1//2   4//3   3//5
			if (slashCount == 6 && containsDoubleSlash)
			{
				(void)sscanf_s(line.c_str(), "f %d//%d %d//%d %d//%d\n", &indices[0], &indices[4], &indices[1], &indices[5], &indices[2], &indices[6]);
				vertex_indices.insert(vertex_indices.end(), { indices[0], indices[1], indices[2] });
				normal_indices.insert(normal_indices.end(), { indices[4], indices[5], indices[6] });
			}

			// case 2
			// vertex only
			// f %d %d %d
			// f  1  2  3
			else if (slashCount == 0)
			{
				(void)sscanf_s(line.c_str(), "f %d %d %d\n", &indices[0], &indices[1], &indices[2]);
				vertex_indices.insert(vertex_indices.end(), { indices[0], indices[1], indices[2] });
			}

			// case 3
			// vertex and texture coord
			// f %d/%d %d/%d %d/%d
			// f  1/2   3/4   4/2
			else if (slashCount == 3)
			{
				sscanf_s(line.c_str(), "f %d/%d %d/%d %d/%d\n", &indices[0], &indices[8], &indices[1], &indices[9], &indices[2], &indices[10]);
				vertex_indices.insert(vertex_indices.end(), { indices[0], indices[1], indices[2] });
				uv_indices.insert(uv_indices.end(), { indices[8], indices[9], indices[10] });
			}

			// case 4
			// vertex, texture coord and normal
			// f %d/%d/%d %d/%d/%d %d/%d/%d
			// f  1/1/1    1/1/1    1/1/1
			else if (slashCount == 6)
			{
				sscanf_s(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d\n", &indices[0], &indices[4], &indices[8], &indices[1], &indices[5], &indices[9], &indices[2], &indices[6], &indices[10]);
				vertex_indices.insert(vertex_indices.end(), { indices[0], indices[1], indices[2] });
				uv_indices.insert(uv_indices.end(), { indices[4], indices[5], indices[6] });
				normal_indices.insert(normal_indices.end(), { indices[8], indices[9], indices[10] });
			}

			// case 5
			// vertex and normal vector, square
			// f %d//%d %d//%d %d//%d %d//%d
			// f  1//2   4//3   3//5   5//3
			else if (slashCount == 8 && containsDoubleSlash)
			{
				sscanf_s(line.c_str(), "f %d//%d %d//%d %d//%d %d\\%d\n", &indices[0], &indices[4], &indices[1], &indices[5], &indices[2], &indices[6], &indices[3], &indices[7]);
				vertex_indices.insert(vertex_indices.end(), { indices[0], indices[1], indices[2], indices[0], indices[2], indices[3] });
				normal_indices.insert(normal_indices.end(), { indices[4], indices[5], indices[6], indices[4], indices[6], indices[7] });
			}
			// TODO other 3 square methods

			// case 6
			// vertex and texture coord, square
			// f %d %d %d %d
			// f  1  2  3  4
			else if (slashCount == 4)
			{
				sscanf_s(line.c_str(), "f %d/%d %d/%d %d/%d %d/%d\n", &indices[0], &indices[8], &indices[1], &indices[9], &indices[2], &indices[10], &indices[3], &indices[11]);
				vertex_indices.insert(vertex_indices.end(), { indices[0], indices[1], indices[2], indices[0], indices[2], indices[3] });
				uv_indices.insert(uv_indices.end(), { indices[8], indices[9], indices[10], indices[8], indices[10], indices[11] });
			}
			// case 7
			// vertex, texture coord and normal, square
			// f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d
			// f  1/1/1     2/2/2    3/3/3    4/4/4
			else if (slashCount == 8)
			{
				sscanf_s(line.c_str(), "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n", &indices[0], &indices[4], &indices[8], &indices[1], &indices[5], &indices[9], &indices[2], &indices[6], &indices[10], &indices[3], &indices[7], &indices[11]);
				vertex_indices.insert(vertex_indices.end(), { indices[0], indices[1], indices[2], indices[0], indices[2], indices[3] });
				uv_indices.insert(uv_indices.end(), { indices[4], indices[5], indices[6], indices[4], indices[6], indices[7] });
				normal_indices.insert(normal_indices.end(), { indices[8], indices[9], indices[10], indices[8], indices[10], indices[11] });
			}


			else
			{
				std::cout << "bruh" << std::endl;
				continue;
			}

		}
	}

	if (vertex_indices.size() != normal_indices.size())
	{
		throw std::exception("Non-matching sizes of indices");
	}

	// get number of triangles
	auto triangle_count = vertex_indices.size();

	//
	for (unsigned int u = 0; u < triangle_count; u++) {
		Vertex vertex{};
		vertex.position = temp_vertices[vertex_indices[u] - 1];
		vertex.tex_coords = temp_uvs[uv_indices[u] - 1];
		vertex.normal = temp_normals[normal_indices[u] - 1];
		vertices.push_back(vertex);
		indices.push_back(u);
	}

	// Print done
	std::cout << "LoadOBJFile: Loaded OBJ file " << filename;

	std::cout << "Model loaded: " << filename.string() << std::endl;

	return true;
}

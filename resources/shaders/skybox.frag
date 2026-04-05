#version 460 core

out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube cubemap; // cubemap texture sampler

void main()
{             
    FragColor = vec4(vec3(texture(cubemap, TexCoords)), 0);
}  
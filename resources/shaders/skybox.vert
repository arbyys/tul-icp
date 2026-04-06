#version 460 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 uP_m;
uniform mat4 uV_m;

void main()
{
    TexCoords = aPos;
    gl_Position = uP_m * uV_m * vec4(aPos, 1.0);
}  
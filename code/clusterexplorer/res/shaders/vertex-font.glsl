// vertex shader
#version max_version

in mediump vec3 i_position;
in mediump vec3 i_normal;
in mediump vec4 i_color;
in mediump vec2 i_tcoord;

out mediump vec2 fs_tcoord;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform float offset;

void main() {
  fs_tcoord = i_tcoord;

  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(i_position.x + offset, i_position.y, i_position.z, 1);
}



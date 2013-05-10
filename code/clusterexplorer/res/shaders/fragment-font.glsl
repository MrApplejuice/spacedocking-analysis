// fragment shader
#version max_version

in mediump vec2 fs_tcoord;

uniform mediump vec4 fs_refColor;
uniform sampler2D fs_texture;

void main() {
  gl_FragColor = texture(fs_texture, vec2(fs_tcoord[0], 1.0 - fs_tcoord[1]), -2.0); // bias of 2 to prevent horrible mipmaps from being used
  if (gl_FragColor.a == 0.0) {
    discard;
  }
  gl_FragColor = gl_FragColor * fs_refColor;
}

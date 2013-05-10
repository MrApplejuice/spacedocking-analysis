/* Includes opengl in a platform indepenent way
*/

#if ((defined TARGET_IPHONE) || (defined TARGET_IPHONE_SIMULATOR))
  #ifndef USE_OPENGLES
    #define USE_OPENGLES 100
  #endif
  
  #if (USE_OPENGLES >= 200)
    // Redfine OES functions to "normal" functions
    //#define glGenerateMipmap glGenerateMipmapOES
    #include <OpenGLES/ES2/gl.h>
    #include <OpenGLES/ES2/glext.h>
    #include <glu.h>
  #else
    #include <OpenGLES/ES1/gl.h>
    #include <OpenGLES/ES1/glext.h>
    #include <glu.h>
  #endif
#else
  #ifndef OPENGL_VERSION
    #define OPENGL_VERSION 120
  #endif

  #if defined (__APPLE__)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <OpenGL/glext.h>
  #elif (defined TARGET_WINDOWS)
    #include <GL/glew.h>
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glext.h>
  #else
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glext.h>
  #endif
#endif // IPHONE/IPHONE_SIMULATOR

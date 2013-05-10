#if ((defined USE_OPENGLES) && (USE_OPENGLES >= 200) || (defined OPENGL_VERSION) && (OPENGL_VERSION >= 330))
#define USE_GL33_GLES20_CODE
#else
#undef USE_GL33_GLES20_CODE
#endif

#if ((defined USE_OPENGLES) && (USE_OPENGLES >= 200))
#define USE_GLES20_CODE
#else
#undef USE_GLES20_CODE
#endif

#if ((defined OPENGL_VERSION) && (OPENGL_VERSION >= 330))
#define USE_GL33_CODE
#else
#undef USE_GL33_CODE
#endif

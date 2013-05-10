#include "DynamicTexture.hpp"

using namespace std;
using namespace boost;

namespace engine {
  DynamicTexture :: DynamicTexture(size_t width, size_t height, GLuint type) : glTexture(0), type(type), width(width), height(height), dataLoaded(false) {
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Called " << __FUNCTION__ << " with uncleard glerror" << endl;
    }
    
    glGenTextures(1, &glTexture);
    if (glGetError() != GL_NO_ERROR) {
      throw "Cannot create texture";
    }
  }

  void DynamicTexture :: loadTexture(const void* data, GLenum dataPacking) {
    assert(data);
    
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Called " << __FUNCTION__ << " with uncleard glerror" << endl;
    }
    
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, type, width, height, 0, type, dataPacking, data);
    if (glGetError() == GL_NO_ERROR) {
      dataLoaded = true;
    } else {
      cerr << "Error: Could not load dynamic texture data (" << __FUNCTION__ << ")" << endl;
    }
  }

  void DynamicTexture :: loadTexture(const void* data, GLenum dataPacking, size_t offsetx, size_t offsety, size_t width, size_t height) {
    assert(data);

    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Called " << __FUNCTION__ << " with uncleard glerror" << endl;
    }

    if (!dataLoaded) {
      cerr << "Error: Partial texture data cannot be loaded until whole texture has been loaded once" << endl;
      return;
    }
    
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, offsetx, offsety, width, height, type, dataPacking, data);
    if (glGetError() == GL_NO_ERROR) {
      dataLoaded = true;
    } else {
      cerr << "Error: Could not load dynamic texture data (" << __FUNCTION__ << ")" << endl;
    }
  }

  void DynamicTexture :: resize(size_t width, size_t height) {
    this->width = width;
    this->height = height;
    this->dataLoaded = false;
  }
  
  void DynamicTexture :: changeDataType(GLenum datatype) {
    this->type = datatype;
    this->dataLoaded = false;
  }
  
  DynamicTexture :: ~DynamicTexture() {
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Called " << __FUNCTION__ << " with uncleard glerror" << endl;
    }

    glDeleteTextures(1, &glTexture);
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Error! Could not unload dynamic texture" << endl;
    }
  }

  DynamicTextureRef DynamicTexture :: create(size_t width, size_t height, GLenum type) {
    try {
      return DynamicTextureRef(new DynamicTexture(width, height, type));
    }
    catch (const char* e) {
      cerr << "Error while creating dynamic texture: " << e << endl;
      return DynamicTextureRef();
    }
  }
} // namespace engine

#include "TextureManager.hpp"

#include <iostream>

using namespace std;
using namespace boost;

namespace engine {
  Texture :: Texture(const std::string& name, GLuint texture, int width, int height) {
    this->name = name;
    this->glTexture = texture;
    this->width = width;
    this->height = height;
  }
  
  Texture :: ~Texture() {
    glDeleteTextures(1, &glTexture);
  }


  TextureRef TextureManager :: getTexture(const std::string& textureName) {
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Uncleared glError while calling " << __FUNCTION__ << endl;
    }
    
    string fullFilename = textureName;
    if (textureName.empty() || (textureName[0] != '/')) {
      fullFilename = resourcePrefix + fullFilename;
    } else {
      fullFilename = fullFilename.substr(1);
    }

    TextureRef result;

    NameTextureMap::const_iterator match = textures.find(fullFilename);
    if (match == textures.end()) {
#ifndef USE_GL33_GLES20_CODE
      bool prevTexEnabled = glIsEnabled(GL_TEXTURE_2D);
#endif

      RawImageData imageData;
      GLuint tex;

#ifndef USE_GL33_GLES20_CODE
      glEnable(GL_TEXTURE_2D);
#endif
      
      glGenTextures(1, &tex);
      
      try {
        if (glGetError() != GL_NO_ERROR) {
          throw "Could not generate gl texture";
        }
      
        glBindTexture(GL_TEXTURE_2D, tex);

        // First attempt the native load
        bool nativeSucceeded = backendRef->nativeLoadTex2D(fullFilename, imageData.width, imageData.height);
        if (!nativeSucceeded) {
          imageData = backendRef->loadImage(fullFilename);

          if ((imageData.width * imageData.height > 0) && (imageData.pixels.get() == NULL)) {
            throw string("Could not load texture: ") + textureName;
          }
          
#ifndef USE_GL33_GLES20_CODE
          glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
#endif
          
          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  //          gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, imageData.width, imageData.height, GL_RGBA, GL_UNSIGNED_BYTE, imageData.pixels.get()); "old version"
          glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageData.width, imageData.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData.pixels.get());

#ifdef USE_GL33_GLES20_CODE
          glGenerateMipmap(GL_TEXTURE_2D);
          if (glGetError() != GL_NO_ERROR) {
            cout << "Warning! Could not generate mipmaps for texture " << textureName << endl;
          }
#else
          if (glGetError() != GL_NO_ERROR) {
            throw "Could not load texture int memory";
          }
          glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_FALSE);
#endif

          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

          if (glGetError() != GL_NO_ERROR) {
            throw "Cannot set texture parameters";
          }
        }

        result = TextureRef(new Texture(textureName, tex, imageData.width, imageData.height)); 
        textures[fullFilename] = result;
      }
      catch (const string& e) {
        glDeleteTextures(1, &tex);
        cerr << e << ": " << textureName << endl;
      }
      catch (const char* e) {
        glDeleteTextures(1, &tex);
        cerr << e << ": " << textureName << endl;
      }
      
      glBindTexture(GL_TEXTURE_2D, 0);
#ifndef USE_GL33_GLES20_CODE
      if (!prevTexEnabled) {
        glDisable(GL_TEXTURE_2D);
      }
#endif
    } else {
      result = match->second;
    }
    return result;
  }

  TextureManager :: TextureManager(const SystemBackendRef& backendRef, std::string resourcePrefix) : resourcePrefix(resourcePrefix), backendRef(backendRef) {
  }
}

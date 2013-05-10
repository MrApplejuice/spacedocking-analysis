#ifndef TEXTUREMANAGER_HPP_
#define TEXTUREMANAGER_HPP_

#include <map>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>

#include <ClassTools.hpp>
#include <OpenGL.hpp>

#include "SystemBackend.hpp"

#include "OpenGLESDetectionHeader.hpp"

namespace engine {
  class AbstractTexture : public virtual boost::noncopyable {
    public:
      GEN_ABSTRACT_GETTER(GLuint, Texture);
      GEN_ABSTRACT_GETTER(int, Width);
      GEN_ABSTRACT_GETTER(int, Height);
  };
  typedef boost::shared_ptr<AbstractTexture> TextureRef;
  
  class Texture : public virtual AbstractTexture {
    private:
      std::string name;
      int width, height;
      GLuint glTexture;
    public:
      GEN_GETTER(const std::string&, Name, name);
      GEN_GETTER(GLuint, Texture, glTexture);
      GEN_GETTER(int, Width, width);
      GEN_GETTER(int, Height, height);
      
      Texture(const std::string& name, GLuint texture, int width, int height);
      virtual ~Texture();
  };
  
  typedef std::map<std::string, TextureRef> NameTextureMap;
  class TextureManager : boost::noncopyable {
    private:
      std::string resourcePrefix;
    
      SystemBackendRef backendRef;

      NameTextureMap textures;
    public:
      /**
       * Loads a texture from the specified resource.
       * 
       * @param textureName
       *   The filename of the texture to load. The texture is
       *   searched in the res/images/ directory. An exception of this
       *   rule accounts for filenames that start with an '/'. Such files
       *   are searched in the bunde base directory (usually
       *   the place where the executable resides).
       * 
       * @result
       *   If the texture was loaded sucessfully, the result is a 
       *   reference to an intialized texture object. Otherwise a
       *   NULL reference is returned.
       */
      virtual TextureRef getTexture(const std::string& textureName);
    
      TextureManager(const SystemBackendRef& backendRef, std::string resourcePrefix);
  };
}

#endif // TEXTUREMANAGER_HPP_

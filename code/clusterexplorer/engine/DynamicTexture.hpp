#ifndef DYNAMICTEXTURE_HPP_
#define DYNAMICTEXTURE_HPP_

#include <boost/shared_ptr.hpp>

#include "TextureManager.hpp"

namespace engine {
  class DynamicTexture;
  typedef boost::shared_ptr<DynamicTexture> DynamicTextureRef;
  
  /**
   * A dynamic texture is a texture that is not created from a file, but
   * loaded dynmaically via its "loadTexture" function. A dynamic texture
   * also does not autoamtically generate mipmaps of the new data, 
   * allowing it to be used for movies or other means where fast changing
   * pixel information needs to be displayed. A dynamic texture is 
   * compatible with standard TextureRef references.
   */
  class DynamicTexture : public virtual AbstractTexture {
    private:
      GLuint glTexture;

      GLenum type;
      size_t width, height;
      
      bool dataLoaded;
    
      DynamicTexture(size_t width, size_t height, GLenum type);
    public:
      GEN_GETTER(GLuint, Texture, glTexture);
      GEN_GETTER(int, Width, width);
      GEN_GETTER(int, Height, height);
      
      GEN_GETTER(GLenum, Type, type);
      
      /**
       * Loads the texture with data using glTexImage2D. The data parameter
       * is assumed to point to a memory area containing enough data to
       * fill the entire texture.
       * 
       * @param data
       *   Pixel data in the desired format, with packed rows. May not 
       *   be NULL
       * @param dataPacking
       *   Specifies how the pixel data is been packed. Corresponds to
       *   the type parameter of glTexImage
       */
      virtual void loadTexture(const void* data, GLenum dataPacking);

      /**
       * Loads a part of the texture iwth data using glTexSubImage2D. 
       * The data parameter is assumed to point to a memory area 
       * containing enough data to fill the desired texture area.
       * 
       * @param data
       *   Pixel data with packed rows of width given as paramater. May
       *   not be NULL
       * @param dataPacking
       *   Specifies how the pixel data is been packed. Corresponds to
       *   the type parameter of glTexImage
       * @param offsetx
       *   The x-offset to copy the subimage to
       * @param offsety
       *   The y-offset to copy the subimage to
       * @param width
       *   Width of the image part to copy
       * @param height
       *   Height of the image part to copy
       */
      virtual void loadTexture(const void* data, GLenum dataPacking, size_t offsetx, size_t offsety, size_t width, size_t height);
      
      /**
       * Changes the size of this texture. After a call to this function 
       * the content of the texture are undefined and the whole
       * texture needs to be reloaded with data before doing anything
       * else.
       * 
       * @param width
       *   New width of the texture
       * @param height
       *   New height of the texture
       */
      virtual void resize(size_t width, size_t height);

      /**
       * Changes the texture storage format. Texture contents are 
       * undefined after calling this function and need to be reloaded
       * using loadTexture.
       * 
       * @param datatype
       *   New data format of the color data that is stored in the 
       *   texture (see internalFormat/format parameter for glTexImage2D).
       */
      virtual void changeDataType(GLenum datatype);
      
      virtual ~DynamicTexture(); ///< destructor
      
      /**
       * Creates a new dynamic texture with given size and of given storage
       * type.
       * 
       * @param width
       *   Width of the new dynamic texture
       * @param height
       *   Height of the new dynamic texture
       * @param type
       *   Storage tyoe of the new dynamic texture. Valid values are the 
       *   as for the internalFormat/format parameter for glTexImage2D
       * @return
       *   On success a new shared_ptr to the create dynamic texture. A
       *   NULL reference when an error ocurred during creation.
       */
      static DynamicTextureRef create(size_t width, size_t height, GLenum type);
  };
}

#endif // DYNAMICTEXTURE_HPP_

#ifndef GLFONT_HPP_
#define GLFONT_HPP_

#include <vector>
#include <string>
#include <climits>
#include <stdint.h>

#include <boost/exception/all.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/function.hpp>

#include "SystemBackend.hpp"
#include "TextureManager.hpp"
#include "Mesh.hpp"

#include "libs/glm/glm.hpp"

namespace engine {
  class GLFont;
  typedef boost::shared_ptr<GLFont> GLFontRef;

  typedef boost::error_info<struct tag_my_info, const std::string> GLFontExceptionData;
  struct GLFontException : public boost::exception, public std::exception {};

  class GLFont {
    private:
      MeshRef letterMeshes[CHAR_MAX - CHAR_MIN + 1];
      float letterWidths[CHAR_MAX - CHAR_MIN + 1];
      
      engine::SystemBackendRef backendRef;

      TextureRef fontTexture;
    public:
      typedef boost::function1<void, const glm::vec4&> SetupColorDrawFunction;
      typedef boost::function1<void, TextureRef> SetupFontDrawFunction;
      typedef boost::function1<void, float> AdvanceXFunction;
      typedef boost::function0<void> ResetFontDrawFunction;
      typedef boost::function0<void> ResetColorDrawFunction;
    
      static SetupColorDrawFunction SetupColorDrawDefaultFunction;
      static SetupFontDrawFunction SetupFontDrawDefaultFunction;
      static AdvanceXFunction AdvanceXDefaultFunction;
      static ResetFontDrawFunction ResetFontDrawDefaultFunction;
      static ResetColorDrawFunction ResetColorDrawDefaultFunction;
    
      virtual void drawString(const std::string& str) const;
      virtual void drawStringWithColor(const std::string& str, uint32_t color) const;
      virtual void drawStringWithColor(const std::string& str, const glm::vec4& color) const;
      virtual float getStringWidth(const std::string& str) const;
    
      GLFont(engine::SystemBackendRef backendRef, TextureManager& textureManager, const std::string& name);
  };
}

#endif // GLFONT_HPP_

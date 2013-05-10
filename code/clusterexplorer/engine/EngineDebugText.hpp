#ifndef ENGINEDEBUGTEXT_HPP_
#define ENGINEDEBUGTEXT_HPP_

#include <string>
#include <vector>
#include <stdint.h>

#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include "GLFont.hpp"

namespace engine {
  struct DebugTextValuePair {
    std::string name, value;
  };
  typedef std::vector<DebugTextValuePair> DebugTextValuePairVector;

  class DebugText {
    private:
      float aspect;
      
      DebugTextValuePairVector debugValues;
      
      virtual void drawLine(const std::string& s, int lno);
    public:
      typedef boost::function0<void> ResetOriginFunc;
      typedef boost::function1<void, float> InitDrawFunc;
      typedef boost::function2<void, float, float> ScaleOriginFunc;
      typedef boost::function2<void, float, float> MoveOriginFunc;
      typedef boost::function3<void, GLFontRef, const std::string&, uint32_t> DrawTextFunc;
      typedef boost::function0<void> ResetDrawFunc;
    
      static ResetOriginFunc ResetOrigin;
      static InitDrawFunc InitDraw;
      static ScaleOriginFunc ScaleOrigin;
      static MoveOriginFunc MoveOrigin;
      static DrawTextFunc DrawText;
      static ResetDrawFunc ResetDraw;
    
      float offsetx, offsety;
    
      engine::GLFontRef font;
      std::string text;
    
      void append(const std::string& s);

      template <typename T>
      void value(const std::string& name, T v) {
        using namespace std;
        using namespace boost;
        
        // Conversion via streaming operator with precision for floats
        string value;
        {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(3);
          oss << v;
          value = oss.str();
        }

        BOOST_FOREACH(DebugTextValuePair& vp, debugValues) {
          if (vp.name == name) {
            vp.value = value;
            return;
          }
        }
        
        DebugTextValuePair valuePair;
        valuePair.name = name;
        valuePair.value = value;
        debugValues.push_back(valuePair);
      }

      virtual void clear();

      virtual void draw();
    
      DebugText(float aspect);
  };
}

#endif // ENGINEDEBUGTEXT_HPP_

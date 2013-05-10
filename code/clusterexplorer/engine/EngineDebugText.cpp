#include "EngineDebugText.hpp"

#include <boost/foreach.hpp>
#include <boost/bind.hpp>

#include <OpenGL.hpp>

#include "EngineWarnings.hpp"

#define foreach BOOST_FOREACH

using namespace std;
using namespace boost;

#define MAX(x, y) (((x) < (y)) ? (y) : (x))

namespace engine {
  class OpenGL12TextRenderer {
    private:
    public:
      void initDraw(float aspectRatio) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glScalef(1 / aspectRatio, 1, 1);
        glMatrixMode(GL_MODELVIEW);
      }
      
      void resetOrigin() {
        glLoadIdentity();
      }
      
      void scaleOrigin(float dx, float dy) {
        glScalef(dx, dy, 1);
      }
      
      void moveOrigin(float x, float y) {
        glTranslatef(x, y, 0);
      }
      
      void drawText(GLFontRef font, const std::string& s, uint32_t color) {
        font->drawStringWithColor(s, color);
      }
      
      void resetDraw() {
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC0_RGB, GL_PREVIOUS);
        GLfloat resetcolor[4] = {1, 1, 1, 1};
        glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, resetcolor);
      }

      OpenGL12TextRenderer() {
      }
  };
  static OpenGL12TextRenderer gOpenGL12TextRenderer;

  DebugText::InitDrawFunc DebugText :: InitDraw = bind<void>(&OpenGL12TextRenderer::initDraw, &gOpenGL12TextRenderer, _1);
  DebugText::ResetOriginFunc DebugText :: ResetOrigin = bind<void>(&OpenGL12TextRenderer::resetOrigin, &gOpenGL12TextRenderer);
  DebugText::ScaleOriginFunc DebugText :: ScaleOrigin = bind<void>(&OpenGL12TextRenderer::scaleOrigin, &gOpenGL12TextRenderer, _1, _2);
  DebugText::MoveOriginFunc DebugText :: MoveOrigin = bind<void>(&OpenGL12TextRenderer::moveOrigin, &gOpenGL12TextRenderer, _1, _2);
  DebugText::DrawTextFunc DebugText :: DrawText = bind<void>(&OpenGL12TextRenderer::drawText, &gOpenGL12TextRenderer, _1, _2, _3);
  DebugText::ResetDrawFunc DebugText :: ResetDraw = bind<void>(&OpenGL12TextRenderer::resetDraw, &gOpenGL12TextRenderer);

  void DebugText :: drawLine(const string& s, int lno) {
    MoveOrigin(0, -lno - 1);
    DrawText(font, s, 0xFFFF0000);
  }

  void DebugText :: append(const string& s) {
    text += s + "\n";
  }

  void DebugText :: clear() {
    text = "";
    debugValues.clear();
  }

  void DebugText :: draw() {
    if (!text.empty() || !debugValues.empty()) {
      if (!font) {
        ENGINE_WARN_ONCE("No font for debug text specified - no debug HUD available");
      } else {
        InitDraw(aspect);
      
        unsigned int lineIndex = 0;
        
        // Nicely format values-list first
        {
          float block1width = 0, block2width = 0;
          
          if (debugValues.size() > 0) {
            float spaceWidth = font->getStringWidth(" ");
            
            foreach (DebugTextValuePair vtx, debugValues) {
              block1width = MAX(font->getStringWidth(vtx.name + ":"), block1width);
              block2width = MAX(font->getStringWidth(vtx.value), block2width);
            }
            
            for (unsigned int i = 0; i < debugValues.size(); i++) {
              ResetOrigin();
              MoveOrigin(offsetx, offsety);
              MoveOrigin(-aspect, 1);
              ScaleOrigin(0.15, 0.15);
              drawLine(debugValues[i].name + ":", i);

              ResetOrigin();
              MoveOrigin(offsetx, offsety);
              MoveOrigin(-aspect, 1);
              ScaleOrigin(0.15, 0.15);
              MoveOrigin(spaceWidth + block1width, 0);

              drawLine(debugValues[i].value, i);
            }
            
            lineIndex += debugValues.size();
          }
          
          // Draw free text
          string& ltext = text;
          if (!ltext.empty()) {
            vector<string> lines;
            split(lines, ltext, is_any_of("\n"), token_compress_off);
            foreach (string& line, lines) {
              if (!line.empty()) {
                ResetOrigin();
                MoveOrigin(offsetx, offsety);
                MoveOrigin(-aspect, 1);
                ScaleOrigin(0.15, 0.15);
                drawLine(line, lineIndex);
              }
              lineIndex++;
            }
          }
        }

        ResetDraw();
      }
    }
  }

  DebugText :: DebugText(float aspect) : aspect(aspect), debugValues(), offsetx(0), offsety(0), font(), text() {
  }
}

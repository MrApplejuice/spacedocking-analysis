#include "GLFont.hpp"

#include <iostream>
#include <sstream>

#include <boost/lexical_cast.hpp>

#include <OpenGL.hpp>

#include "WavefrontMeshDataProvider.hpp"
#include "EditableMesh.hpp"

#include "local/Global.hpp"

using boost::lexical_cast;

using namespace engine;

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

namespace engine {
  struct WidthPair {
    char letter;
    int width;
  };
  typedef vector<WidthPair> WidthPairVector;
  
  struct WidthsFileContent {
    string texture;
    WidthPairVector widths;
  };
  
  typedef boost::error_info<struct tag_my_data, const string> ConversionExceptionData;
  struct ConversionException : public boost::exception, public std::exception {};
  
  static WidthsFileContent parseWidthsFile(std::istream& in) {
    WidthsFileContent result;
    
    while (!in.eof()) {
      if (in.fail()) {
        throw ConversionException() << ConversionExceptionData(string("Failed to read from input stream"));
      }
      
      string line, trimmedline;
      getline(in, line);
      trimmedline = boost::trim_copy(line);
      
      if (trimmedline.size() > 0) {
        if (trimmedline[0] != '#') {
          // Process content
          if (result.texture.size() == 0) {
            result.texture = trimmedline;
          } else {
            try {
              vector<string> splits;
              boost::trim_right(line);
              boost::split(splits, line, boost::is_any_of("\t"), boost::token_compress_on);
              
              if (splits.size() != 2) {
                throw ConversionException() << ConversionExceptionData(string("Invalid amount of \\t columns: ") + line);
              }
              if (splits[0].size() != 1) {
                throw ConversionException() << ConversionExceptionData(string("Expected single char in line: ") + line);
              }
              
              WidthPair pair;
              pair.letter = splits[0][0];
              pair.width = lexical_cast<int>(splits[1]);
              
              result.widths.push_back(pair);
            }
            catch (boost::bad_lexical_cast) {
              throw ConversionException() << ConversionExceptionData(string("Invalid line: ") + line);
            }
          }
        }
      }
    }
    
    return result;
  }
  
  static MeshRef createFontMesh(const WidthsFileContent& content, TextureRef texture) {
    const int textureWidth = texture->getWidth();

    EditableMesh eMesh;
    EditableVertex eVertex;
    
    eVertex.setNormal(EditableVertex::Vector3D(0, 0, 1));
    
    const int gridEdgeLen = (int) ceil(sqrt(content.widths.size()));
    const float relGridDelta = 1.0 / (float) gridEdgeLen;
    int gridX = 0;
    int gridY = 0;
    
    EditableMesh::MaterialTriangleGroupRef materialGroup = eMesh.addMaterialGroup(EditableMesh::MaterialTriangleGroup::create(eMesh, "material"));
    materialGroup->material.setColor(1, 1, 1, 1);
    materialGroup->material.setTexture(0, texture);
    
    for (WidthPairVector::const_iterator it = content.widths.begin(); it != content.widths.end(); it++) {
      EditableMesh::EditableMeshVertexRef emRect[4];
      
      float relGridX = gridX * relGridDelta;
      float relGridY = gridY * relGridDelta;
      float srgDelta = (float) it->width / (float) textureWidth;
      float pixelHeight = 2.0 / (float) textureWidth;

      eVertex.setPosition(EditableVertex::Vector3D(0, 0, 0))
             .setTextureCoordinate(0, EditableVertex::TextureCoodinate(relGridX, 1.0 - (relGridY + relGridDelta - pixelHeight)));
      emRect[0] = eMesh.addVertex(eVertex);       

      eVertex.setPosition(EditableVertex::Vector3D(0, 1, 0))
             .setTextureCoordinate(0, EditableVertex::TextureCoodinate(relGridX, 1.0 - relGridY));
      emRect[1] = eMesh.addVertex(eVertex);       

      eVertex.setPosition(EditableVertex::Vector3D((float) it->width * (float) gridEdgeLen / (float) textureWidth, 0, 0))
             .setTextureCoordinate(0, EditableVertex::TextureCoodinate(relGridX + srgDelta, 1.0 - (relGridY + relGridDelta - pixelHeight)));
      emRect[2] = eMesh.addVertex(eVertex);       

      eVertex.setPosition(EditableVertex::Vector3D((float) it->width * (float) gridEdgeLen / (float) textureWidth, 1, 0))
             .setTextureCoordinate(0, EditableVertex::TextureCoodinate(relGridX + srgDelta, 1.0 - relGridY));
      emRect[3] = eMesh.addVertex(eVertex);       

      EditableMesh::EditableMeshTriangleRef t1 = eMesh.addTriangle(emRect[0], emRect[2], emRect[1]); 
      EditableMesh::EditableMeshTriangleRef t2 = eMesh.addTriangle(emRect[1], emRect[2], emRect[3]); 

      EditableMesh::TriangleGroupRef triangleGroup = eMesh.addTriangleGroup(EditableMesh::TriangleGroup::create(eMesh, lexical_cast<string>((int) it->letter)));
      triangleGroup->addTriangle(t1);
      triangleGroup->addTriangle(t2);
      materialGroup->addTriangle(t1);
      materialGroup->addTriangle(t2);

      gridX++;
      gridY += gridX / gridEdgeLen;
      gridX = gridX % gridEdgeLen;
    }

    return MeshRef(new Mesh(eMesh.getMeshData()));
  }
  
  class OpenGL120DrawFunctions {
    private:
      GLint oTEM, oCRGB, oS0RGB, oS1RGB, oCA, oS0A, oS1A;
      GLboolean oBlend;
      GLint oBS, oBD;

      GLboolean glTextureEnabled;
      GLfloat lastCurrentColor[4];
      GLint boundTexture;
    public:    
      void setupColor(const glm::vec4& color) {
        glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &oTEM);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_COMBINE_RGB, &oCRGB);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_SRC0_RGB, &oS0RGB);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_SRC1_RGB, &oS1RGB);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_COMBINE_ALPHA, &oCA);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_SRC0_ALPHA, &oS0A);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_SRC1_ALPHA, &oS1A);

        glGetBooleanv(GL_BLEND, &oBlend);
        glGetIntegerv(GL_BLEND_SRC, &oBS);
        glGetIntegerv(GL_BLEND_DST, &oBD);
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
        glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_MODULATE);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC0_RGB, GL_CONSTANT);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC1_RGB, GL_TEXTURE);
        glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_ALPHA, GL_MODULATE);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC0_ALPHA, GL_TEXTURE);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC1_ALPHA, GL_CONSTANT);
        glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, glm::value_ptr(color));
      }
    
      void setupDraw(TextureRef fontTexture) {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        glGetBooleanv(GL_BLEND, &glTextureEnabled);
        glGetFloatv(GL_CURRENT_COLOR, lastCurrentColor);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &boundTexture);

        glColor4f(1, 1, 1, 1);

        if (fontTexture) {
          glEnable(GL_TEXTURE_2D);
          glBindTexture(GL_TEXTURE_2D, fontTexture->getTexture());
        } else {
          cerr << "Warning! Drawing font without texture" << endl;
        }
      }
    
      void advanceX(float x) {
        glTranslatef(x, 0, 0);
      }
      
      void resetDraw() {
        glColor4f(lastCurrentColor[0], lastCurrentColor[1], lastCurrentColor[2], lastCurrentColor[3]);

        glBindTexture(GL_TEXTURE_2D, boundTexture);
        if (!glTextureEnabled) {
          glDisable(GL_TEXTURE_2D);
        }
        glPopMatrix();
      }
    
      void resetColor() {
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, oTEM);
        glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, oCRGB);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC0_RGB, oS0RGB);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC1_RGB, oS1RGB);
        glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_ALPHA, oCA);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC0_ALPHA, oS0A);
        glTexEnvi(GL_TEXTURE_ENV, GL_SRC1_ALPHA, oS1A);

        if (!oBlend) {
          glDisable(GL_BLEND);
        }
        glBlendFunc(oBS, oBD);
      }
  };
  static OpenGL120DrawFunctions OGL1200DrawFunctions;
  
  GLFont::SetupColorDrawFunction GLFont :: SetupColorDrawDefaultFunction = bind<void>(&OpenGL120DrawFunctions::setupColor, &OGL1200DrawFunctions, _1);
  GLFont::SetupFontDrawFunction GLFont :: SetupFontDrawDefaultFunction = bind<void>(&OpenGL120DrawFunctions::setupDraw, &OGL1200DrawFunctions, _1);
  GLFont::AdvanceXFunction GLFont :: AdvanceXDefaultFunction = bind<void>(&OpenGL120DrawFunctions::advanceX, &OGL1200DrawFunctions, _1);
  GLFont::ResetFontDrawFunction GLFont :: ResetFontDrawDefaultFunction = bind<void>(&OpenGL120DrawFunctions::resetDraw, &OGL1200DrawFunctions);
  GLFont::ResetColorDrawFunction GLFont :: ResetColorDrawDefaultFunction = bind<void>(&OpenGL120DrawFunctions::resetColor, &OGL1200DrawFunctions);

  void GLFont :: drawStringWithColor(const std::string& str, uint32_t color) const {
    drawStringWithColor(str, glm::vec4((float) (color >> 16 & 0xFF) / (float) 0xFF, (float) (color >> 8 & 0xFF) / (float) 0xFF, (float) (color >> 0 & 0xFF) / (float) 0xFF, (float) (color >> 24 & 0xFF) / (float) 0xFF));
  }
  
  void GLFont :: drawStringWithColor(const std::string& str, const glm::vec4& color) const {
    SetupColorDrawDefaultFunction(color);
    drawString(str);
    ResetColorDrawDefaultFunction();
  }

  void GLFont :: drawString(const string& str) const {
    SetupFontDrawDefaultFunction(fontTexture);
    
    for (string::const_iterator it = str.begin(); it != str.end(); it++) {
      if (MeshRef l = letterMeshes[((int) *it) - CHAR_MIN]) {
        l->drawMaterialGroup(0);
        AdvanceXDefaultFunction(letterWidths[((int) *it) - CHAR_MIN]);
      }
    }
    
    ResetFontDrawDefaultFunction();
  }

  float GLFont :: getStringWidth(const std::string& str) const {
    float result = 0;
    for (string::const_iterator it = str.begin(); it != str.end(); it++) {
      result += letterWidths[((int) *it) - CHAR_MIN];
    }
    return result;
  }

  GLFont :: GLFont(SystemBackendRef backendRef, TextureManager& textureManager, const std::string& name) : backendRef(backendRef), fontTexture() {
    std::fill(letterWidths, letterWidths + (CHAR_MAX - CHAR_MIN), 0);
    std::fill(letterMeshes, letterMeshes + (CHAR_MAX - CHAR_MIN), MeshRef());
    
    WidthsFileContent wFile;
    
    { // Parse widths file
      shared_ptr<std::istream> fontFile = backendRef->openResource(name);
      if (!fontFile) {
        throw GLFontException() << GLFontExceptionData(string("Could not open font file: ") + name);
      }
      try {
        wFile = parseWidthsFile(*fontFile);
      }
      catch (ConversionException& e) {
        string msg = "Could not convert file: ";
        if (const string* emsg = boost::get_error_info<ConversionExceptionData>(e)) {
          msg += *emsg;
        } else {
          msg += "[no additional data]";
        }
        throw GLFontException() << GLFontExceptionData(msg);
      }
    }

    // Try to preload texture
    fontTexture = textureManager.getTexture(wFile.texture);
    MeshRef mesh = createFontMesh(wFile, fontTexture);
    if (!mesh) {
      throw GLFontException() << GLFontExceptionData(string("Could not load generated mesh"));
    }
    
    ObjectNameVector onames = mesh->getObjectNames();
    for (ObjectNameVector::iterator it = onames.begin(); it != onames.end(); it++) {
      int letterIndex = lexical_cast<int>(*it) - CHAR_MIN;
      letterMeshes[letterIndex] = mesh->createObjectMesh(*it);
    }
    
    float gridEdgeLen = ceil(sqrt(wFile.widths.size()));
    float maxBoxWidth = (float) fontTexture->getWidth() / gridEdgeLen;
    for (WidthPairVector::iterator it = wFile.widths.begin(); it != wFile.widths.end(); it++) {
      letterWidths[((int) it->letter) - CHAR_MIN] = (float) it->width / maxBoxWidth;
    }
  }
}

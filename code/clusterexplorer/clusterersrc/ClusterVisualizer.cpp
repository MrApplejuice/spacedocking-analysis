#include "ClusterVisualizer.hpp"

#include <cmath>

#include <algorithm>

#include <boost/shared_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

using namespace std;
using namespace boost;
using namespace engine;

class VertexBufferAccess {
  private:
    size_t itemSize;
    unsigned int indexCount;
    void* dataptr;
    int* indexptr;
  public:
    unsigned int currentDataIndex;

    void writeIndex(int index) {
      *indexptr = index;
      indexptr++;
      indexCount++;
    }
    
    size_t getIndexCount() const {
      return indexCount;
    }
  
    void* getDataPtr() {
      return dataptr;
    }
  
    void advance() {
      currentDataIndex++;
      dataptr = (char*) dataptr + itemSize;
    }
    
    VertexBufferAccess(void* dataBuffer, int* indexBuffer, size_t itemSize) {
      this->itemSize = itemSize;
      dataptr = dataBuffer;
      indexptr = indexBuffer;
      currentDataIndex = 0;
      indexCount = 0;
    }
};

static void writeBracket(VertexBufferAccess& vba, const glm::vec3& startPosition, float width, bool extraLine, glm::vec3* labelPosition = NULL, glm::vec3* endPoints = NULL) {
  const unsigned int lineStartIndex = vba.currentDataIndex;
  vba.writeIndex(lineStartIndex + 0);
  vba.writeIndex(lineStartIndex + 2);
  vba.writeIndex(lineStartIndex + 1);
  vba.writeIndex(lineStartIndex + 3);
  vba.writeIndex(lineStartIndex + 2);
  vba.writeIndex(lineStartIndex + 3);
  if (extraLine) {
    vba.writeIndex(lineStartIndex + 4);
    vba.writeIndex(lineStartIndex + 5);
  }

  glm::vec3 vertXtraOffset(0, 0, 0);
  const glm::vec3 vertOffset(0, -width * 0.25, 0);
  const glm::vec3 lineDelta(width / 2.0, 0, 0);
  if (extraLine) {
    vertXtraOffset += vertOffset;
  }
  
  const glm::vec3 endpoint1 = startPosition - lineDelta + vertOffset + vertXtraOffset;
  const glm::vec3 endpoint2 = startPosition + lineDelta + vertOffset + vertXtraOffset;
  
  memcpy(vba.getDataPtr(), glm::value_ptr(endpoint1), sizeof(float) * 3);
  vba.advance();
  memcpy(vba.getDataPtr(), glm::value_ptr(endpoint2), sizeof(float) * 3);
  vba.advance();
  memcpy(vba.getDataPtr(), glm::value_ptr(startPosition - lineDelta + vertXtraOffset), sizeof(float) * 3);
  vba.advance();
  memcpy(vba.getDataPtr(), glm::value_ptr(startPosition + lineDelta + vertXtraOffset), sizeof(float) * 3);
  vba.advance();
  if (extraLine) {
    memcpy(vba.getDataPtr(), glm::value_ptr(startPosition + vertXtraOffset), sizeof(float) * 3);
    vba.advance();
    memcpy(vba.getDataPtr(), glm::value_ptr(startPosition), sizeof(float) * 3);
    vba.advance();
  }

  if (labelPosition != NULL) {
    *labelPosition = glm::vec3(startPosition.x + vertXtraOffset.x, startPosition.y + vertXtraOffset.y, 0);
  }
  
  if (endPoints != NULL) {
    endPoints[0] = endpoint1;
    endPoints[1] = endpoint2;
  }
}

static void recursiveFillBuffer(VertexBufferAccess& vba, const glm::vec3& startPosition, float width, ClusterVisualizer::DistanceLabelStack& distanceLabels, Cluster::NodeRef node, int depth=0) {
  if (node) {
    const Cluster::NodeRef* children = node->getChildren();

    glm::vec3 childStartPositions[2];
    glm::vec3 labelPos;
    writeBracket(vba, startPosition, width, !node->getParent(), &labelPos, childStartPositions);
    
    {
      ClusterVisualizer::DistanceLabel distanceLabel(lexical_cast<string>(node->getDistance()), labelPos.x, labelPos.y, 0.075 * pow(0.5, depth));
      while (depth >= distanceLabels.size()) {
        distanceLabels.push_back(ClusterVisualizer::DistanceLabelVector());
      }
      distanceLabels[depth].push_back(distanceLabel);
    }

    for (int i = 0; i < 2; i++) {
      if (children[i]) {
        recursiveFillBuffer(vba, childStartPositions[i], width / 2.0, distanceLabels, children[i], depth + 1);
      }
    }
  }
}


void ClusterVisualizer :: DistanceLabel :: draw(engine::GLFontRef font) const {
  float width = font->getStringWidth(label);
  
  ShaderProgram shader = ShaderProgram::current();
  shader.uniforms["modelMatrix"] = glm::translate<float>(x, y, 0) * glm::scale<float>(size, size, size) * glm::translate<float>(-width / 2, -1, 0);
  font->drawStringWithColor(label, glm::vec4(0.75f * glm::vec3(1, 1, 1), 1));
}

float ClusterVisualizer :: DistanceLabel :: getX() const {
  return x;
}

float ClusterVisualizer :: DistanceLabel :: getY() const {
  return y;
}

ClusterVisualizer :: DistanceLabel :: DistanceLabel(const std::string& label, float x, float y, float size) {
  this->label = label;
  this->x = x;
  this->y = y;
  this->size = size;
}


glm::mat4 ClusterVisualizer :: Camera :: getViewMatrix() const {
  const float hs = _size * calcVerticalSizeFactor();
  const float vs = hs;
  return glm::mat4(     1.0 / hs,               0, 0, 0,
                               0,        1.0 / vs, 0, 0,
                               0,               0, 1, 0,
                    -_pos.x / hs,    -_pos.y / vs, 0, 1);
}

glm::vec3& ClusterVisualizer :: Camera :: position() {
  return _pos;
}

const glm::vec3& ClusterVisualizer :: Camera :: position() const {
  return _pos;
}

float& ClusterVisualizer :: Camera :: size() {
  return _size;
}

float ClusterVisualizer :: Camera :: size() const {
  return _size;
}

void ClusterVisualizer :: Camera :: move(float dx, float dy) {
  const float sizeFactor = calcVerticalSizeFactor();
  _pos.x -= dx * sizeFactor;
  _pos.y -= dy * sizeFactor;
}
      
float ClusterVisualizer :: Camera :: calcVerticalSizeFactor() const {
  return pow(0.5, calcDepth());
}

float ClusterVisualizer :: Camera :: calcDepth() const {
  float ypower = 0.0;
  if (_pos.y < 0.0) {
    ypower = log(1 - -_pos.y / 0.25 * (1 - 0.5)) / log(0.5);
  }
  return ypower;
}

bool ClusterVisualizer :: Camera :: checkVisible(float aspect, float x, float y, float graceSize) const {
  x += x < _pos.x ? graceSize : -graceSize;
  y += y < _pos.y ? graceSize : -graceSize;
  float sf = _size * calcVerticalSizeFactor();
  return !((x - _pos.x < -aspect * sf) || (x - _pos.x > aspect * sf) || (y - _pos.y < -sf) || (y - _pos.y > sf));
}

ClusterVisualizer :: Camera :: Camera() {
  _pos = glm::vec3(0, 0, 0);
  _size = 1.0;
}


const static size_t VERTEX_SIZE = sizeof(float) * 6;

void ClusterVisualizer :: draw() const {
  lineShader.install();
  lineShader.uniforms["transformation"] = glm::scale<float>(1.0 / engine.getScreenAspectRatio(), 1.0, 1.0) * camera.getViewMatrix();
  lineShader.uniforms["color"] = glm::vec4(1, 1, 1, 1);

  GL_SIMPLE_ERROR(glBindVertexArray(lineSettingsVertexArray));
  GL_SIMPLE_ERROR(glBindBuffer(GL_ARRAY_BUFFER, lineDataBuffer));
  GL_SIMPLE_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lineDataIndexBuffer));

  GL_SIMPLE_ERROR(glDrawElements(GL_LINES, lineCount * 2, GL_UNSIGNED_INT, NULL));
  GL_SIMPLE_ERROR(glDrawElements(GL_POINTS, lineCount * 2, GL_UNSIGNED_INT, NULL));

  GL_SIMPLE_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
  GL_SIMPLE_ERROR(glBindVertexArray(0));
  GL_SIMPLE_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
  
  { // Draw labels 
    const float aspectRatio = engine.getScreenAspectRatio();    

    fontShader.install();
    fontShader.uniforms["projectionMatrix"] = glm::scale<float>(1.0 / aspectRatio, 1.0, 1.0) * camera.getViewMatrix();
    fontShader.uniforms["viewMatrix"] = glm::mat4(1.0f);

    const static float DEPTH_DISPLAY_MARGIN = 2;
    int depth = (int) round(camera.calcDepth());
    int depthStart = depth - DEPTH_DISPLAY_MARGIN;
    if (depthStart < 0) {
      depthStart = 0;
    }

    for (int d = depthStart; (d <= depthStart + DEPTH_DISPLAY_MARGIN * 2) && (d < distanceLabels.size()); d++) {
      const float graceSize = 0.25 * pow(0.5, d);
      BOOST_FOREACH(const DistanceLabel& label, distanceLabels[d]) {
        if (camera.checkVisible(aspectRatio, label.getX(), label.getY(), graceSize)) {
          label.draw(font);
        }
      }
    }
  }
}

ClusterVisualizer :: ClusterVisualizer(GameEngine& engine, const Cluster& cluster) : engine(engine), lineSettingsVertexArray(0), lineDataBuffer(0), lineDataIndexBuffer(0) {
  GL_SIMPLE_ERROR(glGenBuffers(1, &lineDataBuffer));
  GL_SIMPLE_ERROR(glGenBuffers(1, &lineDataIndexBuffer));
  GL_SIMPLE_ERROR(glGenVertexArrays(1, &lineSettingsVertexArray));

  { // Create draw buffers
    shared_array<char> dataBuffer(new char[6 * VERTEX_SIZE * cluster.getNodeCount()]);
    shared_array<int> indexBuffer(new int[6 * 2 * cluster.getNodeCount()]);
    VertexBufferAccess vba(dataBuffer.get(), indexBuffer.get(), VERTEX_SIZE);
    recursiveFillBuffer(vba, glm::vec3(0, 0.25, 0), 1.0, distanceLabels, cluster.getRoot());

    lineCount = vba.getIndexCount() / 2;
    
    GL_SIMPLE_ERROR(glBindVertexArray(lineSettingsVertexArray));
    GL_SIMPLE_ERROR(glBindBuffer(GL_ARRAY_BUFFER, lineDataBuffer));
    GL_SIMPLE_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lineDataIndexBuffer));

    GL_SIMPLE_ERROR(glBufferData(GL_ARRAY_BUFFER, (size_t) ((char*) vba.getDataPtr() - dataBuffer.get()), dataBuffer.get(), GL_STATIC_DRAW));
    GL_SIMPLE_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, vba.getIndexCount() * sizeof(int), indexBuffer.get(), GL_STATIC_DRAW));

    GL_SIMPLE_ERROR(glEnableVertexAttribArray(0));
    GL_SIMPLE_ERROR(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_SIZE, (GLvoid*) 0));

    GL_SIMPLE_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    GL_SIMPLE_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
    GL_SIMPLE_ERROR(glBindVertexArray(0));
  }  
  
  { // Compile shaders
    {
      MapAttribLocationAssigner attribLocationAssignment = MapAttribLocationAssigner().addMapping("color", 0).addMapping("transformation", 1);
      lineShader = engine.getShaderManager().compileProgram(engine.getShaderManager().getShader("line-vertex-shader.glsl"), engine.getShaderManager().getShader("line-fragment-shader.glsl"), attribLocationAssignment);
    }
    
    {
      MapAttribLocationAssigner attribLocationAssignment = MapAttribLocationAssigner().addMapping("i_position", 0).addMapping("i_normal", 1).addMapping("i_color", 2).addMapping("i_tcoord", 3);
      fontShader = engine.getShaderManager().compileProgram(engine.getShaderManager().getShader("vertex-font.glsl"), engine.getShaderManager().getShader("fragment-font.glsl"), attribLocationAssignment);
    }
  }
  
  font = engine.getFontManager().getFont("arial");
}

ClusterVisualizer :: ~ClusterVisualizer() {
  GL_SIMPLE_ERROR(glDeleteVertexArrays(1, &lineSettingsVertexArray));
  GL_SIMPLE_ERROR(glDeleteBuffers(1, &lineDataBuffer));
  GL_SIMPLE_ERROR(glDeleteBuffers(1, &lineDataIndexBuffer));
}

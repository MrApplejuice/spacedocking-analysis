#pragma once

#include <string>
#include <vector>

#include <engine/Engine.hpp>

#include "Cluster.hpp"

class ClusterVisualizer {
  public:
    class DistanceLabel {
      private:
        std::string label;
        float x;
        float y;
        float size;
      public:
        void draw(engine::GLFontRef font) const;
        
        float getX() const;
        float getY() const;
        
        DistanceLabel(const std::string& label, float x, float y, float size);
    };
    
    typedef std::vector<DistanceLabel> DistanceLabelVector;
    typedef std::vector<DistanceLabelVector> DistanceLabelStack;
  private:
    engine::GameEngine& engine;
    
    Cluster::NodeRef rootNode, topNode;
    
    size_t lineCount;
    
    GLuint lineSettingsVertexArray;
    GLuint lineDataBuffer;
    GLuint lineDataIndexBuffer;
    
    engine::ShaderProgram lineShader;
    
    glm::vec3 lineLabelPosition;
    glm::vec3 lineEndpoints[2];
    
    engine::GLFontRef font;
    engine::ShaderProgram fontShader;
    
    virtual void recursiveDraw(const glm::mat4& projMat, const glm::mat4& viewMat, Cluster::NodeRef node, int depth, DistanceLabelStack& distanceLabels) const;
  public:
    class Camera {
      private:
        glm::vec3 _pos;
        float _size;
      public:
        glm::mat4 getViewMatrix() const;
        
        glm::vec3& position();
        const glm::vec3& position() const;
        
        float& size();
        float size() const;
        
        void move(float dx, float dy);
      
        float calcVerticalSizeFactor() const;
        float calcDepth() const;
        
        bool checkVisible(float aspect, float x, float y, float graceSize = 0.0f) const;
      
        Camera();
    };
    
    Camera camera;
  
    virtual void updateZoom();
    virtual void draw() const;
  
    ClusterVisualizer(engine::GameEngine& engine, const Cluster& cluster);
    virtual ~ClusterVisualizer();
};

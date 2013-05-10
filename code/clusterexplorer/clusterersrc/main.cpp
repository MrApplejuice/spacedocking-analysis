#include <engine/Engine.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include <OpenGL.hpp>

#include "Cluster.hpp"
#include "ClusterVisualizer.hpp"

using std::cout;
using std::cerr; 
using std::endl;

using boost::shared_ptr;
using boost::bind;

using namespace engine;

class FontSetupFunctions {
  private:
    GLint lastTexture;
    float currentX;
    bool lastBlend;
    
    GLboolean oldBlend;
    GLint oldSrcBlendFunc, oldDestBlendFunc;
    
    bool colorConfigured;
  public:
    void setupColorConfig(const glm::vec4& color) {
      ShaderProgram cp = ShaderProgram::current();
      cp.uniforms["fs_refColor"] = color;
      cp.uniforms["fs_texture"] = 0;
      colorConfigured = true;
    }
  
    void setupDrawConfig(TextureRef texture) {
      glGetIntegerv(GL_TEXTURE_BINDING_2D, &lastTexture);
      glBindTexture(GL_TEXTURE_2D, texture->getTexture());

      glGetBooleanv(GL_BLEND, &oldBlend);
      glGetIntegerv(GL_BLEND_SRC, &oldSrcBlendFunc);
      glGetIntegerv(GL_BLEND_DST, &oldDestBlendFunc);
      
      //glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (!colorConfigured) { // NOT using color config
        setupColorConfig(glm::vec4(1, 1, 1, 1)); // ... so draw white text
      }
      
      currentX = 0;
      ShaderProgram::current().uniforms["offset"] = currentX;
    }
  
    void advanceX(float x) {
      currentX += x;
      ShaderProgram::current().uniforms["offset"] = currentX;
    }

    void resetDrawConfig() {
      glBlendFunc(oldSrcBlendFunc, oldDestBlendFunc);
      if (!oldBlend) {
        glDisable(GL_BLEND);
      }

      colorConfigured = false;

      glBindTexture(GL_TEXTURE_2D, lastTexture);
    }

    void resetColorConfig() {
      colorConfigured = false;
    }
    
    FontSetupFunctions() {
    }
};

static FontSetupFunctions gFontSetupFunctions;

class ClusterEngine : public virtual GameEngine {
  private:  
    typedef shared_ptr<ClusterVisualizer> ClusterVisualizerRef;
  
    Cluster cluster;
    ClusterVisualizerRef clusterVisualizer;
    
    bool trackingPoint;
    int trackedPoint;
  public:
    void onPointerPress(const PointerInfo& pi) {
      if (!trackingPoint) {
        trackingPoint = true;
        trackedPoint = pi.id;
      }
    }

    void onPointerMove(const PointerInfo& pi) {
      if (trackingPoint && (pi.id == trackedPoint)) {
        clusterVisualizer->camera.move(pi.x - pi.oldX, pi.y - pi.oldY);
      }
    }

    void onPointerRelease(const PointerInfo& pi) {
      if (trackingPoint && (trackedPoint == pi.id)) {
        trackingPoint = false;
      }
    }
  
    virtual void step(float deltaT) {
      GL_SIMPLE_ERROR(glClearColor(0.5, 0, 0, 0));
      GL_SIMPLE_ERROR(glClear(GL_COLOR_BUFFER_BIT));
      
      GL_SIMPLE_ERROR(glDisable(GL_DEPTH_TEST));
      
      if (clusterVisualizer) {
        clusterVisualizer->draw();
      }
    }

    virtual void initialize() {
      // Setup alternative text drawing functions for GLFont
      GLFont::SetupColorDrawDefaultFunction = bind<void>(&FontSetupFunctions::setupColorConfig, &gFontSetupFunctions, _1);
      GLFont::SetupFontDrawDefaultFunction = bind<void>(&FontSetupFunctions::setupDrawConfig, &gFontSetupFunctions, _1);
      GLFont::AdvanceXDefaultFunction = bind<void>(&FontSetupFunctions::advanceX, &gFontSetupFunctions, _1);
      GLFont::ResetFontDrawDefaultFunction = bind<void>(&FontSetupFunctions::resetDrawConfig, &gFontSetupFunctions);
      GLFont::ResetColorDrawDefaultFunction = bind<void>(&FontSetupFunctions::resetColorConfig, &gFontSetupFunctions);

      cluster.readCluster("testcluster");
      clusterVisualizer = ClusterVisualizerRef(new ClusterVisualizer(*this, cluster));
      
      trackingPoint = false;
      
      getPointerInput().registerOnPress(bind<void>(&ClusterEngine::onPointerPress, this, _1));
      getPointerInput().registerOnMove(bind<void>(&ClusterEngine::onPointerMove, this, _1));
      getPointerInput().registerOnRelease(bind<void>(&ClusterEngine::onPointerRelease, this, _1));
    }
    
    ClusterEngine(SystemBackendRef sbr) : GameEngine(sbr) {
    }
};

namespace engine {
  EngineBaseClassRef createEngine(SystemBackendRef sbr) {
    return EngineBaseClassRef(new ClusterEngine(sbr));
  }
}

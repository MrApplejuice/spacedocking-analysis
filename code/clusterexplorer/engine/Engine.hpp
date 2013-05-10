#ifndef ENGINE_HPP_
#define ENGINE_HPP_

#include <map>
#include <vector>
#include <string>
#include <iostream>

#include <boost/utility.hpp>
#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <ClassTools.hpp>
#include "SystemBackend.hpp"
#include "CallbackListManager.hpp"
#include "ShaderLoader.hpp"
#include "TextureManager.hpp"

#include "libs/glm/glm.hpp"
#include "libs/glm/ext.hpp"

#include "Mesh.hpp"
#include "GLFont.hpp"
#include "EngineDebugText.hpp"

#include "EngineWarnings.hpp"

namespace engine {
  typedef std::map<std::string, GLFontRef> NameGLFontMap;
  
  typedef boost::error_info<struct tag_my_info, const std::string> EngineExceptionData;
  struct EngineException : public virtual boost::exception, public virtual std::exception {};
  
  #define GL_SIMPLE_ERROR(...) \
    __VA_ARGS__; \
    if (const GLenum errorCode = glGetError() != GL_NO_ERROR) { \
      cerr << "GL Error " << errorCode  << " during: " << #__VA_ARGS__ << " on line " << __LINE__ << " file " << __FILE__ << endl; \
    }
    
  #define GL_EXCEPTION(...) \
    __VA_ARGS__; \
    if (const GLenum errorCode = glGetError() != GL_NO_ERROR) { \
      std::string msg = std::string("GL Error ") + boost::lexical_cast<std::string>(errorCode) + std::string(" during: ") + std::string(#__VA_ARGS__) + std::string(" on line ") + boost::lexical_cast<std::string>(__LINE__) + std::string(" file " __FILE__); \
      throw EngineException() << EngineExceptionData(msg); \
    }
    
  struct KeyInfo {
    char character;
    KeyCode keycode;
    bool pressed;
  };
  typedef boost::function1<void, const KeyInfo&> KeyCallback;
  typedef internal::CallbackListManager<KeyCallback> KeyboardCallbackManager;
  typedef KeyboardCallbackManager::CallbackListItemRemover KeyboardCallbackRemover;
  class Keyboard {
    private:
      SystemBackendRef backendRef;
      
      bool keyStates[KEY_CODE_COUNT];
      KeyboardCallbackManager keyCallbacks[KEY_CODE_COUNT];
    protected:
      virtual void poll();
    public:
      virtual bool getKeyState(KeyCode keycode) const {
        assert((keycode >= KEY_UNKNOWN) && (keycode < KEY_CODE_COUNT));
        return keyStates[keycode];
      }
    
      virtual KeyboardCallbackRemover registerOnKey(KeyCode keycode, KeyCallback cb) { 
        assert((keycode >= KEY_UNKNOWN) && (keycode < KEY_CODE_COUNT));
        return keyCallbacks[keycode].addElement(cb); 
      }
    
      Keyboard(SystemBackendRef backendRef);

      friend class GameEngine;
  };
  
  class FontManager {
    private:
      std::string resourcePrefix;
      SystemBackendRef backendRef;
      TextureManager& textureManager;
      
      NameGLFontMap fonts;
    public:
      virtual GLFontRef getFont(const std::string& font);
    
      FontManager(SystemBackendRef backendRef, TextureManager& tmanager, std::string resourcePrefix);
  };
  
  struct PointerInfo {
    int id;
    bool pressed;
    float x, y;
    float oldX, oldY;
    float speedx, speedy;
  };
  typedef std::vector<PointerInfo> PointerInfoVector;
  
  typedef boost::function1<void, const PointerInfo&> PointerCallback;
  typedef internal::CallbackListManager<PointerCallback>::CallbackListItemRemover PointerCallbackRemover;
  
  class PointerInput : boost::noncopyable {
    private:
      internal::CallbackListManager<PointerCallback> onPress, onMove, onRelease;
      SystemBackendRef backendRef;
      
      PointerInfoVector pointerData;
    protected:
      virtual void poll(float deltaT);
    public:
      virtual GEN_CALLBACK_ACCESSOR(PointerCallback, OnPress, onPress);
      virtual GEN_CALLBACK_ACCESSOR(PointerCallback, OnMove, onMove);
      virtual GEN_CALLBACK_ACCESSOR(PointerCallback, OnRelease, onRelease);
    
      virtual const PointerInfoVector& getPointers() const;
    
      PointerInput(const SystemBackendRef& backendRef);
      
      friend class GameEngine;
  };
  
  typedef std::map<std::string, MeshRef> NameMeshMap;
  
  class MeshManager : boost::noncopyable {
    private:
      std::string resourcePrefix;
    
      SystemBackendRef backendRef;
      
      NameMeshMap loadMeshs;
      
      TextureManager& textureManager;
    public:
      virtual MeshRef getMesh(const std::string& name);
    
      MeshManager(const SystemBackendRef& backendRef, TextureManager& _texMan, std::string resourcePrefix);
  };
  
  typedef std::map<std::string, int> SoundNameIdMap;
  typedef std::vector<int> SoundIdVector;
  
  class SoundEngine : boost::noncopyable {
    private:
      std::string resourcePrefix;
    
      SystemBackendRef backendRef;
      
      SoundNameIdMap soundBuffers, soundStreams;
      SoundIdVector playingSounds;
      
      virtual int retrieveBuffer(const std::string& str);
      virtual int retrieveStream(const std::string& str);
    public:
      template<class InputIterator>
      void buffer(InputIterator begin, InputIterator end) {
        for (InputIterator it = begin; it != end; it++) {
          retrieveBuffer(*it);
        }
      }

      virtual int stream(const std::string& sound);
      virtual int play(const std::string& sound);
      virtual void stop(int playid);
    
      SoundEngine(const SystemBackendRef& backendRef, std::string resourcePrefix);
  };
  
  class GameEngine : public EngineBaseClass, boost::noncopyable {
    private:
      std::string resourcePrefix;
    
      SystemBackendRef backend;

      SoundEngine soundEngine;
      TextureManager textureManager;
      MeshManager meshManager;
      PointerInput pointerInput;
      FontManager fontManager;
      DebugText debugText;
      Keyboard keyboard;
      ShaderManager shaderManager;

      int initializationStepCounter;
      
      virtual void internalDoStep(float deltaT);
    protected:
      GEN_GETTER(SystemBackendRef, Backend, backend);

      /**
       * Must be overwritten!!! The implementation of the game loop
       * goes in here.
       * 
       * @param deltaT
       *   The passed time since the last call to step
       */
      virtual void step(float deltaT) = 0;
      
      /**
       * May be overwritten (useful for initial setup, the GameEngine
       * has initialized during the call to this function). Called once
       * before step is called repeatedly but after the engine
       * initialization has been completed (which might not be the case
       * at the moment the constructor finishes).
       */
       virtual void initialize() {}

       /** Specialized function to load and display the first frame before
        * initialize is called */
       virtual void drawFirstFrame();
    public:
      /**
       * Returns the aspect ratio of the screen used for OpenGL.
       * 
       * @return
       *   Aspect ratio of the used screen
       */
      virtual float getScreenAspectRatio() const;

      /**
       * Opens a persistent file for reading and/or writing which remains 
       * persistent between multiple program starts. Subdirectories are
       * automatically created if in openmode::out has been specified.
       * 
       * @param filename
       *   The name of the persistent file to open.
       * @param openmode
       *   Mode to open the persistent file in (see ios_base::openmode
       *   for more info)
       * @return
       *   Reference to open stream object if the file was opened 
       *   successfully. Returns a NULL-reference on error.
       */
      virtual boost::shared_ptr<std::iostream> openPersistentFile(const std::string& filename, std::ios_base::openmode openmode);
      
      /** 
       * Deletes the specified persistent file.
       * 
       * @param filename
       *   The persistent file to delete
       * @return
       *   True on success (the file was deleted), False on failure
       */
      virtual bool deletePersistentFile(const std::string& filename);
    
      /**
       * Asks the system backend if it supports a given extension. If it does
       * this function returns a reference to an initialized extension object.
       * Otherwise the function returns a NULL reference.
       * 
       * @param extensionName
       *   The name of the extension thats support is requested.
       * 
       * @return
       *   If the extension is supported by the system backend a shared_ptr
       *   holding the pointer to the initialized externsion object. 
       *   Otherwise a NULL pointer is returned.
       */
      virtual EngineExtensionRef getExtension(const std::string& extensionName);
      
      /**
       * Opens any file in the game directory res/misc. This function can
       * be used to open files that are not supported by the game engine 
       * itself and interpret them with custom algorithms
       * 
       * @param filename
       *   The name of the misc resource file to open.
       * @param openmode
       *   Mode to open the persistent file in. Only readable is allowed
       *   (for write mode a new persistent file must be created).
       * @return
       *   Reference to the open stream object if the file was opened 
       *   successfully. Returns a NULL-reference on error.
       */
      virtual boost::shared_ptr<std::istream> openMiscResource(const std::string& filename, std::ios_base::openmode openmode);
      
      FLEXIBLE_GEN_GETTER(SoundEngine&, SoundEngine, soundEngine, );
      FLEXIBLE_GEN_GETTER(MeshManager&, MeshManager, meshManager, );
      FLEXIBLE_GEN_GETTER(TextureManager&, TextureManager, textureManager, );
      FLEXIBLE_GEN_GETTER(PointerInput&, PointerInput, pointerInput, );
      FLEXIBLE_GEN_GETTER(FontManager&, FontManager, fontManager, );
      FLEXIBLE_GEN_GETTER(DebugText&, DebugText, debugText, );
      FLEXIBLE_GEN_GETTER(Keyboard&, Keyboard, keyboard, );
      FLEXIBLE_GEN_GETTER(ShaderManager&, ShaderManager, shaderManager, );
    private:
      void init();
    public:
      GameEngine(const SystemBackendRef& backend);
      GameEngine(const SystemBackendRef& backend, std::string customPrefix);
      virtual ~GameEngine();
  };
}

#endif // ENGINE_HPP_

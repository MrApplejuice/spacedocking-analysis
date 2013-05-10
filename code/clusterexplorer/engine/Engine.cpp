#include "Engine.hpp"

#include <boost/shared_ptr.hpp>

#include <OpenGL.hpp>

#include "WavefrontMeshDataProvider.hpp"

#include "OpenGLESDetectionHeader.hpp"

#include "local/Global.hpp"

#define RESOURCE_PREFIX "res/"
#define AUDIO_PREFIX "audio/"
#define MESH_PREFIX "meshes/"
#define IMAGE_PREFIX "images/"
#define FONT_PREFIX "glfonts/"
#define SHADER_PREFIX "shaders/"
#define MISC_PREFIX "misc/"

namespace engine {
  static shared_ptr<std::istream> openResourceFunction(SystemBackendRef backend, string path, string filename) {
    return backend->openResource(path + filename);
  }


  static char translateKeyCode(KeyCode keycode) {
    switch (keycode) {
      #include "TranslateKeycode.cpp.inc"
      default: ;
    }
    
    return '\0';
  }
  
  void Keyboard :: poll() {
    KeyPressChangeVector keyPressChanges = backendRef->getKeyPressChanges();
    foreach (KeyPressChange& pc, keyPressChanges) {
      KeyInfo ki;
      ki.keycode = pc.keycode;
      ki.character = translateKeyCode(ki.keycode);
      ki.pressed = pc.pressState;
      
      if (keyStates[ki.keycode] != ki.pressed) {
        keyStates[ki.keycode] = ki.pressed;
        vector<KeyCallback> callbacks = keyCallbacks[ki.keycode].getCallbacks();
        foreach (KeyCallback& cb, callbacks) {
          cb(ki);
        }
      }
    }
  }
  
  Keyboard :: Keyboard(SystemBackendRef backendRef) : backendRef(backendRef) {
    std::fill(keyStates, keyStates + KEY_CODE_COUNT, false);
  }


  GLFontRef FontManager :: getFont(const std::string& font) {
    string fullname = resourcePrefix + FONT_PREFIX + font;
    GLFontRef result;

    NameGLFontMap::iterator found = fonts.find(fullname);
    if (found != fonts.end()) {
      result = found->second;
    } else {
      try {
        result = GLFontRef(new GLFont(backendRef, textureManager, fullname));
        fonts[fullname] = result;
      }
      catch (GLFontException& e) {
        cerr << "Font load error for font " << font << " : ";
        if (const string* msg = boost::get_error_info<GLFontExceptionData>(e)) {
          cerr << *msg;
        } else {
          cerr << "[no msg]";
        }
        cerr << endl;
      }
    }

    return result;
  }
  
  FontManager :: FontManager(SystemBackendRef backendRef, TextureManager& tmanager, std::string resourcePrefix) : resourcePrefix(resourcePrefix), backendRef(backendRef), textureManager(tmanager) {
  }


  template <typename T, typename N>
  static void callall(T begin, T end, N parameter) {
    for (T it = begin; it != end; it++) {
      (*it)(parameter);
    }
  }
  
  void PointerInput :: poll(float deltaT) {
    const PointerDataVector& backendData = backendRef->getPointerData();
    PointerInfoVector newPointerInfo;

    for (PointerDataVector::const_iterator b_it = backendData.begin(); b_it != backendData.end(); b_it++) {
      PointerInfo pi;
      memset(&pi, 0, sizeof(pi));

      for (PointerInfoVector::iterator p_it = pointerData.begin(); p_it != pointerData.end(); p_it++) {
        if (b_it->id == p_it->id) {
          pi = *p_it;
          pi.oldX = pi.x;
          pi.oldY = pi.y;
          pi.speedx = (b_it->x - pi.oldX) / deltaT;
          pi.speedy = (b_it->y - pi.oldY) / deltaT;
          
          pointerData.erase(p_it);
          break;
        }
      }

      pi.x = b_it->x;
      pi.y = b_it->y;
      bool oldPressed = pi.pressed;
      pi.pressed = b_it->pressed;
      if (pi.pressed && !oldPressed) {
        vector<PointerCallback> callbacks = onPress.getCallbacks();
        callall(callbacks.begin(), callbacks.end(), pi);
      }
      if (pi.speedx || pi.speedy) {
        vector<PointerCallback> callbacks = onMove.getCallbacks();
        callall(callbacks.begin(), callbacks.end(), pi);
      }
      if (!pi.pressed && oldPressed) {
        vector<PointerCallback> callbacks = onRelease.getCallbacks();
        callall(callbacks.begin(), callbacks.end(), pi);
      }
      
      newPointerInfo.push_back(pi);
    }

    // Send release for all pointers that are now untracked
    if (pointerData.size() > 0) {    
      vector<PointerCallback> releaseCallbacks = onRelease.getCallbacks();
      for (PointerInfoVector::iterator p_it = pointerData.begin(); p_it != pointerData.end(); p_it++) {
        p_it->pressed = false;
        callall(releaseCallbacks.begin(), releaseCallbacks.end(), *p_it);
      }
    }
    
    pointerData = newPointerInfo;
  }

  const PointerInfoVector& PointerInput :: getPointers() const {
    return pointerData;
  }

  PointerInput :: PointerInput(const SystemBackendRef& backendRef) {
    this->backendRef = backendRef;
  }


  MeshRef MeshManager :: getMesh(const std::string& name) {
    string fullname = resourcePrefix + MESH_PREFIX + name;
    
    MeshRef result;
    
    NameMeshMap::iterator foundMesh = loadMeshs.find(fullname);
    if (foundMesh != loadMeshs.end()) {
      result = foundMesh->second;
    } else {
      // Mesh not found - try to load it
      try {
        WavefrontMeshDataProviderRef mdpRef(new WavefrontMeshDataProvider(bind< shared_ptr<std::istream> >(openResourceFunction, backendRef, resourcePrefix + MESH_PREFIX, _1), textureManager, name)); 
        result = MeshRef(new Mesh(mdpRef->getMeshData()));
        loadMeshs[fullname] = result;
      }
      catch (WavefrontMeshDataProviderException& e) {
        cerr << "Could not load mesh file into memory: ";
        if (const string* msg = boost::get_error_info<WavefrontMeshDataProviderExceptionData>(e)) {
          cerr << *msg;
        } else {
          cerr << "Could not load mesh file into memory: ";
        }
        cerr << endl;
      }
    }
    
    return result;
  }

  MeshManager :: MeshManager(const SystemBackendRef& backendRef, TextureManager& _texMan, string resourcePrefix) : resourcePrefix(resourcePrefix), backendRef(backendRef), textureManager(_texMan) {
  }


  int SoundEngine :: retrieveBuffer(const string& str) {
    string fullSoundPath = string() + resourcePrefix + AUDIO_PREFIX + str;
    int result = soundBuffers[fullSoundPath];
    if (result) {
      return result;
    }
    
    result = backendRef->allocateSoundBuffer(fullSoundPath);
    if (!result) {
      cerr << "Failed to load sound buffer: " << str << endl;
    } else {
      soundBuffers[fullSoundPath] = result;
    }
    return result;
  }

  int SoundEngine :: retrieveStream(const string& str) {
    string fullSoundPath = string() + resourcePrefix + AUDIO_PREFIX + str;
    int result = soundStreams[fullSoundPath];
    if (result) {
      return result;
    }
    
    result = backendRef->prepareSoundStream(fullSoundPath);
    if (!result) {
      cerr << "Failed to prepare sound stream: " << str << endl;
    } else {
      soundStreams[fullSoundPath] = result;
    }
    return result;
  }

  int SoundEngine :: stream(const string& sound) {
    int soundRef = retrieveStream(sound);
    if (soundRef) {
      return backendRef->play(soundRef);
    }
    return 0;
  }
  
  int SoundEngine :: play(const string& sound) {
    int soundRef = retrieveBuffer(sound);
    if (soundRef) {
      return backendRef->play(soundRef);
    }
    return 0;
  }
  
  void SoundEngine :: stop(int playid) {
    backendRef->stop(playid);
  }

  SoundEngine :: SoundEngine(const SystemBackendRef& backendRef, string resourcePrefix) : resourcePrefix(resourcePrefix), backendRef(backendRef) {
  }

  void GameEngine :: internalDoStep(float deltaT) {
    const static int INIT_FIRSTFRAME = 0;
    const static int INIT_INITIALIZE = 1;
    const static int INIT_RUN = 2;
    
    pointerInput.poll(deltaT);
    keyboard.poll();

    switch (initializationStepCounter) {
      case INIT_FIRSTFRAME:
        initializationStepCounter++;
        drawFirstFrame();
        break;
      case INIT_INITIALIZE:
        deltaT = 0;
        initialize();
        step(deltaT);
        initializationStepCounter++;
        break;
      case INIT_RUN:
        step(deltaT);
        break;
      default:
        initializationStepCounter++;
    }
    
    debugText.draw();
  }
  
  EngineExtensionRef GameEngine :: getExtension(const std::string& extensionName) {
    EngineExtensionRef result = backend->getExtension(extensionName);
    return result;
  }
  
  shared_ptr<std::istream> GameEngine :: openMiscResource(const std::string& filename, std::ios_base::openmode openmode) {
    if ((openmode & (std::ios_base::out | std::ios_base::trunc)) != 0) {
      return shared_ptr<std::iostream>();
    }

    string fullname = resourcePrefix + MISC_PREFIX + filename;
    return backend->openResource(fullname, (openmode & std::ios_base::binary) != 0);
  }
  
  void GameEngine :: drawFirstFrame() {
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
  }
  
  float GameEngine :: getScreenAspectRatio() const {
    return this->backend->getScreenAspectRatio();
  }
  
  shared_ptr<std::iostream> GameEngine :: openPersistentFile(const string& filename, std::ios_base::openmode openmode) {
    return backend->openPersistentFile(filename, openmode);
  }
  
  bool GameEngine :: deletePersistentFile(const std::string& filename) {
    return backend->deletePersistentFile(filename);
  }

  void GameEngine :: init() {
    this->initializationStepCounter = 0;
    
    this->backend->registerStepCallback(bind(&GameEngine::internalDoStep, this, _1));
  }

  GameEngine :: GameEngine(const SystemBackendRef& backend) : resourcePrefix(RESOURCE_PREFIX), backend(backend), soundEngine(backend, resourcePrefix), textureManager(backend, resourcePrefix + IMAGE_PREFIX), meshManager(backend, textureManager, resourcePrefix), pointerInput(backend), fontManager(backend, textureManager, resourcePrefix), debugText(backend->getScreenAspectRatio()), keyboard(backend), shaderManager(bind< shared_ptr<std::istream> >(openResourceFunction, backend, resourcePrefix + SHADER_PREFIX, _1)) {
    init();
  }
  
  GameEngine :: GameEngine(const SystemBackendRef& backend, std::string customPrefix) : resourcePrefix(customPrefix), backend(backend), soundEngine(backend, resourcePrefix), textureManager(backend, resourcePrefix + IMAGE_PREFIX), meshManager(backend, textureManager, resourcePrefix), pointerInput(backend), fontManager(backend, textureManager, resourcePrefix), debugText(backend->getScreenAspectRatio()), keyboard(backend), shaderManager(bind< shared_ptr<std::istream> >(openResourceFunction, backend, resourcePrefix + SHADER_PREFIX, _1)) {
    init();
  }

  GameEngine :: ~GameEngine() {
  }
}

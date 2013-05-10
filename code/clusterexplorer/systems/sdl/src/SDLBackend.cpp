#include "SDLBackend.hpp"

#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#define foreach BOOST_FOREACH

#include "SDL.h"
#include "SDL_mixer.h"

#include <OpenGL.hpp>

#include "IL/il.h"
#include "IL/ilut.h"

// Make screen the same size as the iPhone's screen
#define SCREEN_WIDTH  (480 * 2)
#define SCREEN_HEIGHT (320 * 2)

#define PERSISTENT_DIRECTORY "data/"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

using boost::shared_ptr;
using boost::shared_array;

using namespace engine;
using namespace lbe;

namespace lbe {
  StreamInstance :: StreamInstance(Sound& srcSound) {
    this->srcSound = &srcSound;
  }

  StreamInstanceRef SoundBuffer :: play() {
    return StreamInstanceRef(new SoundBufferStreamInstance(*this));
  }
  
  SoundBuffer :: SoundBuffer(const std::string& filename) {
    sample = Mix_LoadWAV(filename.c_str());
    if (sample) {
      this->filename = filename;
    } else {
      throw LinuxBackendSoundException();
    }
  }

  bool SoundBufferStreamInstance :: SoundBufferStreamInstance :: stop() {
    if ((!stopped) && (Mix_HaltChannel(channel) == 0)) {
      stopped = true;
    }
    return stopped;
  }
  
  bool SoundBufferStreamInstance :: SoundBufferStreamInstance :: finished() {
    return Mix_Playing(channel) == 0;
  }
  
  SoundBufferStreamInstance :: SoundBufferStreamInstance(SoundBuffer& srcSound) : StreamInstance(srcSound) {
    this->channel = Mix_PlayChannel(-1, srcSound.sample, false);
    this->stopped = false;
    
    if (this->channel == -1) {
      throw LinuxBackendSoundException();
    }
  }
  
  StreamInstanceRef StreamedSound :: play() {
    return sharedMusicStream->play(*this);
  }
  
  StreamedSound :: StreamedSound(const std::string& filename, CentralStreamedSoundStreamInstanceRef sharedMusicStream) {
    this->filename = filename;
    this->sharedMusicStream = sharedMusicStream;
    
    this->music = Mix_LoadMUS(filename.c_str());
    if (this->music == NULL) {
      throw LinuxBackendSoundException();
    }
  }

  bool StreamedSoundStreamInstance :: stop() {
    if ((!stopped) && (Mix_HaltMusic())) {
      stopped = true;
    }
    return stopped;
  }
  
  bool StreamedSoundStreamInstance :: finished() {
    return stopped || (Mix_PlayingMusic() == 0);
  }

  StreamedSoundStreamInstance :: StreamedSoundStreamInstance(StreamedSound& srcSound) : StreamInstance(srcSound) {
    if (Mix_PlayMusic(srcSound.music, 1) == -1) {
      throw LinuxBackendSoundException();
    }
  }

  StreamInstanceRef CentralStreamedSoundStream :: play(StreamedSound& music) {
    if (playingMusic.get() != NULL) {
      playingMusic->stop();
      playingMusic.reset();
    }
    StreamInstanceRef result (new StreamedSoundStreamInstance(music));
    return result;
  }
}

int LinuxBackend :: findFreePlaybackIndex() {
  int result = -1;
  for (unsigned int i = 0; i < soundPlaybacks.size(); i++) {
    if ((soundPlaybacks[i].get() == NULL) || (soundPlaybacks[i]->finished())) {
      result = i;
      break;
    }
  }
  if (result < 0) {
    result = soundPlaybacks.size();
    soundPlaybacks.push_back(StreamInstanceRef());
  }
  return result;
}

void LinuxBackend :: registerStepCallback(const StepCallback& callback) {
  stepFuncs.push_back(callback);
}

int LinuxBackend :: allocateSoundBuffer(const std::string& soundpath) {
  try {
    SoundRef sound (new SoundBuffer(soundpath));
    sounds.push_back(sound);
    return sounds.size();
  }
  catch (LinuxBackendSoundException& e) {
    cerr << "Could not load sound buffer: " << soundpath << endl;
  }
  return 0;
}

int LinuxBackend :: prepareSoundStream(const std::string& soundpath) {
  try {
    SoundRef sound (new StreamedSound(soundpath, centralStreamPlayer));
    sounds.push_back(sound);
    return sounds.size();
  }
  catch (LinuxBackendSoundException& e) {
    cerr << "Could not load sound stream: " << soundpath << endl;
  }
  return 0;
}

int LinuxBackend :: play(int soundId) {
  soundId--;
  if ((soundId >= 0) && ((unsigned int) soundId < sounds.size())) {
    int freePlaybackIndex = findFreePlaybackIndex();
    if (freePlaybackIndex < 0) {
      cerr << "Cannot find a free playback slot" << endl;
    } else {
      try {
        soundPlaybacks[freePlaybackIndex] = sounds[soundId]->play();
        return freePlaybackIndex + 1;
      }
      catch (LinuxBackendSoundException e) {
        cerr << "Could not play sound" << endl;
      }
    }
  }
  return 0;
}

bool LinuxBackend :: stop(int playbackId) {
  playbackId--;
  if ((playbackId >= 0) && ((unsigned int) playbackId < soundPlaybacks.size())) {
    if (soundPlaybacks[playbackId].get() != NULL) {
      soundPlaybacks[playbackId]->stop();
      return true;
    }
  }
  
  return false;
}

bool LinuxBackend :: isPlaying(int playbackId) {
  if ((playbackId >= 0) && ((unsigned int) playbackId < soundPlaybacks.size())) {
    if (soundPlaybacks[playbackId].get() != NULL) {
      return !soundPlaybacks[playbackId]->finished();
    }
  }

  return false;
}

shared_ptr<std::istream> LinuxBackend :: openResource(const std::string& filename, bool binary) {
  shared_ptr<std::istream> result;
  
  shared_ptr<std::ifstream> filestream(new std::ifstream(filename.c_str(), (binary ? std::ifstream::in | std::ifstream::binary : std::ifstream::in)));
  if (filestream->is_open()) {
    result = filestream;
  }

  return result;
}

RawImageData LinuxBackend :: loadImage(const std::string& filename) {
  RawImageData result;

  ILuint ilImage = ilGenImage();
  if (ilImage == 0) {
    cerr << "FIXME: Could not create il image" << endl;
  } else {
    ilBindImage(ilImage);
    
    char nonConstFilename[filename.length() + 1];
    strcpy(nonConstFilename, filename.c_str());
    
    if (!ilLoadImage(nonConstFilename)) {
      ILenum error = ilGetError();
      if (error == IL_COULD_NOT_OPEN_FILE) {
        cerr << "Could not open file: " << filename << endl;
      } else {
        cerr << "Could not load/parse file: " << filename << endl;
      }
    } else {
      if (!ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE)) {
        cerr << "FIXME: Could not convert image to RGBA (char) format" << endl;
      } else {
        ChannelByte* srcBytes = ilGetData();
        
        if (srcBytes == NULL) {
          cerr << "Could not lock il image data" << endl;
        } else {
          result.width = ilGetInteger(IL_IMAGE_WIDTH);
          result.height = ilGetInteger(IL_IMAGE_HEIGHT);
          
          int len = result.width * result.height * 4;
          result.pixels = shared_array<ChannelByte>(new ChannelByte[len]);
          std::copy(srcBytes, srcBytes + len, result.pixels.get());
        }
      }
    }
    ilDeleteImage(ilImage);
  }
  
  return result;
}

const PointerDataVector& LinuxBackend :: getPointerData() {
  return pointerData;
}

float LinuxBackend :: getScreenAspectRatio() {
  return (float) SCREEN_WIDTH / (float) SCREEN_HEIGHT;
}

EngineExtensionRef LinuxBackend :: getExtension(const string& extensionName) {
  return SystemBackend::getExtension(extensionName);
}

static string canonicalizeFilename(const string& filename) {
  string result;
  result.reserve(filename.length());

  vector<string> filenameStack;
  
  {
    vector<string> filenameParts;
    boost::split(filenameParts, filename, boost::is_any_of("/"), boost::token_compress_on);
    foreach (const string& fp, filenameParts) {
      if (fp == "..") {
        // Strip last filename stack entry or fail if stack is empty (-> .. goes beyond root)
        if (filenameStack.empty()) {
          throw "Filename goes beyond root";
        } else {
          filenameStack.pop_back();
        }
      } else {
        filenameStack.push_back(fp);
      }
    }
  }
  
  result = boost::algorithm::join(filenameStack, "/");
  
  return result;
}

shared_ptr<std::iostream> LinuxBackend :: openPersistentFile(const string& filename, std::ios_base::openmode openmode) {
  shared_ptr<std::fstream> result;
  
  try {
    // Canonicalize filename and strip all .. going beyond root
    string fullFilename;
   
    { 
      string canonicalizedFilename = canonicalizeFilename(filename);
      fullFilename = string() + PERSISTENT_DIRECTORY + canonicalizedFilename;
      
      // If write mode enabled, make sure that all (sub-)directories exist
      if (openmode & std::ios_base::out) {
        boost::filesystem::path fPath(fullFilename);
        fPath.remove_filename();
        boost::filesystem::create_directories(fPath);
      }
    }
    
    result = shared_ptr<std::fstream>(new std::fstream(fullFilename.c_str(), openmode));
    if (!result->is_open()) {
      result.reset();
      throw "Could not be opened";
    }
  }
  catch (const char* s) {
    cerr << "Could not open persistent file " << filename << ": " << s << endl;
  }
  
  return result;
}

bool LinuxBackend :: deletePersistentFile(const string& filename) {
  try {
    string canonicalizedFilename = canonicalizeFilename(filename);

    string fullFilename = string() + PERSISTENT_DIRECTORY + canonicalizedFilename;
    return remove(fullFilename.c_str()) == 0;
  }
  catch (const char* s) {
    cerr << "Could not delete persistent file " << filename << ": " << s << endl;
    return false;
  }
}

KeyPressChangeVector LinuxBackend :: getKeyPressChanges() {
  return keyChanges;
}

static KeyCode convertKeyboardConstants(SDLKey sdlkey) {
  switch (sdlkey) {
    #include "ConvertKeyboardConstants.cpp.inc"
    default: ;
  }
  
  return KEY_UNKNOWN;
}

void LinuxBackend :: run() {
  Uint32 lastTick = SDL_GetTicks();
  
  while (doRun) {
    {
      SDL_Event event;
      PointerData& sysPointerData = pointerData[0];
      bool gotMouseData = false;
      int newMouseX = INT_MAX, newMouseY = INT_MAX;
      
      keyChanges.clear();
      while (SDL_PollEvent(&event)) {
        switch (event.type) {
          case SDL_QUIT:
            doRun = false;
            break;
          case SDL_MOUSEBUTTONDOWN:
          case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT) {
              sysPointerData.pressed = event.button.state == SDL_PRESSED;
            }
            newMouseX = event.button.x;
            newMouseY = event.button.y;
            gotMouseData = true;
            break;
          case SDL_MOUSEMOTION:
            newMouseX = event.motion.x;
            newMouseY = event.motion.y;
            gotMouseData = true;
            break;
          case SDL_KEYDOWN:
          case SDL_KEYUP:
            {
              KeyPressChange keyPressChange;
              keyPressChange.keycode = convertKeyboardConstants(event.key.keysym.sym);
              if (keyPressChange.keycode != KEY_UNKNOWN) {
                keyPressChange.pressState = (event.key.state == SDL_PRESSED ? true : false);
                keyChanges.push_back(keyPressChange);
              }
            }
            break;
        }
      }
      
      // Recalculate new mouse data to device coordinates
      if (gotMouseData) {
        if ((newMouseX >= 0) && (newMouseX < SCREEN_WIDTH)) {
          if ((newMouseY >= 0) && (newMouseY < SCREEN_HEIGHT)) {
            sysPointerData.x = (float) newMouseX * 2.0f / (float) SCREEN_WIDTH - 1.0f;
            sysPointerData.y = 1.0f - (float) newMouseY * 2.0f / (float) SCREEN_HEIGHT;
            sysPointerData.x *= (float) SCREEN_WIDTH / (float) SCREEN_HEIGHT;
          }
        }
      }
    }
    
    if (doRun) {
      Uint32 now = SDL_GetTicks();
      float deltaT = (now - lastTick) / 1000.0;
      lastTick = now;
      
      const StepCallbackVector stepFuncsCopy = stepFuncs;
      foreach (StepCallback sc, stepFuncsCopy) {
        sc(deltaT);
      }
      
      glFinish();
      SDL_GL_SwapBuffers();
    }
  }
}

LinuxBackend :: LinuxBackend() {
  doRun = true;

  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO))
    throw LinuxBackendException() << LinuxBackendExceptionData("SDL could not be initialized");

  try {  
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);    
    
    if (Mix_OpenAudio(44100, AUDIO_S16SYS, 2, 1024)) 
      throw LinuxBackendException() << LinuxBackendExceptionData("Could initialize sound mixer");

    try {
      if (SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT, 0, SDL_OPENGL) == 0) 
        throw LinuxBackendException() << LinuxBackendExceptionData("Could not allocate framebuffer");

#ifdef TARGET_WINDOWS
      if (glewInit() != GLEW_OK) {
        throw LinuxBackendException() << LinuxBackendExceptionData("Error initializing glew");
      }
#endif

      ilInit();
      ilutInit();
        
      SDL_WM_SetCaption("SDL game window", NULL);
      Mix_AllocateChannels(32);
      
      // Setup "fake" channel for streamed playback (this implementation only allows single "central" streamed playback)
      centralStreamPlayer = CentralStreamedSoundStreamInstanceRef(new CentralStreamedSoundStream());
      
      PointerData sysPointerData = {0, false, -1, -1};
      pointerData.push_back(sysPointerData);
    }
    catch (LinuxBackendException& e) {
      Mix_CloseAudio();
      throw e;
    }
  }
  catch (LinuxBackendException& e) {
    SDL_Quit();
    throw e;
  }
}

LinuxBackend :: ~LinuxBackend() {
  ilShutDown();
  Mix_CloseAudio();
  SDL_Quit();
}

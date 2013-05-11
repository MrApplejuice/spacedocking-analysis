#pragma once

#include <vector>
#include <string>

#include <boost/exception/all.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <SDL_mixer.h>

#include <ClassTools.hpp>
#include <engine/SystemBackend.hpp>

#include "extensions/CommandLineArguments.hpp"

/*
 * IMPORTANT
 *
 * Please note that this backend does not behave like expected by the 
 * SystemBackend description. Only one streaming media can be played at
 * a time with this implementation even though it should be possible to
 * play multiple at once. "Stop" indications are propergated correctly
 * however.
*/

typedef boost::error_info<struct tag_my_info, const std::string> LinuxBackendExceptionData;
struct LinuxBackendException : public boost::exception, public std::exception {};

namespace lbe {
  struct LinuxBackendSoundException : public boost::exception, public std::exception {};

  typedef std::vector<engine::StepCallback> StepCallbackVector;
  
  class Sound;
  class StreamInstance;
  class SoundBuffer;
  class StreamedSound;

  // Sound
  typedef boost::shared_ptr<Sound> SoundRef;
  typedef std::vector<SoundRef> SoundRefVector;

  typedef boost::shared_ptr<StreamInstance> StreamInstanceRef;
  typedef std::vector<StreamInstanceRef> StreamInstanceRefVector;
  
  /**
   * Abstract class describing the capabilities of a playing sound
   */
  class StreamInstance {
    private:
      Sound* srcSound;
    public:
      GEN_GETTER(Sound*, Sound, srcSound);
      
      virtual bool stop() = 0;
      virtual bool finished() = 0;
      
      StreamInstance(Sound& srcSound);
  };
  
  /**
   * Abstract class describing the capabilities of a general sound 
   * (stream or buffer)
   */
  class Sound {
    private:
    public:
      virtual StreamInstanceRef play() = 0;
  };

  /**
   * Representation of a statically load sound sample
   */
  class SoundBuffer : public Sound {
    private:
      std::string filename;
      Mix_Chunk* sample;
    public:
      GEN_GETTER(const std::string&, SoundFilename, filename);
    
      virtual StreamInstanceRef play();
      SoundBuffer(const std::string& filename);
      
      friend class SoundBufferStreamInstance;
  };
  
  /**
   * Representation of a currently playing SoundBuffer using a given
   * channel. The ctor throws an exception if the sound playback could 
   * not be started.
   */
  class SoundBufferStreamInstance : public StreamInstance {
    private:
      int channel;
      bool stopped;
    public:
      virtual bool stop();
      virtual bool finished();
      
      SoundBufferStreamInstance(SoundBuffer& srcSound);
  };

  class StreamedSoundStreamInstance;
  class CentralStreamedSoundStream;
  typedef boost::shared_ptr<CentralStreamedSoundStream> CentralStreamedSoundStreamInstanceRef;

  /**
   * Representation of a streamed sound sample
   */
  class StreamedSound : public Sound {
    private:
      std::string filename;
      Mix_Music* music;
      CentralStreamedSoundStreamInstanceRef sharedMusicStream;
    public:
      GEN_GETTER(const std::string&, MusicFilename, filename);
     
      virtual StreamInstanceRef play();
      StreamedSound(const std::string& filename, CentralStreamedSoundStreamInstanceRef sharedMusicStream);
      
      friend class StreamedSoundStreamInstance;
  };
  
  /**
   * Instance of a playing streamed sound (shared via the 
   * CentralStreamedSoundStream)
   */
  class StreamedSoundStreamInstance : public StreamInstance {
    private:
      bool stopped;
    public:
      virtual bool stop();
      virtual bool finished();

      StreamedSoundStreamInstance(StreamedSound& srcSound);
  };

  /**
   * Facade for the single-channel streamed music player offered by SDL.
   * Used by all streamed sound samples to play themselves. Stops still
   * playing streamed sounds before the new one is played.
   */
  class CentralStreamedSoundStream {
    private:
      StreamInstanceRef playingMusic;
    public:
      virtual StreamInstanceRef play(StreamedSound& music);
  };
}

/**
 * Linux implementation of a linux (SDL-based) backend to be used with
 * the game engine. Note that the audio backend only supports single
 * playback for streamed media, because of restrictions of SDL_mixer.
 */
class LinuxBackend : public engine::SystemBackend, public boost::enable_shared_from_this<LinuxBackend> {
  private:
    bool doRun;
  
    lbe::StepCallbackVector stepFuncs;
    
    lbe::SoundRefVector sounds;
    lbe::StreamInstanceRefVector soundPlaybacks;
    
    lbe::CentralStreamedSoundStreamInstanceRef centralStreamPlayer;
    
    engine::PointerDataVector pointerData;

    engine::KeyPressChangeVector keyChanges;
    
    std::vector<engine::EngineExtensionRef> staticExtensions;
    
    virtual int findFreePlaybackIndex();
  public:
    // Implementation of SystemBackend
    virtual void registerStepCallback(const engine::StepCallback& callback);
    virtual int allocateSoundBuffer(const std::string& soundpath);
    virtual int prepareSoundStream(const std::string& soundpath);
    virtual int play(int soundId);
    virtual bool stop(int playbackId);
    virtual bool isPlaying(int playbackId);
    virtual boost::shared_ptr<std::istream> openResource(const std::string& filename, bool binary=false);
    virtual engine::RawImageData loadImage(const std::string& filename);
    virtual const engine::PointerDataVector& getPointerData();
    virtual float getScreenAspectRatio();
    virtual engine::EngineExtensionRef getExtension(const std::string& extensionName);
    virtual boost::shared_ptr<std::iostream> openPersistentFile(const std::string& filename, std::ios_base::openmode openmode);
    virtual bool deletePersistentFile(const std::string& filename);
    virtual engine::KeyPressChangeVector getKeyPressChanges();
    
    // Runs until the game is terminated (e.g. the SDL window is closed)
    virtual void run();
    
    LinuxBackend(int argc, const char** argv);
    virtual ~LinuxBackend();
};


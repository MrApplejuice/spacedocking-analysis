#ifndef SYSTEMBACKEND_HPP_
#define SYSTEMBACKEND_HPP_

#include <vector>
#include <string>
#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/function.hpp>

namespace engine {
  enum KeyCode {
    KEY_UNKNOWN = 0,
    KEY_0,
    KEY_1,
    KEY_2,
    KEY_3,
    KEY_4,
    KEY_5,
    KEY_6,
    KEY_7,
    KEY_8,
    KEY_9,
    KEY_Q,
    KEY_W,
    KEY_E,
    KEY_R,
    KEY_T,
    KEY_Z,
    KEY_U,
    KEY_I,
    KEY_O,
    KEY_P,
    KEY_A,
    KEY_S,
    KEY_D,
    KEY_F,
    KEY_G,
    KEY_H,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_Y,
    KEY_X,
    KEY_C,
    KEY_V,
    KEY_B,
    KEY_N,
    KEY_M,
    KEY_SPACE,
    KEY_TAB,
    KEY_RETURN,
    KEY_ESC,
    KEY_BACKSPACE,
    KEY_DELETE,
    KEY_INSERT,
    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_PAGEUP,
    KEY_PAGEDOWN,
    KEY_LSHIFT,
    KEY_RSHIFT,
    KEY_LCTRL,
    KEY_RCTRL,
    KEY_LALT,
    KEY_RALT,
    KEY_CODE_COUNT
  };
  struct KeyPressChange {
    KeyCode keycode;
    bool pressState;
  };
  typedef std::vector<KeyPressChange> KeyPressChangeVector;
    
  typedef boost::function1<void, float> StepCallback;
  
  typedef unsigned char ChannelByte;
  
  /**
   * Structure that carries information about a load image.
   */
  struct RawImageData {
    /** Dimensions of the load image */
    unsigned int width, height;
    /** Array of (R,G,B,A) values representing the image */
    boost::shared_array<ChannelByte> pixels;
  };

  /**
   * Pointing device data that describes the state of a pointer
   * interacting with the game.
   */
  struct PointerData {
    /** Unique identifier idendtifying the current pointer (important for
     * multitouch/multi mouse interactions). Use this value for point tracking. */
    int id;
    /** Checks whether the button of the pointer is pressed or not */
    bool pressed;
    /** x-Position of the pointer in the range from -1.0 to 1.0 (from left to right)*/
    float x;
    /** y-Position of the pointer in the range from 1.0 to -1.0 (from top to bottom)*/
    float y;
  };

  /** List of pointer data structures. Multi pointer data can be obtained
   *  on systems that for support for example multitouch. */
  typedef std::vector<PointerData> PointerDataVector;

  /**
   * Empty interface class for GameEngine
   */
  class EngineBaseClass {
    public:
      virtual ~EngineBaseClass() {}
  };
  typedef boost::shared_ptr<EngineBaseClass> EngineBaseClassRef;

  /**
   * <p>Class that can be requested through an engine using the 
   * Engine::getExtension(const std::string&) function. The precise 
   * capabilities depend on the extension.</p>
   * 
   * <p>The Engine class calls as response to 
   * Engine::getExtension(const std::string&) the getExtension functions
   * of its system backend which delivers an instantiated EngineExtension
   * object.</p>
   */
  class EngineExtension {
    private:
    public:
      /**
       * Gets the name of this extension.
       * 
       * @return
       *   The name of this extension
       */
      virtual std::string getName() const = 0;
  };
  typedef boost::shared_ptr<EngineExtension> EngineExtensionRef;

  class SystemBackend {
    private:
    public:
      /**
       * Registers a new callback to be repeatadly called in the main 
       * program loop
       */
      virtual void registerStepCallback(const StepCallback& callback) = 0;
    
      /**
       * Loads a sound sample into memory and prepares to play it
       * 
       * @return
       *   SoundBuffer ID (its a soundid) that identifies the load sound 
       *   sample; 0 on error
       */
      virtual int allocateSoundBuffer(const std::string& soundpath) = 0;
      
      /**
       * Prepares a sound for streamed playback.
       * 
       * @return
       *   soundId for this streaming sound, 0 on error
       */
      virtual int prepareSoundStream(const std::string& soundpath) = 0;
      
      /**
       * Starts playback of a previously loaded sound sample
       * 
       * @param soundId
       *   Sound id to play (stream or buffer)
       * @return
       *   Sound playback id if succeeded, 0 on error
       */
      virtual int play(int soundId) = 0;
      
      /**
       * Allows to stop sound playback of a given sound sample or stream.
       * After calling stop the playbackId can be disposed by the implementation 
       * and should not be used anymore after the call to this function.
       * 
       * @param playbackId
       *   Id of the playback stream that should be stopped
       * 
       * @return
       *   True on success (sound was stopped), False on error (sound
       *   could not be stopped)
       */
      virtual bool stop(int playbackId) = 0;
      
      /**
       * Checks if the given stream still plays.
       * @param playbackId
       *   Playback stream id for that should be checked if it still plays
       * @return
       *   True if it still plays, False if it is/has stopped
       */
      virtual bool isPlaying(int playbackId) = 0;
       
      /**
       * Returns a file input stream to the resource referenced by its 
       * filename.
       * 
       * @param filename
       *   The filename of the resource to load
       * 
       * @return
       *   The open data stream to the resource or a NULL reference on error
       */
      virtual boost::shared_ptr<std::istream> openResource(const std::string& filename, bool binary = false) = 0;
      
      /**
       * Loads raw image data from an image. Raw image data consists of the 
       * dimensions of the load image and the pixel data that consists of 
       * a string of RGBA-values (in that order).
       * 
       * @param filename
       *   Image resource to load
       * @return
       *   RawImageData structure containing the raw image information.
       */
      virtual RawImageData loadImage(const std::string& filename) = 0;
      
      /**
       * Returns data about the system pointer devices that can be used.
       * These can be mouse(s) or touching points for multi touch systems.
       * The engine polls this function and compares the data to the
       * snapshot obtained during the previous cycle to check whether
       * any updates have to be made.
       * 
       * @return
       *   A possibly variable list containing data about all pointer inputs
       *   that currently are available.
       */
      virtual const PointerDataVector& getPointerData() = 0;
      
      /**
       * <p>Called by engine to ask for a specific engine extension. If the extension
       * is supported by the backend a shared_ptr (EngineExtensionRef)
       * can be returned to that object through this function.</p>
       * 
       * <p>This method is default implemented and never returns any 
       * extension references ( shared_ptr(NULL) ).</p>
       * 
       * @param extensionName
       *   The name/identifier of the extension of that an object instance
       *   should be obtained.
       * 
       * @return
       *   A NULL-shared_ptr if the extension is not supported. Otherswise
       *   this will contain a reference to extension object (must be
       *   typecasted to the proper type by the final game engine).
       */
      virtual EngineExtensionRef getExtension(const std::string& extensionName) {
        return EngineExtensionRef();
      }
      
      /**
       * <p>Returns the aspect ratio of the screen that is used for 
       * rendering OpenGL content. This is important for applications
       * to prevent distortions of images during presentation.</p>
       * 
       * @return
       *   The aspect ratio (width/height) of the screen used for 
       *   OpenGL
       */
      virtual float getScreenAspectRatio() = 0;
      
      /**
       * <p>If implemented this method should load a bound OpenGL texture
       * with the texture from the given file. This method can specifically
       * be used to speed up the loading process on handheld devices
       * by using preconverted texture formats instead of the original
       * png files (for example, the iPhone's pvrtc format).</p>
       * 
       * @param filename
       *   Texture filename to load
       * @param width
       *   If loading succeeds, the width of the load texture in pixels
       * @param height
       *   If loading succeeds, the height of the load texture in pixels
       * @return
       *   If the texture was load succesfully by this function the
       *   result is true. If the function is not implemeted or an error
       *   occurred during the loading process, false is returned.
       */
      virtual bool nativeLoadTex2D(const std::string& filename, unsigned int& width, unsigned int& height) {
        return false;
      }
      
      /**
       * Opens a file in read and/or write mode that is persistent after
       * the application has been closed. The path to filename may not 
       * be part of a sub directory (at the moment, since those cannot be
       * created at the current version of the system backend 
       * specification).
       * 
       * @param filename
       *   Filename of the persistent file to open
       * @param openmode
       *   Mode to open the file in
       * @return
       *   On success the shared_ptr holds a reference to the open 
       *   iostream accessing the persistent file. On failure the 
       *   function returns a NULL-pointer.
       */
      virtual boost::shared_ptr<std::iostream> openPersistentFile(const std::string& filename, std::ios_base::openmode openmode) = 0;
      
      /**
       * Deletes a persistent file. The file must not be opened while
       * deleting or this function fails.
       * 
       * @param filename
       *   The persistent file that should be deleted
       * @return
       *   True on success, false on failure
       */
      virtual bool deletePersistentFile(const std::string& filename) = 0;
      
      /**
       * Returns all keyboard events that ocurred since the 
       * last call the step functions.
       * 
       * @return
       *   Array of timely ordered, successive keyboard events.
       */
      virtual KeyPressChangeVector getKeyPressChanges() = 0;
  };

  typedef boost::shared_ptr<SystemBackend> SystemBackendRef;
  extern EngineBaseClassRef createEngine(SystemBackendRef backend);
}

#endif // SYSTEMBACKEND_HPP_

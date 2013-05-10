#include <SDL.h>

#ifdef TARGET_WINDOWS
#include <SDL_main.h>
#endif

#include <SDL_mixer.h>

#include <OpenGL.hpp>

#include <engine/Engine.hpp>

#include "SDLBackend.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

using boost::shared_ptr;

using namespace engine;

int main(int argc, char** argv) {
  try {
    shared_ptr<LinuxBackend> linuxBackend (new LinuxBackend());
    EngineBaseClassRef baseClass = createEngine(linuxBackend);
    linuxBackend->run();
    baseClass.reset();
    linuxBackend.reset();
  }
  catch (LinuxBackendException& e) {
    if (const string* errorMsg = boost::get_error_info<LinuxBackendExceptionData>(e)) {
      cerr << "Error during initialization: " << *errorMsg << endl;
    } else {
      cerr << "Error during error handling: LinuxBackendException had no data" << endl;
    }
  }

  return 0;
}

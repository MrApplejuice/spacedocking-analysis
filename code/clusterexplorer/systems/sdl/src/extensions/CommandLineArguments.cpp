#include "CommandLineArguments.hpp"

namespace lbe {
  const engine::ext::CommandLineArguments::CommandLineArgumentsVector& CommandLineArguments :: getCommandLineArguments() const {
    return arguments;
  }
  
  CommandLineArguments :: CommandLineArguments(int argc, const char** args) {
    arguments.reserve(argc);
    for (const char** arg = args; arg != args + argc; arg++) {
      arguments.push_back(std::string(*arg));
    }
  }
}

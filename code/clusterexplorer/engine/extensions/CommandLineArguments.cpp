#include "CommandLineArguments.hpp"

namespace engine {
  namespace ext {
    const std::string CommandLineArguments :: NAME = "Command Line Arguments Extension";
    
    std::string CommandLineArguments :: getName() const {
      return NAME;
    }
  }
}

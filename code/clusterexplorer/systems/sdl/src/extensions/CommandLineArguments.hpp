#pragma once

#include <engine/extensions/CommandLineArguments.hpp>

namespace lbe {
  class CommandLineArguments : public virtual engine::ext::CommandLineArguments {
    private:
      engine::ext::CommandLineArguments::CommandLineArgumentsVector arguments;
    public:
      virtual const CommandLineArgumentsVector& getCommandLineArguments() const;
      
      CommandLineArguments(int argc, const char** args);
  };
}

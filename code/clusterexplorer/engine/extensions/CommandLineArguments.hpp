#pragma once

#include <string>
#include <vector>

#include <boost//shared_ptr.hpp>

#include "../SystemBackend.hpp"

namespace engine {
  namespace ext {
    class CommandLineArguments : public virtual EngineExtension {
      public:
        const static std::string NAME;
        
        virtual std::string getName() const;
        
        typedef std::vector<std::string> CommandLineArgumentsVector;
        typedef CommandLineArgumentsVector List;
        virtual const CommandLineArgumentsVector& getCommandLineArguments() const = 0;
    };
    
    typedef boost::shared_ptr<CommandLineArguments> CommandLineArgumentsRef;
  }
}

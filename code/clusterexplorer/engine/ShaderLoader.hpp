#ifndef SHADERLOADER_HPP_
#define SHADERLOADER_HPP_

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/exception/all.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include <OpenGL.hpp>

#include "libs/glm/glm.hpp"
#include "libs/glm/ext.hpp"

namespace engine {
  typedef boost::error_info<struct tag_my_info, const std::string> ShaderExceptionData;
  struct ShaderException : public virtual boost::exception, public virtual std::exception {};

  namespace internal {
    class ShaderReferenceContainer : public boost::noncopyable {
      public:
        typedef boost::function0<GLuint> CreateFunction;
        typedef boost::function1<void, GLuint> DeleteFunction;
      private:
        GLuint shaderRef;
        DeleteFunction deleteFunction;
      public:
        virtual GLuint getRef() const;
        
        ShaderReferenceContainer(CreateFunction creationFunction, DeleteFunction deleteFunction);
        virtual ~ShaderReferenceContainer();
    };
    typedef boost::shared_ptr<ShaderReferenceContainer> ShaderReferenceContainerRef;
  }
  
  class Shader {
    private:
      internal::ShaderReferenceContainerRef shaderRef;
    public:
      typedef boost::shared_ptr<std::istream> InstreamRef;
      typedef boost::function1<InstreamRef, std::string> OpenFileFunctionType;

      bool operator<(const Shader& other) const;
      bool operator==(const Shader& other) const;
    
      operator bool() const;
      virtual GLuint getShaderRef() const;
    
      Shader();
      Shader(GLenum shaderType);
      Shader(OpenFileFunctionType openFileFunction, const std::string& filename);
      virtual ~Shader();
  };
  
  class ShaderProgram {
    public:
      class AttribLocationAssigner {
        public:
          struct NameIndexPair {
            std::string name;
            GLuint index;
          };
          typedef std::vector<NameIndexPair> NameIndexPairVector;
          
          virtual NameIndexPairVector getNameIndexPairs() const = 0;
          virtual const std::string& getComparisonString() const = 0;
          
          bool operator==(const AttribLocationAssigner& other) const {
            return getComparisonString() == other.getComparisonString();
          }

          bool operator<(const AttribLocationAssigner& other) const {
            return getComparisonString() < other.getComparisonString();
          }
      };
      typedef boost::shared_ptr<AttribLocationAssigner> AttribLocationAssignerRef;

      struct ProgramConfiguration {
        typedef std::vector<Shader> ShaderVector;
        
        ShaderVector shaders;
        AttribLocationAssignerRef locationAssigner;
        
        bool operator<(const ProgramConfiguration& other) const {
          if (shaders < other.shaders) {
            return true;
          }
          if (!locationAssigner && !(other.locationAssigner)) {
            return false;
          }
          if (!locationAssigner) {
            return true;
          }
          if (!other.locationAssigner) {
            return false;
          }
          if (*locationAssigner < *(other.locationAssigner)) {
            return true;
          }
          return false;
        }
        
        bool operator==(const ProgramConfiguration& other) const {
          if (shaders != other.shaders) {
            return false;
          }
          if (locationAssigner.get() == other.locationAssigner.get()) {
            return true;
          }
          if (!locationAssigner || !other.locationAssigner) {
            return false;
          }
          if (*locationAssigner == *(other.locationAssigner)) {
            return true;
          }
          return false;
        }

        ProgramConfiguration() : shaders(), locationAssigner() {
        }
        
        template <typename Iterator, typename T>
        ProgramConfiguration(Iterator begin, Iterator end, const T& _locationAssigner) : shaders(begin, end), locationAssigner(new T()) {
          std::sort(shaders.begin(), shaders.end()); // Sort vector to make it comparable
          *(locationAssigner.get()) = _locationAssigner;
        }
      };
      
      class UniformAccessor {
        public:
          struct UniformSetter {
            private:
              GLuint vIndex;
              UniformSetter(GLuint vIndex) : vIndex(vIndex) {}
            public:
              /**
               * Handler for single integer
               */
              UniformSetter& operator=(const int& value) {
                glUniform1i(vIndex, value);
                return *this;
              }
              
              /**
               * Handler for single float
               */
              UniformSetter& operator=(const float& value) {
                glUniform1f(vIndex, value);
                return *this;
              }
              
              /**
               * Handler for single double
               */
              UniformSetter& operator=(const double& value) {
                glUniform1f(vIndex, value);
                return *this;
              }
              
              /**
               * Handler for single glm::vec2
               */
              UniformSetter& operator=(const glm::vec2& v) {
                glUniform2fv(vIndex, 1, glm::value_ptr(v));
                return *this;
              }
              
              /**
               * Handler for single glm::vec3
               */
              UniformSetter& operator=(const glm::vec3& v) {
                glUniform3fv(vIndex, 1, glm::value_ptr(v));
                return *this;
              }
              
              /**
               * Handler for single glm::vec4
               */
              UniformSetter& operator=(const glm::vec4& v) {
                glUniform4fv(vIndex, 1, glm::value_ptr(v));
                return *this;
              }
              
              /**
               * Handler for single glm::mat2
               */
              UniformSetter& operator=(const glm::mat2& m) {
                glUniformMatrix2fv(vIndex, 1, GL_FALSE, glm::value_ptr(m));
                return *this;
              }
              
              /**
               * Handler for single glm::mat3
               */
              UniformSetter& operator=(const glm::mat3& m) {
                glUniformMatrix3fv(vIndex, 1, GL_FALSE, glm::value_ptr(m));
                return *this;
              }
              
              /**
               * Handler for single glm::mat4
               */
              UniformSetter& operator=(const glm::mat4& m) {
                glUniformMatrix4fv(vIndex, 1, GL_FALSE, glm::value_ptr(m));
                return *this;
              }
              
              friend class UniformAccessor;
          };
        private:
          GLuint shaderProgram;
          
          UniformAccessor(GLuint shaderProgram) : shaderProgram(shaderProgram) {}
        public:
          UniformSetter operator[](const std::string& uniform) const {
            assert(shaderProgram);
            
            glUseProgram(shaderProgram);
            GLint result = glGetUniformLocation(shaderProgram, uniform.c_str());
            if (result < 0) {
              throw ShaderException() << ShaderExceptionData(std::string("Uniform ") + uniform + " does not exist in shader program");
            }
            return UniformSetter((GLuint) result);
          }
          
          friend class ShaderProgram;
      };
    private:
      ProgramConfiguration programConfig;
      internal::ShaderReferenceContainerRef programReference;
      
      virtual void init(const AttribLocationAssigner& attributeAssignment);
      
      ShaderProgram(GLuint program);
    public:
      UniformAccessor uniforms;
    
      operator bool() const;
      virtual GLuint getProgramRef() const;
    
      virtual void install() const;
      virtual GLint getUniformLocation(const char* str) const;
      virtual GLint getUniformLocation(const std::string& str) const;
    
      ShaderProgram();
      template <typename Iterator, typename T>
      ShaderProgram(Iterator b, Iterator e, const T& attribAssigner) : programConfig(b, e, attribAssigner), programReference(), uniforms(0) {
        init(attribAssigner);
      }
      virtual ~ShaderProgram();
      
      static ShaderProgram current() {
        GLint program;
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        return ShaderProgram(program);
      }
  };
  
  class MapAttribLocationAssigner : public ShaderProgram::AttribLocationAssigner {
    public:
      struct MapPair {
        std::string name;
        GLuint mappedValue;
        
        bool operator==(const MapPair& other) const {
          return (name == other.name) && (mappedValue == other.mappedValue);
        }
        
        bool operator<(const MapPair& other) const {
          return (mappedValue < other.mappedValue) || ((mappedValue == other.mappedValue) && (name < other.name));
        }
        
        MapPair(const std::string& name, GLuint mappedValue) : name(name), mappedValue(mappedValue) {}
      };
      
      typedef std::vector<MapPair> MapType;
    private:
      int index;
      MapType map;
      std::string comparisonString;
      
      virtual void updateComparisonString() {
        comparisonString = "MapAttribLocationAssigner:";
        BOOST_FOREACH(const MapPair& p, map) {
          comparisonString += p.name + "->" + boost::lexical_cast<std::string>(p.mappedValue) + ";";
        }
      }
    public:
      virtual ShaderProgram::AttribLocationAssigner::NameIndexPairVector getNameIndexPairs() const {
        ShaderProgram::AttribLocationAssigner::NameIndexPairVector result;
        BOOST_FOREACH(const MapPair& mp, map) {
          ShaderProgram::AttribLocationAssigner::NameIndexPair nip;
          nip.index = mp.mappedValue;
          nip.name = mp.name;
          result.push_back(nip);
        }
        return result;
      }

      virtual const std::string& getComparisonString() const {
        return comparisonString;
      }

      MapAttribLocationAssigner& addMapping(const std::string name, GLuint index) {
        MapPair mapPair(name, index);
        
        // Insert at correct position to maintain ordening of vector
        MapType::iterator it;
        for (it = map.begin(); (it != map.end()) && (mapPair < *it); it++) {}
        if (it == map.end()) {
          map.push_back(mapPair);
        } else {
          map.insert(it, mapPair);
        }
        
        updateComparisonString();
        
        return *this;
      }
      
      template <typename Iterator>
      MapAttribLocationAssigner(Iterator begin, Iterator end) : index(0), map(begin, end) {
        std::sort(map.begin(), map.end()); // Sort to make comparable
        updateComparisonString();
      }

      MapAttribLocationAssigner() : index(0), map() {}
  };

  class ShaderManager {
    public:
      typedef Shader::InstreamRef InstreamRef;
      typedef Shader::OpenFileFunctionType OpenFileFunctionType;
    private:
      typedef std::map<std::string, Shader> ShaderFilenameMap;
      typedef std::map<ShaderProgram::ProgramConfiguration, ShaderProgram> ProgramShaderMap;
      
      OpenFileFunctionType openfileFunction;
      
      ShaderFilenameMap shaderMap;
      ProgramShaderMap shaderProgramMap;
    public:
      virtual Shader getShader(std::string shaderFileName);
      
      template <typename T>
      ShaderProgram compileProgram(const Shader* s, size_t count, const T& attribAssigner) {
        ShaderProgram::ProgramConfiguration progConfig(s, s + count, attribAssigner);
        
        ProgramShaderMap::iterator found = shaderProgramMap.find(progConfig);
        if (found == shaderProgramMap.end()) {
          try {
            ShaderProgram shaderProgram(s, s + count, attribAssigner);
            shaderProgramMap[progConfig] = shaderProgram;
            return shaderProgram;
          }
          catch (ShaderException& e) {
            std::cerr << "Could not compile shader program :" << std::endl;
            if (const std::string* msg = boost::get_error_info<ShaderExceptionData>(e)) {
              std::cerr << *msg;
            } else {
              std::cerr << "[no msg]";
            }
            std::cerr << std::endl;
          }
          
          return ShaderProgram();
        }
    
        return found->second;
      }
        
      template <typename T>
      ShaderProgram compileProgram(const Shader& s1, const T& attribAssigner) {
        return compileProgram(&s1, 1, attribAssigner);
      }
      
      template <typename T>
      ShaderProgram compileProgram(const Shader& s1, const Shader& s2, const T& attribAssigner) {
        Shader s[2] = {s1, s2};
        return compileProgram(s, 2, attribAssigner);
      }

      template <typename T>
      ShaderProgram compileProgram(const Shader& s1, const Shader& s2, const Shader& s3, const T& attribAssigner) {
        Shader s[3] = {s1, s2, s3};
        return compileProgram(s, 3, attribAssigner);
      }

      ShaderManager(OpenFileFunctionType openfileFunction);
      virtual ~ShaderManager();
  };
}

#endif // SHADERLOADER_HPP_

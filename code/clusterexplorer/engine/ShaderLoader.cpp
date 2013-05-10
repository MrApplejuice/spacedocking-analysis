#include "ShaderLoader.hpp"

#include <string>
#include <vector>
#include <algorithm>

#include <boost/regex.hpp>
#include <boost/bind.hpp>
#include <boost/shared_array.hpp>
#include <boost/algorithm/string.hpp>

#include "local/Global.hpp"

using namespace std;
using namespace boost;

namespace engine {
  static GLenum parseShaderTypeLine(string line) {
    assert((GL_VERTEX_SHADER != 0) && (GL_FRAGMENT_SHADER != 0));
    
    static const regex VertexShaderMarker("^//\\s*vertex shader(\\s.*|)$");
    static const regex FragmentShaderMarker("^//\\s*fragment shader(\\s.*|)$");
    
    trim(line);
    to_lower(line);
    
    if (regex_match(line, VertexShaderMarker)) {
      return GL_VERTEX_SHADER;
    }
    if (regex_match(line, FragmentShaderMarker)) {
      return GL_FRAGMENT_SHADER;
    }
  
    return 0;
  }

  typedef vector<string> LibFileList;
  typedef vector<string> LibCodeStringList;

  static void loadShaderfile(istream& inputStream, GLenum& type, string& content, string& versionLine, LibFileList& libs) {
    static const regex VersionConstDetector("^\\s*#version\\s+max_version\\s*$");
    static const regex VersionLine("^\\s*#version\\s+.*$");
    static const regex PreprocessorCommand("^\\s*#[a-zA-Z].*$");
    static const regex LibIncludeLine("^\\s*#include\\s+\"([^\"]+)\"\\s*$");

    content = "";
    versionLine = "";
    libs.clear();
    
    if (inputStream.good()) {
      while (!inputStream.eof() && inputStream.good()) {
        string line;
        getline(inputStream, line);
        bool doCommentLine = false;
        
        if (!type) {
          type = parseShaderTypeLine(line);
        }

        // Try to detect version string variable
        if (regex_match(line, VersionConstDetector)) {
          static const string SHADER_VERSION_STRING = "#version "
#if (USE_OPENGLES >= 200)
"100"
#elif (OPENGL_VERSION >= 330)
"150 core"
#endif
;
          line = SHADER_VERSION_STRING;
        }

        // Parse comment preprocessor command
        if (regex_match(line, PreprocessorCommand)) {
          doCommentLine = true;
          match_results<string::const_iterator> matches;
          if (regex_match(line, matches, LibIncludeLine)) {
            const string libName = matches[1];
            if (find(libs.begin(), libs.end(), libName) == libs.end()) { // Only add every lib once
              libs.push_back(libName);
            }
          } else {
            doCommentLine = false;
          }
        }

        // extract version line
        if (versionLine.empty()) {
          if (regex_match(line, VersionLine)) {
            versionLine = line;
            doCommentLine = true;
          }
        }
        
        // Comment line if it was processed by internal parser
        if (doCommentLine) {
          line = string("// ") + line;
        }

        content += line + "\n";
      }

      if (!inputStream.eof()) {
        content = "";
        throw ShaderException() << ShaderExceptionData(string("Failed to load shader"));
      }
      if (!type) {
        throw ShaderException() << ShaderExceptionData(string("Missing or invalid shader type definition line"));
      }
    } else {
      throw ShaderException() << ShaderExceptionData(string("Stream not ready"));
    }
  }

  static void loadLibraries(Shader::OpenFileFunctionType openfileFunction, LibCodeStringList& libTexts, LibFileList& loadedLibs, const LibFileList& libsToLoad, LibFileList& loadingTrace) {
    foreach (const string& libToLoad, libsToLoad) {
      if (find(loadingTrace.begin(), loadingTrace.end(), libToLoad) != loadingTrace.end()) {
        throw ShaderException() << ShaderExceptionData(libToLoad + " Error: circular dependency");
      }
      
      if (find(loadedLibs.begin(), loadedLibs.end(), libToLoad) == loadedLibs.end()) { // Check if lib was already loaded
        try {
          loadingTrace.push_back(libToLoad);
          
          Shader::InstreamRef instreamRef = openfileFunction(libToLoad);
          if (!instreamRef) {
            throw ShaderException() << ShaderExceptionData(string("Cannot open shader lib: ") + libToLoad);
          }
          
          GLenum shaderType = GL_FRAGMENT_SHADER; // Set dummy type - lib files do not need type
          LibFileList subLibs;
          string libText, versionLine;
          loadShaderfile(*instreamRef, shaderType, libText, versionLine, subLibs); // Throws if anything fails
          
          // Load "subLibs"
          loadLibraries(openfileFunction, libTexts, loadedLibs, subLibs, loadingTrace);
          
          libTexts.push_back(libText);
          loadedLibs.push_back(libToLoad);

          loadingTrace.pop_back();
        }
        catch (ShaderException& e) { // Create trace
          string message;
          if (const string* m = get_error_info<ShaderExceptionData>(e)) {
            message = *m;
          } else {
            message = "[No message]";
          }
          throw ShaderException() << ShaderExceptionData(libToLoad + " -> " + message);
        }
      }
    }
  }

  static bool compileShader(string shaderCode, string versionLine, GLuint shader, string& error, const LibCodeStringList& libTexts) {
    char* infoBuffer = NULL;
    bool result = false;
    
    try {
      static const char* SHADER_DEFINES = "#define TARGET_" TARGET_PLATFORM_NAME "\n";
      
      const string definesSource = versionLine + "\n" + SHADER_DEFINES;
      
      // Create program text with includes
      {
        shared_array<const char*> sourceCodePtrs(new const char*[2 + libTexts.size()]);
        unsigned int sourceCodePtrsI = 0;
        sourceCodePtrs[sourceCodePtrsI++] = definesSource.c_str();
        foreach (const string& libText, libTexts) {
          sourceCodePtrs[sourceCodePtrsI++] = libText.c_str();
        }
        sourceCodePtrs[sourceCodePtrsI++] = shaderCode.c_str();
        
        glShaderSource(shader, sourceCodePtrsI, sourceCodePtrs.get(), NULL);
      }
      
      glCompileShader(shader);
      {
        GLint compileState;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compileState);
        if (compileState == GL_FALSE) {
          const char* BASE_ERROR_MSG = "Error:\n";
          const size_t BASE_ERROR_MSG_LEN = strlen(BASE_ERROR_MSG);
          const char* EMPTY_MSG = "[None]";
          const size_t EMPTY_MSG_LEN = strlen(EMPTY_MSG);
          
          GLint shaderInfoLength;
          glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &shaderInfoLength);
          
          if (shaderInfoLength <= 1) { // Ensure there is at least enough space for a "no message" message
            shaderInfoLength = EMPTY_MSG_LEN + 1;
            
            infoBuffer = new char[shaderInfoLength + BASE_ERROR_MSG_LEN];
            copy(EMPTY_MSG, EMPTY_MSG + EMPTY_MSG_LEN, infoBuffer + BASE_ERROR_MSG_LEN);
          } else {
            infoBuffer = new char[shaderInfoLength + BASE_ERROR_MSG_LEN];
            glGetShaderInfoLog(shader, shaderInfoLength, NULL, infoBuffer + BASE_ERROR_MSG_LEN);
          }
          
          copy(BASE_ERROR_MSG, BASE_ERROR_MSG + BASE_ERROR_MSG_LEN, infoBuffer);
          infoBuffer[BASE_ERROR_MSG_LEN + shaderInfoLength - 1] = 0; // Ensure 0-zerominatedness
          
          throw (const char*) infoBuffer;
        }
      }
      
      result = true;
    }
    catch (const char* m) {
      error = string(m);
    }
    
    if (infoBuffer) {
      delete[] infoBuffer;
    }
    
    return result;
  }


  namespace internal {
    GLuint ShaderReferenceContainer :: getRef() const {
      return shaderRef;
    }

    
    ShaderReferenceContainer :: ShaderReferenceContainer(CreateFunction creationFunction, DeleteFunction deleteFunction) : shaderRef(0), deleteFunction(deleteFunction) {
      shaderRef = creationFunction();
      if (!shaderRef) {
        throw ShaderException() << ShaderExceptionData("Cannot create new OpenGL shader");
      }
    }

    ShaderReferenceContainer :: ~ShaderReferenceContainer() {
      deleteFunction(shaderRef);
    }
  }

  using namespace internal;

  bool Shader :: operator<(const Shader& other) const {
    return getShaderRef() < other.getShaderRef(); // Compare integer references (zero result from getShaderRef covers invalid refs)
  }

  bool Shader :: operator==(const Shader& other) const {
    return getShaderRef() == other.getShaderRef();
  }

  Shader :: operator bool() const {
    return (bool) shaderRef;
  }
  
  GLuint Shader :: getShaderRef() const {
    if (shaderRef) {
      return shaderRef->getRef();
    }

    // Can be reached through defualt constructor    
    return 0; // Return invalid shader ref
  }

  Shader :: Shader() {
  }
  
  Shader :: Shader(GLenum shaderType) : shaderRef(new ShaderReferenceContainer(bind<GLuint>(glCreateShader, shaderType), glDeleteShader)) {
  }

  Shader :: Shader(OpenFileFunctionType openfileFunction, const std::string& filename) {
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Called Shader::Shader with uncleared glError" << endl;
    }
        
    string shaderText, versionLine;
    GLenum shaderType = 0;
    LibFileList libs;
    
    {
      InstreamRef instreamRef = openfileFunction(filename);
      if (!instreamRef) {
        throw ShaderException() << ShaderExceptionData(string("Cannot open shader: ") + filename);
      }
      loadShaderfile(*instreamRef, shaderType, shaderText, versionLine, libs); // Throws if anything fails
    }

    LibCodeStringList libTexts;
    try {
      LibFileList loadingTrace, loadedLibs;
      loadLibraries(openfileFunction, libTexts, loadedLibs, libs, loadingTrace);
    }
    catch (ShaderException& e) {
      string message = "Error: Cannot load libraries for " + filename + ": ";
      if (const string* s = get_error_info<ShaderExceptionData>(e)) {
        message += *s;
      } else {
        message += "[No message]";
      }
      throw ShaderException() << ShaderExceptionData(message);
    }

    ShaderReferenceContainerRef tmpShaderRef(new ShaderReferenceContainer(bind<GLuint>(glCreateShader, shaderType), glDeleteShader));
    string compileError;
    if (compileShader(shaderText, versionLine, tmpShaderRef->getRef(), compileError, libTexts)) {
      shaderRef = tmpShaderRef;
    } else {
      throw ShaderException() << ShaderExceptionData(string("Error compiling shader: ") + compileError);
    }
  }

  Shader :: ~Shader() {
    // Everything is autoreleased
  }


  void ShaderProgram :: init(const AttribLocationAssigner& attributeAssignment) {
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Called ShaderProgram::ShaderProgram with uncleared glError" << endl;
    }
    
    programReference = ShaderReferenceContainerRef(new ShaderReferenceContainer(glCreateProgram, glDeleteProgram));
    
    if (programReference) { // Try to initialize and link program
      GLuint renderProgram = programReference->getRef();
    
      // Attach shaders
      foreach (Shader& shader, programConfig.shaders) {
        glAttachShader(renderProgram, shader.getShaderRef());
      }

      // Load mappings
      foreach (const ShaderProgram::AttribLocationAssigner::NameIndexPair& nip, attributeAssignment.getNameIndexPairs()) {
        //cout << "Binding index " << nip.index << " to " << nip.name << endl;
        glBindAttribLocation(renderProgram, nip.index, nip.name.c_str());
        if (glGetError() != GL_NO_ERROR) {
          cerr << "Warning! Cannot bind attribute index " << nip.index << " to shader variable " << nip.name << endl;
        }
      }

      glLinkProgram(renderProgram);
      {
        string errorString;
        
        GLint linkStatus;
        glGetProgramiv(renderProgram, GL_LINK_STATUS, &linkStatus);
        if (linkStatus == GL_FALSE) {
          errorString = "Error linking program:\n";
          
          GLint programLogLength; 
          glGetProgramiv(renderProgram, GL_INFO_LOG_LENGTH, &programLogLength);
          
          if (programLogLength > 0) {
            char* infoBuffer = new char[programLogLength];
            glGetProgramInfoLog(renderProgram, programLogLength, NULL, infoBuffer);
            errorString += infoBuffer;
            delete[] infoBuffer;
          }
          
          throw ShaderException() << ShaderExceptionData(errorString); // Linking failed
        }
        
        uniforms = UniformAccessor(getProgramRef()); 
      }
    }
  }
  
  static GLuint returnGLuint(GLuint u) {
    return u;
  }
  
  static void skip() {}
  
  ShaderProgram :: ShaderProgram(GLuint program) : programConfig(), programReference(), uniforms(0) {
    if (!program || !glIsProgram(program)) {
      throw ShaderException() << ShaderExceptionData("Tried to wrap invalid program");
    }
    programReference = internal::ShaderReferenceContainerRef(new internal::ShaderReferenceContainer(bind<GLuint>(returnGLuint, program), bind<void>(skip)));
    uniforms = UniformAccessor(getProgramRef()); 
  }
  
  ShaderProgram :: operator bool() const {
    return (bool) programReference;
  }
  
  GLuint ShaderProgram :: getProgramRef() const {
    if (programReference) {
      return programReference->getRef();
    }
    
//    cerr << "Warning! Invalid programReference in " << __FUNCTION__ << " " << __FILE__ << ":" << __LINE__ << " - execution should never reach this code" << endl;
    return 0; // Return invalid program ref
  }
  
  void ShaderProgram :: install() const {
    if (programReference) {
      glUseProgram(getProgramRef());
    } else {
      cerr << "Warning! Called install on program with invalid program reference" << endl;
    }
  }
  
  GLint ShaderProgram :: getUniformLocation(const char* str) const {
    if (programReference) {
      return glGetUniformLocation(getProgramRef(), str);
    } else {
      cerr << "Warning! Called install on program with invalid program reference" << endl;
      return -2;
    }
  }

  GLint ShaderProgram :: getUniformLocation(const std::string& str) const {
    return getUniformLocation(str.c_str());
  }
  
  ShaderProgram :: ShaderProgram() : programConfig(), programReference(), uniforms(0) {
  }

  ShaderProgram :: ~ShaderProgram() {
    // Everything is autoreleased
  }


  Shader ShaderManager :: getShader(std::string shaderFileName) {
    ShaderFilenameMap::iterator found = shaderMap.find(shaderFileName);
    if (found == shaderMap.end()) {
      // Try to load new one
      try {
        Shader newShader(openfileFunction, shaderFileName);
        shaderMap[shaderFileName] = newShader;
        return newShader;          
      }
      catch (ShaderException& e) {
        cerr << "Could not load shader " << shaderFileName << ":" << endl;
        if (const string* msg = get_error_info<ShaderExceptionData>(e)) {
          cerr << *msg;
        } else {
          cerr << "[no msg]";
        }
        cerr << endl;
      }
      
      return Shader();
    }
    
    return found->second;
  }
  
  ShaderManager :: ShaderManager(OpenFileFunctionType openfileFunction) : openfileFunction(openfileFunction), shaderMap() {
  }
  
  ShaderManager :: ~ShaderManager() {
  }
}

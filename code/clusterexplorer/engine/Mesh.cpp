/** \file
 * 
 * Changelog
 *   10-04-2012 Added caching of sub-object meshes
 *                (Paul Konstantin Gerke)
 */
#include "Mesh.hpp"

#include <functional>
#include <limits>
#include <fstream>

#include <OpenGL.hpp>
  
#include "local/Global.hpp"

#ifdef DEBUG
#define DEBUG_STRING(s) cerr << "Mesh DEBUG: " << s << endl
#else
#define DEBUG_STRING(s) do {} while(0)
#endif

using namespace std;
using namespace boost;

/** */
namespace engine {
  /* Attribute indexes as used by Mesh-class to serialize data to shaders*/
  namespace internal {
    typedef unsigned short IndexBufferIndexType;
    
    static GLuint retrieveGLRef(const GLBuffer& buffer) {
      return buffer.glRef;
    }
    
    void MeshMaterialGLBuffers :: freeBuffers() throw() {
      GLuint bufferRefs[indexBuffers.size()];
      transform(indexBuffers.begin(), indexBuffers.end(), bufferRefs, &retrieveGLRef);
      glDeleteBuffers(indexBuffers.size(), bufferRefs);
      indexBuffers.clear();
    }
    
    void MeshMaterialGLBuffers :: addBuffer(const MeshData::MaterialGroup& materialGroup) {
      try {
        IndexBufferIndexType vertexIndexes[materialGroup.faces.size() * 3]; // All triangles
        
        {
          GLBuffer newBuffer;
          newBuffer.len = materialGroup.faces.size() * 3;
          newBuffer.size = newBuffer.len * sizeof(*vertexIndexes);

          glGenBuffers(1, &newBuffer.glRef);
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not create OpenGL index buffer");
          }
          indexBuffers.push_back(newBuffer);
        }
        GLBuffer& newBuffer = indexBuffers[indexBuffers.size() - 1];
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, newBuffer.glRef);
        if (glGetError() != GL_NO_ERROR) {
          throw MeshException() << MeshExceptionData("Could not bind new index buffer");
        }

        // Accumulate indexes
        {
          int i = 0;
          foreach (const MeshData::TriangleIndexes& tri, materialGroup.faces) {
            copy(tri.indexes, tri.indexes + 3, vertexIndexes + i * 3);
            i++;
          }
        }
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, newBuffer.size, vertexIndexes, GL_STATIC_DRAW);
        if (glGetError() != GL_NO_ERROR) {
          throw MeshException() << MeshExceptionData("Could fill indexbuffer with data");
        }
        
        // Unbind again to prevent bugs
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
      }
      catch (...) {
        freeBuffers();
        throw;
      }
    }

    void MeshMaterialGLBuffers :: draw(unsigned int i) const {
      if ((i < 0) || (i >= indexBuffers.size())) {
        throw out_of_range("MeshMaterialGLBuffers :: draw(i) - i out of bounds");
      }
      
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffers[i].glRef);
      if (glGetError() != GL_NO_ERROR) {
        throw MeshException() << MeshExceptionData("Could not bind index buffer");
      }

      try {      
        glDrawElements(GL_TRIANGLES, indexBuffers[i].len, GL_UNSIGNED_SHORT, NULL);
        if (glGetError() != GL_NO_ERROR) {
          throw MeshException() << MeshExceptionData("Could not draw primitives");
        }
      }
      catch (...) {
        // Unbind buffer - do not care as much about errors - there however should not be any
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glGetError(); // Don't leave any errors on the "error stack"
        throw;
      }
    }
    
    MeshMaterialGLBuffers :: MeshMaterialGLBuffers() : indexBuffers() {
    }

    MeshMaterialGLBuffers :: ~MeshMaterialGLBuffers() {
      freeBuffers();
    }


    void MeshVertexBufferDrawConfig :: cleanup() const {
      const MeshVertexFormat& format = vb->getMeshData()->getFormat();
//      size_t stride = vb->getMeshData()->getVertexStride();
      
      if (format.getVertexCount() > 0) {
        #ifdef USE_GL33_GLES20_CODE
          glDisableVertexAttribArray(ATI_VERTEX); 
        #else
          glDisableClientState(GL_VERTEX_ARRAY);
        #endif
      }
      
      if (format.getNormalCount() > 0) {
        #ifdef USE_GL33_GLES20_CODE
          glDisableVertexAttribArray(ATI_NORMAL); 
        #else
          glDisableClientState(GL_NORMAL_ARRAY);
        #endif
      }
      
      if (format.getColorCount() > 0) { 
        #ifdef USE_GL33_GLES20_CODE
          glDisableVertexAttribArray(ATI_COLOR);
        #else
          glDisableClientState(GL_COLOR_ARRAY);
        #endif
      } 
      
      if (format.getTextureCoordinateCount() > 0) {
        #ifdef USE_GL33_GLES20_CODE
          for (unsigned int i = 0; i < format.getTextureCoordinateCount(); i++) { 
            glDisableVertexAttribArray(ATI_TEXTURE_BASE + i);
          }
        #else
          glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        #endif
      }

      #ifdef USE_GL33_CODE
        glBindVertexArray(0);
      #endif

      glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    
    MeshVertexBufferDrawConfig :: MeshVertexBufferDrawConfig(const MeshVertexBuffer* _vb) : vb(_vb) {
      if (glGetError() != GL_NO_ERROR) {
        cerr << "Warning! Calling " << __FUNCTION__ << " with uncleared glGetError()" << endl;
      }
      
      try {
        glBindBuffer(GL_ARRAY_BUFFER, vb->ref);
        if (glGetError() != GL_NO_ERROR) {
          throw MeshException() << MeshExceptionData("Could not bind vertex array");
        }
        
        #ifdef USE_GL33_CODE
          glBindVertexArray(vb->vertexArrayRef);
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not bind as OpenGL 3.3/OpenGLES 2.0 vertex array");
          }
        #endif

        const MeshVertexFormat& format = vb->getMeshData()->getFormat();
        size_t stride = vb->getMeshData()->getVertexStride();
        
        if (format.getVertexCount() > 0) {
          #ifdef USE_GL33_GLES20_CODE
            glEnableVertexAttribArray(ATI_VERTEX);
          #else
            glEnableClientState(GL_VERTEX_ARRAY);
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not enable client state (vertex array)");
          }
          
          #ifdef USE_GL33_GLES20_CODE
            glVertexAttribPointer(ATI_VERTEX, 3, GL_FLOAT, false, stride, (GLvoid*) (format.getVertexOffset(0)));
          #else
            glVertexPointer(3, GL_FLOAT, stride, (GLvoid*) (format.getVertexOffset(0)));
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not specify vertex buffer offsets");
          }
        } else {
          #ifdef USE_GL33_GLES20_CODE
            glVertexAttrib3f(ATI_VERTEX, 0, 0, 0);
          #endif
        }
        
        if (format.getNormalCount() > 0) {
          #ifdef USE_GL33_GLES20_CODE
            glEnableVertexAttribArray(ATI_NORMAL);
          #else
            glEnableClientState(GL_NORMAL_ARRAY);
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not enable client state (normal array)");
          }
          
          #ifdef USE_GL33_GLES20_CODE
            glVertexAttribPointer(ATI_NORMAL, 3, GL_FLOAT, false, stride, (GLvoid*) (format.getNormalOffset(0)));
          #else
            glNormalPointer(GL_FLOAT, stride, (GLvoid*) (format.getNormalOffset(0)));
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not specify normal buffer offsets");
          }
        } else {
          #ifdef USE_GL33_GLES20_CODE
            glVertexAttrib3f(ATI_NORMAL, 0, 0, 0);
          #endif
        }
        
        if (format.getColorCount() > 0) {
          #ifdef USE_GL33_GLES20_CODE
            glEnableVertexAttribArray(ATI_COLOR);
          #else
            glEnableClientState(GL_COLOR_ARRAY);
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not enable client state (color array)");
          }
          
          #ifdef USE_GL33_GLES20_CODE
            glVertexAttribPointer(ATI_COLOR, 4, GL_FLOAT, false, stride, (GLvoid*) (format.getColorOffset(0)));
          #else
            glColorPointer(4, GL_FLOAT, stride, (GLvoid*) (format.getColorOffset(0)));
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not specify color buffer offsets");
          }
        } else {
          #ifdef USE_GL33_GLES20_CODE
            glVertexAttrib4f(ATI_COLOR, 1, 1, 1, 1);
          #endif
        }
        
        for (unsigned int tex_index = 0; tex_index < format.getTextureCoordinateCount(); tex_index++) {
          #ifdef USE_GL33_GLES20_CODE
          #else
            glClientActiveTexture(GL_TEXTURE0 + tex_index);
          #endif

          #ifdef USE_GL33_GLES20_CODE
            glEnableVertexAttribArray(ATI_TEXTURE_BASE + tex_index);
          #else
            glEnableClientState(GL_TEXTURE_COORD_ARRAY);
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not enable client state (texture array)");
          }
          
          #ifdef USE_GL33_GLES20_CODE
            glVertexAttribPointer(ATI_TEXTURE_BASE + tex_index, 2, GL_FLOAT, false, stride, (GLvoid*) (format.getTextureCoordinateOffset(tex_index)));
          #else
            glTexCoordPointer(2, GL_FLOAT, stride, (GLvoid*) (format.getTextureCoordinateOffset(tex_index)));
          #endif
          if (glGetError() != GL_NO_ERROR) {
            throw MeshException() << MeshExceptionData("Could not specify texture coordinate buffer offsets");
          }
        }
        #ifndef USE_GL33_GLES20_CODE
          glClientActiveTexture(GL_TEXTURE0);
        #endif
      }
      catch (...) {
        cleanup();
        throw;
      }
    }
    
    MeshVertexBufferDrawConfig :: ~MeshVertexBufferDrawConfig() {
      cleanup();
    }


    void MeshVertexBuffer :: freeBuffer() {
      meshData.reset();
      if (ref) {
        glDeleteBuffers(1, &ref);
      }
      ref = 0;
      #ifdef USE_GL33_CODE
        if (vertexArrayRef) {
          glDeleteVertexArrays(1, &vertexArrayRef);
        }
        vertexArrayRef = 0;
      #endif
    }
    
    MeshVertexBufferDrawConfigRef MeshVertexBuffer :: setupDrawConfig() {
      MeshVertexBufferDrawConfigRef result = drawConfig.lock();
      if (!result) {
        result = MeshVertexBufferDrawConfigRef(new MeshVertexBufferDrawConfig(this));
        drawConfig = MeshVertexBufferDrawConfigWeakRef(result);
      }
      return result;
    }
    
    void MeshVertexBuffer :: createBuffer(const MeshDataRef& buffer) {
      freeBuffer();
      
      meshData = buffer;
      if (meshData->getVertexCount() > numeric_limits<IndexBufferIndexType>::max()) {
        cerr << "Warning! More vertices than the index buffer can address: "  << meshData->getVertexCount() << " > " << numeric_limits<IndexBufferIndexType>::max() << endl
             << "  (the mesh you are trying to load is too large for the iPhone)" << endl;
      }
      
      glGenBuffers(1, &ref);
      if (glGetError() != GL_NO_ERROR) {
        throw MeshException() << MeshExceptionData("Failed to create vertex buffer");
      }
  
      #ifdef USE_GL33_CODE
        glGenVertexArrays(1, &vertexArrayRef);
        if (glGetError() != GL_NO_ERROR) {
          throw MeshException() << MeshExceptionData("Failed to create vertex array");
        }
      #endif

      try {
        MeshVertexBufferDrawConfigRef config = setupDrawConfig();

/*Write data testwise into a file (DEBUGGING ROCKS)
        fstream f(lexical_cast<string>(ref).c_str(), fstream::out | fstream::binary);
        f.write(meshData->getMeshData(), buffer->getVertexDataSize());
        f.close();
//*/
        glBufferData(GL_ARRAY_BUFFER, buffer->getVertexDataSize(), meshData->getMeshData(), GL_STATIC_DRAW);
        if (glGetError() != GL_NO_ERROR) {
          throw MeshException() << MeshExceptionData("Failed copy data to vertex buffer");
        }
      }
      catch (...) {
        freeBuffer();
        throw;
      }
    }
    
    const MeshDataRef& MeshVertexBuffer :: getMeshData() const {
      return meshData;
    }

    MeshVertexBuffer :: ~MeshVertexBuffer() {
      freeBuffer();
    }
  }

  using namespace internal;

  template <typename T>
  struct _minimum : binary_function<T, T, T> {
    T operator()(const T& a, const T& b) {
      return (a < b) ? (a) : (b);
    }
  };
  template <typename T>
  struct _maximum : binary_function<T, T, T> {
    T operator()(const T& a, const T& b) {
      return (a > b) ? (a) : (b);
    }
  };
  void Mesh :: updateNormalizationMatrix(const MeshData::FaceGroup& group) {
    float mins[3] = {0, 0, 0};
    float maxs[3] = {0, 0, 0};
    bool first = true;
    
    if (meshData->getFormat().getVertexCount() > 0) {
      const char* rawVertexData = meshData->getMeshData() + meshData->getFormat().getVertexOffset(0);
      const size_t stride = meshData->getVertexStride();
      
      foreach (const MeshData::TriangleIndexes& tri, group.faces) {
        for (int i = 0; i < 3; i++) {
          const float* v = (const float*) (rawVertexData + tri.indexes[i] * stride);
          
          if (first) {
            copy(v, v + 3, mins);
            copy(v, v + 3, maxs);
            first = false;
          } else {
            transform(mins, mins + 3, v, mins, _minimum<float>());
            transform(maxs, maxs + 3, v, maxs, _maximum<float>());
          }
        }
      }
    }
    
    float size[3];
    transform(maxs, maxs + 3, mins, size, minus<float>());
    
    float maxSize = size[0];
    maxSize = (maxSize > size[1]) ? maxSize : size[1];
    maxSize = (maxSize > size[2]) ? maxSize : size[2];
    
    if (maxSize == 0) {
      maxSize = 1.0;
    }
    
    normalizationMatrix[0] = 1.0 / maxSize;
    normalizationMatrix[5] = 1.0 / maxSize;
    normalizationMatrix[10] = 1.0 / maxSize;
    normalizationMatrix[12] =  -(maxs[0] + mins[0]) / 2.0 / maxSize;
    normalizationMatrix[13] =  -(maxs[1] + mins[1]) / 2.0 / maxSize;
    normalizationMatrix[14] =  -(maxs[2] + mins[2]) / 2.0 / maxSize;
  }

  glm::mat4 Mesh :: getNormalizationGLMMatrix() const {
    return glm::mat4(normalizationMatrix[0], normalizationMatrix[4], normalizationMatrix[8],  normalizationMatrix[12], 
                      normalizationMatrix[1], normalizationMatrix[5], normalizationMatrix[9],  normalizationMatrix[13], 
                      normalizationMatrix[2], normalizationMatrix[6], normalizationMatrix[10], normalizationMatrix[14], 
                      normalizationMatrix[3], normalizationMatrix[7], normalizationMatrix[11], normalizationMatrix[15]);
  }
 
  MeshDrawConfigRef Mesh :: setupDrawConfig() const {
    return vertexBuffer->setupDrawConfig();
  }
  
  void Mesh :: draw() const {
    try {
      // Bind vertex buffer
      
      // "Misuse" the draw config for filling the vertex buffer
      MeshVertexBufferDrawConfigRef vertexBufferConfig = vertexBuffer->setupDrawConfig();
      
      for (unsigned int i = 0; i < meshData->getMeshMaterialGroupCount(); i++) {
        const MeshData::MaterialGroup& matGroup = meshData->getMeshMaterialGroup(i);
        {
          // Set material parameters
          if (matGroup.material->getColorCount() > 0) {
            const float* color = matGroup.material->getColor(0);
            #ifdef USE_GL33_GLES20_CODE
              glVertexAttrib4f(ATI_COLOR, color[0], color[1], color[2], color[3]); // Magic value "2" is the index
            #else
              glColor4f(color[0], color[1], color[2], color[3]);
            #endif
          } else {
            #ifdef USE_GL33_GLES20_CODE
              glVertexAttrib4f(ATI_COLOR, 1, 1, 1, 1);
            #else
              glColor4f(1, 1, 1, 1);
            #endif
          }

          for (unsigned int ti = 0; ti < matGroup.material->getTextureCount(); ti++) {
            glActiveTexture(GL_TEXTURE0 + ti);
            #ifndef USE_GL33_GLES20_CODE
              glEnable(GL_TEXTURE_2D);
            #endif
            glBindTexture(GL_TEXTURE_2D, matGroup.material->getTexture(ti)->getTexture());
          }
        }
        
        materialIndexBuffers.draw(i);
        
        for (unsigned int ti = 0; ti < matGroup.material->getTextureCount(); ti++) {
          glActiveTexture(GL_TEXTURE0 + ti);
          glBindTexture(GL_TEXTURE_2D, 0);
          #ifndef USE_GL33_GLES20_CODE
            glDisable(GL_TEXTURE_2D);
          #endif
        }
        glActiveTexture(GL_TEXTURE0);
      }
    }
    catch (MeshException& e) {
      cerr << "Failed to draw mesh: ";
      if (const string* errorMsg = get_error_info<MeshExceptionData>(e)) {
        cerr << *errorMsg;
      } else {
        cerr << "(no reason given)";
      }
      cerr << endl;
    }
  }

  void Mesh :: drawMaterialGroup(unsigned int i) const {
    MeshVertexBufferDrawConfigRef vertexBufferConfig = vertexBuffer->setupDrawConfig();
    materialIndexBuffers.draw(i);
  }
  
  void Mesh :: drawMaterialGroup(const std::string& matName) const {
    for (unsigned int i = 0; i < meshData->getMeshMaterialGroupCount(); i++) {
      if (meshData->getMeshMaterialGroup(i).name == matName) {
        drawMaterialGroup(i);
        return;
      }
      i++;
    }
    
    cerr << "Failed to draw mesh material group '" << matName << "'" << endl;
  }
  
  std::string Mesh :: getMaterialGroupName(unsigned int i) const {
    return meshData->getMeshMaterialGroup((unsigned int) i).name;
  }
  
  int Mesh :: getMaterialGroupCount() const {
    return meshData->getMeshMaterialGroupCount();
  } 

  static bool equalsGetTaggedSubMeshName(SubMeshNameTag& tag, const string& name) {
    return tag.name == name;
  }
  MeshRef Mesh :: createObjectMesh(const std::string& name) {
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Calling " << __FUNCTION__ << " with uncleared glError" << endl;
    }
    
    {
      NameSubMeshTagVector::iterator found = find_if(subMeshs.begin(), subMeshs.end(), bind<bool>(equalsGetTaggedSubMeshName, _1, name));
      if (found != subMeshs.end()) {
        return found->mesh;
      }
    }
    
    {
      MeshRef result;
      
      for (unsigned int i = 0; i < meshData->getMeshGroupCount(); i++) {
        const MeshData::FaceGroup& fg = meshData->getMeshGroup(i);
        if (fg.name == name) {
          result = MeshRef(new Mesh(*this, fg));
          subMeshs.push_back(SubMeshNameTag(name, result));
          break;
        }
      }
      
      return result;
    }
  }
  
  ObjectNameVector Mesh :: getObjectNames() const {
    ObjectNameVector result(meshData->getMeshGroupCount());
    for (unsigned int i = 0; i < meshData->getMeshGroupCount(); i++) {
      result[i] = meshData->getMeshGroup(i).name;
    }
    return result;
  }

  Mesh :: Mesh(const Mesh& baseMesh, const MeshData::FaceGroup& subMeshFaces) {
    this->meshData = baseMesh.meshData;
    this->vertexBuffer = baseMesh.vertexBuffer;
    
    MeshData::FaceGroup allFacesGroup;
    for (unsigned int i = 0; i < meshData->getMeshMaterialGroupCount(); i++) {
      MeshData::MaterialGroup matGroup = meshData->getMeshMaterialGroup(i);
      MeshData::FaceGroup subGroup = matGroup * subMeshFaces;
      matGroup.faces = subGroup.faces;
      materialIndexBuffers.addBuffer(matGroup);
      allFacesGroup += subGroup;
    }
    updateNormalizationMatrix(allFacesGroup);
  }
  
  const MeshDataRef& Mesh :: getMeshData() const {
    return meshData;
  }
  
  Mesh :: Mesh(MeshDataRef meshData) : meshData(meshData), vertexBuffer(new MeshVertexBuffer()), materialIndexBuffers(), subMeshs() {
    if (glGetError() != GL_NO_ERROR) {
      cerr << "Warning! Calling " << __FUNCTION__ << " with uncleared glError" << endl;
    }
    
    fill(normalizationMatrix, normalizationMatrix + 16, 0);
    normalizationMatrix[0] = 1;
    normalizationMatrix[5] = 1;
    normalizationMatrix[10] = 1;
    normalizationMatrix[15] = 1;

    vertexBuffer->createBuffer(meshData);

    MeshData::FaceGroup allFacesGroup;
    
    for (unsigned int i = 0; i < meshData->getMeshMaterialGroupCount(); i++) {
      const MeshData::MaterialGroup& materialGroup = meshData->getMeshMaterialGroup(i);
      materialIndexBuffers.addBuffer(materialGroup);
      allFacesGroup += materialGroup;
    }
    
    updateNormalizationMatrix(allFacesGroup);
  }
}

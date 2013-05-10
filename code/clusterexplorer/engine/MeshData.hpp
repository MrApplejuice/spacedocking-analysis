#ifndef MESHDATA_HPP_
#define MESHDATA_HPP_

#include <cstdio>
#include <stdint.h>

#include <vector>

#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

#include <OpenGL.hpp>

#include "TextureManager.hpp"

namespace engine {
  class MeshMaterialData {
    public:
      virtual std::string getName() const = 0;
      
      virtual size_t getColorCount() const = 0;
      virtual const float* getColor(unsigned int i) const = 0;

      virtual size_t getTextureCount() const = 0;
      virtual TextureRef getTexture(unsigned int i) const = 0;
  };
  typedef boost::shared_ptr<MeshMaterialData> MeshMaterialDataRef;

  class MeshVertexFormat {
    public:
      virtual int getVertexOffset(unsigned int i) const = 0;
      virtual int getNormalOffset(unsigned int i) const = 0;
      virtual int getColorOffset(unsigned int i) const = 0;
      virtual int getTextureCoordinateOffset(unsigned int i) const = 0;
      
      virtual size_t getVertexCount() const = 0;
      virtual size_t getNormalCount() const = 0;
      virtual size_t getColorCount() const = 0;
      virtual size_t getTextureCoordinateCount() const = 0;
  };

  class MeshData {
    public:
      typedef unsigned int TriangleIndex;
      struct TriangleIndexes {
        TriangleIndex indexes[3];
      
        void normalize();  
        bool operator<(const TriangleIndexes& other) const;
        bool operator==(const TriangleIndexes& other) const;
      };
      
      struct FaceGroup {
        typedef std::vector<MeshData::TriangleIndexes> TriangleVector;
        
        std::string name;
        TriangleVector faces;
        
        void normalize();  
        FaceGroup operator*(const FaceGroup& other) const;
        FaceGroup& operator+=(const FaceGroup& other);
      };
    
      struct MaterialGroup : public FaceGroup {
        MeshMaterialDataRef material;
      };
      
      virtual const char* getMeshData() const = 0;
      virtual size_t getVertexDataSize() const = 0;
      virtual size_t getVertexCount() const = 0;
      virtual size_t getVertexStride() const = 0;
      
      virtual const MeshVertexFormat& getFormat() const = 0;
      
      virtual size_t getMeshMaterialGroupCount() const = 0;
      virtual const MaterialGroup& getMeshMaterialGroup(unsigned int i) const = 0;

      virtual size_t getMeshGroupCount() const = 0;
      virtual const FaceGroup& getMeshGroup(unsigned int i) const = 0;
  };

  typedef boost::shared_ptr<MeshData> MeshDataRef;

  class MeshDataProvider {
    public:
      virtual MeshDataRef getMeshData() = 0;
  };
}

#endif // MESHDATA_HPP_

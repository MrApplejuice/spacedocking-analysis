#include "WavefrontMeshDataProvider.hpp"

#include <boost/regex.hpp>

#include "MeshData.hpp"

#include "local/Global.hpp"

namespace engine {
  namespace internal {
    class WavefrontMeshMaterialData : public virtual MeshMaterialData {
      private:
        typedef float ColorArray[4];
      
        string name;
        TextureRef texture;
        
        ColorArray colors[1];
      public:
        virtual std::string getName() const {
          return name;
        }
        
        virtual size_t getColorCount() const {
          return 1;
        }
        
        virtual const float* getColor(unsigned int i) const {
          return colors[0];
        }

        virtual size_t getTextureCount() const {
          return texture ? 1 : 0;
        }
        
        virtual TextureRef getTexture(unsigned int i) const {
          return texture;
        }
        
        WavefrontMeshMaterialData(string name, TextureRef texture, const float* diffuse) : name(name), texture(texture) {
          std::copy(diffuse, diffuse + 4, colors[0]);
        }
    };

    class WavefrontMeshVertexFormat : public virtual MeshVertexFormat {
      private:
        bool hasVertexNormal, hasTextureCoordinates;
      public:
        virtual int getVertexOffset(unsigned int i) const {
          return 0;
        }
        
        virtual int getNormalOffset(unsigned int i) const {
          return 3 * sizeof(float);
        }

        virtual int getColorOffset(unsigned int i) const {
          return -1;
        }
        
        virtual int getTextureCoordinateOffset(unsigned int i) const {
          return 3 * sizeof(float) + (hasVertexNormal ? 3 * sizeof(float) : 0) + i * 2 * sizeof(float);
        }
        
        virtual size_t getVertexCount() const {
          return 1;
        }
        
        virtual size_t getNormalCount() const {
          return hasVertexNormal ? 1 : 0;
        }
        
        virtual size_t getColorCount() const {
          return 0;
        }
        
        virtual size_t getTextureCoordinateCount() const {
          return hasTextureCoordinates ? 1 : 0;
        }
        
        WavefrontMeshVertexFormat() : hasVertexNormal(false), hasTextureCoordinates(false) {
        }

        WavefrontMeshVertexFormat(bool hasVertexNormal, bool hasTextureCoordinates) : hasVertexNormal(hasVertexNormal), hasTextureCoordinates(hasTextureCoordinates) {
        }
    };
    
    static void copyVertexIndexes(const WavefrontFaceGroup& wfg, MeshData::FaceGroup& faceGroup) {
      faceGroup.faces.clear();
      faceGroup.faces.reserve(wfg.getFaceCount());
      for (unsigned int i = 0; i < (unsigned int) wfg.getFaceCount(); i++) { // Bah! Someone fix the WavefrontLibrary!!!
        const WavefrontTriangle& tri = wfg.getFace(i);
        MeshData::TriangleIndexes ttri;
        
        for (unsigned int j = 0; j < 3; j++) {
          ttri.indexes[j] = tri.indexes[j];
        }
        
        faceGroup.faces.push_back(ttri);
      }
    }
    
    class WavefrontMeshData : public virtual MeshData {
      private:
        WavefrontMeshVertexFormat vertexFormat;
      
        size_t vertexCount;
        shared_ptr<char[]> vertexBuffer;
        
        vector<MaterialGroup> materialFaceGroups;
        vector<FaceGroup> faceGroups;
      public:
        virtual const char* getMeshData() const {
          return vertexBuffer.get();
        }
        
        virtual size_t getVertexDataSize() const {
          return getVertexStride() * getVertexCount();
        }
        
        virtual size_t getVertexCount() const {
          return vertexCount;
        }
        
        virtual size_t getVertexStride() const {
          return 3 * sizeof(float) + vertexFormat.getNormalCount() * 3 * sizeof(float) + vertexFormat.getTextureCoordinateCount() * 2 * sizeof(float);
        }
        
        virtual const MeshVertexFormat& getFormat() const {
          return vertexFormat;
        }
        
        virtual size_t getMeshMaterialGroupCount() const {
          return materialFaceGroups.size();
        }
        
        virtual const MaterialGroup& getMeshMaterialGroup(unsigned int i) const {
          return materialFaceGroups[i];
        }

        virtual size_t getMeshGroupCount() const {
          return faceGroups.size();
        }
        
        virtual const FaceGroup& getMeshGroup(unsigned int i) const {
          return faceGroups[i];
        }
        
        WavefrontMeshData(WavefrontMesh& wavefrontMesh, WavefrontMaterialLibrary& materialLibrary, TextureManager& textureManager) {
          bool hasNormals = false, hasTextureCoords = false;
          WavefrontVertexVector& vertexes = *(wavefrontMesh.getVertices());
          vertexCount = vertexes.size();
          
          foreach (WavefrontVertex& v, vertexes) {
            if (!hasNormals) {
              hasNormals = (v.normal[0] != 0) || (v.normal[1] != 0) || (v.normal[2] != 0);
            }
            if (!hasTextureCoords) {
              hasTextureCoords = (v.textureCoordinate[0] != 0) || (v.textureCoordinate[1] != 0);
            }
          }
          
          vertexFormat = WavefrontMeshVertexFormat(hasNormals, hasTextureCoords);

          vertexBuffer = shared_ptr<char[]>(new char[getVertexDataSize()]);
          float* vbDataPointer = (float*) vertexBuffer.get();
          foreach (WavefrontVertex& v, vertexes) {
            std::copy(v.coordinate, v.coordinate + 3, vbDataPointer);
            vbDataPointer += 3;
            if (hasNormals) {
              std::copy(v.normal, v.normal + 3, vbDataPointer);
              vbDataPointer += 3;
            }
            if (hasTextureCoords) {
              std::copy(v.textureCoordinate, v.textureCoordinate + 2, vbDataPointer);
              vbDataPointer += 2;
            }
          }
          
          // Process material groups
          foreach (WavefrontFaceGroup wfg, wavefrontMesh.getMaterialGroups()) {
            MeshData::MaterialGroup materialGroup;
            materialGroup.name = wfg.getName();
            
            // Convert material
            WavefrontMaterialRef matRef = materialLibrary.getMaterial(wfg.getName());
            TextureRef texture;
            float diffuseColor[4] = {1, 1, 1, 1};
            
            if (matRef) {
              if (!matRef->diffuseTex.empty()) {
                texture = textureManager.getTexture(matRef->diffuseTex);
              }
              std::copy(matRef->diffuse, matRef->diffuse + 3, diffuseColor);
              diffuseColor[3] = 1.0;
            }
            
            materialGroup.material = shared_ptr<WavefrontMeshMaterialData>(new WavefrontMeshMaterialData(wfg.getName(), texture, diffuseColor));

            // Convert vertex indexes
            copyVertexIndexes(wfg, materialGroup);
            
            materialFaceGroups.push_back(materialGroup);
          }

          foreach (WavefrontFaceGroup wfg, wavefrontMesh.getObjectGroups()) {
            MeshData::FaceGroup faceGroup;
            faceGroup.name = wfg.getName();

            // Convert vertex indexes
            copyVertexIndexes(wfg, faceGroup);
            
            faceGroups.push_back(faceGroup);
          }
        }
    };
    
    typedef shared_ptr<WavefrontMeshData> WavefrontMeshDataRef;
  }

  using namespace internal;

  MeshDataRef WavefrontMeshDataProvider :: getMeshData() {
    return meshDataRef;
  }

  WavefrontMeshDataProvider :: WavefrontMeshDataProvider(StreamOpenFunction openFunction, TextureManager& textureManager, const std::string& meshName) {
    const static boost::regex EXTRACT_PATH_REGEX("^((?:[^/]+/)*)[^/]+$");
    
    string meshPath = "";
    {
      boost::match_results<string::const_iterator> matches;
      boost::regex_match(meshName, matches, EXTRACT_PATH_REGEX);
      if (matches.size() > 1) {
        meshPath = matches[1];
      }
    }
    
    WavefrontMesh wavefrontMesh;
    WavefrontMaterialLibrary materialLibrary;

    try {
      bool loaded = false;
    
      if (!loaded) {
        shared_ptr<std::istream> inputStream = openFunction(meshName + ".bin");
        if (inputStream) {
          wavefrontMesh.loadBinary(*inputStream);
          loaded = true;
        }
      }
      
      if (!loaded) {
        shared_ptr<std::istream> inputStream = openFunction(meshName);
        if (inputStream) {
          wavefrontMesh.load(*inputStream);
          loaded = true;
        }
      }

      if (!loaded) {
        throw WavefrontMeshDataProviderException() << WavefrontMeshDataProviderExceptionData("Could not load mesh");
      }

      foreach (WavefrontMaterialLibraryName matName, wavefrontMesh.getMaterialLibraries()) {
        shared_ptr<std::istream> inputStream = openFunction(meshPath + matName);
        if (!inputStream) {
          throw WavefrontMeshDataProviderException() << WavefrontMeshDataProviderExceptionData("Could not load material library " + matName);
        }
        materialLibrary.load(*inputStream);
      }

      WavefrontMeshDataRef wmdRef(new WavefrontMeshData(wavefrontMesh, materialLibrary, textureManager));
      meshDataRef = wmdRef;
    }
    catch (WavefrontMeshException& e) {
      string data;
      if (const string* eData = boost::get_error_info<WavefrontMeshExceptionData>(e)) {
        data = "Error loading mesh " + meshName + ": " + *eData;
      } else {
        data = "Loading of mesh " + meshName + " failed";
      }
      throw WavefrontMeshDataProviderException() << WavefrontMeshDataProviderExceptionData(data);
    }
    catch (...) {
      throw;
    }
  }
}

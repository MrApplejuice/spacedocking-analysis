/*
  Implementation according to http://en.wikipedia.org/wiki/Wavefront_.obj_file
   
  Additions by 12-03-2012
    Added parsing support for material files and refractored the .obj parser 
    to incorporate allow implementations to use index buffers. (Previously
    the parse only formed one long vector of vertices describing the 
    triangles of the whole mesh. Now there is a separate vertex vertex buffer
    and indexes into that vertex buffer describe the mesh)
     
    Paul K. Gerke

  Additions by 29-03-2012
    Completed TODOs
      - Do grouping correctly (propagate, delete empty groups, reinsert into existing groups)
        - Comment: What does "propagate mean?!"
    
    Paul K. Gerke

  Additions by 03-04-2012
    Added ability to track drawing materials when combining WavefrontFaceGroup
    via the implemented +, -, * operators
    
*/
#include "WavefrontObj.hpp"

#include <map>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "local/Global.hpp"

#ifdef DEBUG
#define DEBUG_STRING(s) cerr << "WavefrontObj DEBUG: " << s << endl
#else
#define DEBUG_STRING(s) do {} while(0)
#endif

#define CHECK_DEFINE(splits, name, paramcount) (((splits)[0] == (name)) && ((splits).size() >= (paramcount) + 1))

using namespace std;
using namespace boost;

static bool checkIsEmpty(engine::internal::WavefrontFaceGroup a) {
  return a.getFaceCount() == 0;
}

static bool equalWavefrontFaceGroupNames(const engine::internal::WavefrontFaceGroup& a, const engine::internal::WavefrontFaceGroup& b) {
  return a.getName() == b.getName();
}

static engine::internal::WavefrontFaceGroup reuseFaceGroup(const engine::internal::WavefrontFaceGroupVector& objs, const engine::internal::WavefrontFaceGroup& newFacegroup) {
  engine::internal::WavefrontFaceGroupVector::const_iterator result = find_if(objs.begin(), objs.end(), bind<bool>(&equalWavefrontFaceGroupNames, _1, newFacegroup));
  if (result == objs.end()) {
    return newFacegroup;
  }
  return *result;
}

namespace engine {
  namespace internal {
    WavefrontMaterialNameVector WavefrontMaterialLibrary :: getMaterialNames() const {
      WavefrontMaterialNameVector result;
      result.reserve(materials.size());
      for (NameMaterialRefMap::const_iterator it = materials.begin(); it != materials.end(); it++) {
        result.push_back(it->first);
      }
      return result;
    }
    
    WavefrontMaterialRef WavefrontMaterialLibrary :: getMaterial(const WavefrontMaterialName& matName) const {
      WavefrontMaterialRef result;
      
      NameMaterialRefMap::const_iterator match = materials.find(matName);
      if (match != materials.end()) {
        result = match->second;
      }
      
      return result;
    }
    
    void WavefrontMaterialLibrary :: clear() {
      materials.clear();
    }
    
    void WavefrontMaterialLibrary :: load(std::istream& inputstream) {
      if (!inputstream.good()) {
        throw WavefrontMeshException() << WavefrontMeshExceptionData("Stream cannot be accessed");
      }
      
      WavefrontMaterialRef currentMaterial;
      while (inputstream.good()) {
        string line;
        getline(inputstream, line);
        trim(line);

        if ((line.empty()) || (line[0] == '#')) {
        } else {
          vector<string> lineSplits;
          split(lineSplits, line, is_any_of(" "), token_compress_on);

          bool processed = false;
          if (lineSplits.size() > 0) {
            try {
              if (CHECK_DEFINE(lineSplits, "newmtl", 1)) {
                currentMaterial = WavefrontMaterialRef(new WavefrontMaterial());
                currentMaterial->name = lineSplits[1];
                materials[currentMaterial->name] = currentMaterial;
                processed = true;
              }
              
              // Only proceed if "current color" can be written
              if (currentMaterial.get() != NULL) {
                if (CHECK_DEFINE(lineSplits, "Ka", 3)) {
                  WavefrontColor color;
                  color[0] = lexical_cast<float>(lineSplits[1]);
                  color[1] = lexical_cast<float>(lineSplits[2]);
                  color[2] = lexical_cast<float>(lineSplits[3]);
                  
                  copy(color, color + 3, currentMaterial->ambient);
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "Kd", 3)) {
                  WavefrontColor color;
                  color[0] = lexical_cast<float>(lineSplits[1]);
                  color[1] = lexical_cast<float>(lineSplits[2]);
                  color[2] = lexical_cast<float>(lineSplits[3]);
                  
                  copy(color, color + 3, currentMaterial->diffuse);
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "Ks", 3)) {
                  WavefrontColor color;
                  color[0] = lexical_cast<float>(lineSplits[1]);
                  color[1] = lexical_cast<float>(lineSplits[2]);
                  color[2] = lexical_cast<float>(lineSplits[3]);
                  
                  copy(color, color + 3, currentMaterial->specular);
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "Ns", 1)) {
                  currentMaterial->specularCoefficient = lexical_cast<float>(lineSplits[1]);
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "d", 1) || CHECK_DEFINE(lineSplits, "Tr", 1)) {
                  currentMaterial->transparency = lexical_cast<float>(lineSplits[1]);
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "illum", 1)) {
                  currentMaterial->illumination = (WavefrontIlluminationModel) lexical_cast<int>(lineSplits[1]);
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "map_Ka", 1)) {
                  currentMaterial->ambientTex = lineSplits[lineSplits.size() - 1];
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "map_Kd", 1)) {
                  currentMaterial->diffuseTex = lineSplits[lineSplits.size() - 1];
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "map_Ks", 1)) {
                  currentMaterial->specularTex = lineSplits[lineSplits.size() - 1];
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "map_Ns", 1)) {
                  currentMaterial->highlightTex = lineSplits[lineSplits.size() - 1];
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "map_d", 1)) {
                  currentMaterial->alphaMap = lineSplits[lineSplits.size() - 1];
                  processed = true;
                }
                if (CHECK_DEFINE(lineSplits, "map_bump", 1) || CHECK_DEFINE(lineSplits, "bump", 1)) {
                  currentMaterial->bumpMap = lineSplits[lineSplits.size() - 1];
                  processed = true;
                }
              }
            }
            catch (bad_lexical_cast) {
              processed = false;
            }

            if (!processed) {
              warnings.push_back(string("Warning! Line not interpretable: ") + line);
            }
          }
        }
      }
      if (inputstream.fail() && !inputstream.eof()) {
        clear();
        throw WavefrontMeshException() << WavefrontMeshExceptionData("IO Error");
      }
    }
    
    WavefrontMaterialLibrary :: WavefrontMaterialLibrary() : warnings(), materials() {
    }
    
    WavefrontMaterialLibrary :: ~WavefrontMaterialLibrary() {
      clear();
    }

    bool operator<(const WavefrontVertexIndexTriple& t1, const WavefrontVertexIndexTriple& t2) {
      bool result = (t1.vi < t2.vi) || 
                    ((t1.vi == t2.vi) && ((t1.ti < t2.ti) || 
                    ((t1.ti == t2.ti) && (t1.ni < t2.ni))));
      return result;
    }
    
    bool operator==(const WavefrontVertexIndexTriple& t1, const WavefrontVertexIndexTriple& t2) {
      bool result = memcmp(&t1, &t2, sizeof(t1));
      return result;
    }

    bool operator<(const WavefrontTriangle& t1, const WavefrontTriangle& t2) {
      bool result = (t1.indexes[0] < t2.indexes[0]) || 
                    ((t1.indexes[0] == t2.indexes[0]) && ((t1.indexes[1] < t2.indexes[1]) || 
                    ((t1.indexes[1] == t2.indexes[1]) && (t1.indexes[2] < t2.indexes[2]))));
      return result;
    }

    void WavefrontFaceGroup :: addTriangle(int i) {
      if (!finished) {
        faceIndexes.push_back(i);
      }
    }
    
    void WavefrontFaceGroup :: finishGroup() {
      if (!finished) {
        sort(faceIndexes.begin(), faceIndexes.end());
        finished = true;
      }
    }
    
    WavefrontFaceGroup WavefrontFaceGroup :: operator*(const WavefrontFaceGroup& other) const {
      WavefrontFaceGroup result("(" + name + " * " + other.name + ")", (material != "" ? material : other.material), vertices, triangles);

      unsigned int ti = 0; // This index
      unsigned int oi = 0; // Other index
      while ((ti < this->faceIndexes.size()) && (oi < other.faceIndexes.size())) {
        int tv = this->faceIndexes[ti];
        int ov = other.faceIndexes[oi]; 
        
        if (tv == ov) {
          result.addTriangle(tv);
          ti++;
          oi++;
        } else {
          if (tv < ov) {
            ti++;
          } else {
            oi++;
          }
        }
      }
      
      result.finishGroup();
      return result;
    }
    
    WavefrontFaceGroup WavefrontFaceGroup :: operator+(const WavefrontFaceGroup& other) const {
      WavefrontFaceGroup result("(" + name + " * " + other.name + ")", "", vertices, triangles);

      unsigned int ti = 0; // This index
      unsigned int oi = 0; // Other index
      while ((ti < this->faceIndexes.size()) || (oi < other.faceIndexes.size())) {
        if (ti >= this->faceIndexes.size()) {
          result.addTriangle(other.faceIndexes[oi++]);
          continue;
        }
        if (oi >= other.faceIndexes.size()) {
          result.addTriangle(this->faceIndexes[ti++]);
          continue;
        }
        
        int tv = this->faceIndexes[ti];
        int ov = other.faceIndexes[oi]; 
        
        if (tv == ov) {
          result.addTriangle(tv);
          ti++;
          oi++;
        } else {
          if (tv < ov) {
            result.addTriangle(tv);
            ti++;
          } else {
            result.addTriangle(ov);
            oi++;
          }
        }
      }
      
      result.finishGroup();
      return result;
    }
    
    WavefrontFaceGroup WavefrontFaceGroup :: operator-(const WavefrontFaceGroup& other) const {
      WavefrontFaceGroup result("(" + name + " * " + other.name + ")", material, vertices, triangles);

      unsigned int ti = 0; // This index
      unsigned int oi = 0; // Other index
      while (ti < this->faceIndexes.size()) {
        if (oi >= other.faceIndexes.size()) {
          result.addTriangle(this->faceIndexes[ti++]);
          continue;
        }
        
        int tv = this->faceIndexes[ti];
        int ov = other.faceIndexes[oi]; 
        
        if (tv == ov) {
          ti++;
          oi++;
        } else {
          if (tv < ov) {
            result.addTriangle(tv);
            ti++;
          } else {
            oi++;
          }
        }
      }
      
      result.finishGroup();
      return result;
    }
    
    int WavefrontFaceGroup :: getFaceCount() const {
      return this->faceIndexes.size();
    }
    
    const WavefrontTriangle& WavefrontFaceGroup :: getFace(int i) const {
      return this->triangles->at(this->faceIndexes[i]);
    }
    
    WavefrontFaceGroup :: WavefrontFaceGroup(const std::string& name, const std::string& material, WavefrontVertexVectorRef vertices, WavefrontTriangleVectorRef triangles) : finished(false), name(name), material(material), vertices(vertices), triangles(triangles), faceIndexes() {
    }

    WavefrontVertexIndexTriple WavefrontMesh :: parseFaceIndex(const std::string& faceDesc, const std::string& line, int vertexCount, int textureCoordinateCount, int normalCount) {
      WavefrontVertexIndexTriple result = {-1, -1, -1};
      
      vector<string> splits;
      split(splits, faceDesc, is_any_of("/"), token_compress_off);
      if (splits.size() == 0) {
        warnings.push_back(string("Erronous face description: ") + line);
      } else {
        int vertexIndex = atoi(splits[0].c_str()) - 1;
        int textureIndex = -1;
        int normalIndex = -1;

        if (splits.size() > 1) {
          if (!splits[1].empty()) {
            textureIndex = atoi(splits[1].c_str()) - 1;
          }
          if (splits.size() > 2) {
            if (!splits[2].empty()) {
              normalIndex = atoi(splits[2].c_str()) - 1;
            }
          }
        }

        DEBUG_STRING("  vertex index check: " << vertexIndex << " < " << vertexCount);

        if (vertexIndex >= vertexCount) {
          warnings.push_back(string("Erronous vertex index: ") + line);
          vertexIndex = -1;
        } else {
          DEBUG_STRING("Adding vertex data using index " << vertexIndex);

          if (textureIndex >= 0) {
            if (textureIndex >= textureCoordinateCount) {
              warnings.push_back(string("Erronous texture coordinate index: ") + line);
              vertexIndex = -1;
            } else {
              DEBUG_STRING("Adding texture data using index " << textureIndex);
            }
          }

          if (normalIndex >= 0) {
            if (normalIndex >= normalCount) {
              warnings.push_back(string("Erronous normal index: ") + line);
              vertexIndex = -1;
            } else {
              DEBUG_STRING("Adding normal data using index " << normalIndex);
            }
          }
        }
        
        if (vertexIndex >= 0) {
          result.vi = vertexIndex;
          result.ti = textureIndex;
          result.ni = normalIndex;
        }
      }
      
      return result;
    }
    
    void WavefrontMesh :: parseData(std::istream& inputstream) {
      vector< vector<float> > vertexCoordinates;
      vector< vector<float> > vertexNormals;
      vector< vector<float> > vertexTCoordinates;
      
      // Add default vertex groups
      objects.push_back(WavefrontFaceGroup("", "", this->vertices, this->triangles));
      vgroups.push_back(WavefrontFaceGroup("", "", this->vertices, this->triangles));
      materialgroups.push_back(WavefrontFaceGroup("", "", this->vertices, this->triangles));
      
      typedef map<WavefrontVertexIndexTriple, int> TripleIndexMap;
      TripleIndexMap wfVertexIndexTriplets;

      if (inputstream.fail()) {
        throw WavefrontMeshException() << WavefrontMeshExceptionData("Stream cannot be accessed");
      }
      
      while (inputstream.good()) {
        string line;
        getline(inputstream, line);
        trim(line);

        if ((line.empty()) || (line[0] == '#')) {
        } else {
          vector<string> lineSplits;
          split(lineSplits, line, is_any_of(" \t"), token_compress_on);

          if (lineSplits.size() == 0) {
            warnings.push_back(string("Warning! Line not interpretable: ") + line);
          } else {
            DEBUG_STRING("Processing line: " << line);
            DEBUG_STRING(" --> First split: " << dec << lineSplits[0]);

            string typeString = lineSplits[0];

            bool processed = false;

            if (typeString == "v") {
              processed = true;

              // Vertex Coordinate
              if (lineSplits.size() < 4) {
                warnings.push_back(string("Warning! v argument needs at least 3 coordinates: ") + line);
              } else {
                vector<float> v;
                v.push_back(atof(lineSplits[1].c_str()));
                v.push_back(atof(lineSplits[2].c_str()));
                v.push_back(atof(lineSplits[3].c_str()));
                vertexCoordinates.push_back(v);
              }
            }

            if (typeString == "vt") {
              processed = true;

              // Texture Coordinate
              if (lineSplits.size() < 3) {
                warnings.push_back(string("Warning! vt argument needs at least 2 coordinates: ") + line);
              } else {
                vector<float> v;
                v.push_back(atof(lineSplits[1].c_str()));
                v.push_back(atof(lineSplits[2].c_str()));
                vertexTCoordinates.push_back(v);
              }
            }

            if (typeString == "vn") {
              processed = true;

              // Vertex Normal
              if (lineSplits.size() < 4) {
                warnings.push_back(string("Warning! vn argument needs at least 3 coordinates: ") + line);
              } else {
                vector<float> v;
                v.push_back(atof(lineSplits[1].c_str()));
                v.push_back(atof(lineSplits[2].c_str()));
                v.push_back(atof(lineSplits[3].c_str()));
                vertexNormals.push_back(v);
              }
            }

            if (typeString == "f") {
              processed = true;

              #define __MAX_FACE_TRIANGLES 6
              
              WavefrontVertexIndexTriple triangles[__MAX_FACE_TRIANGLES];
              for (int i = 0; i < __MAX_FACE_TRIANGLES; i++) {
                triangles[i].vi = -1;
              }

              int triangleIndex = 0;
              
              // Face
              if (lineSplits.size() >= 3 + 1) {
                for (vector<string>::iterator fv = lineSplits.begin() + 1; fv < lineSplits.begin() + 4; fv++) {
                  triangles[triangleIndex] = parseFaceIndex(*fv, line, vertexCoordinates.size(), vertexTCoordinates.size(), vertexNormals.size());
                  if (triangles[triangleIndex].vi != -1) {
                    triangleIndex++;
                  }
                }
                
                if (lineSplits.size() == 4 + 1) {
                  triangles[triangleIndex++] = triangles[0];
                  triangles[triangleIndex++] = triangles[2];
                  triangles[triangleIndex] = parseFaceIndex(lineSplits[4], line, vertexCoordinates.size(), vertexTCoordinates.size(), vertexNormals.size());
                  if (triangles[triangleIndex].vi != -1) {
                    triangleIndex++;
                  }
                } else if (lineSplits.size() > 4 + 1) {
                  warnings.push_back(string("Warning! This wavefront obj implementation only supports triangular or quad faces: ") + line);
                }
              }
              
              int cornerCount = triangleIndex;
              DEBUG_STRING("Found " << cornerCount << " corners");
              
              // Check if correct corner count
              if ((cornerCount % 3) == 0) {
                WavefrontIndexType faceIndexes[__MAX_FACE_TRIANGLES];
                for (int i = 0; (i < __MAX_FACE_TRIANGLES) && (triangles[i].vi != -1); i++) {
                  DEBUG_STRING("Looking up corner " << i << ":");
                  DEBUG_STRING("  --> " << triangles[i].vi << " "  << triangles[i].ti << " "  << triangles[i].ni);

                  TripleIndexMap::iterator lookup = wfVertexIndexTriplets.find(triangles[i]);
                  int vertexIndex = -1;
                  
                  if (lookup == wfVertexIndexTriplets.end()) {
                    DEBUG_STRING("  Not found... adding:");
                    DEBUG_STRING("  vi = " << triangles[i].vi << " ti = " << triangles[i].ti << " ni = "  << triangles[i].ni);
                    vertexIndex = vertices->size();
                    
                    WavefrontVertex vertex;
                    // Indexes should be okay - checked by parseFaceIndex
                    DEBUG_STRING("    Copying vertex data");
                    copy(vertexCoordinates[triangles[i].vi].begin(), vertexCoordinates[triangles[i].vi].end(), vertex.coordinate);
                    if (triangles[i].ti != -1) {
                      DEBUG_STRING("    Copying texture coordinate data");
                      copy(vertexTCoordinates[triangles[i].ti].begin(), vertexTCoordinates[triangles[i].ti].end(), vertex.textureCoordinate);
                    }
                    if (triangles[i].ni != -1) {
                      DEBUG_STRING("    Copying normal data");
                      copy(vertexNormals[triangles[i].ni].begin(), vertexNormals[triangles[i].ni].end(), vertex.normal);
                    }
                    
                    vertices->push_back(vertex);
                    wfVertexIndexTriplets[triangles[i]] = vertexIndex;

                    DEBUG_STRING("  Added -> vertex index = " << vertexIndex);
                  } else {
                    vertexIndex = lookup->second;
                    DEBUG_STRING("  Found -> vertex index = " << vertexIndex);
                  }
                  
                  faceIndexes[i] = vertexIndex;
                }
              
                // Now form real WavefrontTriangles and insert them into triangles
                DEBUG_STRING("Now preparing triangle indexes from face indexes");
                for (int i = 0; i < (cornerCount / 3); i++) {
                  DEBUG_STRING("  Face triangle " << i << " with:");
                  DEBUG_STRING("    " << faceIndexes[i * 3 + 0] << " " << faceIndexes[i * 3 + 1] << " " << faceIndexes[i * 3 + 2]);

                  WavefrontTriangle tr = {{faceIndexes[i * 3 + 0], faceIndexes[i * 3 + 1], faceIndexes[i * 3 + 2]}};
                  int triangleIndex = this->triangles->size();
                  this->triangles->push_back(tr);
                  
                  // Add to current material-/object-/vertexgroups
                  objects[objects.size() - 1].addTriangle(triangleIndex);
                  vgroups[vgroups.size() - 1].addTriangle(triangleIndex);
                  materialgroups[materialgroups.size() - 1].addTriangle(triangleIndex);
                }
              } else {
                warnings.push_back(string("Something went wrong during the allocation of face vertices: ") + line);
              }
            } // if (typeName == "f")
            
            if (typeString == "mtllib") {
              processed = true;
              
              if (lineSplits.size() < 2) {
                warnings.push_back(string("No material lib specified for mttlib: ") + line);
              } else {
                materialLibraries.push_back(lineSplits[1]);
              }
            }

            if (typeString == "o") {
              if (lineSplits.size() < 2) {
                warnings.push_back(string("No object name specified: ") + line);
              } else {
                objects.push_back(reuseFaceGroup(objects, WavefrontFaceGroup(lineSplits[1], "", this->vertices, this->triangles)));
                processed = true;
              }
            }

            if (typeString == "g") {
              if (lineSplits.size() < 2) {
                warnings.push_back(string("No vertex group name specified: ") + line);
              } else {
                vgroups.push_back(reuseFaceGroup(vgroups, WavefrontFaceGroup(lineSplits[1], "", this->vertices, this->triangles)));
                processed = true;
              }
            }

            if (typeString == "usemtl") {
              if (lineSplits.size() < 2) {
                warnings.push_back(string("No material name specified: ") + line);
              } else {
                materialgroups.push_back(reuseFaceGroup(materialgroups, WavefrontFaceGroup(lineSplits[1], lineSplits[1], this->vertices, this->triangles)));
                processed = true;
              }
            }

            if (!processed) {
              warnings.push_back(string("Wavefront mesh warning: Unknown command: ") + line);
            }
          }
        }
      }

      if (inputstream.fail() && !inputstream.eof()) {
        clear();
        throw WavefrontMeshException() << WavefrontMeshExceptionData("IO Error");
      }
 

      WavefrontFaceGroupVector* faceGroups[3] = {&objects, &vgroups, &materialgroups};
      for (WavefrontFaceGroupVector** it = faceGroups; it != faceGroups + 3; it++) {
        // Remove empty face groups
        WavefrontFaceGroupVector::iterator newEnd = remove_if((*it)->begin(), (*it)->end(), &checkIsEmpty);
        **it = WavefrontFaceGroupVector((*it)->begin(), newEnd);
        
        // "Fix" all vertex groups
        for_each((*it)->begin(), (*it)->end(), bind<void>(&WavefrontFaceGroup::finishGroup, _1));
      }

      DEBUG_STRING("Finished loading wavefront file!");
    }
    
    void WavefrontMesh :: clear() {
      vertices->clear();
      triangles->clear();
      warnings.clear();
      objects.clear();
      vgroups.clear();
      materialgroups.clear();
      materialLibraries.clear();
    }
    
    void WavefrontMesh :: load(std::istream& inputstream) {
      clear();
      parseData(inputstream);
    }

    #include "WavefrontObj-BinaryWriteHelpers.icpp"    
    
    void WavefrontMesh :: saveBinary(std::ostream& outputstream, Endianness endian) {
      BufferedWriter writer (outputstream, endian);
      
      if (!outputstream.good()) {
        throw WavefrontMeshException() << WavefrontMeshExceptionData("Cannot write to output stream");
      }
      
      // Write endianness byte
      if (endian == ENDIAN_BIG) {
        writer((uint8_t) 0x01);
      } else if (endian == ENDIAN_LITTLE) {
        writer((uint8_t) 0x00);
      } else {
        throw WavefrontMeshException() << WavefrontMeshExceptionData("Invalid endianess specified");
      }
      
      // Write material library names
      writer((uint32_t) materialLibraries.size()); 
      for_each(materialLibraries.begin(), materialLibraries.end(), PackAndWriteString(writer));
      
      // Write vertex data (Count + packed vertex data)
      writer((uint32_t) vertices->size()); 
      for_each(vertices->begin(), vertices->end(), PackAndWriteVertex(writer));
      
      // Write triangle data
      writer((uint32_t) triangles->size()); 
      for_each(triangles->begin(), triangles->end(), PackAndWriteTriangle(writer));
      
      // Save all face groups
      WavefrontFaceGroupVector* faceGroups[3] = {&objects, &vgroups, &materialgroups};
      for (WavefrontFaceGroupVector** fgv_it = faceGroups; fgv_it != faceGroups + 3; fgv_it++) {
        writer((uint32_t) (**fgv_it).size());
        
        for (WavefrontFaceGroupVector::iterator fg_it = (**fgv_it).begin(); fg_it != (**fgv_it).end(); fg_it++) {
          (PackAndWriteString(writer))(fg_it->name);
          (PackAndWriteString(writer))(fg_it->material);
          writer((uint32_t) fg_it->faceIndexes.size());
          
          for (FaceIndexVector::iterator it = fg_it->faceIndexes.begin(); it != fg_it->faceIndexes.end(); it++) {
            writer((uint32_t) *it);
          }
        }
      }
    }
    
    #include "WavefrontObj-BinaryReadHelpers.icpp"    

    void WavefrontMesh :: loadBinary(std::istream& inputstream) {
      clear();
      
      if (!inputstream.good()) {
        throw WavefrontMeshException() << WavefrontMeshExceptionData("Cannot read from input stream");
      }
      
      // Determine endianness
      Endianness endian;
      {
        char c = inputstream.get();
        if (c == 0x01) {
          endian = ENDIAN_BIG;
        } else if (c == 0x00) {
          endian = ENDIAN_LITTLE;
        } else {
          throw WavefrontMeshException() << WavefrontMeshExceptionData("Invalid endianness marker");
        }
      }
      
      BufferedReader reader(inputstream, endian);
      
      // Read material library names
      {
        size_t count = reader((uint32_t) 0);
        materialLibraries.reserve(count);
        for (unsigned int i = 0; i < count; i++) {
          materialLibraries.push_back(readAndUnpackString(reader));
        }
      }
      
      // Read vertex data (Count + packed vertex data)
      {
        size_t count = reader((uint32_t) 0);
        vertices->reserve(count);
        for (unsigned int i = 0; i < count; i++) {
          vertices->push_back(readAndUnpackVertex(reader));
        }
      }
            
      // Read triangle data
      {
        size_t count = reader((uint32_t) 0);
        triangles->reserve(count);
        for (unsigned int i = 0; i < count; i++) {
          triangles->push_back(readAndUnpackTriangle(reader));
        }
      }

      // Load all face groups
      WavefrontFaceGroupVector* faceGroups[3] = {&objects, &vgroups, &materialgroups};

      for (WavefrontFaceGroupVector** fgv_it = faceGroups; fgv_it != faceGroups + 3; fgv_it++) {
        size_t count = reader((uint32_t) 0);
        (**fgv_it).reserve(count);
        
        for (unsigned int i = 0; i != count; i++) {
          string name = readAndUnpackString(reader);
          string material = readAndUnpackString(reader);
          (**fgv_it).push_back(WavefrontFaceGroup(name, material, vertices, triangles));
          
          WavefrontFaceGroup& fg = (**fgv_it)[(**fgv_it).size() - 1];

          size_t fcount = reader((uint32_t) 0);
          fg.faceIndexes.reserve(fcount);
          for (unsigned int f = 0; f < fcount; f++) {
            fg.addTriangle(reader((uint32_t) 0));
          }
          fg.finishGroup();
        }
      }
    }
    
    WavefrontMesh :: WavefrontMesh() : vertices(new WavefrontVertexVector()), triangles(new WavefrontTriangleVector()), warnings(), materialLibraries(), objects(), vgroups(), materialgroups() {
    }

    WavefrontMesh :: ~WavefrontMesh() {
      // NOOOO - no clear!!! Its horrible if you try to copy this object :D
    }
  }
}


#include "EditableMesh.hpp"

#include <cassert>
#include <cstdlib>
#include <stdint.h>

#include <iostream>
#include <algorithm>

#include <boost/shared_array.hpp>

#include "local/Global.hpp"

namespace engine {

using namespace std;
using namespace boost;
using namespace engine;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

EditableVertex::Format EditableVertex :: Format :: operator+(const Format& other) const {
  FlagType newFlags = (flags & (~FLAG_TEXTURE_MASK)) | (other.flags & (~FLAG_TEXTURE_MASK));
  newFlags |= MAX(flags & FLAG_TEXTURE_MASK, other.flags & FLAG_TEXTURE_MASK);
  return Format(newFlags);
}

bool EditableVertex :: Format :: operator==(const Format& other) const {
  return flags == other.flags;
}

EditableVertex::Format& EditableVertex :: Format :: operator=(const Format& other) {
  flags = other.flags;
  return *this;
}

EditableVertex::Format& EditableVertex :: Format :: setFormat(FlagType flags) {
  assert((flags & (~VALID_FLAG_MASK)) == 0);
  
  this->flags = flags;
  
  return *this;
}

EditableVertex::Format& EditableVertex :: Format :: enablePosition(bool t) {
  flags = ( t ? flags | FLAG_POSITION : flags & (~FLAG_POSITION) );
  return *this;
}

EditableVertex::Format& EditableVertex :: Format :: enableColor(bool t) {
  flags = ( t ? flags | FLAG_COLOR : flags & (~FLAG_COLOR) );
  return *this;
}

EditableVertex::Format& EditableVertex :: Format :: enableNormal(bool t) {
  flags = ( t ? flags | FLAG_NORMAL : flags & (~FLAG_NORMAL) );
  return *this;
}

EditableVertex::Format& EditableVertex :: Format :: enableTextures(unsigned int count) {
  assert(count <= MAX_TEXTURE_COUNT);
  flags = (flags & (~FLAG_TEXTURE_MASK)) | (count << TEXTURE_FLAG_SHIFT);
  return *this;
}

bool EditableVertex :: Format :: includesFormat(const Format& other) const {
  return includesFormat(other.flags);
}

bool EditableVertex :: Format :: includesFormat(FlagType type) const {
  return (((~FLAG_TEXTURE_MASK) & flags & type) == ((~FLAG_TEXTURE_MASK) & type)) && ((type & FLAG_TEXTURE_MASK) <= (flags & FLAG_TEXTURE_MASK));
}

bool EditableVertex :: Format :: hasPosition() const {
  return (flags & FLAG_POSITION) != 0;
}

bool EditableVertex :: Format :: hasColor() const {
  return (flags & FLAG_COLOR) != 0;
}

bool EditableVertex :: Format :: hasNormal() const {
  return (flags & FLAG_NORMAL) != 0;
}

unsigned int EditableVertex :: Format :: getTextureCount() const {
  return (flags & FLAG_TEXTURE_MASK) >> TEXTURE_FLAG_SHIFT;
}

size_t EditableVertex :: Format :: getSize() const {
  return (hasPosition() ? 3 * sizeof(float)   : 0) +
          (hasColor()    ? 4 * sizeof(float)   : 0) +
          (hasNormal()   ? 3 * sizeof(float)   : 0) +
          getTextureCount() * 2 * sizeof(float);
}

unsigned int EditableVertex :: Format :: getByteOffset(FlagType type) const {
  assert(type != 0);

  unsigned int byteOffset = 0;
  
  if (includesFormat(FLAG_POSITION)) {
    if (type == FLAG_POSITION) return byteOffset;
    byteOffset += 3 * sizeof(float);
  } else if (type == FLAG_POSITION) {
    assert(false || "FLAG_POSITION offset requested but not part of vertex format");
  }

  if (includesFormat(FLAG_COLOR)) {
    if (type == FLAG_COLOR) return byteOffset;
    byteOffset += 4 * sizeof(float);
  } else if (type == FLAG_COLOR) {
    assert(false || "FLAG_COLOR offset requested but not part of vertex format");
  }
  
  if (includesFormat(FLAG_NORMAL)) {
    if (type == FLAG_NORMAL) return byteOffset;
    byteOffset += 3 * sizeof(float);
  } else if (type == FLAG_NORMAL) {
    assert(false || "FLAG_NORMAL offset requested but not part of vertex format");
  }

  if ((type & (~FLAG_TEXTURE_MASK)) == 0) {
    const unsigned int textureIndex = (flags >> TEXTURE_FLAG_SHIFT) - 1;
    assert(textureIndex < getTextureCount());
    byteOffset += textureIndex * 2 * sizeof(float);
    return byteOffset;
  }
  byteOffset += getTextureCount() * 2 * sizeof(float);
  
  assert(false && "Invalid flags for EditableVertex::Format::getByteOffset");
  cerr << "Error in " << __FILE__ << " - Invalid flags for EditableVertex::Format::getByteOffset"  << endl;
  exit(1);
}

EditableVertex :: Format :: Format() : flags(0) {
}

EditableVertex :: Format :: Format(const Format& other) : flags(0) {
  *this = other;
}

EditableVertex :: Format :: Format(FlagType flags) : flags(0) {
  setFormat(flags);
}

EditableVertex :: Format :: Format(const MeshVertexFormat& format) : flags(0) {
  setFormat((format.getVertexCount() > 0 ? FLAG_POSITION : 0)
            | (format.getColorCount() > 0 ? FLAG_COLOR : 0)
            | (format.getNormalCount() > 0 ? FLAG_NORMAL : 0)
            | ((format.getTextureCoordinateCount() - 1) > 0 ? MIN(format.getTextureCoordinateCount() + 1, MAX_TEXTURE_COUNT) << TEXTURE_FLAG_SHIFT : 0));
}


EditableVertex& EditableVertex :: merge(const EditableVertex& other) {
  *format = *format + *(other.format);
  
  if (other.format->hasPosition()) position = other.position;
  if (other.format->hasNormal())   normal =   other.normal;
  if (other.format->hasColor())    color =    other.color;
  copy(other.textureCoordinates, other.textureCoordinates + other.format->getTextureCount(), textureCoordinates);
  
  return *this;
}

EditableVertex& EditableVertex :: merge(const char* data, const MeshVertexFormat& f) {
  if (f.getVertexCount() > 0) {
    format->enablePosition();
    const float* d = (const float*) (data + f.getVertexOffset(0));
    position = Vector3D(d);
  }
  if (f.getNormalCount() > 0) {
    format->enableNormal();
    const float* d = (const float*) (data + f.getNormalOffset(0));
    normal = Vector3D(d);
  }
  if (f.getColorCount() > 0) {
    format->enableColor();
    const float* d = (const float*) (data + f.getColorOffset(0));
    color = Color(d);
  }
  
  const size_t texturesToCopy = MIN(f.getTextureCoordinateCount(), Format::MAX_TEXTURE_COUNT - 1);
  if (format->getTextureCount() <= texturesToCopy) {
    format->enableTextures(texturesToCopy);
  }
  for (unsigned int i = 0; i < texturesToCopy; i++) {
    const float* d = (const float*) (data + f.getTextureCoordinateOffset(i));
    textureCoordinates[i] = TextureCoodinate(d);
  }
  return *this;
}

const EditableVertex::Format& EditableVertex :: getFormat() const {
  return *format;
}

EditableVertex& EditableVertex :: setPosition(const Vector3D& v) {
  format->enablePosition();
  position = v;
  return *this;
}

const EditableVertex::Vector3D& EditableVertex :: getPosition() const {
  return position;
}

EditableVertex& EditableVertex :: setNormal(const Vector3D& v) {
  format->enableNormal();
  normal = v;
  return *this;
}

const EditableVertex::Vector3D& EditableVertex :: getNormal() const {
  return normal;
}

EditableVertex& EditableVertex :: setColor(const EditableVertex::Color& c) {
  format->enableColor();
  color = c;
  return *this;
}

const EditableVertex::Color& EditableVertex :: getColor() const {
  return color;
}

EditableVertex& EditableVertex :: setTextureCoordinate(unsigned int i, const TextureCoodinate& tc) {
  i++; // Make 1-based
  if (i >= format->getTextureCount()) {
    format->enableTextures(i);
  }
  assert(i <= (EditableVertex::Format::FLAG_TEXTURE_MASK >> EditableVertex::Format::TEXTURE_FLAG_SHIFT));
  if (format->getTextureCount() < i) {
    format->enableTextures(i);
  }
  textureCoordinates[i - 1] = tc;
  return *this;
}

const EditableVertex::TextureCoodinate& EditableVertex :: getTextureCoordinate(unsigned int i) const {
  i++; // Make 1-based
  assert(i <= (EditableVertex::Format::FLAG_TEXTURE_MASK >> EditableVertex::Format::TEXTURE_FLAG_SHIFT));
  return textureCoordinates[i - 1];
}

size_t EditableVertex :: getSize() const {
  return format->getSize();
}
      
void EditableVertex :: packInto(char* buffer) const {
  assert(buffer != NULL);
  
  if (format->hasPosition()) {
    const float* f = position.getData();
    copy((const char*) f, ((const char*) f) + 3 * sizeof(float), buffer + format->getByteOffset(EditableVertex::Format::FLAG_POSITION));
  }
  if (format->hasNormal()) {
    const float* f = normal.getData();
    copy((const char*) f, ((const char*) f) + 3 * sizeof(float), buffer + format->getByteOffset(EditableVertex::Format::FLAG_NORMAL));
  }
  if (format->hasColor()) {
    const float* c = color.getData();
    copy((const char*) c, ((const char*) c) + 4 * sizeof(float), buffer + format->getByteOffset(EditableVertex::Format::FLAG_COLOR));
  }
  for (unsigned int i = 0; i < format->getTextureCount(); i++) {
    const EditableVertex::Format::FlagType flag = (i + 1) << EditableVertex::Format::TEXTURE_FLAG_SHIFT;
    const float* f = textureCoordinates[i].getData();
    copy((const char*) f, ((const char*) f) + 2 * sizeof(float), buffer + format->getByteOffset(flag));
  }  
}

EditableVertex :: EditableVertex() : format(new EditableVertex::Format()) {
}

EditableVertex :: EditableVertex(boost::shared_ptr<Format> sharedFormat) : format(sharedFormat) {
}

bool EditableMesh :: EditableMeshTriangle :: operator==(const EditableMesh::EditableMeshTriangle& other) const {
  if (sizeof(int) <= sizeof(v[0].get())) { // If possible to a fast checksum check
    if ((((long int) v[0].get()) * ((long int) v[1].get()) * ((long int) v[2].get())) != (((long int) other.v[0].get()) * ((long int) other.v[1].get()) * ((long int) other.v[2].get()))) {
      return false; // Checksum can only falsify
    }
  }
  
  // Now the precise check, respecting vertex sequence
  for (int sequence = 0; sequence < 3; sequence++) {
    bool equal = true;
    for (int i = 0; (i < 3) && equal; i++) {
      if (v[(i + sequence) % 3] != other.v[(i + sequence) % 3]) {
        equal = false;
      }
    }
    if (equal) {
      return true;
    }
  }
  return false;
}

EditableMesh::EditableMeshVertexRef EditableMesh :: EditableMeshTriangle :: getVertex(unsigned int i) const {
  assert(i < 3);
  return v[i];
}

void EditableMesh :: EditableMeshTriangle :: setVertex(unsigned int i, EditableMesh::EditableMeshVertexRef v) {
  assert(v);
  assert(i < 3);
  assert(v->checkOwner(this->v[i]->getOwner()));
  this->v[i] = dynamic_pointer_cast<IndexedEditableMeshVertex>(v);
}

bool EditableMesh :: EditableMeshTriangle :: isValid() const {
  return valid && v[0]->valid && v[1]->valid && v[2]->valid;
}

void EditableMesh :: EditableMeshTriangle :: invalidate() {
  valid = false;
}

const EditableMesh& EditableMesh :: EditableMeshTriangle :: getOwner() const {
  return v[0]->getOwner(); // Owner is determined by vertices this triangle consists of 
}

EditableMesh :: EditableMeshTriangle :: EditableMeshTriangle(EditableMesh::IndexedEditableMeshVertexRef v1, EditableMesh::IndexedEditableMeshVertexRef v2, EditableMesh::IndexedEditableMeshVertexRef v3) {
  assert(v1 && v2 && v3);
  assert(v1->checkOwner(v2->getOwner()) && v2->checkOwner(v3->getOwner()));
  
  v[0] = v1;
  v[1] = v2;
  v[2] = v3;
  
  valid = true;
}


EditableMesh::Material& EditableMesh :: Material :: operator=(const EditableMesh::Material& other) {
  name = other.name;
  _hasColor = other._hasColor;
  copy(other.color, other.color + 4, color);
  currentTextureCount = other.currentTextureCount;
  copy(other.textures, other.textures + MAX_TEXTURES, textures);
  
  return *this;
}

EditableMesh::Material& EditableMesh :: Material :: operator=(const MeshMaterialData& other) {
  name = other.getName();
  
  _hasColor = other.getColorCount() > 0;
  if (_hasColor) {
    const float* c = other.getColor(0); 
    copy(c, c + 4, color);
  }
  
  currentTextureCount = MIN(other.getTextureCount(), MAX_TEXTURES);
  for (unsigned int i = 0; i < currentTextureCount; i++) {
    textures[i] = other.getTexture(i);
  }
  
  return *this;
}

std::string EditableMesh :: Material :: getName() const {
  return name;
}

size_t EditableMesh :: Material :: getColorCount() const {
  return _hasColor ? 1 : 0;
}

const float* EditableMesh :: Material :: getColor(unsigned int i) const {
  assert(i == 0);
  return color;
}

size_t EditableMesh :: Material :: getTextureCount() const {
  return currentTextureCount;
}

TextureRef EditableMesh :: Material :: getTexture(unsigned int i) const {
  assert(i < currentTextureCount);
  return textures[i];
}

bool EditableMesh :: Material :: hasColor() const {
  return _hasColor;
}

void EditableMesh :: Material :: removeColor() {
  _hasColor = false;
}

void EditableMesh :: Material :: setColor(uint32_t c) {
  color[0] = (float) ((c >> 16) & 0xFF) / (float) 0xFF;
  color[1] = (float) ((c >> 8)  & 0xFF) / (float) 0xFF;
  color[2] = (float) ((c >> 0)  & 0xFF) / (float) 0xFF;
  color[3] = (float) ((c >> 24) & 0xFF) / (float) 0xFF;
}

void EditableMesh :: Material :: setColor(float r, float g, float b, float a) {
  color[0] = r;
  color[1] = g;
  color[2] = b;
  color[3] = a;
  _hasColor = true;
}

void EditableMesh :: Material :: setTexture(unsigned int i, TextureRef texture) {
  if (i < MAX_TEXTURES) {
    textures[i] = texture;
    
    if (texture && (i >= currentTextureCount)) {
      currentTextureCount = i + 1;
    } else if (!texture && (i >= currentTextureCount - 1)) {
      while ((currentTextureCount > 0) && (!textures[currentTextureCount - 1])) {
        currentTextureCount--;
      }
    }
  } else {
    cerr << __FUNCTION__ << ": Material only supports " << MAX_TEXTURES << " textures" << endl;
  }
}

EditableMesh :: Material :: Material(const Material& other) {
  *this = other;
}

EditableMesh :: Material :: Material(const MeshMaterialData& other) {
  *this = other;
}

EditableMesh :: Material :: Material(std::string name) : name(name), _hasColor(false), currentTextureCount(0) {
  fill(color, color + 4, 0);
}



const std::string& EditableMesh :: TriangleGroup :: getName() const {
  return name;
}

const EditableMesh::EditableMeshTriangleRefVector& EditableMesh :: TriangleGroup :: getTriangles() const {
  return triangles;
}

void EditableMesh :: TriangleGroup :: addTriangle(EditableMesh::EditableMeshTriangleRef triangle) {
  assert(&(triangle->getOwner()) == &owner);
  
  triangles.push_back(triangle);
}

bool EditableMesh :: TriangleGroup :: removeTriangle(EditableMesh::EditableMeshTriangleRef triangle) {
  EditableMeshTriangleRefVector::iterator found = find(triangles.begin(), triangles.end(), triangle);
  
  if (found != triangles.end()) {
    triangles.erase(found);
    return true;
  }
  
  return false;
}

void EditableMesh :: TriangleGroup :: clean() {
  EditableMeshTriangleRefVector tmpTriangles = triangles;
  triangles.clear();
  triangles.reserve(tmpTriangles.size());
  foreach (EditableMeshTriangleRef& t, tmpTriangles) {
    if (t->isValid()) {
      triangles.push_back(t);
    }
  }
}

EditableMesh :: TriangleGroup :: TriangleGroup(EditableMesh& owner, std::string name) : owner(owner), name(name), triangles() {
}

EditableMesh :: MaterialTriangleGroup :: MaterialTriangleGroup(EditableMesh& owner, std::string materialName) : TriangleGroup(owner, materialName), material(materialName) {
}


void EditableMesh :: recache() {
  if (cacheDirty) {
    cachedVertexesConversion = EditableMeshVertexRefVector(vertexes.begin(), vertexes.end());
    cacheDirty = false;
  }
}

EditableMesh::EditableMeshVertexRef EditableMesh :: addVertex(EditableVertex& vertex) {
  EditableMesh::IndexedEditableMeshVertexRef newVertex (new EditableMesh::IndexedEditableMeshVertex(*this, format));
  newVertex->merge(vertex);
  vertexes.push_back(newVertex);
  vertexesIndexed = false;
  if (!cacheDirty) { // No need to rebuild all of the cache if it is not dirty anyway
    cachedVertexesConversion.push_back(newVertex);
  }
  return newVertex;
}

bool EditableMesh :: removeVertex(EditableMeshVertexRef v) {
  assert(v->checkOwner(*this));
  IndexedEditableMeshVertexRef iv = dynamic_pointer_cast<IndexedEditableMeshVertex>(v);
  iv->valid = false; // Mark as invalid
  
  vertexesIndexed = false;
  
  // Find position to erase
  IndexedEditableMeshVertexRefVector::iterator it = find(vertexes.begin(), vertexes.end(), iv);
  if (it != vertexes.end()) {
    vertexes.erase(it);
    cacheDirty = true;
    
    return true;
  }
  return false;
}

const EditableMesh::EditableMeshVertexRefVector& EditableMesh :: getVertexes() {
  recache();
  return cachedVertexesConversion;
}

size_t EditableMesh :: getVertexCount() const {
  return vertexes.size();
}

int EditableMesh :: getVertexIndex(const EditableMesh::EditableMeshVertexRef& v) const {
  assert(v->checkOwner(*this));
  
  if (!isIndexed()) {
    return -1;
  }
  
  IndexedEditableMeshVertexRef iv = dynamic_pointer_cast<IndexedEditableMeshVertex>(v);
  return iv->index;
}

EditableVertex::Format& EditableMesh :: getFormat() {
  return *format;
}

const EditableVertex::Format& EditableMesh :: getFormat() const {
  return *format;
}

EditableMesh::EditableMeshTriangleRef EditableMesh :: addTriangle(EditableMesh::EditableMeshVertexRef v1, EditableMesh::EditableMeshVertexRef v2, EditableMesh::EditableMeshVertexRef v3) {
  assert(v1->checkOwner(*this));
  assert(v2->checkOwner(*this));
  assert(v3->checkOwner(*this));
  
  IndexedEditableMeshVertexRef iv1 = dynamic_pointer_cast<IndexedEditableMeshVertex>(v1);
  IndexedEditableMeshVertexRef iv2 = dynamic_pointer_cast<IndexedEditableMeshVertex>(v2);
  IndexedEditableMeshVertexRef iv3 = dynamic_pointer_cast<IndexedEditableMeshVertex>(v3);
  
  assert(iv1->valid && iv2->valid && iv3->valid);
  
  EditableMesh::EditableMeshTriangleRef newTriangle (new EditableMesh::EditableMeshTriangle(iv1, iv2, iv3));
  triangles.push_back(newTriangle);
  return newTriangle;
}

EditableMesh::EditableMeshTriangleRef EditableMesh :: addTriangle(unsigned int i1, unsigned int i2, unsigned int i3) {
  assert(i1 < vertexes.size());
  assert(i2 < vertexes.size());
  assert(i3 < vertexes.size());
  
  return addTriangle(vertexes[i1], vertexes[i2], vertexes[i3]);
}

bool EditableMesh :: removeTriangle(EditableMesh::EditableMeshTriangleRef v) {
  EditableMeshTriangleRefVector::iterator found = find(triangles.begin(), triangles.end(), v);
  if (found != triangles.end()) {
    (*found)->invalidate();
    triangles.erase(found);
    return true;
  }
  return false;
}

const EditableMesh::EditableMeshTriangleRefVector& EditableMesh :: getTriangles() const {
  return triangles;
}

EditableMesh::TriangleGroupRef EditableMesh :: addTriangleGroup(EditableMesh::TriangleGroupRef triangleGroup) {
  faceGroups.push_back(triangleGroup);
  return triangleGroup;
}

bool EditableMesh :: removeTriangleGroup(EditableMesh::TriangleGroupRef triangleGroup) {
  TriangleGroupRefVector::iterator found = find(faceGroups.begin(), faceGroups.end(), triangleGroup);
  
  if (found != faceGroups.end()) {
    faceGroups.erase(found);
    return true;
  }
  return false;
}

const EditableMesh::TriangleGroupRefVector& EditableMesh :: getTriangleGroups() const {
  return faceGroups;
}

EditableMesh::MaterialTriangleGroupRef EditableMesh :: addMaterialGroup(EditableMesh::MaterialTriangleGroupRef materialGroup) {
  materialGroups.push_back(materialGroup);
  return materialGroup;
}

bool EditableMesh :: removeTriangleGroup(EditableMesh::MaterialTriangleGroupRef materialGroup) {
  MaterialTriangleGroupRefVector::iterator found = find(materialGroups.begin(), materialGroups.end(), materialGroup);
  
  if (found != materialGroups.end()) {
    materialGroups.erase(found);
    return true;
  }
  return false;
}

const EditableMesh::MaterialTriangleGroupRefVector& EditableMesh :: getMaterialGroups() const {
  return materialGroups;
}

void EditableMesh :: clean() {
  {
    // Filter vertexes (should be none but just to be safe)
    IndexedEditableMeshVertexRefVector tmpVertexes = vertexes;
    vertexes.clear();
    vertexes.reserve(tmpVertexes.size());
    foreach (IndexedEditableMeshVertexRef& v, tmpVertexes) {
      if (v->valid) {
        vertexes.push_back(v);
      } else {
        cacheDirty = true;
      }
    }
  }
  
  // Filter triangles/faces
  {
    EditableMeshTriangleRefVector tmpTriangles = triangles;
    triangles.clear();
    triangles.reserve(tmpTriangles.size());
    foreach (EditableMeshTriangleRef& t, tmpTriangles) {
      if (t->isValid()) {
        triangles.push_back(t);
      } else {
        t->invalidate();
      }
    }
  }
  
  // Filter triangles of face and material groups
  {
    foreach (TriangleGroupRef& fg, faceGroups) {
      fg->clean();
    }
    foreach (MaterialTriangleGroupRef& fg, materialGroups) {
      fg->clean();
    }
  }
}

class EditableMeshMeshData : virtual public MeshData {
  private:
    class MeshVertexFormat_imp : public virtual MeshVertexFormat {
      private:
        EditableVertex::Format& format;
      public:
        virtual int getVertexOffset(unsigned int i) const {
          return format.getByteOffset(EditableVertex::Format::FLAG_POSITION);
        }
        
        virtual int getNormalOffset(unsigned int i) const {
          return format.getByteOffset(EditableVertex::Format::FLAG_NORMAL);
        }
        
        virtual int getColorOffset(unsigned int i) const {
          return format.getByteOffset(EditableVertex::Format::FLAG_COLOR);
        }
        
        virtual int getTextureCoordinateOffset(unsigned int i) const {
          return format.getByteOffset((i + 1) << EditableVertex::Format::TEXTURE_FLAG_SHIFT);
        }
        
        virtual size_t getVertexCount() const {
          return format.hasPosition() ? 1 : 0;
        }
        
        virtual size_t getNormalCount() const {
          return format.hasNormal() ? 1 : 0;
        }
        
        virtual size_t getColorCount() const {
          return format.hasColor() ? 1 : 0;
        }
        
        virtual size_t getTextureCoordinateCount() const {
          return format.getTextureCount();
        }
        
        MeshVertexFormat_imp(EditableVertex::Format& format) : format(format) {}
    };
  
    size_t vertexBufferSize, vertexCount;
    shared_array<char> vertexBuffer;
  
    EditableVertex::Format format;
    MeshVertexFormat_imp mdFormat;
  
    std::vector<MeshData::FaceGroup> faceGroups;
    std::vector<MeshData::MaterialGroup> materialGroups;
  public:
    virtual const char* getMeshData() const {
      return vertexBuffer.get();
    }
    
    virtual size_t getVertexDataSize() const {
      return vertexBufferSize;
    }
    
    virtual size_t getVertexCount() const {
      return vertexCount;
    }
    
    virtual size_t getVertexStride() const {
      return format.getSize();
    }
    
    virtual const MeshVertexFormat& getFormat() const {
      return mdFormat;
    }
    
    virtual size_t getMeshMaterialGroupCount() const {
      return materialGroups.size();
    }
    
    virtual const MaterialGroup& getMeshMaterialGroup(unsigned int i) const {
      return materialGroups[i];
    }

    virtual size_t getMeshGroupCount() const {
      return faceGroups.size();
    }
    
    virtual const FaceGroup& getMeshGroup(unsigned int i) const {
      return faceGroups[i];
    }
    
    EditableMeshMeshData(const EditableMesh& mesh) : vertexBuffer(), format(mesh.getFormat()), mdFormat(format), faceGroups(), materialGroups() {
      assert(mesh.isIndexed());
      
      { // Pack vertex data
        vertexCount = mesh.getVertexCount();
        vertexBufferSize = format.getSize() * mesh.getVertexCount();
        vertexBuffer = shared_array<char>(new char[vertexBufferSize]);
        mesh.packVertexDataInto(vertexBuffer.get(), vertexBufferSize);
      }
      
      { // Convert face groups
        foreach (const EditableMesh::TriangleGroupRef& g, mesh.getTriangleGroups()) {
          FaceGroup mdfg;
          mdfg.name = g->getName();
          
          foreach (const EditableMesh::EditableMeshTriangleRef& t, g->getTriangles()) {
            TriangleIndexes ti;
            for (unsigned int i = 0; i < 3; i++) {
              ti.indexes[i] = mesh.getVertexIndex(t->getVertex(i));
            }
            mdfg.faces.push_back(ti);
          }
          
          faceGroups.push_back(mdfg);
        }
      }

      { // Convert material groups
        foreach (const EditableMesh::MaterialTriangleGroupRef& g, mesh.getMaterialGroups()) {
          MaterialGroup mdmg;
          mdmg.name = g->getName();
          mdmg.material = MeshMaterialDataRef(new EditableMesh::Material(g->material));
          
          foreach (const EditableMesh::EditableMeshTriangleRef& t, g->getTriangles()) {
            TriangleIndexes ti;
            for (unsigned int i = 0; i < 3; i++) {
              ti.indexes[i] = mesh.getVertexIndex(t->getVertex(i));
            }
            mdmg.faces.push_back(ti);
          }
          
          materialGroups.push_back(mdmg);
        }
      }
    }
};

void EditableMesh :: indexVertexes() {
  int i = 0;
  foreach (IndexedEditableMeshVertexRef& v, vertexes) {
    v->index = i++;
  }
  vertexesIndexed = true;
}

bool EditableMesh :: isIndexed() const {
  return vertexesIndexed;
}

void EditableMesh :: packVertexDataInto(char* buffer, size_t size) const {
  const size_t vertexSize = format->getSize();
  foreach (const IndexedEditableMeshVertexRef& v, vertexes) {
    size -= vertexSize;
    if (size < 0) {
      break;
    }
    
    v->packInto(buffer);
    buffer += vertexSize;
  }
}

MeshDataRef EditableMesh :: getMeshData() {
  clean();
  indexVertexes();
  return MeshDataRef(new EditableMeshMeshData(*this));
}

EditableMesh :: EditableMesh() : format(new EditableVertex::Format()),
                                     vertexes(),
                                     vertexesIndexed(false),
                                     triangles(),
                                     faceGroups(),
                                     materialGroups(),
                                     cacheDirty(false),
                                     cachedVertexesConversion() {
}

EditableMesh :: EditableMesh(MeshDataRef meshData) : format(new EditableVertex::Format()),
                                                          vertexes(),
                                                          vertexesIndexed(false),
                                                          triangles(),
                                                          faceGroups(),
                                                          materialGroups(),
                                                          cacheDirty(false),
                                                          cachedVertexesConversion() {
  if (meshData) {
    format = boost::shared_ptr<EditableVertex::Format>(new EditableVertex::Format(meshData->getFormat()));
    
    vertexes.reserve(meshData->getVertexCount());
    const char* data = meshData->getMeshData();
    
    EditableVertex vertex;
    for (unsigned int i = 0; i < meshData->getVertexCount(); i++) {
      vertex.merge(data + i * meshData->getVertexStride(), meshData->getFormat());
      addVertex(vertex);
    }
    
    for (unsigned int i = 0; i < meshData->getMeshMaterialGroupCount(); i++) {
      const MeshData::MaterialGroup& mdMatGroup = meshData->getMeshMaterialGroup(i);
      MaterialTriangleGroupRef matGroup = MaterialTriangleGroup::create(*this, mdMatGroup.name);

      if (mdMatGroup.material) {
        matGroup->material = Material(*mdMatGroup.material);
      }

      foreach (MeshData::TriangleIndexes tri, mdMatGroup.faces) {
        EditableMeshTriangleRef addedTriangle = addTriangle(tri.indexes[0], tri.indexes[1], tri.indexes[2]);
        matGroup->addTriangle(addedTriangle);
      }
      
      addMaterialGroup(matGroup);
    }

    for (unsigned int i = 0; i < meshData->getMeshGroupCount(); i++) {
      const MeshData::FaceGroup& mdFaceGroup = meshData->getMeshGroup(i);
      TriangleGroupRef triGroup = TriangleGroup::create(*this, mdFaceGroup.name);

      foreach (MeshData::TriangleIndexes tri, mdFaceGroup.faces) {
        EditableMeshTriangleRef addedTriangle = addTriangle(tri.indexes[0], tri.indexes[1], tri.indexes[2]);
        triGroup->addTriangle(addedTriangle);
      }
      
      addTriangleGroup(triGroup);
    }
  } else {
    cerr << "Warning! Initialized EditableMesh with meshData = NULL" << endl;
  } 
}

} // namespace engine

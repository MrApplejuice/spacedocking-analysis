#ifndef EDITABLEMESH_HPP_
#define EDITABLEMESH_HPP_

#include <cstdio>
#include <stdint.h>

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

#include "MeshData.hpp"

namespace engine {
  class EditableMesh;
  class EditableVertex;
  typedef boost::shared_ptr<EditableVertex> EditableVertexRef;
  
  class EditableVertex : public virtual boost::noncopyable {
    public:
      class Format {
        public:
          typedef unsigned int FlagType;
          static const FlagType FLAG_POSITION     = 0x00000001;
          static const FlagType FLAG_COLOR        = 0x00000002;
          static const FlagType FLAG_NORMAL       = 0x00000004;
          static const FlagType FLAG_TEXTURE_MASK = 0x000000F0;
          static const FlagType TEXTURE_FLAG_SHIFT = 4;
        private:
          static const FlagType VALID_FLAG_MASK = 0x000000F7;
          FlagType flags;
        public:
          static const size_t MAX_TEXTURE_COUNT = FLAG_TEXTURE_MASK >> TEXTURE_FLAG_SHIFT;
        
          /**
           * Unifies the properties of two formats. Creates a new one 
           * that has enough properties to cover all properties
           * of the formats that are combined.
           * @param other
           *   Other format to combine with *this
           * @return
           *   New format covering all properties of the combined formats
           */
          Format operator+(const Format& other) const;
          
          /**
           * Compares if two formats are identical.
           * @param other
           *   Other format that *this format should be compared to.
           * @return
           *   True if the formats are equal, false otherwise
           */
          bool operator==(const Format& other) const;
          
          /**
           * Copies another Format into *this.
           * @param other
           *   Other format to be copied
           * @return
           *   *this
           */
          Format& operator=(const Format& other);
          
          /**
           * Sets the format to the format specified by the flags.
           * @param
           *   Combination of format flags specifying a vertex format.
           * @return
           *   *this
           */
          Format& setFormat(FlagType flags);
          
          /**
           * Modifies the vertex format by enabling or disabling
           * the vertex position vector.
           * @param e
           *   Indication to enable (true) or disable (false) the
           *   position vector (defaults to true).
           * @return
           *   *this
           */
          Format& enablePosition(bool e = true);
          
          /**
           * Modifies the vertex format by enabling or disabling
           * the vertex color value.
           * @param e
           *   Indication to enable (true) or disable (false) the
           *   vertex color value (defaults to true).
           * @return
           *   *this
           */
          Format& enableColor(bool e = true);
          
          /**
           * Modifies the vertex format by enabling or disabling
           * the vertex normal vector.
           * @param e
           *   Indication to enable (true) or disable (false) the
           *   normal vector (defaults to true).
           * @return
           *   *this
           */
          Format& enableNormal(bool e = true);
          
          /**
           * Modifies the vertex format by setting the number
           * of supported texture coordinates.
           * @param count
           *   Number of texture coordinates to activate (defaults to 1)
           * @return
           *   *this
           */
          Format& enableTextures(unsigned int count = 1);
          
          /**
           * Checks if *this covers all formats that are defined for 
           * other.
           * @param other
           *   Format that should be checked if it is covered by *this.
           * @return
           *   true if other is covered the format of *this
           */
          bool includesFormat(const Format& other) const;
          
          /**
           * Checks if *this covers all formats given by the flags of
           * type.
           * @param type
           *   Format that should be checked if it is covered by *this.
           * @return
           *   true if other is covered the format of *this
           */
          bool includesFormat(FlagType type) const;
          
          /**
           * Checks if *this has a position. 
           */
          bool hasPosition() const;
          
          /**
           * Checks if *this has a color. 
           */
          bool hasColor() const;
          
          /**
           * Checks if *this has a normal. 
           */
          bool hasNormal() const;
          
          /**
           * Checks how many texture coodinates are defined in *this. 
           */
          unsigned int getTextureCount() const;
          
          /**
           * Returns the size of a vertex with *this format.
           * @return
           *   Size of a vertex with the format represented by *this.
           */
          size_t getSize() const;
          
          /**
           * <p>Gets the byte offset of a property of the described 
           * format.</p>
           * 
           * <p>Only a single flag in the paramter must be set. A flag is
           * either a "real" flag like FLAG_POSITION or a texture index
           * shifted into the bits covered by FLAG_TEXTURE_MASK. Texture
           * indexes are handled as 1-based to avoid abiguous zero parameters.</p>
           * 
           * <p>Note that the function assert-s that the requested
           * property is covered by the format of *this.</p>
           * 
           * <code>
           * Examples:
           * 
           *   getByteOffset(FLAG_POSITION) // Offset of position vector
           *
           *   getByteOffset(1 << TEXTURE_FLAG_SHIFT) // Offset of the first! texture coordinate
           * </code>
           * 
           * @param type
           *   Flag or texture index thats offset is queried.
           * @return
           *   Byte offset to the flag spcified in the parameter type
           */
          unsigned int getByteOffset(FlagType type) const;
          
          /**
           * Returns the flags of *this
           */
          FlagType getFlags() const;
          
          /**
           * Creates an empty format with no properties.
           */
          Format();

          /**
           * Copies a format.
           */
          Format(const Format& other);

          /**
           * Creates a format from the given combination of flags.
           */
          Format(FlagType flags);
          
          /**
           * Creates from a MeshVertexFormat
           */
          Format(const MeshVertexFormat& format);
          
          friend class EditableMesh;
      };
      
      template <size_t C>
      class FloatArrayValue {
        private:
          float v[C];
        protected:
          /**
           * Returns the modifiable! data of this vector.
           * @return 
           *   Modifieable vector data of three components
           */
          float* getData() {
            return v;
          }
        public:
          /**
           * Assignment operator
           * @param other
           *   Other vector to take the data from
           * @return
           *   *this
           */
          FloatArrayValue<C>& operator=(const FloatArrayValue<C>& other) {
            std::copy(other.v, other.v + 3, v);
            
            return *this;
          }
        
          /**
           * Returns the data of this vector.
           * @return 
           *   Vector data of three components
           */
          const float* getData() const {
            return v;
          }

          /**
           * Creates a zero vector.
           */        
          FloatArrayValue() {
            std::fill(v, v + C, 0);
          }

          /**
           * Copy constructor.
           */        
          FloatArrayValue(const FloatArrayValue<C>& v) {
            *this = v;
          }

          /**
           * Creates a vector from the presented data.
           * @param v
           *   Pointer to three successive floats to initialize this
           *   vector with.
           */        
          FloatArrayValue(const float* v) {
            std::copy(v, v + C, this->v);
          }

          /**
           * Creates a vector from the given iterator-based data.
           * @param begin
           *   Iterator over floats that must at least be able to iterate
           *   over C items
           */        
          template<typename T>
          FloatArrayValue(T begin) {
            std::copy(begin, begin + C, this->v);
          }
      };

      class Vector3D : public FloatArrayValue<3> {
        public:
          Vector3D() : FloatArrayValue<3>::FloatArrayValue() {}
          Vector3D(const FloatArrayValue<3>& v) : FloatArrayValue<3>::FloatArrayValue(v) {}
          Vector3D(const float* v) : FloatArrayValue<3>::FloatArrayValue(v) {}
          template<typename T> Vector3D(T begin) : FloatArrayValue<3>::FloatArrayValue(begin) {}

          /**
           * Creates a vector from the given data.
           * @param x
           *   x vector component
           * @param y
           *   y vector component
           * @param z
           *   z vector component
           */        
          Vector3D(float x, float y, float z) : FloatArrayValue<3>::FloatArrayValue() {
            float* v = getData();
            v[0] = x; v[1] = y; v[2] = z;
          }
      };
      
      class TextureCoodinate : public FloatArrayValue<2> {
        public:
          TextureCoodinate() : FloatArrayValue<2>::FloatArrayValue() {}
          TextureCoodinate(const FloatArrayValue<2>& v) : FloatArrayValue<2>::FloatArrayValue(v) {}
          TextureCoodinate(const float* v) : FloatArrayValue<2>::FloatArrayValue(v) {}
          template<typename T> TextureCoodinate(T begin) : FloatArrayValue<2>::FloatArrayValue(begin) {}

          /**
           * Creates a texture coodinate from the given data.
           * @param u
           *   u texture coordinate component
           * @param v
           *   v texture coordinate component
           */        
          TextureCoodinate(float u, float v) : FloatArrayValue<2>::FloatArrayValue() {
            float* _v = getData();
            _v[0] = u; _v[1] = v; 
          }
      };

      class Color : public FloatArrayValue<4> {
        public:
          Color() : FloatArrayValue<4>::FloatArrayValue() {}
          Color(const FloatArrayValue<4>& v) : FloatArrayValue<4>::FloatArrayValue(v) {}
          Color(const float* v) : FloatArrayValue<4>::FloatArrayValue(v) {}
          template<typename T> Color(T begin) : FloatArrayValue<4>::FloatArrayValue(begin) {}

          /**
           * Creates a color from the given data.
           * @param r
           * @param g
           * @param b
           * @param a
           */        
          Color(float r, float g, float b, float a) : FloatArrayValue<4>::FloatArrayValue() {
            float* _v = getData();
            _v[0] = r; _v[1] = g; _v[2] = b; _v[3] = a; 
          }

          /**
           * Creates a color from the given data.
           * @param color
           *   0xAARRGGBB encoded 32-bit color value
           */        
          Color(uint32_t color) : FloatArrayValue<4>::FloatArrayValue() {
            float* _v = getData();
            _v[0] = (float) ((color >> 16) & 0xFF) / (float) 0xFF; 
            _v[1] = (float) ((color >> 8)  & 0xFF) / (float) 0xFF;
            _v[2] = (float) ((color >> 0)  & 0xFF) / (float) 0xFF; 
            _v[3] = (float) ((color >> 24) & 0xFF) / (float) 0xFF;
          }
      };
    private:
      boost::shared_ptr<Format> format;
      
      Vector3D position, normal;
      Color color;
      TextureCoodinate textureCoordinates[Format::MAX_TEXTURE_COUNT];
    public:
      /**
       * Copies the format and data from another vertex. Note that the
       * format is copied using the union operator. So any formats 
       * defined before will remain when merging the new data.
       * @param other
       *   Other vertex to copy data from
       * @return
       *   *this
       */
      virtual EditableVertex& merge(const EditableVertex& other);

      /**
       * Merges vertex data from any valid MeshVertexData-source into
       * this EditableVertex.
       * @param data
       *   Pointer to raw vertex data to be merged into this EditableVertex
       * @param f
       *   Format of the vertex to be merged.
       * @return 
       *   *this
       */
      virtual EditableVertex& merge(const char* data, const MeshVertexFormat& f);
    
      /** 
       * Returns the format of this Vertex 
       * @return
       *   Format object with the current vertex format.
       */
      virtual const Format& getFormat() const;
      
      /**
       * Sets the position component of this vertex (if not set,
       * this function also sets the format to cover this property)
       * @param v
       *   Vector for the new position component.
       * @return
       *   *this
       */
      virtual EditableVertex& setPosition(const Vector3D& v);

      /**
       * Return the position component of this vertex. If the queried 
       * component is not part of this vector format, the function 
       * returns undefined data. Use the Format from getFormat to see, 
       * whether the queried format is supported.
       * @return
       *   The position component of this vertex, if supported by the 
       *   vertex format.
       */
      virtual const Vector3D& getPosition() const;

      /**
       * Sets the normal component of this vertex (if not set,
       * this function also sets the format to cover this property)
       * @param v
       *   Vector for the new normal component.
       * @return
       *   *this
       */
      virtual EditableVertex& setNormal(const Vector3D& v);

      /**
       * Return the normal component of this vertex. If the queried 
       * component is not part of this vector format, the function 
       * returns undefined data. Use the Format from getFormat to see, 
       * whether the queried format is supported.
       * @return
       *   The normal component of this vertex, if supported by the 
       *   vertex format.
       */
      virtual const Vector3D& getNormal() const;

      /**
       * Sets the color component of this vertex (if not set,
       * this function also sets the format to cover this property)
       * @param v
       *   Value for the new color component.
       * @return
       *   *this
       */
      virtual EditableVertex& setColor(const Color& c);

      /**
       * Return the color component of this vertex. If the queried 
       * component is not part of this vector format, the function 
       * returns undefined data. Use the Format from getFormat to see, 
       * whether the queried format is supported.
       * @return
       *   The color component of this vertex, if supported by the 
       *   vertex format.
       */
      virtual const Color& getColor() const;

      /**
       * Sets the i-th texture coordinate component of this vertex 
       * (if not set, this function also sets the format to cover 
       * this property). It must be that i < 15, the maximum supported
       * number of texture coordinate components in this implementation.
       * @param i
       *   The 0-based texture index to set.
       * @param v
       *   New texture coordinate for texture index i
       * @return
       *   *this
       */
      virtual EditableVertex& setTextureCoordinate(unsigned int i, const TextureCoodinate& tc);
      
      /**
       * Return texture coordinate component i. If the queried component
       * is not part of this vector format, the function returns
       * undefined data. Use the Format from getFormat to see, whether
       * the queried format is supported.
       * @param i
       *   The 0-based texture index to query.
       * @return
       *   The texture coordinate i, if i is supported by the vertex
       *   format.
       */
      virtual const TextureCoodinate& getTextureCoordinate(unsigned int i) const;

      /**
       * Returns the size of this vertex in bytes. The same as
       * getFormat().getSize().
       * @return
       *   Size of this vertex in bytes.
       */
      virtual size_t getSize() const;      
      
      /**
       * Packs the data stored in this vertex into the given buffer.
       * The buffer should be able to hold at least enough bytes to
       * store this vertex ( check getSize() ).
       * @param buffer
       *   Buffer receiveing the data from this vertex.
       */
      virtual void packInto(char* buffer) const;

      /**
       * Creates a new, empty vertex.
       */
      EditableVertex();

      /**
       * Cretes a new vertex, sharing the data by the common format
       * pointed by the sharedFormat parameter.
       * @param sharedFormat
       *   Format shared among multiple vertexes.
       */
      EditableVertex(boost::shared_ptr<Format> sharedFormat);
  };

  typedef std::vector<EditableVertexRef> EditableVertexRefVector;
  class EditableMesh : public virtual boost::noncopyable, public virtual MeshDataProvider {
    public:
      /**
       * <p>Special vertex class that is used by the class EditableMesh. 
       * This specialized class also saves a reference to its owner. A
       * EditableMeshVertex cannot be owned by multiple EidtableMesh
       * instances at once.</p>
       * 
       * <p>the constructor of this class is protected. The class
       * is instantiated by EditableMesh when adding a new vertex to
       * the internal vertex list via {@link EditableMesh::addVertex}.
       * The returned reference can be edited as usual.</p>
       * 
       * <p>IMPORTANT!!! This class is only used as a public interface
       * for IndexedEditableMeshVertex objects. Never try to instantiate
       * EditableMeshVertex objects on your own and adding them to
       * a EditableMesh - it will crash you application</p>
       */
      class EditableMeshVertex : public EditableVertex  {
        private:
          const EditableMesh& owner;
        protected:
          /**
           * Creates a new editable vertex with the specified owner
           * and shared format.
           * 
           * @param owner
           *   The owner of the newly crated editable mesh
           * @param sharedFormat
           *   The shared format to be used by this EditableVertex.
           */
          EditableMeshVertex(const EditableMesh& owner, boost::shared_ptr<Format> sharedFormat) : EditableVertex(sharedFormat), owner(owner) {}
        public:
          /**
           * Gets the reference to the owner of this EditableVertex.
           * 
           * @return
           *   Reference to the owner of this vertex
           */
          virtual const EditableMesh& getOwner() const {
            return owner;
          }
        
          /**
           * Checks if this EditableVertex is owned by the  given 
           * EditableMesh.
           * 
           * @param checkingObject
           *   Object to check if it owns this vertex
           * @return
           *   True if the EditableMesh checkingObject owns this vertex
           */
          virtual bool checkOwner(const EditableMesh& checkingObject) const {
            return &owner == &checkingObject;
          }
      };
      typedef boost::shared_ptr<EditableMeshVertex> EditableMeshVertexRef;
      typedef std::vector<EditableMeshVertexRef> EditableMeshVertexRefVector;
    private:
      /**
       * Even more specialized instance of an EditableVertex, used
       * internally by EditableMesh to index vertices when serializing
       * them into a vertex buffer. EditableMeshVertex is the public
       * interface of this class, the rest is internal.
       */
      class IndexedEditableMeshVertex : public EditableMeshVertex {
        public:
          /** Saves whether the current vertex is valid or invalid.
           * Invalid vertices that will not be serialized and are removed
           * from any vertex lists in EditableMesh.
           */
          bool valid;
          /**
           * Saves the index of this vertex in the serialized buffer.
           * This is used to construct the index arrays for faces based
           * on the vertex buffer.
           */
          unsigned int index;

          /**
           * Creates a new IndextEditableMeshVertex, owned by owner
           * and using the shared format sharedForamt
           * 
           * @param owner
           *   The owner of the new indexed editable vertex
           * @param sharedFormat
           *   The shared format used by the new editable vertex.
           */
          IndexedEditableMeshVertex(const EditableMesh& owner, boost::shared_ptr<Format> sharedFormat) : EditableMeshVertex(owner, sharedFormat), valid(true), index(0) {}
      };
      typedef boost::shared_ptr<IndexedEditableMeshVertex> IndexedEditableMeshVertexRef;
      typedef std::vector<IndexedEditableMeshVertexRef> IndexedEditableMeshVertexRefVector;
    public:
      /**
       * <p>A EditableMeshTriangle represents a face of an edited mesh.
       * EditableMeshes only support triangles as faces to compose 
       * objects from.</p>
       * 
       * <p>You can edit these triangle groups by setting the 
       * corner points with vertices using the function setVertex.
       * All vertices of a EditableMeshTriangle must be owned by
       * the same EditableMesh, or the program will abort.</p>
       * 
       * <p>The class uses a "invalidation" technique for performance 
       * reasons. If a triangle is removed, the triangle is marked
       * as "invalid" with the invalidate method of this class. 
       * Invalidated triangles are then collected from container 
       * classes when the corresponding "clear" method is called.
       * The clear method is automatically called, when an EditableMesh
       * is serialized. Note that a EditableMeshTriangle is also marked
       * as invalid as soons as one of its corner points becomes 
       * invalid.</p>
       */
      class EditableMeshTriangle {
        private:
          IndexedEditableMeshVertexRef v[3];
          bool valid;
        public:
          /**
           * <p>Checks if this triangle is equal to another triangle. The
           * function uses a "normalized" version of the triangles for
           * this, ignoring differences in phasing of the corner
           * points of the triangles but not the direction. </p>
           * 
           * <p>For example, a triangle consisting of corner points
           * [1,2,3] and another with corner points [2,3,1] would be
           * equal (same "direction"/sequence, but different phase), but
           * a third triangle [3,2,1] would be different (different
           * sequence of corner points).</p>
           * 
           * @param other
           *   Other triangle to compare this triangle with
           * @return 
           *   Retruns if *this and other are equal.
           */
          bool operator==(const EditableMeshTriangle& other) const;

          /**
           * Gets the vertex at corner point i.
           * 
           * @param i
           *   Index of the corner point to get the vertex from. It must 
           *   be that: 0 <= i <= 2.
           * @return
           *   The corner vertex at index i
           */
          virtual EditableMeshVertexRef getVertex(unsigned int i) const;
          
          /**
           * Sets the coner vertex at corner i.
           * 
           * @param i
           *   Index of the corner point to set the vertex. It must 
           *   be that: 0 <= i <= 2.
           * @param vertex
           *   New vertex for corner i. The owner of this vertex must be 
           *   same as the owner for the vertex that is replaced by this
           *   function call.
           */
          virtual void setVertex(unsigned int i, EditableMeshVertexRef v);
          
          /**
           * Checks if this triangle has been invalidated. Invalidated 
           * triangles are never serialized and are removed from any
           * triangle lists during a "clear" run of a triangle list.
           * (It is more or less a remove, but only marking the triangle
           * for future removal)
           * 
           * @return
           *   True if this triangle is invalid (has an invalid corner or
           *   was marked as invalid).
           */
          virtual bool isValid() const;
          
          /**
           * Marks the triangle as invalid (cannot be undone).
           */
          virtual void invalidate();
          
          /**
           * Returns the owner of this triangle. The owner of a triangle
           * is defined by the owner of its corners, which all must be
           * the same
           * 
           * @return
           *   A reference to the owner owning this EditableTriangle.
           */
          virtual const EditableMesh& getOwner() const; 
        private:  
          /**
           * Private constructor creatign a new EditableMeshTriangle
           * with the given corners. Use the EditableMesh::addTriangle
           * functions to instantiate a new EditableMeshTriangle.
           */
          EditableMeshTriangle(IndexedEditableMeshVertexRef v1, IndexedEditableMeshVertexRef v2, IndexedEditableMeshVertexRef v3);
          
          friend class EditableMesh;
      };
      typedef boost::shared_ptr<EditableMeshTriangle> EditableMeshTriangleRef;
      typedef std::vector<EditableMeshTriangleRef> EditableMeshTriangleRefVector;

      class Material : public MeshMaterialData {
        private:
          const static size_t MAX_TEXTURES = 8;
        
          std::string name;

          bool _hasColor;
          float color[4];
          
          size_t currentTextureCount;
          TextureRef textures[MAX_TEXTURES];
        public:
          Material& operator=(const Material& other);
          Material& operator=(const MeshMaterialData& other);
        
          virtual std::string getName() const;
          virtual size_t getColorCount() const;
          virtual const float* getColor(unsigned int i) const;
          virtual size_t getTextureCount() const;
          virtual TextureRef getTexture(unsigned int i) const;
          
          virtual bool hasColor() const;
          virtual void removeColor();
          virtual void setColor(uint32_t c);
          virtual void setColor(float r, float g, float b, float a=1.0);
          virtual void setTexture(unsigned int i, TextureRef texture);
          
          Material(const Material& other);
          Material(const MeshMaterialData& other);
          Material(std::string name);
      };

      class TriangleGroup;
      typedef boost::shared_ptr<TriangleGroup> TriangleGroupRef;
      class TriangleGroup {
        private:
          EditableMesh& owner;
          std::string name;
          EditableMeshTriangleRefVector triangles;
        protected:
          TriangleGroup(EditableMesh& owner, std::string name);
        public:
          /**
           * Returns the name of this face group
           * @return
           *   Name of the mesh group
           */
          virtual const std::string& getName() const;
          
          /**
           * Returns the vector of all triangles that are part of this triangle group.
           * @return
           *   std::vector containing shared_ptr references to the triangles
           *   that are part of this face group.
           */
          virtual const EditableMeshTriangleRefVector& getTriangles() const;
          
          /**
           * Adds a triangle to this triangle group. Triangles must be valid
           * and owned by the same EditableMesh this triangle group was
           * created for.
           * @param triangle
           *   The triangle to add to this group
           */
          virtual void addTriangle(EditableMeshTriangleRef triangle);
          
          /**
           * Removes a triangle from this triangle group
           * @param triangle
           *   The triangle to remove from this group
           * @return
           *   true if the triangle was removed from this group, false if
           *   it could not be removed (probably because it was not part of
           *   this group in the first place)
           */
          virtual bool removeTriangle(EditableMeshTriangleRef triangle);
          
          /**
           * Removes all invalid triangles from this group.
           */
          virtual void clean();
          
          /**
           * Creates a new triangle group with this given name and
           * given owner.
           * @param owner
           *   Owner of this triangle group. Only triangles owned by
           *   this editable mesh may be added to this triangle group.
           * @param name
           *   Name of the new triangle group
           */
          static TriangleGroupRef create(EditableMesh& owner, std::string name) {
            return TriangleGroupRef(new TriangleGroup(owner, name));
          }
      };
      typedef std::vector<TriangleGroupRef> TriangleGroupRefVector;

      class MaterialTriangleGroup;
      typedef boost::shared_ptr<MaterialTriangleGroup> MaterialTriangleGroupRef;
      class MaterialTriangleGroup : public TriangleGroup {
        public:
          Material material;
        protected:
          MaterialTriangleGroup(EditableMesh& owner, std::string materialName);
        public:
          static MaterialTriangleGroupRef create(EditableMesh& owner, std::string materialName) {
            return MaterialTriangleGroupRef(new MaterialTriangleGroup(owner, materialName));
          }
      };
      typedef std::vector<MaterialTriangleGroupRef> MaterialTriangleGroupRefVector;
    
      boost::shared_ptr<EditableVertex::Format> format;
      IndexedEditableMeshVertexRefVector vertexes;
      bool vertexesIndexed;
      EditableMeshTriangleRefVector triangles;
      
      TriangleGroupRefVector faceGroups;
      MaterialTriangleGroupRefVector materialGroups;
      
      bool cacheDirty;
      EditableMeshVertexRefVector cachedVertexesConversion;
      
      virtual void recache();
    public:
      virtual EditableVertex::Format& getFormat(); 
      virtual const EditableVertex::Format& getFormat() const; 
    
      virtual EditableMeshVertexRef addVertex(EditableVertex& vertex);
      virtual bool removeVertex(EditableMeshVertexRef v);
      virtual const EditableMeshVertexRefVector& getVertexes();
      virtual size_t getVertexCount() const;
      virtual int getVertexIndex(const EditableMeshVertexRef& v) const;
      
      virtual EditableMeshTriangleRef addTriangle(EditableMeshVertexRef v1, EditableMeshVertexRef v2, EditableMeshVertexRef v3);
      virtual EditableMeshTriangleRef addTriangle(unsigned int v1, unsigned int v2, unsigned int v3);
      virtual bool removeTriangle(EditableMeshTriangleRef v);
      virtual const EditableMeshTriangleRefVector& getTriangles() const;

      virtual TriangleGroupRef addTriangleGroup(TriangleGroupRef triangleGroup);
      virtual bool removeTriangleGroup(TriangleGroupRef triangleGroup);
      virtual const TriangleGroupRefVector& getTriangleGroups() const;

      virtual MaterialTriangleGroupRef addMaterialGroup(MaterialTriangleGroupRef materialGroup);
      virtual bool removeTriangleGroup(MaterialTriangleGroupRef materialGroup);
      virtual const MaterialTriangleGroupRefVector& getMaterialGroups() const;

      virtual void clean();
      virtual void indexVertexes();
      virtual bool isIndexed() const;
      virtual void packVertexDataInto(char* buffer, size_t size) const;

      // Inheritance from MeshDataProvider
      virtual MeshDataRef getMeshData();

      EditableMesh();
      EditableMesh(MeshDataRef meshData);
  };
  typedef boost::shared_ptr<EditableMesh> EditableMeshRef;
}

#endif // EDITABLEMESH_HPP_

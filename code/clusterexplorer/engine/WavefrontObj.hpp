/** \file
 * <p>Implementation of a parser for parsing wavefront .obj files without. This does not
 * present a complete implementation of the old wavefront specifications. Most
 * notably there is no support for texture options in the material library yet.</p>
 * 
 * <p>The central class which parses a wavefront .obj file is engine::internal::WavefrontMesh.
 * An instantiated objects loads a wavefront .obj file via the function load from a 
 * std::istream. During the loading process vertex data is separated from face
 * corner indexes. The loading process also assures that only a single copy of a
 * coordinate-normal-texturecoordinate combination is stored (so a face 'f 1/2/1 1/2/1 1/2/1'
 * will only lead to a vertex buffer of size 1 and an index buffer containing one face 
 * consisting of indexes [1 1 1]). Quads are triangularized during the loading process.</p>
 * 
 * <p>After a wavefront .obj file has been load, the vertex buffer can be 
 * accessed using the function getVertices(). Index buffers are groups according
 * to the objects (o), vertex groups (g), and materials (usemtl) the
 * contained faces belong to. They can be accessed though the functions
 * getObjectGroups(), getVertexGroups(), and getMaterialGroups(), respectively.</p>
 * 
 * <p>Should the class fail to load data from the inputstream provided
 * for the load-function, a WavefrontMeshException will be thrown. Parsing errors 
 * are signalled as warnings, which can be requested using the function getWarnings() 
 * after a loading process has finished. Materials libraries that are referenced
 * (mtllib) by this wavefront .obj file are listed internally and their names
 * can be obtained usingthe function getMaterialLibraries().</p>
 * 
 * <p>Parsing of material libraries is also supported by this wavefront
 * implementation. WavefrontMesh objects do not automatically parse those
 * because they cannot open them (they only are provided with an inputstream
 * for loading the specific .obj file - the implementation does not make
 * assumptions about the data source). In order to load a material file
 * you can use the WavefrontMaterialLibrary class. Just like the WavefrontMesh
 * file, the class utilizes a std::istream to load its data from. However,
 * unlike the WavefrontMesh, a WavefrontMaterialLibrary does not clear its
 * internal data when repeatedly calling load. This gives the opportunity to
 * instantiate just one WavefrontMaterialLibrary object, which contains all
 * data from all material libraries referenced by a .obj file.</p>
 */

#ifndef WAVEFRONTOBJ_HPP_
#define WAVEFRONTOBJ_HPP_

#include <iostream>
#include <vector>
#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <boost/exception/all.hpp>

#include <ClassTools.hpp>

/** */
namespace engine {
  namespace internal {
    typedef boost::error_info<struct tag_my_info, const std::string> WavefrontMeshExceptionData;
    struct WavefrontMeshException : public virtual boost::exception, public virtual std::exception {};

    /** Stack for stacking parser warnings */
    typedef std::vector<std::string> WavefrontWarningStack;

    /** Illumination model as defined by the Wavefront.obj standard */
    enum WavefrontIlluminationModel {
      WAVEFRONT_ILLUMNATION_COLOR = 0,
      WAVEFRONT_ILLUMNATION_AMBIENT = 1,
      WAVEFRONT_ILLUMNATION_HIGHLIGHT = 2,
      WAVEFRONT_ILLUMNATION_RT_REFLECTION = 3,
      WAVEFRONT_ILLUMNATION_RT_TRANSPARENCY = 4,
      WAVEFRONT_ILLUMNATION_RT_FRESNEL = 5,
      WAVEFRONT_ILLUMNATION_RT_REFRACTION_WITHOUT_FRESNEL = 6,
      WAVEFRONT_ILLUMNATION_RT_REFRACTION = 7,
      WAVEFRONT_ILLUMNATION_REFLECTION = 8,
      WAVEFRONT_ILLUMNATION_TRANSPARENCY = 9,
      WAVEFRONT_ILLUMNATION_CAST_SHADOW_ON_INVISIBLE = 10
    };

    /** RGB values representing a color as found in wavefront material libraries */     
    typedef float WavefrontColor[3];
    
    typedef std::string WavefrontMaterialName;
    typedef std::vector<WavefrontMaterialName> WavefrontMaterialNameVector;
    
    /**
     * Representation of a wavefront material. Only a subset of the whole 
     * set of options is implemented. In particular loading options for textures
     * are not implemented.
     * 
     * Texture maps are saved as strings, colors as vector of 3 float values
     * designated by the type WavefrontColor.
     */
    struct WavefrontMaterial {
      WavefrontMaterialName name;
      
      std::string ambientTex, diffuseTex, specularTex, highlightTex, alphaMap, bumpMap;
      WavefrontColor ambient, diffuse, specular;
      float specularCoefficient;
      float transparency;
      
      WavefrontIlluminationModel illumination;
      
      /**
       * Constructor makes sure everything is zero initialized
       */
      WavefrontMaterial() {
        name = "";
        ambientTex = "";
        diffuseTex = "";
        specularTex = "";
        highlightTex = "";
        alphaMap = "";
        bumpMap = "";
        
        std::fill(ambient, ambient + 3, 0);
        std::fill(diffuse, diffuse + 3, 0);
        std::fill(specular, specular + 3, 0);
        
        specularCoefficient = 0;
        transparency = 0;
        
        illumination = WAVEFRONT_ILLUMNATION_AMBIENT;
      }
    };
    
    typedef boost::shared_ptr<WavefrontMaterial> WavefrontMaterialRef;
    typedef std::map<WavefrontMaterialName, WavefrontMaterialRef> NameMaterialRefMap;
    
    /**
     * Parser for Wavefront \.mtl files. Calling load with a stream that
     * contains Wavefront material library data will load all materials from 
     * that stream (until eof is reached) and unpack them into 
     * {@link WavefrontMaterial} structures. These structures are accessible
     * via the functions getMaterial(std::string matName) of this object. You can 
     * use getWavefrontMaterialNames() to enumerate all load materials of this file.
     * 
     * getWarnings() will give you a list of all parsing errors/warnings that 
     * occurred during parsing the stream. Severe parsing errors will be reported via
     * exceptions: see {@link WavefrontMeshException}
     */
    class WavefrontMaterialLibrary {
      private:
        WavefrontWarningStack warnings;
        
        NameMaterialRefMap materials;
      public:
        /**
         * Gets the vector of all warnings that occurred during the last loading process.
         * 
         * @return
         *   Vector of strings containing the readable (english) messages
         */
        GEN_GETTER(const WavefrontWarningStack&, Warnings, warnings);
      
        /**
         * Returns a vector containing all material names that were load from
         * the material libraries that were load using the method load.
         * 
         * @return
         *   Vector of strings containing all material names defined in this
         *   material library.
         */
        virtual WavefrontMaterialNameVector getMaterialNames() const;
        
        /**
         * Returns a reference to a load material or a zero reference, if 
         * no material with the given name could be found in this material library.
         * 
         * @param matName
         *   The name of the material that should be retrieved
         * 
         * @return
         *   Reference to the returned material. If matName does not exist, a
         *   NULL reference is returned.
         */
        virtual WavefrontMaterialRef getMaterial(const WavefrontMaterialName& matName) const;
      
        /**
         * Clears all data of this material library (empties it).
         */
        virtual void clear();
        
        /**
         * Adds data from a wavfront material library file, accessible through 
         * the provided std::istream to this material library.
         * 
         * @param inputstream
         *   The inputstream to load the material library data from
         */
        virtual void load(std::istream& inputstream);
      
        WavefrontMaterialLibrary();
        virtual ~WavefrontMaterialLibrary();
    };
    
    typedef int WavefrontIndexType;

    /**
     * Structure describing a vertex of the mesh. Implementations can decide whether to use
     * normal values or texture coordinates. If they were not present in the wavefront file
     * they are set to zero.
     */
    struct WavefrontVertex {
      /** Position of this vertex */
      float coordinate[3];
      /** Normal of this vertex, set to zero if not defined */
      float normal[3];
      /** Texture coordinate of this vertex, set to zero if not defined */
      float textureCoordinate[2];
    };
    
    /**
     * An "index triple" describing the corner of a triangle or quad in a wavefront obj
     * file. Not set indexes are set to -1. Only used for internal purposes.
     */
    struct WavefrontVertexIndexTriple {
      WavefrontIndexType vi, ti, ni;
    };
    bool operator<(const WavefrontVertexIndexTriple& t1, const WavefrontVertexIndexTriple& t2);
    bool operator==(const WavefrontVertexIndexTriple& t1, const WavefrontVertexIndexTriple& t2);
    
    /**
     * Triangle indexes into the vertex buffer of this mesh. Not that there is no
     * representation for a quad since quads are translated into triangles during loading
     * for easier presentation purposes.
     */
    struct WavefrontTriangle {
      WavefrontIndexType indexes[3];
    };
    bool operator<(const WavefrontTriangle& t1, const WavefrontTriangle& t2);
    
    typedef std::vector<WavefrontVertex> WavefrontVertexVector;
    typedef boost::shared_ptr<WavefrontVertexVector> WavefrontVertexVectorRef;

    typedef std::vector<WavefrontTriangle> WavefrontTriangleVector;
    typedef boost::shared_ptr<WavefrontTriangleVector> WavefrontTriangleVectorRef;
    
    typedef std::vector<WavefrontIndexType> FaceIndexVector;

    typedef std::string WavefrontMaterialLibraryName;
    typedef std::vector<WavefrontMaterialLibraryName> WavefrontMaterialLibraryNameVector;
    
    class WavefrontMesh;
    
    /**
     * <p>A face group implements the possibility of a Wavefront obj file to
     * bundle faces into objects, vertexgroups and material (groups). Every face
     * group has a name that can be accessed by the getName function.</p>
     * 
     * <p>Every face group contains indexes to faces that part of that corresponding 
     * group. Face groups can be combined using the operators *, + and -
     * which correspond to the set operators of intersection, union and 
     * difference, respectively.</p>
     * 
     * <p>The triangular faces of a group can be accessed by getting the triangle indexes
     * using the function getFace(int i) (where 0 <= i < getFaceCount() ). Then a lookup
     * in the original WavefrontMesh vertex array will ( getVertices() ) will give
     * the corner vertexes for the particular triangle.</p>
     * 
     * <p>Material groups are a special representation of groups of vertices that are
     * drawn using the same material. For material face groups it holds that the
     * group name corresponds to the wavefront material name that should be used 
     * for rendering.</p>
     */
    class WavefrontFaceGroup {
      private:
        bool finished;
      
        std::string name, material;
        WavefrontVertexVectorRef vertices;
        WavefrontTriangleVectorRef triangles;
        
        FaceIndexVector faceIndexes;
        
        /**
         * Adds a triangle to the internal face index buffer
         * 
         * @param i
         *   The index pointing inside the referenced list 
         *   WavefrontFaceGroup::triangles
         */
        virtual void addTriangle(int i);
        
        /**
         * "Seals" this face group and prevents any further changes to it.
         */
        virtual void finishGroup();
      public:
        /** Returns the name of this face */
        GEN_GETTER(const std::string&, Name, name);
        /** Returns the name of the material this face group
         * should be drawn with */
        GEN_GETTER(const std::string&, Material, material);
        
        /**
         * Combines two face groups using the intersection operator.
         * 
         * @param other
         *   The other face group to compute the intersection with
         * 
         * @return
         *   A new WavefrontFaceGroup only containing faces of this and other.
         */
        virtual WavefrontFaceGroup operator*(const WavefrontFaceGroup& other) const;

        /**
         * Combines two face groups using the union operator.
         * 
         * @param other
         *   The other face group to compute the union with
         * 
         * @return
         *   A new WavefrontFaceGroup containing faces that are part of this or the other
         *   WavefrontFaceGroup.
         */
        virtual WavefrontFaceGroup operator+(const WavefrontFaceGroup& other) const;

        /**
         * Combines two face groups using the difference operator.
         * 
         * @param other
         *   The other face group to compute the difference with
         * 
         * @return
         *   A new WavefrontFaceGroup containing faces that are part of this bot not 
         *   the other WavefrontFaceGroup.
         */
        virtual WavefrontFaceGroup operator-(const WavefrontFaceGroup& other) const;
        
        /**
         * Returns how many (triangular) faces are part of this WavefrontFaceGroup.
         * 
         * @return
         *   The number of faces in this face group.
         */
        virtual int getFaceCount() const;
        
        /**
         * Returns the vertex indexes for face i.
         * 
         * @param i
         *   The face index to get the vertex indexes for
         * 
         * @return
         *   The WavefrontIndexes of triangular face i.
         */
        virtual const WavefrontTriangle& getFace(int i) const;
        
        /**
         * Creates a new (empty) WavefrontFaceGroup with a given name, referencing
         * the given vertex buffer and face buffer.
         * 
         * @param name
         *   The name of the new WavefrontFaceGroup
         * @param material
         *   The material name to draw this face group with
         * @param vertices
         *   A reference to the vertex buffer referenced by vertex indexes 
         *   that can be queried from this object.
         * @param triangles
         *   The face (triangle) buffer that is referenced by the face
         *   indexes that will be stored in this object.
         */
        WavefrontFaceGroup(const std::string& name, const std::string& material, WavefrontVertexVectorRef vertices, WavefrontTriangleVectorRef triangles);
        
        friend class WavefrontMesh;
    };
    
    typedef std::vector<WavefrontFaceGroup> WavefrontFaceGroupVector;

    /**
     * <p>Central class which parses wavefront .obj data and stores 3d
     * data in a vertex buffer accessible via WavefrontMesh::getVertices()
     * and WavefrontFaceGroup objects that more or less correspond to
     * index buffers.</p>
     * 
     * <p>After instantiation a WavefrontMesh object contains no information.
     * You have to load a file using the WavefrontMesh::load method. After
     * a file has been load, you can access the vertex buffer and face groups
     * which represent wavefront .obj-objects, vertex groups, and faces
     * that share the same material settings. You can also query which material
     * libraries are referenced by the load wavefront .obj file using
     * WavefrontMesh::getMaterialLibraries().</p>
     */
    class WavefrontMesh : boost::noncopyable {
      private:
        WavefrontVertexVectorRef vertices;
        WavefrontTriangleVectorRef triangles;
        
        WavefrontWarningStack warnings;
        
        WavefrontMaterialLibraryNameVector materialLibraries;
        WavefrontFaceGroupVector objects, vgroups, materialgroups;

        virtual WavefrontVertexIndexTriple parseFaceIndex(const std::string& faceDesc, const std::string& line, int vertexCount, int textureCoordinateCount, int normalCount);
        virtual void parseData(std::istream& inputstream);
      public:
        /** Endianness enum to specify the endianess of a binary mesh file during saving */
        enum Endianness {ENDIAN_LITTLE, ENDIAN_BIG};
      
        /** Returns a string vector of material library names referenced by the load WavefrontFile. */
        GEN_GETTER(const WavefrontMaterialLibraryNameVector&, MaterialLibraries, materialLibraries);
        /** Returns the VertexBuffer containing all vertices of the read
         * mesh (not necessarily in the order of the read file */
        GEN_GETTER(const WavefrontVertexVectorRef&, Vertices, vertices);
        /** String-vector containing all parsing problems that ocurred during loading the .obj file */
        GEN_GETTER(const WavefrontWarningStack&, Warnings, warnings);
        /** Returns all face groups grouping faces of each wavefront object
         * (marker 'o') together. */
        GEN_GETTER(const WavefrontFaceGroupVector&, ObjectGroups, objects);
        /** Returns all face groups grouping faces of each wavefront vertex groups
         * (marker 'v') together. */
        GEN_GETTER(const WavefrontFaceGroupVector&, VertexGroups, vgroups);
        /** Returns all face groups grouping faces that are drawn with the
         * same material (marker 'usemtl') together */
        GEN_GETTER(const WavefrontFaceGroupVector&, MaterialGroups, materialgroups);
        
        /**
         * Clears all internal data buffers. Object contains no data after this.
         */
        virtual void clear();
        
        /**
         * Loads a wavefront .obj file from the given inputstream.
         * 
         * @param inputstream
         *   Open inputstream to wavefront .obj data.
         * 
         * @throw
         *   The function throws a WavefrontMeshException if an error
         *   occurs regarding the input stream, which prevents the function
         *   to receive more data (except for eof at which the function
         *   exits normally).
         */ 
        virtual void load(std::istream& inputstream);

        /**
         * Loads a mesh data from a binary data source (see saveBinary).
         * 
         * @param inputstream
         *   Open inputstream to binary mesh data.
         * 
         * @throw
         *   The function throws a WavefrontMeshException if an error
         *   occurs while reading the input stream (either parsing or
         *   IOError)
         */ 
        virtual void loadBinary(std::istream& inputstream);

        /**
         * Saves the contents of this wavefront mesh file in a binary
         * format that needs no parsing and can be loaded quickly.
         * 
         * @param outputstream
         *   Output stream that the binary mesh should be written to
         * 
         * @throw
         *   If an error occurs during the write process, an error
         *   is thrown (e.g. during an IOError)
         */
        virtual void saveBinary(std::ostream& outputstream, Endianness endian=ENDIAN_LITTLE);
        
        WavefrontMesh();
        virtual ~WavefrontMesh();
    };
  }
}

#endif // WAVEFRONTOBJ_HPP_

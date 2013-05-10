#ifndef MESH_HPP_
#define MESH_HPP_

#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/exception/all.hpp>
#include <boost/exception/all.hpp>
#include <boost/utility.hpp>

#include <ClassTools.hpp>

#include <OpenGL.hpp>

#include "libs/glm/glm.hpp"

#include "TextureManager.hpp"
#include "MeshData.hpp"

#include "OpenGLESDetectionHeader.hpp"

namespace engine {
  class Mesh;
  typedef boost::shared_ptr<Mesh> MeshRef;

  /** Message data that is thrown together with a MeshException */
  typedef boost::error_info<struct tag_my_info, const std::string> MeshExceptionData;
  /** MeshException are thrown by the Mesh-class to signal errors to the
   * Engine object */
  struct MeshException : public virtual boost::exception, public virtual std::exception {};
  
  static const unsigned int ATI_VERTEX = 0;
  static const unsigned int ATI_NORMAL = 1;
  static const unsigned int ATI_COLOR = 2;
  static const unsigned int ATI_TEXTURE_BASE = 3;

  namespace internal {
    class MeshVertexBufferDrawConfig;
  }
  
  typedef boost::shared_ptr<internal::MeshVertexBufferDrawConfig> MeshVertexBufferDrawConfigRef;
  typedef MeshVertexBufferDrawConfigRef MeshDrawConfigRef;
  
  namespace internal {
    /**
     * GLBuffer is a container for a reference to a OpenGL vertex or index buffer.
     */
    struct GLBuffer {
      /** Amount of elements and physical size of the buffer (in bytes) */
      int len, size;
      /** Open gl handle to the buffer  */
      GLuint glRef;
    };
    typedef std::vector<GLBuffer> GLBufferVector;

    /** 
     * The class is a container for index buffers that hold the references
     * to triangles that are all drawn with the same material settings.
     */
    class MeshMaterialGLBuffers : private boost::noncopyable {
      private:
        /** Vector of index buffers which are drawn with different material settings each */
        GLBufferVector indexBuffers;
      public:
        /** Frees all index buffers stored in indexBuffers */
        virtual void freeBuffers() throw(); 

        /**
         * 
         */
        virtual void addBuffer(const MeshData::MaterialGroup& materialGroup); 
        
        /**
         * Draws the polygons referenced by the index buffer with index i.
         * 
         * @param i
         *   The index of the index buffer to draw. Note that the index
         *   corresponds to the material index of the original 
         *   WavefrontMesh object and contains all polygons that are drawn
         *   with that same material.
         * 
         * @see WavefrontMesh::materialgroups 
         */
        virtual void draw(unsigned int i) const;
        
        MeshMaterialGLBuffers();
        virtual ~MeshMaterialGLBuffers();
    };

    class MeshVertexBuffer;
    /**
     * <p>RAII class for vertex buffer settings. Prepares a vertex buffer for 
     * being drawn and cleans up all OpenGL settings when deallocated.</p>
     * 
     * <p>This class only has a private constructor and is spawned by
     * MeshVertexBuffer::setupDrawConfig().</p>
     */
    class MeshVertexBufferDrawConfig {
      private:
        const MeshVertexBuffer* vb;

        /**
         * Function the "cleans up" all OpenGL settings altered by the
         * constructor.
         */
        virtual void cleanup() const;

        /**
         * Prepares the referenced MeshVertexBuffer for being drawn using
         * an index buffer.
         * 
         * @param _vb
         *   The vertex buffer that should be prepared for being drawn.
         */
        MeshVertexBufferDrawConfig(const MeshVertexBuffer* _vb);
      public:
        /**
         * Destructor cleaning up all OpenGL settings
         */
        virtual ~MeshVertexBufferDrawConfig();
        
        friend class MeshVertexBuffer;
    };
    typedef boost::weak_ptr<MeshVertexBufferDrawConfig> MeshVertexBufferDrawConfigWeakRef;
    typedef boost::shared_ptr<MeshVertexBufferDrawConfig> MeshVertexBufferDrawConfigRef;
    
    /**
     * Vertex buffer container, creating an OpenGL vertex buffer for
     * all unique vertices of a WavefrontMesh object.
     */
    class MeshVertexBuffer {
      private:
        /** Mesh data this vertex is created from */
        MeshDataRef meshData;
      
        /** Reference to the OpenGL vertex buffer instance */
        GLuint ref;
        
        /** Reference to the initialized Draw Config */
        MeshVertexBufferDrawConfigWeakRef drawConfig;    

#ifdef USE_GL33_CODE
        /** Vertex array buffer when using OpenGL >=3.3 */
        GLuint vertexArrayRef;
#endif

        /**
         * Internal function to free the OpenGL vertex buffer.
         */
        virtual void freeBuffer();
      public:
        /**
         * This function prepares the current OpengGL context so that
         * vertices from this VertexBuffer can be drawn using an index 
         * buffer.
         * 
         * @return
         *   A shared_ptr to the MeshVertexBufferDrawConfig class. Hold
         *   on to this reference until your drawing from this vertex 
         *   buffer is done. When the object is deallocated it resets 
         *   the OpenGL configuration to stop drawing from this 
         *   VertexBuffer.
         */
        virtual MeshDrawConfigRef setupDrawConfig();
        
        /**
         * Instantiates and fills the internal OpenGL vertex buffer with
         * vertices from the provided WavefrontMesh.
         * 
         * @param buffer
         *   The mesh data for which the OpenGL vertex buffer should
         *   be created.
         */
        virtual void createBuffer(const MeshDataRef& buffer);
        
        /**
         * Returns the mesh data this vertex buffer was created for.
         * 
         * @return
         *   Mesh data this vertex buffer is created from.
         */
        virtual const MeshDataRef& getMeshData() const;
        
        /**
         * Frees internal resources (like the OpenGL buffer).
         */
        virtual ~MeshVertexBuffer();
        
        friend class MeshVertexBufferDrawConfig;
    };
    typedef boost::shared_ptr<MeshVertexBuffer> MeshVertexBufferRef; 
    
    struct SubMeshNameTag {
      std::string name;
      MeshRef mesh;
      
      SubMeshNameTag(std::string name, MeshRef mesh) : name(name), mesh(mesh) {};
    };
    typedef std::vector<SubMeshNameTag> NameSubMeshTagVector;
  } // namespace internal
  
  typedef std::vector<std::string> ObjectNameVector;
    
  /**
   * <p>The core class of this header. A Mesh creates OpenGL structures
   * for drawing from a WavefrontMesh file. These structures include
   * vertex buffers and index buffers for efficient drawing of polygons
   * and textures for textured surfaces.</p>
   * 
   * <p>The class furthermore offers function for drawing the load mesh
   * with and without applying material settings before drawing.</p>
   * 
   * <p>TODOs:
   *   (1) (allow?) Normalize mesh after loading
   * </p>
   */
  class Mesh : boost::noncopyable {
    private:
      MeshDataRef meshData;
    
      internal::MeshVertexBufferRef vertexBuffer;
      internal::MeshMaterialGLBuffers materialIndexBuffers;

      internal::NameSubMeshTagVector subMeshs;
      
      GLfloat normalizationMatrix[16];

      /**
       * Updates the internal normalization matrix using the vertices
       * referenced by the indices in the given selection of faces.
       * 
       * @param group
       *   A selection of triangles for which to generate the 
       *   normalization matrix.
       */
      virtual void updateNormalizationMatrix(const MeshData::FaceGroup& group);

      /**
       * Alternative internal constructor to create a copy of the current
       * mesh without the faces of a certain face group.
       * 
       * @param baseMesh
       *   The mesh to create the new mesh from
       * @param subMeshFaces
       *   The selection of faces that should be included in the 
       *   newly created mesh.
       */
      Mesh(const Mesh& baseMesh, const MeshData::FaceGroup& subMeshFaces);
    public:
      GEN_GETTER(const GLfloat*, NormalizationMatrix, normalizationMatrix);

      /**
       * Retruns the mesh's normalizations matrix a a glm::mat4.
       * @return
       *   Normalization matrix of this mesh as glm::mat4
       */
      virtual glm::mat4 getNormalizationGLMMatrix() const;
      
      /**
       * <p>Loads all settings for a drawing this mesh. This function
       * can be invoked manually to make changes to the default
       * draw configuration and afterwards call one of the draw
       * functions.</p>
       * 
       * <p>Note that material parameters are still set, when calling
       * draw(). Furthermore, if calling the setupDrawConfig() method of 
       * different meshes in an interleaved way, note that draw
       * configuration will overwrite each other (should not be done!).</p>
       * 
       * @return
       *   The draw configuration object handle. If it goes out of scope,
       *   the draw configuration is reset.
       */
      virtual MeshVertexBufferDrawConfigRef setupDrawConfig() const;

      /**
       * The simples version of draw. Draws the whole mesh applying all
       * materials associated to the vertices.
       */
      virtual void draw() const;
      
      /**
       * Selectively draws a material group of vertices without applying
       * the associated material. Can be used to override the material 
       * settings provided in the mesh material database.
       * 
       * @param i
       *   The index of the material group of vertices to draw. 
       *   Corresponds to the indexes to WavefrontMesh::getMaterialGroups.
       */
      virtual void drawMaterialGroup(unsigned int i) const;

      /**
       * Selectively draws a material group of vertices without applying
       * the associated material. Can be used to override the material 
       * settings provided in the mesh material database.
       * 
       * @param matName
       *   The material name of the material group to draw. The material
       *   name is the same as returned by a 
       *   WavefrontMesh::getMaterialGroups()[...].getName() of the
       *   WavefrontMesh that is load.
       */
      virtual void drawMaterialGroup(const std::string& matName) const;
      
      /**
       * Retrieves the name of a material group with index i.
       * 
       * @return
       *   The name the material referenced by index i.
       */
      virtual std::string getMaterialGroupName(unsigned int i) const; 
      
      /**
       * Returns the number of load material index groups.
       * 
       * @return
       *   Number of different material groups 
       */
      virtual int getMaterialGroupCount() const; 
      
      /**
       * Creates a new Mesh object containing only the vertices of 
       * mesh object designated by the first parameter.
       * 
       * @param name
       *   The wavefront object to create a new mesh for.
       * 
       * @return
       *   A null reference if the object 'name' does not exist. 
       *   Otherwise a newly created mesh only drawing the wavefront
       *   sub-object.
       */
      virtual MeshRef createObjectMesh(const std::string& name);
    
      /**
       * Returns a list of all objects that are part of this wavefront
       * object file. Contains at least the default object-group.
       * 
       * @return
       *   A vector of object names
       */
      virtual ObjectNameVector getObjectNames() const;

      /**
       * Returns the mesh data object this Mesh is built from.
       * 
       * @return
       *   The MeshData the object is built from.
       */
      virtual const MeshDataRef& getMeshData() const;
    
      /**
       * Creates a new mesh object from MeshData.
       * 
       * @param meshData
       *   The MeshData implementing object providing
       *   the data to create a mesh from.
       * 
       * @throw MeshException
       *   Object construction failed because of an OpenGL error.
       */
      Mesh(MeshDataRef meshData);
  };
}

#endif // MESH_HPP_

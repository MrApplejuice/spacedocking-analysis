#include "MeshData.hpp"

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/exception/all.hpp>

#include "Engine.hpp"
#include "WavefrontObj.hpp"

namespace engine {
  typedef boost::error_info<struct tag_my_info, const std::string> WavefrontMeshDataProviderExceptionData;
  struct WavefrontMeshDataProviderException : public virtual boost::exception, public virtual std::exception {};
  
  typedef boost::function1<boost::shared_ptr<std::istream>, const std::string&> StreamOpenFunction;

  class WavefrontMeshDataProvider;
  typedef boost::shared_ptr<WavefrontMeshDataProvider> WavefrontMeshDataProviderRef;
  class WavefrontMeshDataProvider : public virtual MeshDataProvider {
    private:
      MeshDataRef meshDataRef;
    public:
      virtual MeshDataRef getMeshData();

      WavefrontMeshDataProvider(StreamOpenFunction openFunction, TextureManager& textureManager, const std::string& meshName);
  };
}

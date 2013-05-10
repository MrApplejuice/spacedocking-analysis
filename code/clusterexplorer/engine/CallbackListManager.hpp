#ifndef CALLBACKLISTMANAGER_HPP_
#define CALLBACKLISTMANAGER_HPP_

#include <vector>
#include <algorithm>
#include <stdexcept>

#include <boost/shared_ptr.hpp>

#define GEN_CALLBACK_ACCESSOR(T, name, varname) \
  engine::internal::CallbackListManager<T>::CallbackListItemRemover register ## name(T cb) { \
    return varname.addElement(cb); \
  }


namespace engine {
  namespace internal {
    template <typename T>
    class TypedFunctionPointerWrapper {
      public:
        T callback;
        
        TypedFunctionPointerWrapper(T callback) {
          this->callback = callback;
        }
    };
  
    template <typename T>
    struct TypedFunctionPointerWrapperRef {
      typedef boost::shared_ptr< TypedFunctionPointerWrapper<T> > type;
    };

    template <typename T>
    struct TypedFunctionPointerWrapperRefVector {
      typedef std::vector< typename TypedFunctionPointerWrapperRef<T>::type > type;
    };
        
    template <class T>
    class CallbackListManager {
      private:
        int invalidIndexCount;
        typename TypedFunctionPointerWrapperRefVector<T>::type functionList;
        
        int insertAtNewListIndex(typename TypedFunctionPointerWrapperRef<T>::type& f) {
          int i = 0;
          if (invalidIndexCount) {
            for (typename TypedFunctionPointerWrapperRefVector<T>::type::iterator it = functionList.begin(); it != functionList.end(); it++) {
              if (!(*it)) {
                *it = f;
                invalidIndexCount--;
                return i;
              }
              i++;
            }
          } else {
            i = functionList.size();
          }
          
          functionList.push_back(f);
          return i;
        }

        void removeElement(int id) {
          if ((id < 0) || ((unsigned int) id >= functionList.size())) {
            throw std::out_of_range("id out of range");
          }
          if (!functionList[id]) {
            throw std::out_of_range("invalid id");
          }
          functionList[id].reset();
          invalidIndexCount++;
        }
      
      public:
        class CallbackListItemRemover {
          private:
            int id;
            CallbackListManager* manager;
            boost::shared_ptr<bool> removed;
            
            CallbackListItemRemover(CallbackListManager& manager, int id) : id(id), manager(&manager), removed(new bool(false)) {}
          public:
            CallbackListItemRemover& operator=(const CallbackListItemRemover& other) {
              id = other.id;
              manager = other.manager;
              removed = other.removed;
              
              return *this;
            }
          
            void remove() {
              if (removed) {
                if (!*removed) {
                  manager->removeElement(id);
                  *removed = true;
                }
              }
            }
            
            CallbackListItemRemover() : id(0), manager(NULL), removed() {
            }
            
            friend class CallbackListManager;
        };

        CallbackListItemRemover addElement(T& item) {
          typename TypedFunctionPointerWrapperRef<T>::type tFuncPtrRef(new TypedFunctionPointerWrapper<T>(item));
          return CallbackListItemRemover(*this, insertAtNewListIndex(tFuncPtrRef));
        }
        
        std::vector<T> getCallbacks() {
          typename std::vector<T> result;
          result.reserve(functionList.size() - invalidIndexCount);
          for (typename TypedFunctionPointerWrapperRefVector<T>::type::iterator it = functionList.begin(); it != functionList.end(); it++) {
            if (*it) {
              result.push_back((*it)->callback);
            }
          }
          return result;
        }

        CallbackListManager() {
          invalidIndexCount = 0;
        }
        
        friend class CallbackListItemRemover;
    };
  }
}

#endif // CALLBACKLISTMANAGER_HPP_

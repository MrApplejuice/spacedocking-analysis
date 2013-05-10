#ifndef CLASSTOOLS_HPP_
#define CLASSTOOLS_HPP_

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_UNIQUE(x) CONCATENATE(x, __COUNTER__)

#define FLEXIBLE_GEN_GETTER(type, name, variable, funcmod) \
  virtual type get ## name() funcmod { \
    return variable; \
  }
#define GEN_GETTER(type, name, variable) FLEXIBLE_GEN_GETTER(type, name, variable, const)

#define FLEXIBLE_GEN_ABSTRACT_GETTER(type, name, funcmod) \
  virtual type get ## name() funcmod = 0
#define GEN_ABSTRACT_GETTER(type, name) FLEXIBLE_GEN_ABSTRACT_GETTER(type, name, const)

#define GEN_SETTER_DETAIL(type, name, variable, uniq) \
  virtual void set ## name(type uniq) { \
    variable = uniq; \
  }
#define GEN_SETTER(type, name, variable) GEN_SETTER_DETAIL(type, name, variable, MAKE_UNIQUE(__setterArgument__))

#define GEN_ABSTRACT_SETTER(type, name) \
  virtual void set ## name(type uniq) = 0

#define GEN_GETTER_AND_SETTER(type, name, variable) \
  GEN_GETTER(type, name, variable) \
  GEN_SETTER(type, name, variable)

#endif // CLASSTOOLS_HPP_

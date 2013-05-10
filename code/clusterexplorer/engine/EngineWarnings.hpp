#ifndef ENGINEWARNINGS_HPP_
#define ENGINEWARNINGS_HPP_

#include <iostream>

#define ENGINE_WARN_ONCE_INTERNAL(s, c) \
   { \
     static bool c = false; \
     if (!c) { \
       std::cerr << s << std::endl; \
       c = true; \
     } \
   }

#define ENGINE_WARN_ONCE(s) \
  ENGINE_WARN_ONCE_INTERNAL(s, warned_once_ ## __COUNTER__)

#endif // ENGINEWARNINGS_HPP_

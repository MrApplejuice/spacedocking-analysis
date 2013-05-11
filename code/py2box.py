#!/usr/bin/env python2

# Loads the all files for interactive mode in python 2

import sys
import os
sys.path += [os.path.join(os.path.realpath("."), os.path.dirname(sys.argv[0]), p) for p in ["py2", "universal"]]

from readdata import *
from clustering import *

importedBox = True
if len(sys.argv) > 1:
   if sys.argv[1].lower() == 'pkg':
     from pkgbox import *
   elif sys.argv[1].lower() == 'cluster':
     from cluster import *
     cluster_main(sys.argv[:1] + sys.argv[2:])
   elif sys.argv[1].lower() == 'guido':
     from guidobox import *
   else:
     print "Warning! unknown box", "'" + sys.argv[1] + "'", "- Falling back to guido"
     importedBox = False
if not importedBox:
  from guidobox import *
  

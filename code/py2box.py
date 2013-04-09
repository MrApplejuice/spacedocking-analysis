#!/usr/bin/env python2

# Loads the all files for interactive mode in python 2

import sys
import os
sys.path += [os.path.join(os.path.realpath("."), os.path.dirname(sys.argv[0]), p) for p in ["py2", "universal"]]

from readdata import *
from clustering import *
from guidobox import *

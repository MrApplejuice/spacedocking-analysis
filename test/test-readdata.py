#!/usr/bin/env python3

import sys

sys.path.insert(0, "..")

from readdata import loadData


data = loadData("../../data/data-2013.02.22.txt")
print("Loaded", len(data), "samples")

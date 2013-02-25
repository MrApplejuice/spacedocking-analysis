#!/usr/bin/env python3

import json

def loadData(filename):
  result = []
  f = open(filename, 'r')
  for line in f:
    line = line.strip()
    if (len(line) > 0) and (line[0] != '#'):
      result.append(json.loads(line))
  f.close()
  return result

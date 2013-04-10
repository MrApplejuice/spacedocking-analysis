#!/usr/bin/env python3

import json

def loadData(filename, max_samples=None):
  result = []
  f = open(filename, 'r')
  
  sample = 0
  for line in f:
    if ((max_samples is None) or (sample < max_samples)): 
      line = line.strip()
      if (len(line) > 0) and (line[0] != '#'):
        result.append(json.loads(line))
    sample += 1
  f.close()
  
  return result
  


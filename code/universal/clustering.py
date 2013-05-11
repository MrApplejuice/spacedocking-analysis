#!/usr/bin/env python3

from pylab import *

def euclidean(x, y):
  return sqrt(sum((array(x) - array(y))**2))

def pairer(data, distanceFunction=euclidean):
  """
    Note: Returns fully clustered tree in form of lists
  """
  data = [array(x) for x in data]
  
  def clusterDistance(c1, c2):
    if type(c1) is list:
      return min([clusterDistance(x, c2) for x in c1])
    elif type(c2) is list:
      return clusterDistance(c2, c1)
    else:
      return distanceFunction(data[c1], data[c2])
      
  indexes = list(range(len(data)))
  while len(indexes) > 2:
    if len(indexes) % 1000 == 0:
      sys.stderr.write(str(len(indexes)) + "\n")
    sys.stderr.write(str(len(indexes)) + "\n")
    
    distances = []
    for i in range(len(indexes)):
      print "  ",i
      for j in range(i + 1, len(indexes)):
        distances.append((i, j, clusterDistance(indexes[i], indexes[j])))
    i, j, minDistance = min(distances, key=lambda x: x[2])
    
    # Unify pair
    indexes.append([indexes[i], indexes[j]])
    del indexes[j];
    del indexes[i];
  return indexes

if __name__ == '__main__':
  targetFigure = figure()
  targetPlot = targetFigure.add_subplot(1, 1, 1)
  targetPlot.xaxis.set_view_interval(-1, 1)
  targetPlot.yaxis.set_view_interval(-1, 1)
  
  ion()
  targetFigure.show()
  
  points = targetFigure.ginput(0)
  ioff()
  
  print(pairer(points))

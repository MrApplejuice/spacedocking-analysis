#!/usr/bin/env python3

from pylab import *

def euclidean(x, y):
  return sqrt(sum((array(x) - array(y))**2))

def PKGCluster(data, distanceFunc=euclidean):
  data = [array(x) for x in data]
  
  distances = zeros((len(data), len(data)))
  for i in range(0, len(data)):
    for j in range(0, len(data)):
      if i != j:
        if i > j:
          distances[i, j] = distances[j, i]
        else:
          distances[i, j] = distanceFunc(data[i], data[j])
    
  cluster = array([None] * len(data))
  clusterFilter = ndarray(cluster.shape, dtype=bool) + True
  valid = array(eye(len(data))) < 1
  indexLookup = array(range(len(data)))
  newClusterIndex = 0

  while clusterFilter.any():
    minValue = distances[valid].min()
    markedValues = distances == minValue
    
    # Get one marked closest index pair in matrix with i < j
    i = indexLookup[markedValues.max(0)].min()
    j = indexLookup[markedValues[i]].min()
    
    if (cluster[i] == None) and (cluster[j] == None):
      cluster[i], cluster[j] = newClusterIndex, newClusterIndex
      newClusterIndex += 1
    else:
      if cluster[i] == None:
        cluster[i] = cluster[j]
      else:
        cluster[j] = cluster[i]
        
    clusterFilter[i] = False
    clusterFilter[j] = False
    
    valid *= -outer(-clusterFilter, -clusterFilter)
  
  return cluster
  

if __name__ == '__main__':
  targetFigure = figure()
  targetPlot = targetFigure.add_subplot(1, 1, 1)
  targetPlot.xaxis.set_view_interval(-1, 1)
  targetPlot.yaxis.set_view_interval(-1, 1)
  
  ion()
  targetFigure.show()
  
  points = targetFigure.ginput(0)
  
  print(PKGCluster(points))

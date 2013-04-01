#!/usr/bin/env python3

from pylab import *

def euclidean(x, y):
  return sqrt(sum((array(x) - array(y))**2))

def MeanSquareLengthLimitTracer(data, distanceFunc=lambda x, y: euclidean(x, y)**1):
  data = [array(x) for x in data]

  vIsNone = vectorize(lambda x: x is None)

  distances = zeros((len(data), len(data))) - 1
  for i in range(0, len(data)):
    for j in range(0, len(data)):
      if i != j:
        if i > j:
          distances[i, j] = distances[j, i]
        else:
          distances[i, j] = distanceFunc(data[i], data[j])
          
  distances_distribution = {}
  distances_distribution["mean"] = mean(distances[distances >= 0])
  distances_distribution["stddev"] = std(distances[(distances >= 0) * (distances < distances_distribution["mean"])])
  
  cluster = array([None] * len(data))
  clusterIndex = 0

  indexLookup = array(range(len(data)))
  limit = distances_distribution["mean"] - distances_distribution["stddev"]
  short_distances = (distances < limit)

  print(limit)
  f = figure()  
  p = f.add_subplot(1, 1, 1)
  p.hold(True)
  p.hist(distances.flatten())
  f.show()
  show()
  
  while vIsNone(cluster).any():
    seed = indexLookup[short_distances.max(0)].min(0)
    searchStack = [seed]
    
    # Search all elements from one cluster
    clusterElements = []
    while len(searchStack) > 0:
      e = searchStack.pop(0)
      
      subElements = short_distances[e,:]
      if not subElements.any():
        continue
      
      clusterElements.append(e)
      searchStack += list(indexLookup[subElements])
      
      short_distances[:,e] = False
      short_distances[e,:] = False
      
    # Write labels for elements and continue
    vIsInElements = vectorize(lambda x: x in clusterElements);
    cluster[vIsInElements(indexLookup)] = clusterIndex
    clusterIndex += 1
    
  return cluster

def PKGCluster(data, distanceFunc=euclidean, labels=None):
  """
  Has problems - 3 items are ALWAYS clusterd into ONE cluster even if 
  a cluster of two and one with a single element would be more optimal. 
  
  Example:
    x x                           x
    
  """
  data = [array(x) for x in data]
  
  if labels != None:
    labels = array(labels)
  
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

    if labels is None:
      clusterFilter[i] = False
      clusterFilter[j] = False
    else:
      f = labels == labels[i]
      cluster[f] = cluster[i]
      clusterFilter[f] = False
      
      f = labels == labels[j]
      cluster[f] = cluster[j]
      clusterFilter[f] = False
    
    valid *= -outer(-clusterFilter, -clusterFilter)
  
  return cluster
  
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
    distances = []
    for i in range(len(indexes)):
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

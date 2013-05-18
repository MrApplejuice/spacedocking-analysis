#!/usr/bin/env python2

from pylab import *
from opencl import generateOpenCLContext
import pyopencl as cl

clContext = None

def euclidean(x, y):
  return sqrt(sum((array(x) - array(y))**2))

pairerDistanceProg = None
def pairer(data):
  """
    Note: Returns fully clustered tree in form of lists
  """
  global clContext
  if clContext is None:
    clContext = generateOpenCLContext(cl.device_type.CPU)
  clQueue = cl.CommandQueue(clContext)
    
  global pairerDistanceProg
  if pairerDistanceProg is None:
    pairerDistanceCode = """
    float euclidean(int width, __global float data1[], __global float data2[]) {
      float result = 0.0f;
      for (int i = 0; i < width; i++) {
        result += pow(data1[i] - data2[i], 2);
      }
      return sqrt(result);
    }
    
    #define DISTANCE_FUNCTION euclidean
    
    __kernel void computeDistance(int width, int indexCount, int groupCount, __global float data[], __global int groupIndexes[], __global int indexes[], __global float distanceMat[]) {
      int data_i = get_global_id(0);
      int data_j = get_global_id(1);

      // Ony fill the lower left side of the matrix:
      //  0 0 0 0 0
      //  x 0 0 0 0 
      //  x x 0 0 0 
      //  x x x 0 0 
      //  x x x x 0 
      if (data_i < data_j) {
        if ((groupIndexes[data_i] > 0) || (groupIndexes[data_j] > 0)) { // Only positive group indexes mean "include"
          const int groupId_i = abs(groupIndexes[data_i]);
          const int groupId_j = abs(groupIndexes[data_j]);
          
          // Only consider starting points for groups
          if ((data_i == 0) || (abs(groupIndexes[data_i - 1]) != groupId_i)) {
            if ((data_j == 0) || (abs(groupIndexes[data_j - 1]) != groupId_j)) {
              float minDistance = -1 - groupIndexes[data_j];
              const int startJ = data_j;
              
              while ((data_i < indexCount) && (abs(groupIndexes[data_i]) == groupId_i)) {
                data_j = startJ;
                while ((data_j < indexCount) && (abs(groupIndexes[data_j]) == groupId_j)) {
                  const float d = DISTANCE_FUNCTION(width, data + indexes[data_i] * width, data + indexes[data_j] * width);
                  if ((minDistance < 0) || (d < minDistance)) {
                    minDistance = d;
                  }
                  data_j++;
                }
                data_i++;
              }
              
              distanceMat[(groupId_i - 1) * groupCount + (groupId_j - 1)] = minDistance;
            }
          }
        } 
      }
    }
    """
    
    pairerDistanceProg = cl.Program(clContext, pairerDistanceCode).build()

  data = array([x for x in data], dtype=float32)
  
  def clusterDistance(c1, c2):
    if type(c1) is list:
      result = None
      for x in c1:
        d = clusterDistance(x, c2)
        if (result is None) or (d < result):
          result = d
      return result
    elif type(c2) is list:
      return clusterDistance(c2, c1)
    else:
      return euclidean(data[c1], data[c2])
      
  def calcDistanceMatrix(binIndexes, doIncludeFunc=None):
    # Calculate with python
    #distanceMatrix = ndarray((len(binIndexes), len(binIndexes)))
    #distanceMatrix[:,:] = inf
    #for i in range(len(binIndexes)):
    #  for j in range(i + 1, len(binIndexes)):
    #    if (doIncludeFunc is None) or any([doIncludeFunc(x) for x in (i, j)]): 
    #      distanceMatrix[i, j] = clusterDistance(binIndexes[i], binIndexes[j])
    #      
    #dm1 = distanceMatrix
    
    # Calculate with opencl
    distanceMatrix = ndarray((len(binIndexes), len(binIndexes)), dtype=float32)
    distanceMatrix[:,:] = inf
    
    distanceMatrixBufferCl = cl.Buffer(clContext, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=distanceMatrix);
    dataBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data);
    
    def flt(x):
      if type(x) is list:
        return list(flatten(x))
      return [x]

    def negExclusives(i):
      if (doIncludeFunc is None) or (doIncludeFunc(i - 1)):
        return i
      else:
        return -i
    groupIndexes = hstack([array([negExclusives(i + 1) for j in flt(b)], dtype=int32) for i, b in enumerate(binIndexes)])
    groupIndexBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=groupIndexes)
    
    binIndexesBuffer = hstack(map(lambda x: array(flt(x), dtype=int32), binIndexes))
    indexBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=binIndexesBuffer)
    
    prev = pairerDistanceProg.computeDistance(clQueue, [binIndexesBuffer.shape[0], binIndexesBuffer.shape[0]], None, int32(data.shape[1]), int32(binIndexesBuffer.shape[0]), int32(len(binIndexes)), dataBufferCl, groupIndexBufferCl, indexBufferCl, distanceMatrixBufferCl)
    cl.enqueue_copy(clQueue, distanceMatrix, distanceMatrixBufferCl, is_blocking=True, wait_for=(prev,))
    
    #dm2 = distanceMatrix
    
    # DEBUG: Compare results
    #odifMat = dm1 - dm2
    #difMat = abs(odifMat[nonzero(-isnan(odifMat))])
    #print difMat
    #print dm1
    #print dm2
    #print "Average difference cl<->python: ", sum(difMat) / len(difMat)
    #if any(difMat > 0.1):
    #  w = where((odifMat > 0.1) * -isnan(odifMat))
    #  #print zip(*w)
    #  print zip(binIndexesBuffer, groupIndexes)
    #  print w[0][0], w[1][0]
    #  print "py:",dm1[w[0][0], w[1][0]]
    #  print "cl:",dm2[w[0][0], w[1][0]]
    #  print binIndexes[w[0][0]-1:w[0][0]+2]
    #  print binIndexes[w[1][0]-1:w[1][0]+2]
    #  sys.exit(0)
    
    return distanceMatrix
      
  #calcDistanceMatrix([0, 1, 2, 3, 4, 16])
  #calcDistanceMatrix([0, 1, 2, [3, 7], 4, 16])
  #sys.exit()
  ######## DEBUG END ##########
  

  indexes = list(range(len(data)))
  
  # Start with binned pairing
  if len(data) > 10000:
    pol_data_covar = cov(data, rowvar=0)
    pcs = eig(pol_data_covar)
    
    # Calculate groups and sort data samples into the groups
    TARGET_GROUP_SIZE = 300
    prinComp = pcs[1][0]
    projections = [dot(pcs[1][0], d) for d in data]
    bin_limits = [x(projections) for x in (min, max)]
    bin_width = float(bin_limits[1] - bin_limits[0]) / float(len(projections) / TARGET_GROUP_SIZE)
    bins = map(list, [[]] * (len(projections) / TARGET_GROUP_SIZE))
    
    # Distribute data over bins
    for sample_i, proj in enumerate(projections):
      bin_i = max(min(int((proj - bin_limits[0]) / bin_width), len(bins) - 1), 0)
      bins[bin_i].append(sample_i)
      
    print "Average size:",float(sum(map(len, bins))) / len(bins)
    print "Bin count:",len(bins)
    
    # Perform bin cleanup: unify bins with 1 or less elements with neighbor bins
    bin_i = 0
    while (bin_i < len(bins)) and (len(bins) > 1):
      if len(bins[bin_i]) <= TARGET_GROUP_SIZE / 2:
        if bin_i <= len(bins) / 2:
          bins[bin_i] = bins[bin_i] + bins[bin_i + 1]
          bins.pop(bin_i + 1)
        else:
          bins[bin_i] = bins[bin_i] + bins[bin_i - 1]
          bins.pop(bin_i - 1)
          bin_i -= 1
      else:
        bin_i += 1
        
    print "Average size:",float(sum(map(len, bins))) / len(bins)
    print "Bin count:",len(bins)
        
    # Iterate over bins and create pairs for every bin until it would
    # be paired with a node from a neighbor bin
    for bin_i in range(len(bins)):
      print "Before:", len(bins[bin_i])
      
      bin_offsets = [0]
      bin_data = list(bins[bin_i])
      bin_data_src_offset = [0] * len(bins[bin_i])
      if bin_i > 0:
        bin_data += bins[bin_i - 1]
        bin_data_src_offset += [-1] * len(bins[bin_i - 1])
        bin_offsets.append(-1)
      if bin_i < len(bins) - 1:
        bin_data += bins[bin_i + 1]
        bin_data_src_offset += [1] * len(bins[bin_i + 1])
        bin_offsets.append(1)

      # Create a distance matrix (woohoo, will fit into memory now :D)
      distanceMatrix = calcDistanceMatrix(bin_data, doIncludeFunc=lambda i: bin_data_src_offset[i] == 0)
      sys.exit(0)
              
      # Got all data... pair!
      doPairing = True
      while doPairing:
        i, j = unravel_index(distanceMatrix.argmin(), distanceMatrix.shape)
          
#        if not all([bin_data_src_offset[x] == 0 for x in (i, j)]):
#          print "Paired with other group:", [bin_data_src_offset[x] == 0 for x in (i, j)]
#          print "                        ", [bin_data_src_offset[x] for x in (i, j)]
#          print "                        ", [i, j]
        doPairing = all([bin_data_src_offset[x] == 0 for x in (i, j)]) or (len(bin_data) <= 1)
        bin_data_src_offset[i] = 0
        
        minCol = hstack((distanceMatrix[[i],:].T, distanceMatrix[:,[i]], distanceMatrix[[j],:].T, distanceMatrix[:,[j]])).min(1)
        distanceMatrix[[i],:] = array([minCol])
        distanceMatrix[:,[i]] = array([minCol]).T
        
        bin_data[i] = [bin_data[i], bin_data[j]]

        distanceMatrix = delete(delete(distanceMatrix, (j, i), 1), (j, i), 0)
        bin_data.pop(j)
        bin_data_src_offset.pop(j)
        
      # Repack bins
      for offset in bin_offsets:
        print "  Bef ass",bin_i + offset,offset,len(bins[bin_i + offset])
        bins[bin_i + offset] = [x for x, o in zip(bin_data, bin_data_src_offset) if o == offset]
        print "  After ass",bin_i + offset,offset,len(bins[bin_i + offset])
      
      print "After:", len(bins[bin_i])

      
  return ()

  while len(indexes) > 2:
#    if len(indexes) % 10 == 0:
#      sys.stderr.write(str(len(indexes)) + "\n")
    
    minDistance = None
    for i in range(len(indexes)):
      print "  ",i
      for j in range(i + 1, len(indexes)):
        distance = (i, j, clusterDistance(indexes[i], indexes[j]))
        if (minDistance is None) or (distance[2] < minDistance[2]):
          minDistance = distance
    i, j, minDistance = minDistance
    
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


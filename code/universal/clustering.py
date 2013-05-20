#!/usr/bin/env python2

from pylab import *
from opencl import generateOpenCLContext
import pyopencl as cl

clContext = None

def euclidean(x, y):
  return sqrt(sum((array(x) - array(y))**2))

pairerProg = None
def pairer(data):
  """
    Note: Returns fully clustered tree in form of lists
  """
  global clContext
  if clContext is None:
    clContext = generateOpenCLContext(cl.device_type.CPU)
  clQueue = cl.CommandQueue(clContext)
    
  global pairerProg
  if pairerProg is None:
    pairerCode = """
    float euclidean(int width, __global const float data1[], __global const float data2[]) {
      float result = 0.0f;
      for (int i = 0; i < width; i++) {
        result += pow(data1[i] - data2[i], 2);
      }
      return sqrt(result);
    }
    
    #define DISTANCE_FUNCTION euclidean
    
    __kernel void computeDistance(int maxDim, int width, int indexCount, int groupCount, __global const float data[], __global const int groupIndexes[], __global const int indexes[], __global float distanceMat[]) {
      int data_i = get_global_id(0);
      int data_j = get_global_id(1);
      
      if ((data_i >= maxDim) || (data_j >= maxDim)) {
        return;
      }

      // Ony fill the lower left side of the matrix:
      //  0 0 0 0 0
      //  x 0 0 0 0 
      //  x x 0 0 0 
      //  x x x 0 0 
      //  x x x x 0 
      const int groupId_i = abs(groupIndexes[data_i]);
      const int groupId_j = abs(groupIndexes[data_j]);
      
      // Only consider starting points for groups
      if ((data_i == 0) || (abs(groupIndexes[data_i - 1]) != groupId_i)) {
        if ((data_j == 0) || (abs(groupIndexes[data_j - 1]) != groupId_j)) {
          float minDistance = INFINITY;
          
          if (groupId_i < groupId_j) {
            if ((groupIndexes[data_i] > 0) || (groupIndexes[data_j] > 0)) { // Only positive group indexes mean "include"
              const int startJ = data_j;
              
              while ((data_i < indexCount) && (abs(groupIndexes[data_i]) == groupId_i)) {
                data_j = startJ;
                while ((data_j < indexCount) && (abs(groupIndexes[data_j]) == groupId_j)) {
                  const float d = DISTANCE_FUNCTION(width, data + indexes[data_i] * width, data + indexes[data_j] * width);
                  if (d < minDistance) {
                    minDistance = d;
                  }
                  data_j++;
                }
                data_i++;
              }
            }
          }
          
          distanceMat[(groupId_i - 1) * groupCount + (groupId_j - 1)] = minDistance;
        }
      }
    }
    
    __kernel void minCompressRow(int matsize, int matstride, __global const float mat[], __global float out[]) {
      const int id = get_global_id(0);
      
      float currentMin = INFINITY;
      
      for (int i = 0; i < matsize; i++) {
        if (mat[i * matstride + id] < currentMin) {
          currentMin = mat[i * matstride + id];
        }
      }
      
      out[id] = currentMin;
    }

    __kernel void minCompressCol(int matsize, int matstride, __global const float mat[], __global float out[]) {
      const int id = get_global_id(0);
      __global const float* row = mat + matstride * id;
      
      float currentMin = INFINITY;
      
      for (int i = 0; i < matsize; i++) {
        if (row[i] < currentMin) {
          currentMin = row[i];
        }
      }
      
      out[id] = currentMin;
    }
    
    __kernel void reduceMinIndex(const int vectorLen, __global const float vec[], int outIndex, __global int out[]) {
      float currentMin = INFINITY;
      int currentMinIndex = -1;

      for (int i = 0; i < vectorLen; i++) {
        if (vec[i] < currentMin) {
          currentMin = vec[i];
          currentMinIndex = i;
        }
      }
      
      out[outIndex] = currentMinIndex;
    }
    
    __kernel void reduceMinRowIndex(const int matrixSize, const int matrixStride, __global const float mat[], __global int out[]) {
      float currentMin = INFINITY;
      int currentMinIndex = -1;
      
      __global const float* row = mat + matrixStride * out[0];

      for (int i = 0; i < matrixSize; i++) {
        if (row[i] < currentMin) {
          currentMin = row[i];
          currentMinIndex = i;
        }
      }
      
      out[1] = currentMinIndex;
    }
    
    __kernel void minUnifyDistances(int ui_min, int ui_max, int matstride, __global float mat[]) {
      const int id = get_global_id(0);
      
      if (id != ui_min) { // Distance to one self does not exist (=> stays INFINITY)
        const int ui_min_index[2] = {min(id, ui_min), max(id, ui_min)};
        const int ui_max_index[2] = {min(id, ui_max), max(id, ui_max)};
       
        mat[ui_min_index[0] * matstride + ui_min_index[1]] =
              min(mat[ui_min_index[0] * matstride + ui_min_index[1]], 
                  mat[ui_max_index[0] * matstride + ui_max_index[1]]);
      }
    }
    
    __kernel void deleteRowAndCol(int index, int matsize, int matstride, __global float mat[]) {
      const int id = get_global_id(0);
      
      if (id < index) {
        // Shift horizontal
        for (int i = index; i < matsize - 1; i++) {
          mat[id * matstride + i] = mat[id * matstride + i + 1];
        }
        // Shift vertical
        for (int i = index; i < matsize - 1; i++) {
          mat[i * matstride + id] = mat[i * matstride + id + 1];
        }
      } else if (id > index) {
        const int offset = id - (index + 1);

        // Shift diagonal
        for (int i = 0; i < matsize - id; i++) {
          mat[(id - 1 + i) * matstride + index + i] = mat[(id - 1 + i + 1) * matstride + index + i + 1];
        }
        if (offset > 0) {
          for (int i = 0; i < matsize - id; i++) {
            mat[(index + i) * matstride + id - 1 + i] = mat[(index + i + 1) * matstride + id - 1 + i + 1];
          }
        }
      }
    }
    """
    pairerProg = cl.Program(clContext, pairerCode).build()

  data = array([x for x in data], dtype=float32)
  dataBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data);
  
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
    distanceMatrixBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE, size=distanceMatrix.nbytes);
    
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
    
    lpacketsize = int(min(sqrt(clContext.devices[0].max_work_group_size), sqrt(min([clContext.devices[0].max_work_item_sizes[0], clContext.devices[0].max_work_item_sizes[1]]))))
    gpacketdim = int(ceil(float(binIndexesBuffer.shape[0]) / float(lpacketsize)))

    prev = pairerProg.computeDistance(clQueue, (gpacketdim * lpacketsize, gpacketdim * lpacketsize), (lpacketsize, lpacketsize), int32(binIndexesBuffer.shape[0]), int32(data.shape[1]), int32(binIndexesBuffer.shape[0]), int32(len(binIndexes)), dataBufferCl, groupIndexBufferCl, indexBufferCl, distanceMatrixBufferCl)
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
      #w = where((odifMat > 0.1) * -isnan(odifMat))
      #print zip(*w)
      #print zip(binIndexesBuffer, groupIndexes)
      #print w[0][0], w[1][0]
      #print "py:",dm1[w[0][0], w[1][0]]
      #print "cl:",dm2[w[0][0], w[1][0]]
      #print binIndexes[w[0][0]-1:w[0][0]+2]
      #print binIndexes[w[1][0]-1:w[1][0]+2]
      #sys.exit(0)
    
    return distanceMatrix, distanceMatrixBufferCl
      
  #print calcDistanceMatrix([0, 1, 2, 3, 4, 16])
  #print calcDistanceMatrix([0, 1, 2, [3, 7], 4, 16])
  #sys.exit(0)
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
        
      set_printoptions(linewidth=200)
      
      ## Test stuff
      #bin_data = [[0, 63], [1, 5], [2, 68], [3, 60], [4, 2], [5, 70], [6, 37], [7, 61], [8, 83], [9, 67], [10, 8], [11, 85], [12, 20], [13, 48], [14, 64], [15, 32], [16, 3], [17, 33], [18, 94], [19, 50], [20, 14], [21, 96], [22, 46], [23, 10], [24, 24], [25, 7], [26, 45], [27, 97], [28, 63], [29, 84], [30, 68], [31, 0], [32, 89], [33, 91], [34, 65], [35, 19], [36, 62], [37, 22], [38, 92], [39, 5], [40, 47], [41, 92], [42, 73], [43, 2], [44, 52], [45, 17], [46, 55], [47, 37], [48, 83], [49, 35], [50, 93], [51, 27], [52, 72], [53, 80], [54, 37], [55, 41], [56, 28], [57, 50], [58, 25], [59, 82], [60, 83], [61, 99], [62, 93], [63, 36], [64, 52], [65, 99], [66, 66], [67, 69], [68, 84], [69, 5], [70, 83], [71, 67], [72, 58], [73, 99], [74, 83], [75, 34], [76, 31], [77, 6], [78, 46], [79, 19], [80, 92], [81, 70], [82, 15], [83, 1], [84, 45], [85, 67], [86, 50], [87, 75], [88, 56], [89, 19], [90, 70], [91, 95], [92, 10], [93, 35], [94, 43], [95, 1], [96, 9], [97, 18], [98, 48], [99, 56]]
      #bin_data = [0, 1, 2, 3, 4, 5]
      #print bin_data
      #bin_offsets = [0]
      #bin_data_src_offset = [0] * len(bin_data)

      ## Create a distance matrix (woohoo, will fit into memory now :D)
      distanceMatrix, clDistanceMatrix = calcDistanceMatrix(bin_data, doIncludeFunc=lambda i: bin_data_src_offset[i] == 0)
      distanceMatrixStride = len(bin_data)
      distanceMatrixSize = len(bin_data)
      
      # Got all data... pair!
      doPairing = True
      
      minPosition = ndarray((2), dtype=int32)
      minPosition[:] = 0
      clMinPosition = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=minPosition)
      
      clColBuffer = cl.Buffer(clContext, cl.mem_flags.READ_WRITE, size=distanceMatrixStride * float32().nbytes)
      clRowBuffer = cl.Buffer(clContext, cl.mem_flags.READ_WRITE, size=distanceMatrixStride * float32().nbytes)
      
      prevs = ()
      while doPairing:
        ## Find minimum in distance matrix
        
        # Python way
        #i_py, j_py = unravel_index(distanceMatrix.argmin(), distanceMatrix.shape)
        
        # OpenCL way
        # Create minimum vectors
        if len(prevs) > 0:
          cl.wait_for_events(prevs)
        prevs = \
          pairerProg.minCompressCol(clQueue, [distanceMatrixSize], None, int32(distanceMatrixSize), int32(distanceMatrixStride), clDistanceMatrix, clColBuffer),
        
        # Compress these vectors to matrix offsets
        prevs = \
          pairerProg.reduceMinIndex(clQueue, [1], None, int32(distanceMatrixSize), clColBuffer, int32(0), clMinPosition, wait_for=prevs),
        prevs = \
          pairerProg.reduceMinRowIndex(clQueue, [1], None, int32(distanceMatrixSize), int32(distanceMatrixStride), clDistanceMatrix, clMinPosition, wait_for=prevs),

        # Load indexes
        cl.enqueue_copy(clQueue, minPosition, clMinPosition, is_blocking=True, wait_for=prevs)
        i, j = minPosition
        
        ## Do pairing
        i, j = min(i, j), max(i, j)
        
        ## Immediately start opencl matrix update even if uneccessary
        ## OpenCL way
        prevs = (pairerProg.minUnifyDistances(clQueue, [distanceMatrixSize], None, int32(i), int32(j), int32(distanceMatrixStride), clDistanceMatrix),)
        prevs = (pairerProg.deleteRowAndCol(clQueue, [distanceMatrixSize], None, int32(j), int32(distanceMatrixSize), int32(distanceMatrixStride), clDistanceMatrix, wait_for=prevs),)
        distanceMatrixSize -= 1
        
        # Track if this is a valid pairing with other bins
        doPairing = all([bin_data_src_offset[x] == 0 for x in (i, j)])
        bin_data_src_offset[i] = 0
        
        # !!!Can perhaps merge more points by marking points as dirty if they cannot be matched!!!
        
        if doPairing:        
          bin_data[i] = [bin_data[i], bin_data[j]]
          bin_data.pop(j)
          bin_data_src_offset.pop(j)

          ## Matrix update
          # Started above
          
          # Py way
          #minCol = hstack((distanceMatrix[[i],:].T, distanceMatrix[:,[i]], distanceMatrix[[j],:].T, distanceMatrix[:,[j]])).min(1)
          #distanceMatrix[[i],:] = array([hstack((array([inf] * (i + 1)), minCol[i + 1:]))])
          #distanceMatrix[:,[i]] = array([hstack((minCol[:i], array([inf] * (distanceMatrix.shape[0] - i))))]).T
          #distanceMatrix = delete(delete(distanceMatrix, (j,), 1), (j,), 0)
          
          doPairing = distanceMatrixSize > 1
          
          # Debug stuff
          #print "cl:",i, j, " py:",i_py, j_py
          #print distanceMatrix[i,j], distanceMatrix[i_py,j_py]
          #print distanceMatrix.argmin()

          #print "py:"
          #print distanceMatrix

          #distanceMatrixDeb = ndarray((distanceMatrixStride, distanceMatrixStride), dtype=float32)
          #cl.enqueue_copy(clQueue, distanceMatrixDeb, clDistanceMatrix, is_blocking=True, wait_for=prevs)
          #print "cl:"
          #print distanceMatrixDeb[:distanceMatrixSize,:distanceMatrixSize]

      cl.wait_for_events(prevs)        
        
      # Repack bins
      for offset in bin_offsets:
        #print "  Bef ass",bin_i + offset,offset,len(bins[bin_i + offset])
        bins[bin_i + offset] = [x for x, o in zip(bin_data, bin_data_src_offset) if o == offset]
        #print "  After ass",bin_i + offset,offset,len(bins[bin_i + offset])
      
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


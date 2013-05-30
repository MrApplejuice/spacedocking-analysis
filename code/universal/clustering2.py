#!/usr/bin/env python2

from pylab import *
from opencl import generateOpenCLContext
import pyopencl as cl

clContext = None

def euclidean(x, y):
  return sqrt(sum((array(x) - array(y))**2))

pairerProg = None

class DeleteModification:
  def __init__(self, index):
    self.__i = index
    
  def modIndex(self, index):
    if index >= self.__i:
      index += 1
    return index
    
  def modLen(self, l):
    return l - 1
    
  def modifyVector(self, v):
    #print "DELETING", self.__i
    return delete(v, self.__i)

class SetItemModification:
  def __init__(self, index, value):
    self.__i = index
    self.__v = value
    
  def modIndex(self, index):
    return index
    
  def modLen(self, l):
    return l
    
  def modifyVector(self, v):
    #print "SETTING", self.__i, "=", self.__v
    v[self.__i] = self.__v
    return v


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
    
    __kernel void computeDistance(int dataVectorLen, int groupStartOffset, int indexCount, int matrixWidth, __global const float data[], __global const int groupIndexes[], __global const int indexes[], __global float distanceMat[]) {
      int data_i = get_global_id(0);
      int data_j = get_global_id(1) + groupStartOffset;
      
      if ((data_i >= indexCount) || (data_j >= indexCount)) {
        return;
      }

      const int groupId_i = abs(groupIndexes[data_i]);
      const int groupId_j = abs(groupIndexes[data_j]);
      
      // Only consider starting points for groups
      if ((data_i == 0) || (abs(groupIndexes[data_i - 1]) != groupId_i)) {
        if ((data_j == 0) || (abs(groupIndexes[data_j - 1]) != groupId_j)) {
          float minDistance = INFINITY;
        
          if (groupId_i != groupId_j)  {
            if ((groupIndexes[data_i] > 0) || (groupIndexes[data_j] > 0)) { // Only positive group indexes mean "include"
              const int startJ = data_j;
              
              while ((data_i < indexCount) && (abs(groupIndexes[data_i]) == groupId_i)) {
                data_j = startJ;
                while ((data_j < indexCount) && (abs(groupIndexes[data_j]) == groupId_j)) {
                  const float d = DISTANCE_FUNCTION(dataVectorLen, data + indexes[data_i] * dataVectorLen, data + indexes[data_j] * dataVectorLen);
                  if (d < minDistance) {
                    minDistance = d;
                  }
                  data_j++;
                }
                data_i++;
              }
            }
          }
          
          distanceMat[(groupId_i - 1) * matrixWidth + (groupId_j - abs(groupIndexes[groupStartOffset]))] = minDistance;
        }
      }
    }

    __kernel void findColMins(int width, int height, __global const float mat[], __global int out[]) {
      const int id = get_global_id(0);
      
      float currentMin = INFINITY;
      int currentMinIndex = -1;
      
      for (int i = 0; i < height; i++) {
        if (mat[i * width + id] < currentMin) {
          currentMin = mat[i * width + id];
          currentMinIndex = i;
        }
      }
      
      out[id] = currentMinIndex;
    }

    """
    pairerProg = cl.Program(clContext, pairerCode).build()

  data = array([x for x in data], dtype=float32)
  dataBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data);
  
  def calcDistanceMatrix(indexes, offset, count=100):
    # Calculate with opencl
    distanceMatrixBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE, size=count * len(indexes) * float32().nbytes)
    
    def flt(x):
      if type(x) is list:
        return list(flatten(x))
      return [x]

    groupIndexes = hstack([array([i + 1] * len(flt(b)), dtype=int32) for i, b in enumerate(indexes)])
    groupIndexBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=groupIndexes)
    
    # Find starting offset
    groupStartOffset = min(where(groupIndexes == offset + 1)[0])

    indexBuffer = hstack(map(lambda x: array(flt(x), dtype=int32), indexes))
    indexBufferCl = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=indexBuffer)
    
    lpacketsize = int(min(clContext.devices[0].max_work_group_size, clContext.devices[0].max_work_item_sizes[0]))
    gpacketdim = int(ceil(float(indexBuffer.shape[0]) / float(lpacketsize)))

    prev = pairerProg.computeDistance(clQueue, (gpacketdim * lpacketsize, count), (lpacketsize, 1), int32(data.shape[1]), int32(groupStartOffset), int32(indexBuffer.shape[0]), int32(count), dataBufferCl, groupIndexBufferCl, indexBufferCl, distanceMatrixBufferCl)
    
    return distanceMatrixBufferCl, prev
    
  def findMinima(clMatrix, width, height, prevs=()):
    clMinIndexBuffer = cl.Buffer(clContext, cl.mem_flags.WRITE_ONLY, size=int32().nbytes * width)
    
    prev = pairerProg.findColMins(clQueue, [width], None, int32(width), int32(height), clMatrix, clMinIndexBuffer, wait_for=prevs)
    
    return clMinIndexBuffer, prev
    
  class SerializeCache:
    def __init__(self, size=500):
      self.size = size
      self.__cache = []
      
    def manageObject(self, o):
      o.serializationCacheIndex = None

    def use(self, o):
      try:
        if o.serializationCacheIndex is None:
          # Place at start - freshly loaded :D
          self.__cache.insert(0, o)
          o.serializationCacheIndex = 0
          for co in self.__cache[1:]:
            co.serializationCacheIndex += 1

          o.load()
          
          while len(self.__cache) > self.size:
            decachedObject = self.__cache[-1]
            decachedObject.unload()
            self.__cache.remove(decachedObject)
            decachedObject.serializationCacheIndex = None
        else:
          # Move up by one place if data is used
          targetIndex = o.serializationCacheIndex - 5
          if targetIndex > 0: # Ignore first n places - it would still be at the top of the cache list
            self.__cache[targetIndex].serializationCacheIndex = o.serializationCacheIndex
            self.__cache[o.serializationCacheIndex] = self.__cache[targetIndex]
            self.__cache[targetIndex] = o
            o.serializationCacheIndex = targetIndex
          
      except AttributeError:
        raise Exception(str(o) + " is not managed")
    
  class SerializedColumn:
    CACHE = SerializeCache(200)
    
    def _getFilename(self):
      return self.__filename;
    
    def save(self):
      if self.__distances is None:
        SerializedColumn.CACHE.use(self)
      
      self._tryApplyMods() # Use updated data so that saved values are "unpatched"
      
      #print "Writing",self._getFilename()
      
      f = file(self._getFilename(), "w")
      
      f.write(str(self.getMinIndex()) + " " + str(self.getMinValue()) + "\n")
      np.save(f, self.__distances)
      
      f.close()
      
    def load(self):
      if self.__distances is None:
        #print "Reading",self._getFilename()
        
        f = file(self._getFilename(), "r")
        
        line = ""
        while not "\n" in line:
          line += f.read(1024)
        line = line.split("\n")[0]
        
        if not self.__wasMinimumUpdated: # Harddisk data has higher precedence - use harddisk data
          splits = line.strip().split(" ")
          self.minimumIndex = int(splits[0])
          self.minimumValue = float(splits[1])
        
        f.seek(len(line) + 1)

        self.__distances = np.load(f)
        self.__itemCount = len(self.__distances)
        
        f.close()
        
      self._tryApplyMods()
      
    def unload(self):
      self.save()
      self.__distances = None
    
    def getVector(self):
      SerializedColumn.CACHE.use(self)
      return self.__distances
    
    def __getitem__(self, index):
      if type(index) is slice:
        raise Exception("Slices are not supported")
        
      SerializedColumn.CACHE.use(self)
        
      for m in reversed(self.__modifications):
        index = m.modIndex(index)
        
      return self.__distances[index]
    
    def __len__(self):
      itemCount = self.__itemCount
      for m in self.__modifications:
        itemCount = m.modLen(itemCount)
      return itemCount
      
    def deleteItem(self, i):
      if (i < 0) or (i >= len(self)):
        raise IndexError("Index out of range: " + str(i))
          
      if not self.minimumIndex is None:
        if i == self.getMinIndex():
          self.invalidateMinimumValues()
        elif i < self.getMinIndex():
          self.minimumIndex -= 1
          self.__wasMinimumUpdated = True
          
      self.__modifications.append(DeleteModification(i))
      self._tryApplyMods()
      
    def __setitem__(self, index, value):
      if (i < 0) or (i >= len(self)):
        raise IndexError("Index out of range: " + str(i))
      
      if not self.minimumValue is None:
        if value < self.minimumValue:
          minimumIndex = index
          minimumValue = value
          self.__wasMinimumUpdated = True
        else:
          if index == self.minimumIndex:
            self.invalidateMinimumValues()
      
      self.__modifications.append(SetItemModification(index, value))
      self._tryApplyMods()
      
    def _recomputeMinimum(self):
      SerializedColumn.CACHE.use(self)
      self._tryApplyMods()
      
      self.minimumIndex = argmin(self.__distances)
      self.minimumValue = self.__distances[self.minimumIndex]
      self.__wasMinimumUpdated = True
      
    def invalidateMinimumValues(self):
      self.minimumIndex = None
      self.minimumValue = None
      self.__wasMinimumUpdated = True
      
    def getMinValue(self):
      if self.minimumValue is None:
        self._recomputeMinimum()
      return self.minimumValue
    
    def getMinIndex(self):
      if self.minimumIndex is None:
        self._recomputeMinimum()
      return self.minimumIndex
    
    def _tryApplyMods(self):
      if len(self.__modifications) > 250: # Hard limit to modification count
        SerializedColumn.CACHE.use(self)
        
      if not self.__distances is None:
        if len(self.__modifications) > 0:
          for m in self.__modifications:
            self.__distances = m.modifyVector(self.__distances)
          self.__itemCount = len(self.__distances)
          self.__modifications = []
        
    def __init__(self, index, distances, minimumIndex=None):
      SerializedColumn.CACHE.manageObject(self)

      self.__modifications = []
      
      self.index = index
      self.__distances = array(distances)
      self.__itemCount = len(self.__distances)
      self.__filename = "tmp/v" + str(self.index) + ".dat"
      
      self.__wasMinimumUpdated = False # Checks if memory or harddisk data should be used for minimum values
      
      self.minimumIndex = minimumIndex
      if self.minimumIndex is None:
        self.minimumValue = None
      else:
        self.minimumValue = self.__distances[self.minimumIndex]
      
      self.save()
      SerializedColumn.CACHE.use(self)
    
  indexes = range(len(data))
  
  sdmat = []
  
  count = 500
  distanceBuffer = ndarray((data.shape[0], count), dtype=float32)
  minimaBuffer= ndarray((count), dtype=int32)

  def pcd(i):
    if not copiesCompleted is None:
      lDistanceBuffer = distanceBuffer[:,:min(count, data.shape[0] - i * count)]
      for j in range(lDistanceBuffer.shape[1]):
        svec = SerializedColumn(i * count + j, lDistanceBuffer[:,j].T, minimaBuffer[j])
        sdmat.append(svec)
  processCopiedData = lambda n: 0

  # Calculate distance matrix  
  last_i = None
  for i in range(int(ceil(float(data.shape[0]) / float(count)))):
    print "Computing ",i*count, "to",(i+1)*count-1
    
    clDistanceBuffer, prevEvent = calcDistanceMatrix(indexes, i * count, count=count)
    processCopiedData(last_i)
    
    copiesCompleted = []
    copiesCompleted.append(cl.enqueue_copy(clQueue, distanceBuffer, clDistanceBuffer, wait_for=(prevEvent,)))
    
    clMinima, minimaWait = findMinima(clDistanceBuffer, count, len(indexes), prevs=(prevEvent,))
    copiesCompleted.append(cl.enqueue_copy(clQueue, minimaBuffer, clMinima, wait_for=(minimaWait,)))

    cl.wait_for_events(copiesCompleted)

    processCopiedData = pcd
    last_i = i
    
  processCopiedData(last_i)
  
  # Created sorted list of minima
  distanceSortedColumns = list(sdmat)
  distanceSortedColumns.sort(key=lambda x: x.getMinValue())

  def sortInColumn(col):
    try:
      distanceSortedColumns.remove(col)
    except ValueError:
      pass
      
    minValue = col.getMinValue()
    for i in range(len(distanceSortedColumns)):
      if distanceSortedColumns[i].getMinValue() >= minValue:
        distanceSortedColumns.insert(i, col)
        minValue = None
        break
    if not minValue is None:
      distanceSortedColumns.append(col)
    
  
  # Start pairing process
  MEMORY = 10000000 # Number of simultaneously loaded values
  while len(sdmat) > 1:
    # Increase used memory size if possible
    SerializedColumn.CACHE.size = max(SerializedColumn.CACHE.size, MEMORY / len(sdmat))
    print "Cache size is",SerializedColumn.CACHE.size
    
    # Find shortest distance
    col = distanceSortedColumns.pop(0)
    distance = col.getMinValue()
    thisColIndex = sdmat.index(col) # Obtain "this col index"
    otherColIndex = col.getMinIndex()
    otherCol = sdmat[otherColIndex]
    
    thisDistance = (otherColIndex, col[otherColIndex], col.getMinValue(), thisColIndex, otherCol[thisColIndex], otherCol.getMinIndex(), otherCol.getMinValue(), argmin(col.getVector()), min(col.getVector()))
    
    # Delete all otherColIndex distance values
    for o in sdmat:
      o.deleteItem(otherColIndex)
    
    # Remove otherCol
    sdmat.pop(otherColIndex)

    # Unify columns - we know the col index of "otherCol" so "otherCol" is removed
    distanceSortedColumns.remove(otherCol)

    col.index = [col.index, otherCol.index, distance]
    print "Pairing", col.index, " ", len(sdmat), "left", " ", "distance", thisDistance
    
    thisColIndex = sdmat.index(col) # Update "this col index" after sdmat.pop(otherColIndex) has been executed
    for i in range(min(len(col), len(sdmat))):
      if col[i] != inf:
        col[i] = min(col[i], otherCol[i])
        sdmat[i][thisColIndex] = col[i]
        
        # Recompute minimum value if necessary
        if sdmat[i].minimumValue is None:
          sortInColumn(sdmat[i])
    col.invalidateMinimumValues()
    
    # Insert updated min values into sorted list
    sortInColumn(col)
    
    # Test: Rows and columns equal?
    #for i in range(min(len(col), len(sdmat))):
    #  if col[i] != sdmat[i][thisColIndex]:
    #    print thisColIndex, i, col.getVector()[i-2:i-2+5], sdmat[i].getVector()[thisColIndex-3:thisColIndex-3+7]
    #    raise Exception("Shiz happenz")
    
    # Slow fix: Force sorting
    #distanceSortedColumns.sort(key=lambda x: x.getMinValue())
    
  return sdmat[0].index

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


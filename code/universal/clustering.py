#!/usr/bin/env python3

from pylab import *
from opencl import generateOpenCLContext
import pyopencl as cl

clContext = None

def euclidean(x, y):
  return sqrt(sum((array(x) - array(y))**2))

def cartesianize(data):
  for i in range(len(data)):
    accumSin = 1.0
    r = data[i, 0]
    for j in range(1, len(data[i])):
      s = sin(data[i, j])
      data[i, j - 1] = r * accumSin * cos(data[i, j])
      accumSin *= s
    data[i, -1] = r * accumSin

def polarize(data):
  orgvec = array(data)
  
  def vert(i, j, r):
    print "VERTING"
    d = array([hstack(([r], data[i][:j], [pi / 2]))])
    cartesianize(d)
    result = hstack((d[0], zeros((len(data[i]) - len(d[0])))))
    #result = zeros((len(data[i])))
    #result[j] = d[0, j]
    return result
  
  for i in range(len(data)):
    r = norm(data[i])
    
    for j in range(len(data[i]) - 2):
      n = norm(data[i][j + 1:])

      if abs(data[i, j]) < 0.00001:
        data[i, j] = pi / 2
      else:
        if data[i, j] < 0:
          data[i, j] = pi + arctan(n / data[i, j])
        else:
          data[i, j] = arctan(n / data[i, j])
    
    n = norm(data[i, -2:]) + data[i, -2]
    if abs(n) < 0.00001:
      data[i, -2] = pi
    else:
      data[i, -2] = 2 * arctan(data[i, -1] / n)
    
    data[i, :] = array(hstack((array([r]), data[i, :-1])))
  
"""
print "Polarize test:"

for i in range(10000000):
  test = array([[10.0 * (random() * random() * random() * 2.0 - 1.0) for i in range(10)]], dtype=float32)
  d = array(test)
  polarize(d)
  cartesianize(d)
  if norm(test - d) > 0.1:
    print "Error: "
    print test
    print d

asdflnkasdjklf
"""



def pairer_cl(data, distanceFunction=euclidean):
  """
    Note: Returns fully clustered tree in form of lists
  """
  global clContext
  if clContext is None:
    clContext = generateOpenCLContext(cl.device_type.CPU)
  
  minDistanceClProg = """
  float dotProdDist(int size, __global float d1[], __global float d2[]) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
      result += d1[i] * d2[i];
    }
    return result;
  }
  
  __kernel void calculateBestDistances(int inputVectorLength, int compare1, int compare2offset, __global float inputData[], __global float outVector[], __global int outCombinations[]) {
    const int outOffset = get_global_id(0);
    const int compare2 = compare2offset + get_global_id(0);
    
    const float distance = dotProdDist(inputVectorLength, inputData + inputVectorLength * compare1, inputData + inputVectorLength * compare2);
    if ((outVector[outOffset] > distance) || (outVector[outOffset] < 0)) {
      outVector[outOffset] = distance;
      outCombinations[outOffset * 2 + 0] = compare1;
      outCombinations[outOffset * 2 + 1] = compare2;
    }
  }
  """
  
  data = array([array(x) for x in data], dtype=float32)
  data_max_length = max(map(len, data))
  data = array(map(lambda x: hstack((x, array([0.0] * (data_max_length - len(x))))), data), dtype=float32)
  dataClBuffer = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data.flatten())

  COMPUTATION_PACKET_SIZE = 100000
  
  outDistanceBuffer = array([-1] * COMPUTATION_PACKET_SIZE, dtype=float32)
  outDistanceClBuffer = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=outDistanceBuffer.flatten())

  outCombinationBuffer = array([0, 0] * COMPUTATION_PACKET_SIZE, dtype=int32)
  outCombinationClBuffer = cl.Buffer(clContext, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=outCombinationBuffer.flatten())
  
  prog = cl.Program(clContext, minDistanceClProg).build()
  
  clQueue = cl.CommandQueue(clContext)

  #//outDistanceBuffer[:] = -1.0
  #//cl.enqueue_copy(clQueue, outD)
  cl.enqueue_fill_buffer(clQueue, outDistanceClBuffer, int32(-1), 0, len(outDistanceBuffer))
  for j in range(data.shape[0]):
    for i in range(j, data.shape[0], COMPUTATION_PACKET_SIZE):
      compPacket = min(COMPUTATION_PACKET_SIZE, len(data) - i)
      print i, compPacket, " ",
      prog.calculateBestDistances(clQueue, [compPacket], None, int32(data.shape[1]), int32(j), int32(i), dataClBuffer, outDistanceClBuffer, outCombinationClBuffer).wait()
    print
  
  return ()
  
  indexes = list(range(len(data)))
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

def pairer(data, distanceFunction=euclidean):
  """
    Note: Returns fully clustered tree in form of lists
  """
  data = [array(x) for x in data]
  
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
      return distanceFunction(data[c1], data[c2])
      
  indexes = list(range(len(data)))
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


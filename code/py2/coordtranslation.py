#!/usr/bin/env python2

from pylab import *
from opencl import generateOpenCLContext
import pyopencl as cl

clContext = None
polarizeProg = None

def cartesianize(data):
  data = array(data)
  for i in range(len(data)):
    accumSin = 1.0
    r = data[i, 0]
    for j in range(1, len(data[i])):
      s = sin(data[i, j])
      data[i, j - 1] = r * accumSin * cos(data[i, j])
      accumSin *= s
    data[i, -1] = r * accumSin
  return data

def polarize(data):
  data = array(data)
  orgvec = array(data)
  
  def vert(i, j, r):
    d = array([hstack(([r], data[i][:j], [pi / 2]))])
    cartesianize(d)
    result = hstack((d[0], zeros((len(data[i]) - len(d[0])))))
    #result = zeros((len(data[i])))
    #result[j] = d[0, j]
    return result
  
  for i in range(len(data)):
    if (i + 1) % 1000 == 0:
      print i
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
    
  return data
  
def cl_polarize(data):
  global clContext, polarizeProg
  if clContext is None:
    clContext = generateOpenCLContext(cl.device_type.CPU)

  data = array(data, dtype=float32)
  inputClBuffer = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data.flatten())
  outputClBuffer = cl.Buffer(clContext, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=data.flatten())

  polarizeProgramCode = """
  __kernel void polarize(int width, __global float input[], __global float outputBuffer[]) {
    __global float* in = input + get_global_id(0) * width;
    __global float* out = outputBuffer + get_global_id(0) * width;
    
    // Calculate length
    float sum = 0.0;
    for (int i = 0; i < width; i++) {
      sum += in[i] * in[i];
    }
    out[0] = sqrt(sum);

    // Calculate angles
    float sum2 = 0.0;
    for (int i = 0; i < width - 2; i++) {
      sum2 += in[i] * in[i];
      const float len = sqrt(sum - sum2);

      if (fabs(in[i]) <= 0.00001f) {
        out[i + 1] = M_PI_2;
      } else {
        float output = atan(len / in[i]);
        if (in[i] < 0) output += M_PI; 
        
        out[i + 1] = output;
      }
    }
    
    const float2 vec = (float2)(in[width - 2], in[width - 1]);
    const float n = length(vec) + in[width - 2];
    if (fabs(n) <= 0.00001f) {
      out[width - 1] = M_PI;
    } else {
      out[width - 1] = 2.0f * atan(in[width - 1] / n);
    }
  }
  
  """
  
  if polarizeProg is None:
    polarizeProg = cl.Program(clContext, polarizeProgramCode).build()
    
  clQueue = cl.CommandQueue(clContext)
  
  polarizeProg.polarize(clQueue, [len(data)], None, int32(data.shape[1]), inputClBuffer, outputClBuffer)
  cl.enqueue_copy(clQueue, data, outputClBuffer, is_blocking=True)

  return data

  for i in range(len(data)):
    if i % 1000 == 0:
      print i
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
import sys
print "Polarize test:"

def prnd(zerochance=0.1):
  if random() < zerochance:
    return 0.0
  else:
    return 10.0 * (random() * random() * random() * 2.0 - 1.0)
    
for i in range(10000000):
  test = array([[prnd() for i in range(1000)]], dtype=float32)
  d = array(test)
  pd = polarize(d)
  clpd = cl_polarize(d)
  if norm(pd - clpd) > 0.001:
    print "Error: OpenCL unprecise"
    print pd
    print clpd
    print "difference is",norm(pd - clpd)
    sys.exit(0)
  
  cpd = cartesianize(pd)
  if norm(test - cpd) > 0.1:
    print "Error: Cartesian reconstruction failed"
    print test
    print d
    sys.exit(0)

sys.exit(0)
#"""



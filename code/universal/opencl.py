import pyopencl as cl

def generateOpenCLContext(favorite=cl.device_type.GPU):
  result = None
  
  allDevices = []
  for platform in cl.get_platforms():
    try:
      allDevices += platform.get_devices(favorite)
    except:
      pass
      
  if len(allDevices) <= 0:
    for platform in cl.get_platforms():
      try:
        allDevices += platform.get_devices(cl.device_type.ALL)
      except:
        pass
        
  if len(allDevices) > 0:
    result = cl.Context(allDevices)
  
  return result


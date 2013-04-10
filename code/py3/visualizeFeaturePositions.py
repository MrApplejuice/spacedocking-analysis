from pylab import *

def showFeaturePositions(samples, title=""):
  fig = figure()
  fig.suptitle(title)
  plot = fig.add_subplot(1, 1, 1)
  
  extents = [None, None, None, None]
  def none_extrema(ef, *args):
    if all([x is None for x in args]):
      return None
    else:
      return ef(filter(lambda x: not x is None, args))
  
  for s in samples:
    for frame in s["frames"]:
      for f in frame["features"]["features"]:
        extents = [none_extrema(min, f["x"], extents[0]), none_extrema(min, f["y"], extents[1]), \
                   none_extrema(max, f["x"], extents[2]), none_extrema(max, f["y"], extents[3])]
        plot.plot(f["x"], f["y"], ',b')
  if all([not x is None for x in extents]):
    plot.plot([extents[0], extents[2], extents[2], extents[0], extents[0]], \
              [extents[1], extents[1], extents[3], extents[3], extents[1]], '-k')
    print("X: min : %f, max : %f" % (extents[0], extents[2]))
    print("Y: min : %f, max : %f" % (extents[1], extents[3]))
    
  plot.invert_yaxis()
  fig.show()

if __name__ == '__main__':
  import sys
  import os
  sys.path.append(os.path.join(os.path.realpath("."), os.path.dirname(sys.argv[0]), "../universal"))
  
  from readdata import loadData
  data = loadData(sys.argv[1])
  
  # AR Drone 1
  print("AR Drone 1")
  showFeaturePositions(filter(lambda x: "Drone 1" in x["device_version"], data), title="AR Drone 1")
  
  # AR Drone 2
  print("AR Drone 2")
  showFeaturePositions(filter(lambda x: "Drone 2" in x["device_version"], data), title="AR Drone 2")
  
  show()

from pylab import *
import pdb;

def showFeaturePositions(samples, title=""):
  fig = figure(facecolor='white', edgecolor='white');
  fig.suptitle(title)
  gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1,4], height_ratios=[1,4]);

#ax1 = plt.subplot(gs[0])
#ax2 = plt.subplot(gs[1])
#ax3 = plt.subplot(gs[2])
#ax4 = plt.subplot(gs[3])
  
  #plot = fig.add_subplot(2, 2, 4)
  
  plot = fig.add_subplot(gs[3])
  
  extents = [None, None, None, None]
  def none_extrema(ef, *args):
    if all([x is None for x in args]):
      return None
    else:
      return ef(filter(lambda x: not x is None, args))
  
  xs = []; ys = [];
  for s in samples:
    for frame in s["frames"]:
      for f in frame["features"]["features"]:
        extents = [none_extrema(min, f["x"], extents[0]), none_extrema(min, f["y"], extents[1]), \
                   none_extrema(max, f["x"], extents[2]), none_extrema(max, f["y"], extents[3])]
        plot.plot(f["x"], f["y"], ',b'); xs.append(f["x"]);	ys.append(f["y"]);
  if all([not x is None for x in extents]):
    plot.plot([extents[0], extents[2], extents[2], extents[0], extents[0]], \
              [extents[1], extents[1], extents[3], extents[3], extents[1]], '-k')
    print("X: min : %f, max : %f" % (extents[0], extents[2]))
    print("Y: min : %f, max : %f" % (extents[1], extents[3]))
    
  plot.invert_yaxis();
  #plot = fig.add_subplot(2, 2, 2);
  plot = fig.add_subplot(gs[1]);
  hist(xs, 30);
  tick_params(axis='y', labelleft='off')
  tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
  #plot = fig.add_subplot(2, 2, 3);
  plot = fig.add_subplot(gs[2]);
  hist(ys, 30, orientation='horizontal');
  tick_params(axis='y', which='both', left='off', right='off', labelleft='off');
  tick_params(axis='x', labelbottom='off');
  plot.invert_yaxis();
  plot.invert_xaxis();
  fig.show();

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

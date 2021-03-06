from pylab import *
import pdb

def plotMarker(plt, sz = 0.6):
	rect1 = matplotlib.patches.Rectangle((-sz/2.0,-sz/6.0), sz/3.0, sz/3.0, color='orange')
	rect2 = matplotlib.patches.Rectangle((-sz/6.0,-sz/6.0), sz/3.0, sz/3.0, color='blue')
	rect3 = matplotlib.patches.Rectangle((sz/6.0,-sz/6.0), sz/3.0, sz/3.0, color='orange')
	plt.add_patch(rect1)
	plt.add_patch(rect2)
	plt.add_patch(rect3)

def plotFlownPaths(data, doShow=True):

  def plotPathArrows(plt, arrowSize=0.05):
    plt.hold(True)
    for sample in data:
      points = [frame["position"][0:2] for frame in sample["frames"]]
      plt.plot(list(map(lambda x: x[1], points)), list(map(lambda x: x[0], points)), ',k')
      for i in range(len(points) - 1):
        plt.arrow(points[i][1], points[i][0], points[i + 1][1] - points[i][1], points[i + 1][0] - points[i][0], fc="k", ec="k", head_width=arrowSize, head_length=arrowSize)

  result = []

  # Show all data
  fig = figure(facecolor='white', edgecolor='white')
  plt = fig.add_subplot(1, 1, 1)
  
  plotPathArrows(plt, 0.1)
  
  lims = plt.get_xaxis().get_view_interval()
  plt.set_xlim([-max(abs(lims[0]), abs(lims[1])), max(abs(lims[0]), abs(lims[1]))])
  plotMarker(plt);
  xlabel('X [m]', fontsize=16);
  ylabel('Y [m]', fontsize=16);
  fig.show()
  result.append(fig)

  # Show interest region only
  fig = figure(facecolor='white', edgecolor='white')
  plt = fig.add_subplot(1, 1, 1)
  plotPathArrows(plt, 0.1)
  plt.set_xlim(-6, 6)
  plt.set_ylim(-11, 1)
  
  # add target to figure:
  plotMarker(plt);
  xlabel('X [m]', fontsize=16);
  ylabel('Y [m]', fontsize=16);

  fig.show()
  result.append(fig)

  if type(doShow) is str:
    for n, f in zip(("-far.png", "-near.png"), result):
      f.savefig(doShow + n, dpi=180)
  elif doShow:
    show()
    
  return result


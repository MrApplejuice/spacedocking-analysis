import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import numpy as np

def plotDownloads():
	
	# weekly downloads:
	downloads = np.asarray([662, 2710, 742, 504, 592, 495, 616, 423, 328, 281, 292, 306, 289, 298, 306, 246, 291, 302, 248]);
	weeks_downloads = np.cumsum(np.asarray([1]*len(downloads))) - 1;
	
	# uploads database:
	weeks_uploads = np.asarray([0,3,7,13,17,18]);
	uploads = np.asarray([100,200,200,60,100,78]);
	
	pl.figure();
	pl.plot(weeks_downloads, np.cumsum(downloads), color=(0,0,1));
	pl.hold(True);
	pl.plot(weeks_uploads, np.cumsum(uploads), color=(0,1,0));
	pl.title('Uploads and downloads')
	pl.show();
	
	
	
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import numpy as np
import pdb

def plotDownloads():
	
	# weekly downloads:
	downloads = np.asarray([662, 2710, 742, 504, 592, 495, 616, 423, 328, 281, 292, 306, 289, 298, 306, 246, 291, 302, 248, 248, 401, 421]);
	weeks_downloads = np.cumsum(np.asarray([1]*len(downloads))) - 1;
	
	# uploads database:
	weeks_uploads = np.asarray([0,3,7,13,17,18,21]);
	uploads = np.asarray([100,200,200,60,100,78,28]); # 766
	
	pl.figure(facecolor='white', edgecolor='white');
	pl.plot(weeks_downloads, np.cumsum(downloads), 's-', color=(0,0,1));
	pl.hold(True);
	pl.plot(weeks_uploads, np.cumsum(uploads), 's-', color=(0,1,0));
	pl.yticks(np.arange(0, 12000, 1000));
	pl.grid(axis='y');
	pl.legend(('App downloads', 'Uploaded samples'), shadow=True,fancybox=True,loc='upper left');
	pl.xlabel('Weeks after release')
	pl.ylabel('Number of downloads / uploads')
	pl.show();
	
	
	
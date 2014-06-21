# you need numpy, scipy, opencv, pygmo (boost 1.48.0, + boost system, serialization, thread)

from py2box import *

def analyze_data():
	# Figure 2:
	plotDatabaseStatistics(test_dir="./", data_name="output_aug_2013.txt", selectSubset=True, filterData = False, analyze_TTC=False, storeKohonenHistograms=False);
	# Figure 3:
	# distanceVariationAnalysis(data_file = "output_aug_2013.txt", ARDrone1 = False);

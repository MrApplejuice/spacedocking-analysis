# you need numpy, scipy, opencv, pygmo (boost 1.48.0, + boost system, serialization, thread)

from py2box import *

def analyze_data():
	# Figure 2:
	# plotDatabaseStatistics(test_dir="./", data_name="../data/output_dec_2013.txt", selectSubset=False, filterData = False, analyze_TTC=False, storeKohonenHistograms=False);
	# Figure 3:
	# distanceVariationAnalysis(data_file = "../data/output_dec_2013.txt", ARDrone1 = False);
	checkHypothesisDecreasingFeatures(parent_dir_name = '../data_GDC/drone2_seqs_constvel/', DISTANCE_PLOT = True, article_plot=True);

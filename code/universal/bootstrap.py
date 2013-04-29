import numpy.random as npr
import numpy as np
from matplotlib import pyplot as pl

# check whether two data vectors come from the same distribution:
def bootstrap_compare(data1, data2, n_runs = 10000):

	# difference between 1 and 2:
	diff_1_2 = np.mean(data1) - np.mean(data2);
	print 'The difference between the means is %f' % diff_1_2
	sz1 = len(data1);
	data = np.array(list(data1) + list(data2));
	diffs = np.array([0.0] * n_runs);
	for r in range(n_runs):
		# shuffle the data to assume that they are from the same distribution:
		npr.shuffle(data);
		# split in two and determine the difference:
		d1 = data[:sz1];
		d2 = data[sz1:];
		diffs[r] = np.mean(d1) - np.mean(d2);
	
	# sort the array of differences and find the index of the first value bigger than diff_1_2:
	sorted_diffs = np.sort(diffs);
	index = n_runs;
	for r in range(n_runs):
		if(sorted_diffs[r] > diff_1_2):
			print 'found!'
			index = r;
			break;
	alpha = float(index) / n_runs;
	print 'The difference between the mean of data1 and data2 has an alpha of %f' % alpha
	
	pl.figure();
	pl.hist(sorted_diffs, 30, facecolor=(0.7, 0.7,0.7));
	pl.text(diff_1_2, 100, '*', color=(0,0,0), weight='bold')
	
	
	
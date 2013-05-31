import numpy as np
import pdb
import scipy.io

def runKohonenOnFile(filename='FTS_descr.txt',  k=10, alpha=0.1, min_alpha = 0.0001, decay = 0.9999, n_iterations=2):
	X = np.loadtxt(filename);
	Kohonen = KohonenClustering(X,  k, alpha, min_alpha, decay, n_iterations);
	# save the Kohonen clusters:
	np.savetxt('Kohonen.txt', Kohonen);
	scipy.io.savemat('Kohonen.mat', mdict={'Kohonen': Kohonen});

def KohonenClustering(X, k=10, alpha=0.1, min_alpha = 0.0001, decay = 0.9999, n_iterations=2):
	""" Cluster the data matrix X with k clusters
			X: n x dims data matrix (list of lists)
			k: number of clusters
			alpha: amount that the node is moved toward the new samples
			min_alpha: don't decreas alpha below this rate
			decay: how much alpha is decreased after each sample
	"""
	n = len(X);

	# initialize Kohonen nodes from data points:
	Kohonen = [];
	for i in range(k):
		Kohonen.append(np.array(X[i]));

	# iterate over the data, refining the Kohonen clusters:
	distances = np.array([0.0] * k);
	for it in range(n_iterations):
		print 'Iteration %d / %d' % (it, n_iterations);
		for m in range(n):
			
			if(np.mod(m, 1000) == 0):
				print 'sample %d / %d' % (m, n);

			# get sample:
			sample = np.array(X[m]);
			# find closest cluster:
			for i in range(k):
				distances[i] = np.linalg.norm(Kohonen[i] - sample);
			min_ind = np.argmin(distances);
			# adapt closest cluster:
			Kohonen[min_ind] = (1-alpha) * Kohonen[min_ind] + alpha * sample;
			# adjust alpha to gradually learn less and less
			if(alpha >= min_alpha):
				# decay update rate:
				alpha *= decay;
			
	return Kohonen;
						
			
			

	
	

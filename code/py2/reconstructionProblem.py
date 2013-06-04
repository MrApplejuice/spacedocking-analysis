from PyGMO import *
from guidobox import *
from VisualOdometry import *

class reconstructionProblem extends problem

	def obj_fun_impl(genome):

		# the image points are necessary for the fitness evaluation:
		params = np.loadtxt('params.dat');
		n_cameras = params[0];
		n_world_points = params[1];
		IPs1 = np.loadtxt('IPs1.dat');
		IPs2 = np.loadtxt('IPs2.dat');
		IPs = IPs1.append(IPs2[-1]);
		K = np.loadtxt('K.dat');
		
		# get the rotations, translations, and world points from the genome
		(Rs, Ts, X) = transformGenomeToMatrices(genome, n_cameras, n_world_points);

		# calculate the reprojection error:
		err = calculateReprojectionError(Rs, Ts, X, IPs, n_cameras, n_world_points, K);

		return err;
			

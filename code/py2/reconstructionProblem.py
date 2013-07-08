from PyGMO import *
from PyGMO.problem import base
from guidobox import *
import VisualOdometry as VO

class reconstructionProblem(base):

	n_cameras = 2;
	n_world_points = 100;
	t_scale = 3.0;
	world_scale = 10.0;
	K = [];
	IPs = [];
	genome_size = 0;

	def __init__(self, n_views=2, n_points=100, IPs=[], t_scale = 3.0, world_scale=10.0, K = []):
		# per view, there are 6 parameters: phi, theta, psi, t1, t2, t3
		# per point, there are 3 parameters: X, Y, Z
		self.genome_size = (n_views-1) * 6 + n_points * 3;
		self.n_cameras = n_views;
		self.n_world_points = n_points;
		self.IPs = IPs; # IPs contains per camera the image points
		self.K = K;
		self.t_scale = t_scale;
		self.world_scale = world_scale;
		super(reconstructionProblem,self).__init__(self.genome_size);
		
	def getGenomeSize(self):
		return self.genome_size;

	def setK(self,K):
		self.K = K;

	def _objfun_impl(self,genome):

		# get the rotations, translations, and world points from the genome
		(Rs, Ts, X) = self.transformGenomeToMatrices(genome, self.n_cameras, self.n_world_points);

		# calculate the reprojection error:
		# What if points are behind the cameras?
		# Should be adapted to except 'missing' points
		(err, errors_per_point) = VO.calculateReprojectionError(Rs, Ts, X, self.IPs, self.n_cameras, self.n_world_points, self.K);

		# return the fitness:
		return [float(err)];
			
	def transformGenomeToMatrices(self, genome, n_cameras, n_points):
		""" Decode the genome to the rotation, translation and world point information.
		"""		

		# decode cameras:
		Rs = [];
		Ts = [];
		# first camera:
		R = VO.getRotationMatrix(0.0, 0.0, 0.0);
		t = np.zeros([3,1]);
		Rs.append(R);
		Ts.append(t);

		for c in range(n_cameras-1):
			# decode rotation:
			phi = genome[c*6] * np.pi;
			theta = genome[c*6+1] * np.pi;
			psi = genome[c*6+2] * np.pi;
			R = VO.getRotationMatrix(phi, theta, psi);
			Rs.append(R);
			# decode translation:
			t = np.zeros([3,1]);
			t[0] = genome[c*6+3] * self.t_scale;
			t[1] = genome[c*6+4] * self.t_scale;
			t[2] = genome[c*6+5] * self.t_scale;
			Ts.append(t);
		
		# decode world points:
		offset = (n_cameras-1)*6;
		X = np.zeros([n_points, 3]);
		for p in range(n_points):
			X[p,0] = genome[offset+p*3] * self.world_scale;
			X[p,1] = genome[offset+p*3+1] * self.world_scale;
			X[p,2] = genome[offset+p*3+2] * self.world_scale;
		
		return (Rs, Ts, X);

		

	

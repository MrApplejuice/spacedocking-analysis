from PyGMO import *
from PyGMO.problem import base
from guidobox import *
import VisualOdometry as VO

class multiReconstructionProblem(base):

	snapshot_time_interval = 0.25;
	n_cameras = 5;
	n_world_points = 100;
	t_scale = 3.0;
	world_scale = 10.0;
	K = [];
	IPs = [];
	genome_size = 0;

	def __init__(self, n_views=5, n_points=100, IPs=[], t_scale = 3.0, world_scale=10.0, K = [], snapshot_time_interv=0.25):
		# per view, there are 6 parameters: roll, yaw, pitch, tx, ty, tz
		# per point, there are 3 parameters: X, Y, Z
		self.genome_size = n_views * 6 + n_points * 3;
		self.n_cameras = n_views;
		self.n_world_points = n_points;
		self.sti = snapshot_time_interv;
		self.IPs = IPs; # IPs contains per camera the image points
		self.K = K;
		self.t_scale = t_scale;
		self.world_scale = world_scale;
		super(multiReconstructionProblem,self).__init__(self.genome_size);
		
	def getGenomeSize(self):
		return self.genome_size;

	def setK(self,K):
		self.K = K;

	def _objfun_impl(self,genome):

		# get the rotations, translations, and world points from the genome
		(Rs, Ts, X) = self.transformMultiGenomeToMatrices(genome, self.n_cameras, self.n_world_points);

		# calculate the reprojection error:
		# What if points are behind the cameras?
		# Should be adapted to except 'missing' points
		(err, errors_per_point) = VO.calculateReprojectionError(Rs, Ts, X, self.IPs, self.n_cameras, self.n_world_points, self.K);


		pdb.set_trace(); # should err not be a normal float?
		# return the fitness:
		return [float(err)];
			
	def transformMultiGenomeToMatrices(self, genome, n_cameras, n_points):
		""" Decode the genome to the rotation, translation and world point information.
		"""		

		angle_normalizer = 180.0;
		roll = [];
		yaw = [];
		pitch = [];
		vx = [];
		vy = [];
		vz = [];

		for c in range(n_cameras):
			# decode rotation:
			roll.append(genome[c*6] * angle_normalizer);
			yaw.append(genome[c*6+1] * angle_normalizer);
			pitch.append(genome[c*6+2] * angle_normalizer);
			# decode translation:
			vx.append((genome[c*6+3] * self.t_scale * 1000.0) / self.sti);
			vy.append((genome[c*6+4] * self.t_scale * 1000.0) / self.sti);
			vz.append((genome[c*6+5] * self.t_scale * 1000.0) / self.sti);
		
		(Rot, Transl, Rotations, Translations) = VO.convertFromDroneToCamera(roll, yaw, pitch, vx, vy, vz, self.sti);
		
		# decode world points:
		offset = (n_cameras)*6;
		X = np.zeros([n_points, 3]);
		for p in range(n_points):
			X[p,0] = genome[offset+p*3] * self.world_scale;
			X[p,1] = genome[offset+p*3+1] * self.world_scale;
			X[p,2] = genome[offset+p*3+2] * self.world_scale;
		
		return (Rotations, Translations, X);

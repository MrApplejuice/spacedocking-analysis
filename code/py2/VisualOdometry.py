import pdb
import cv2
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
from evolveMultiReconstruction import *
from evolveReconstruction import *
# from guidobox import *

def performStructureFromMotion(image_points1, image_points2, K, W, H):
	""" Performs structure from motion on the basis of image matches (image_points1, image_points2, both Nx2),
			a calibration matrix K, and a given width and height of an image. 
			
			It returns (R, t, X), i.e., a rotation matrix R, translation t, and world points X.
	"""

	n_points = len(image_points1);

	# location camera 1:
	l1 = np.zeros([3,1]);
	R1 = np.eye(3);
	rvec1 = cv2.Rodrigues(R1);
	rvec1 = rvec1[0];

	# determine the rotation and translation between the two views:
	(R21, R22, t21, t22) = determineTransformation(image_points1, image_points2, K, W, H);

	# 3D-reconstruction:
	ip1 = cv2.convertPointsToHomogeneous(image_points1.astype(np.float32));
	ip2 = cv2.convertPointsToHomogeneous(image_points2.astype(np.float32));

	# P1 is at the origin
	P1 = np.zeros([3, 4]);
	P1[:3,:3] = np.eye(3);
	P1 = np.dot(K, P1);

	# P2 is rotated and translated with respect to camera 1
	# determine the right R, t:
	# reproject a point to the 3D-world
	# exclude the camera if the point falls behind it. 	
	# first determine all 4 projection matrices:
	R2s = [];
	R2s.append(R21);
	R2s.append(R22);
	t2s = [];
	t2s.append(t21);
	t2s.append(t22);

	# reproject the points into the 3D-world:
	index_r = 0; index_t = 0;
	for ir in range(2):

		# clean up the rotation matrix, i.e., don't allow mirroring of any axis:
		R2s[ir] = cleanUpR(R2s[ir]);

		for it in range(2):
			point_behind = infeasibleP2(ip1, ip2, R1, l1, R2s[ir], t2s[it], K);

			if(point_behind == 0):
				index_r = ir;
				index_t = it;
				print 'ir, it = %d, %d' % (ir, it)

	P2_est = getProjectionMatrix(R2s[index_r], t2s[index_t], K);
	R2_est = R2s[index_r]; t2_est = t2s[index_t];

	# triangulate the image points to obtain world coordinates:
	X_est = triangulate(ip1, ip2, P1, P2_est);

	# BUNDLE ADJUSTMENT:
	bundle_adjustment = True;
	if(bundle_adjustment):
		# evolve a solution:
		IPs = [];
		IPs.append(image_points1);
		IPs.append(image_points2);
		# seed the evolution with some pre-knowledge:
		phis = [0.0];
		thetas = [0.0];
		psis = [0.0];
		#Ts = [t];
		t2e = np.zeros([3,1]);
		for i in range(3):
			t2e[i,0] = t2_est[i];
		Ts = [t2e];
		# points;
		W = np.zeros([n_points, 3]);
		for p in range(n_points):
			for i in range(3):
				W[p, i] = X_est[p][i];

		# calculate reprojection error before further optimization:
		Rs = [R1]; Rs.append(R2_est);
		Ts = [l1]; Ts.append(t2e);
		(err, errors_per_point) = calculateReprojectionError(Rs, Ts, W, IPs, 2, n_points, K);

		# determine the genome on the above information:
		genome = constructGenome(phis, thetas, psis, Ts, n_points, W);
	
		# Get rotations, translations, X_est:
		(Rs, Ts, X_est) = evolveReconstruction('test', 2, n_points, IPs, 3.0, 10.0, K, genome);
		R2_est = Rs[1];
		t2_est = Ts[1];


	print 't_est = %f, %f, %f' % (t2_est[0], t2_est[1], t2_est[2]);
	print 'R = '
	printRotationMatrix(R2_est);	

	# now we have R2, t2, and X, which we return:
	return (R2_est, t2_est, X_est, errors_per_point);	

def getK(W=640.0, H=480.0):
	""" Constructs a standard calibration matrix given a width and height in pixels. """

	K = np.zeros([3,3]);
	K[0,0] = W;
	K[0,2] = W / 2.0;
	K[1,1] = H;
	K[1,2] = H / 2.0;
	K[2,2] = 1.0;

	return K;

def getKdrone1():
	""" Calibration matrix of drone 1 """
	#AR Drone 1: 320x240  (4:3 aspect ratio)
	K = np.zeros([3,3]);
	K[0,0] = 320.0;
	K[0,1] = 0.0;
	K[0,2] = 160.0;
	K[1,0] = 0.0;
	K[1,1] = 320.0;
	K[1,2] = 120.0;
	K[2,2] = 1.0;
	return K;
	
def getKdrone2():
	""" Calibration matrix of drone 2 from cvdrone """
	#AR Drone 2: 640x360  (16:9 aspect ratio)

	K = np.zeros([3,3]);
	K[0,0] = 5.81399719e+002;
	K[0,1] = 0.0;
	K[0,2] = 3.17410492e+002;
	K[1,0] = 0.0;
	K[1,1] = 5.78456116e+002;
	K[1,2] = 1.37808365e+002;
	K[2,2] = 1.0;
	return K;

def testCoordinateSystem(X = 0.0, Y = 0.0, Z = 5.0, tx = 0.0, ty = 0.0, tz = 0.0, Rx = 0.0, Ry = 0.0, Rz = 0.0):
	""" Tests whether the OpenCV coordinate system is left-handed with Z to the front, 
			X to the right, and Y up.
	"""
	
	print '(X,Y,Z) = (%f, %f, %f)' % (X,Y,Z);
	print '(tx,ty,tz) = (%f, %f, %f)' % (tx,ty,tz);
	print '(Rx,Ry,Rz) = (%f, %f, %f)' % (Rx,Ry,Rz);

	# get calibration matrix:
	K = getKdrone1();
	print 'K = '
	printRotationMatrix(K);

	# create world point:
	n_points = 1;
	points_world = np.zeros([n_points, 3]);
	points_world[0,0] = X;
	points_world[0,1] = Y;
	points_world[0,2] = Z;

	# create translation vector:
	transl = np.array([0.0]*3);
	transl[0] = tx;
	transl[1] = ty;
	transl[2] = tz;

	# create rotation matrix:
	Rx = np.deg2rad(Rx);
	R_Rx = np.zeros([3,3]);
	R_Rx[0,0] = 1;
	R_Rx[1,1] = np.cos(Rx);
	R_Rx[1,2] = -np.sin(Rx);
	R_Rx[2,1] = np.sin(Rx);
	R_Rx[2,2] = np.cos(Rx);

	Ry = np.deg2rad(Ry);
	R_Ry = np.zeros([3,3]);
	R_Ry[1,1] = 1;
	R_Ry[0,0] = np.cos(Ry);
	R_Ry[2,0] = -np.sin(Ry);
	R_Ry[0,2] = np.sin(Ry);
	R_Ry[2,2] = np.cos(Ry);

	Rz = np.deg2rad(Rz);
	R_Rz = np.zeros([3,3]);
	R_Rz[0,0] = np.cos(Rz);
	R_Rz[0,1] = -np.sin(Rz);
	R_Rz[1,0] = np.sin(Rz);
	R_Rz[1,1] = np.cos(Rz);
	R_Rz[2,2] = 1;

	# multiply the matrices:
	# what order?
	# first x, then y, then z:
	R2 = np.dot(R_Rz, np.dot(R_Ry, R_Rx));
	# R2 = np.dot(R_Ry, np.dot(R_Rx, R_Rz));
	print 'R = '
	printRotationMatrix(R2);

	# print the translated / rotated point:
	C = np.zeros([3,4]);
	C[:,:3] = R2;
	C[:,3] = transl;
	PW = np.ones([4,1]);
	for c in range(3):
		PW[c,0] = points_world[0,c];
	transformed_X = np.dot(C, PW);
	print 'transformed world point: (%f, %f, %f)' % (transformed_X[0], transformed_X[1], transformed_X[2]);

	# get the vector form of the rotation:
	rvec2 = cv2.Rodrigues(R2);
	rvec2 = rvec2[0];

	# distortion coefficients:
	distCoeffs = np.zeros([4]); 

	# project the world points in the cameras:
	result = cv2.projectPoints(points_world, rvec2, transl, K, distCoeffs);	
	image_points = result[0];

	print 'image point = %f, %f' % (image_points[0][0][0], image_points[0][0][1])

def testVisualOdometry(n_points=100):

	# location camera 1:
	l1 = np.zeros([3,1]);
	R1 = np.eye(3);
	rvec1 = cv2.Rodrigues(R1);
	rvec1 = rvec1[0];

	# translation vector
	t = np.zeros([3,1]);#np.random.rand(3,1);
	t[2] = 0;
	t[1] = 1;
	print 't = %f, %f, %f' % (t[0], t[1], t[2]);
	scale = np.linalg.norm(t);
	print 'scale = %f' % scale;
	l2 = l1 + t;

	# Rotation matrix:
	phi = 0.2 * np.pi;#0.001*(np.random.random(1)*2-1) * np.pi;
	theta = 0.1 * np.pi;#0.001*(np.random.random(1)*2-1) * np.pi;
	psi = 0.0;#0.001*(np.random.random(1)*2-1) * np.pi;

	R_phi = np.zeros([3,3]);
	R_phi[0,0] = 1;
	R_phi[1,1] = np.cos(phi);
	R_phi[1,2] = np.sin(phi);
	R_phi[2,1] = -np.sin(phi);
	R_phi[2,2] = np.cos(phi);

	R_theta = np.zeros([3,3]);
	R_theta[1,1] = 1;
	R_theta[0,0] = np.cos(theta);
	R_theta[2,0] = -np.sin(theta);
	R_theta[0,2] = np.sin(theta);
	R_theta[2,2] = np.cos(theta);

	R_psi = np.zeros([3,3]);
	R_psi[0,0] = np.cos(psi);
	R_psi[0,1] = np.sin(psi);
	R_psi[1,0] = -np.sin(psi);
	R_psi[1,1] = np.cos(psi);
	R_psi[2,2] = 1;

	R2 = np.dot(R_psi, np.dot(R_theta, R_phi));
	
	print 'R = '
	printRotationMatrix(R2);

	rvec2 = cv2.Rodrigues(R2);
	rvec2 = rvec2[0];

	# create X, Y, Z points:
	size = 3;
	distance = 5;
	transl = np.zeros([1,3]);
	transl[0,2] = distance; # is Z in the direction of the principal axis?
	points_world = np.zeros([n_points, 3]);
	for p in range(n_points):
		points_world[p, :] = size * (np.random.rand(1,3)*2-np.ones([1,3])) + transl;

	# camera calibration matrix:
	K = np.zeros([3,3]);
	W = 320.0;
	H = 240.0;
	K[0,0] = W;
	K[1,1] = H;
	K[2,2] = 1.0;
	
	distCoeffs = np.zeros([4]); # no clue what this means

	# project the world points in the cameras:
	result = cv2.projectPoints(points_world, rvec1, np.array([0.0]*3), K, distCoeffs);	
	image_points1 = result[0];

	result = cv2.projectPoints(points_world, rvec2, t, K, distCoeffs);
	image_points2 = result[0];

	add_noise = True;
	if(add_noise):
		for p in range(n_points):
			image_points1[p][0][0] += np.random.normal(0.0, 0.5);
			image_points1[p][0][1] += np.random.normal(0.0, 0.5);			
			image_points2[p][0][0] += np.random.normal(0.0, 0.5);
			image_points2[p][0][1] += np.random.normal(0.0, 0.5);			


	# determine the rotation and translation between the two views:
	(R21, R22, t21, t22) = determineTransformation(image_points1, image_points2, K, W, H);

	# 'wrong' solutions have a negative element on the diagonal, but calling determineTransformation repetitively does not help... 
	# ws = wrongSolution(R21, R22);
	# while(ws == 1):
			# (R21, R22, t21, t22) = determineTransformation(image_points1, image_points2, K, W, H);
			# ws = wrongSolution(R21, R22);

	print 'R21 = '
	printRotationMatrix(R21);
	print 'R22 = '
	printRotationMatrix(R22);
	#R21 = R2.T;
	#R22 = R2;
	#t21 = -t;
	#t22 = t;


	# 3D-reconstruction:
	ip1 = cv2.convertPointsToHomogeneous(image_points1.astype(np.float32));
	ip2 = cv2.convertPointsToHomogeneous(image_points2.astype(np.float32));

	# P1 is at the origin
	P1 = np.zeros([3, 4]);
	P1[:3,:3] = np.eye(3);
	P1 = np.dot(K, P1);

	# P2 is rotated and translated with respect to camera 1
	# determine the right R, t:
	# iterate over all points, reproject them to the 3D-world
	# exclude one of the options as soon as 	
	# first determine all 4 projection matrices:
	R2s = [];
	R2s.append(R21);
	R2s.append(R22);
	t2s = [];
	t2s.append(t21);
	t2s.append(t22);

	# reproject the points into the 3D-world:
	index_r = 0; index_t = 0;
	for ir in range(2):
		# clean up the rotation matrix, i.e., don't allow mirroring of any axis:
		R2s[ir] = cleanUpR(R2s[ir]);
		for it in range(2):
			point_behind = infeasibleP2(ip1, ip2, R1, l1, R2s[ir], t2s[it], K);

			if(point_behind == 0):
				index_r = ir;
				index_t = it;
				print 'ir, it = %d, %d' % (ir, it)

	P2_est = getProjectionMatrix(R2s[index_r], t2s[index_t], K);
	R2_est = R2s[index_r]; t2_est = t2s[index_t];

	# triangulate the image points to obtain world coordinates:
	X_est = triangulate(ip1, ip2, P1, P2_est);

	# We could determine the reprojection error already here. It is close to 0 for a good estimate
	# and in the 10,000s for a bad estimate.

	# BUNDLE ADJUSTMENT:
	bundle_adjustment = True;

	if(bundle_adjustment):
		# evolve a solution:
		IPs = [];
		IPs.append(image_points1);
		IPs.append(image_points2);
		# seed the evolution with some pre-knowledge:
		phis = [phi];
		thetas = [theta];
		psis = [psi];
		#Ts = [t];
		t2e = np.zeros([3,1]);
		for i in range(3):
			t2e[i,0] = t2_est[i];
		Ts = [t2e];
		# points;
		W = np.zeros([n_points, 3]);
		for p in range(n_points):
			for i in range(3):
				W[p, i] = X_est[p][i];

		genome = constructGenome(phis, thetas, psis, Ts, n_points, W);
	
		# Get rotations, translations, X_est
		(Rs, Ts, X_est) = evolveReconstruction('test', 2, n_points, IPs, 3.0, 10.0, K, genome);
		R2_est = Rs[1];
		t2_est = Ts[1];

	scales = np.array([0.0] * n_points);
	for i in range(n_points):
		scales[i] = X_est[i][0] / points_world[i][0];

	print 't_est = %f, %f, %f' % (t2_est[0], t2_est[1], t2_est[2]);
	print 'R = '
	printRotationMatrix(R2_est);
	sc = np.mean(scales);
	print 'Scale = %f, Mean scale = %f' % (scale, 1.0/sc);

	# scale:
	# X_est = X_est * (1.0/sc);	

	# show visually:

	# calculate reprojection error:
	Rs = [R1]; Rs.append(R2_est);
	Ts = [l1]; Ts.append(t2_est);
	(err, errors_per_point) = calculateReprojectionError(Rs, Ts, W, IPs, 2, n_points, K);

	fig = pl.figure()
	ax = fig.gca(projection='3d')
	M_world = np.matrix(points_world);
	M_est = np.matrix(X_est);
	M_est = M_est[:,:3];

	x = np.array(M_est[:,0]); y = np.array(M_est[:,1]); z = np.array(M_est[:,2]);
	x = flatten(x); y = flatten(y); z = flatten(z);
	#ax.scatter(x, y, z, '*', color=(1.0,0,0));
	cm = pl.cm.get_cmap('hot')
	ax.scatter(x, y, z, '*', c=errors_per_point, cmap=cm);
	fig.hold = True;

	#x = (1.0/sc)*np.array(M_est[:,0]); y = (1.0/sc)*np.array(M_est[:,1]); z = (1.0/sc)*np.array(M_est[:,2]);
	#x = flatten(x); y = flatten(y); z = flatten(z);
	#ax.scatter(x, y, z, 's', color=(0,0,1.0));

	x = np.array(M_world[:,0]); y = np.array(M_world[:,1]); z = np.array(M_world[:,2]);
	x = flatten(x); y = flatten(y); z = flatten(z);
	ax.scatter(x, y, z, 'o', color=(0.0,1.0,0.0));

	ax.axis('tight');
	pl.show();

	# if(1.0/sc > 0.99 * scale and 1.0/sc < 1.01 * scale):
	#	pdb.set_trace()

	# now we have R2, t2, and X, which we return:
	return (R2_est, t2_est, X_est);	

def selectCorrectRotationTranslation(image_points1, image_points2, K, R21, R22, t21, t22, R1 = [], l1 = []):
	""" The 8-point algorithm gives back four options for the rotation and translation matrix.
		The matrices that result in all points lying in front of the cameras should be selected.
	"""

	if(len(R1) == 0 or len(l1) == 0):
		# location camera 1:
		l1 = np.zeros([3,1]);
		R1 = np.eye(3);
	
	# 3D-reconstruction:
	ip1 = cv2.convertPointsToHomogeneous(image_points1.astype(np.float32));
	ip2 = cv2.convertPointsToHomogeneous(image_points2.astype(np.float32));

	# P1 is at the origin
	P1 = np.zeros([3, 4]);
	P1[:3,:3] = np.eye(3);
	P1 = np.dot(K, P1);

	# P2 is rotated and translated with respect to camera 1
	# determine the right R, t:
	# iterate over all points, reproject them to the 3D-world
	# exclude one of the options as soon as 	
	# first determine all 4 projection matrices:
	R2s = [];
	R2s.append(R21);
	R2s.append(R22);
	t2s = [];
	t2s.append(t21);
	t2s.append(t22);

	# reproject the points into the 3D-world:
	index_r = 0; index_t = 0;
	for ir in range(2):
		# clean up the rotation matrix, i.e., don't allow mirroring of any axis:
		R2s[ir] = cleanUpR(R2s[ir]);
		for it in range(2):
			point_behind = infeasibleP2(ip1, ip2, R1, l1, R2s[ir], t2s[it], K);

			if(point_behind == 0):
				index_r = ir;
				index_t = it;
				print 'ir, it = %d, %d' % (ir, it)

	P2_est = getProjectionMatrix(R2s[index_r], t2s[index_t], K);
	R2_est = R2s[index_r]; t2_est = t2s[index_t];
	
	return (P2_est, R2_est, t2_est);


def test3DReconstructionParrot(n_frames=5, n_points=30, bvx=0.0, bvy =0.0, b_roll=0.0, b_pitch =0.0, b_yaw = 0.0):
	""" Method to test the 3D-reconstruction for AstroDrone. It creates a drone movement
			consisting of n_frames camera poses, and creates n_wp world points. It projects the points 
			into the images, introducing some noise and missing observations.
			
			Subsequently, it applies the algorithm for reconstruction and checks how well it
			works. 
	"""

	# Algorithm outline:
	#
	# GROUND-TRUTH:
	# 1) create movement and poses in drone coordinates
	# 2) construct the world
	# 3) translate the drone coordinates to camera coordinates 
	# 4) project the world points into the images
	#
	# MEASUREMENT / ESTIMATION:
	# 5) induce some noise in movement perception / world point perception
	# 6) reconstruct with the algorithm

	graphics = True;

	# *************
	# GROUND-TRUTH:
	# *************

	# time in between snap shots:
	snapshot_time_interval = 0.25;

	# camera calibration matrix:
	K = getKdrone1();
	# distortion coefficients:
	distCoeffs = np.zeros([1,4]); 

	# 1) create movement and poses in drone coordinates
	# We make this similar to the way in which data arrives from the AstroDrone database:

	# The drone uses a right-handed coordinate system with x to the right and y to the front.
	# roll, yaw, pitch is all 0.0 when level and straight ahead, set them in degrees:
	roll = np.array([0.0] * n_frames);
	yaw = np.array([0.0] * n_frames);
	pitch = np.array([0.0] * n_frames);
	for fr in range(n_frames):
		roll[fr] = 0.0 + b_roll * fr;
		yaw[fr] = 0.0 + b_yaw * fr;
		pitch[fr] = 0.0 + b_pitch * fr;

	# basic movement:
	vx = np.array([0.0] * n_frames);
	vy = np.array([0.0] * n_frames);
	vz = np.array([0.0] * n_frames);
	if(bvx == 0.0 and bvy == 0.0):
		# little sideward motion, considerable forward motion, in m/s:
		bvx = 0.1 * (np.random.rand(1) * 2.0 - 1.0);
		bvy = 0.25 + np.random.rand(1) * 0.5;
	# add variations and multiply with 1000.0 (mm / s):
	for fr in range(n_frames):
		vx[fr] = bvx * 1000.0;
		vy[fr] = bvy * 1000.0;
			

	# 2) construct the world
	size = 2;
	distance = 5;
	transl = np.zeros([1,3]);
	transl[0,2] = distance; # is Z in the direction of the principal axis?
	points_world = np.zeros([n_points, 3]);
	for p in range(n_points):
		points_world[p, :] = size * (np.random.rand(1,3)*2-np.ones([1,3])) + transl;

	# 3) translate the drone coordinates to camera coordinates
	(Rotations, Translations, Rot, Transl) = convertFromDroneToCamera(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval);

	# 4) project the world points into the images
	IPs = [];
	R = np.eye(3);
	transl = np.zeros([3,1]);
	
	if(graphics):
		# make a figure to plot positions and camera axes:
		pl.figure();
		pl.hold(True);
		camera_axis = np.array([0,0,0.1]);
		
	for fr in range(n_frames):

		# rotations and translations should still be made incremental:		
		
		# some nasty questions:
		# a) first translate and then rotate?
		# b) do vx, vy, vz give velocities in body coordinates? Or in the inertial reference frame?
		
		# get the translation:
		transl = np.dot(np.linalg.inv(R), Translations[fr]) + transl;
		
		# rotation:		
		R = np.dot(Rotations[fr], R);
		# get the vector form of the rotation:
		rvec = cv2.Rodrigues(R);
		rvec = rvec[0];

		if(graphics):
			# plot camera position and camera axis in the X,Z-plane:
			# since translations / rotations are of world points, we have to negate them /
			# invert them to get the camera translations / rotations:
			pl.plot(-transl[0], -transl[2], 'x');
			cR = np.dot(np.linalg.inv(R), camera_axis)
			pl.plot([-transl[0], -transl[0]+cR[0]], [-transl[2], -transl[2]+cR[2]]);
			

		# project the world points in the cameras:
		result = cv2.projectPoints(points_world, rvec, transl, K, distCoeffs);
		image_points = result[0];

		IPs.append(image_points);

	if(graphics):
		# enforce equal axes
		pl.axis('equal');
		pl.title('Positions and orientations');
		pl.show();
		
		showPointsOverTime(IPs);

	# *************************
	# MEASUREMENT / ESTIMATION:
	# *************************

	# 5) induce some noise in movement perception / world point perception

	# noise on image points:
	stv = 0.3;
	p_missing = 0.15; 
	for fr in range(n_frames):
		for p in range(n_points):
			if(np.random.rand() < p_missing):
				IPs[fr][p][0][0] = -1;
				IPs[fr][p][0][1] = -1;
			else:
				noise = stv * np.random.randn();
				IPs[fr][p][0][0] += noise;
				noise = stv * np.random.randn();
				IPs[fr][p][0][1] += noise;
	
	# no noise on movements and rotations yet
	# so roll, yaw, pitch, vx, vy, vz
	stv_angle = 0.5;
	stv_vel = 25.0;

	for fr in range(n_frames):
		noise = stv_angle * np.random.randn();
		roll[fr] += noise;
		noise = stv_angle * np.random.randn();
		yaw[fr] += noise;
		noise = stv_angle * np.random.randn();
		pitch[fr] += noise;
		noise = stv_vel * np.random.randn();
		vx[fr] += noise;
		noise = stv_vel * np.random.randn();
		vy[fr] += noise;
		noise = stv_vel * np.random.randn();
		vz[fr] += noise;
	
	# How close are the estimates from the 8-point algorithm to the onboard estimates?
	# If close enough, we may utilize them for better initialization.
	# a) get rotations and translations per pair (in camera reference frame)
	image_points1 = [];
	image_points2 = [];
	Rs_VO = [];
	Ts_VO = [];
	for fr1 in range(n_frames-1):
		fr2 = fr1 + 1;
		for p in range(n_points):
			if(observed(IPs[fr1][p]) and observed(IPs[fr2][p])):
				image_points1.append(IPs[fr1][p]);
				image_points2.append(IPs[fr2][p]);
		pdb.set_trace();
		(R21, R22, t21, t22) = determineTransformation(np.array(image_points1), np.array(image_points2), K);
		(P2_est, R2_est, t2_est) = selectCorrectRotationTranslation(np.array(image_points1), np.array(image_points2), K, R21, R22, t21, t22);
		Rs_VO.append(R2_est);
		Ts_VO.append(t2_est);
	# b) compare with the drone info (in drone coordinates):
	
	
	
	# 6) reconstruct with the algorithm
	
	# First get an estimate for the world coordinates:
	(MRot, MTransl, MRotations, MTranslations) = convertFromDroneToCamera(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval);
	X = estimateWorldPoints(MRotations, MTranslations, IPs, K);
	
	# Adapt the evolutionary algorithm, so that it can take missing image points.
	# Also, the genome constructor should directly take:
	# roll, yaw, pitch, vx, vy, vz, snapshot_time_interval
	genome = constructMultiGenome(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval, X);
	(Rs, Ts, X_est) = evolveMultiReconstruction('test', n_frames, n_points, IPs, 3.0, 10.0, K, genome);
	
	# what is the total error / error per point
	(err, errors_per_point) = calculateReprojectionError(Rs, Ts, X, IPs, n_frames, n_points, K);
	
	# and the errors in the world coordinates (distance estimate to ground truth):
	M_world = np.matrix(points_world);
	M_est = np.matrix(X_est);
	M_est = M_est[:,:3];
	world_distances = np.array([0.0] * n_points);
	for p in range(n_points):
		world_distances[p] = np.linalg.norm(M_world[p] - M_est[p]);
	
	# show relation between image error and world error:
	pl.figure();
	pl.plot(errors_per_point, world_distances, 'x');
	pl.title('Relation image error and world error');
	pl.show();
	
	# show resulting reconstruction:
	fig = pl.figure()
	ax = fig.gca(projection='3d')

	x = np.array(M_est[:,0]); y = np.array(M_est[:,1]); z = np.array(M_est[:,2]);
	x = flatten(x); y = flatten(y); z = flatten(z);
	#ax.scatter(x, y, z, '*', color=(1.0,0,0));
	cm = pl.cm.get_cmap('jet')
	ax.scatter(x, y, z, '*', c=errors_per_point, cmap=cm);
	fig.hold = True;

	#x = (1.0/sc)*np.array(M_est[:,0]); y = (1.0/sc)*np.array(M_est[:,1]); z = (1.0/sc)*np.array(M_est[:,2]);
	#x = flatten(x); y = flatten(y); z = flatten(z);
	#ax.scatter(x, y, z, 's', color=(0,0,1.0));

	x = np.array(M_world[:,0]); y = np.array(M_world[:,1]); z = np.array(M_world[:,2]);
	x = flatten(x); y = flatten(y); z = flatten(z);
	ax.scatter(x, y, z, 'o', color=(0.0,1.0,0.0));

	pl.xlabel('X'); pl.ylabel('Y'); 
	pl.title('Real vs. estimated world points (camera reference frame).');
	# pl.show();
	
	pl.figure();
	pl.hist(errors_per_point);
	pl.title('Image errors histogram');
	pl.show();
	
	pl.figure();
	pl.hist(world_distances);
	pl.title('World distances histogram');
	pl.show();
	
def estimateWorldPoints(Rotations, Translations, IPs, K):
	""" Takes the rotations and translations in a single reference frame and 
		corresponding image matches. It then does a triangulation for each image pair,
		and averages the resulting world coordinate estimates.
	"""
	
	# number of frames and points
	n_frames = len(IPs);
	# even unobserved points should take up place in each IPs[frame]:
	n_points = len(IPs[0]);
	
	# WP will contain per point a list of coordinate estimates:
	WP = [];
	for p in range(n_points):
		WP.append([]);
	
	for i in range(n_frames):
	
		# rotation and translation of the first frame
		R1 = Rotations[i];
		t1 = Translations[i];
				
		for j in range(i+1, n_frames):
		
			# rotation and translation of the second frame
			R2 = Rotations[j];
			t2 = Translations[j];
			
			# get common points:
			(image_points1, image_points2, indices) = getCommonPoints(IPs, i, j, n_points);
			
			# triangulate the points
			W = getTriangulatedPoints(image_points1, image_points2, R2, t2, K, R1, t1);
			
			# Add coordinate to world point list:
			for ind in range(len(indices)):
				WP[indices[ind]].append(W[ind]);
				
	
	# average over the coordinates to get an estimate:
	standard_coordinate = np.zeros([1,3]);
	standard_coordinate[0,2] = 3;
	X = np.zeros([n_points, 3]);
	for p in range(n_points):
		
		n_coords = len(WP[p]);
		
		if(n_coords == 0):
			# if no estimate: put a standard estimate in (this actually means we should not even estimate it)
			X[p,:] = standard_coordinate;
		else:
			# else average over the estimates:
			C = np.zeros([1, 3]);
			for c in range(n_coords):
				C += WP[p][c][:3];
			C /= n_coords;
			X[p, :] = C;
	
	return X;

def getCommonPoints(IPs, frame1, frame2, n_points):
	""" Get the points visible in both images.
	"""
	
	image_points1 = [];
	image_points2 = [];
	indices = [];
	for p in range(n_points):
		
		if(observed(IPs[frame1][p]) and observed(IPs[frame2][p])):
			indices.append(p);
			image_points1.append([IPs[frame1][p][0][0], IPs[frame1][p][0][1]]);
			image_points2.append([IPs[frame2][p][0][0], IPs[frame2][p][0][1]]);
	
	image_points1 = np.array(image_points1);				
	image_points2 = np.array(image_points2);
	
	return (image_points1, image_points2, indices);

def showPointsOverTime(IPs):
	""" Shows image points over time. One world point is tracked and shown as a line.
	"""
		
	# create the figure:
	pl.figure();
	pl.hold(True)
	n_frames = len(IPs);
	n_points = len(IPs[0]);
	for p in range(n_points):
		x = np.array([0.0] * n_frames);
		y = np.array([0.0] * n_frames);
		for fr in range(n_frames):
			x[fr] = IPs[fr][p][0][0];
			y[fr] = IPs[fr][p][0][1];
		pl.plot(x, y, 'x-');
		pl.plot(x[-1], y[-1], 'or');
	
	pl.show();

def limit_angle(angle):
	""" Makes sure that the angle (in rad) is in the interval [-pi, pi].
	"""
	# angle will be in [-pi, pi]
	while(angle >= np.pi):
		angle -= 2*np.pi;

	while(angle < -np.pi):
		angle += 2*np.pi;

	return angle;

def convertFromDroneToCamera(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval, graphics=False):
	""" This method takes the Euler angles and velocities from the AR drone
			and translates them to "camera" rotations and translations (actually
			translations and rotations of the world points).
			
		It returns (Rot, Transl, Rotations, Translations)
		- Rot, Transl: rotations and translations between subsequent views
		- Rotations, Translations: rotations and translations in a common frame (defined by the first view)
	"""

	# We have n_frames with for each frame the velocity and Euler angles
	# This has to be translated to (n_frames-1) position / attitude changes, 
	# with the first camera being at (0,0,0) and being level.
	n_frames = len(roll);

	Rot = [];
	Transl = [];
	
	# first camera:
	# only valid if if yaw[0]. roll[0], pitch[0] are 0
	# however, we define our world in terms of this first attitude, so it is only a definition question:
	t1 = np.zeros([3,1]);
	Transl.append(t1);
	R1 = np.eye(3);
	Rot.append(R1);		

	for fr in range(n_frames-1):
	
		# get the rotation from the drone data: 
		delta_phi = limit_angle(np.deg2rad(roll[fr+1] - roll[fr]));
		delta_theta = limit_angle(np.deg2rad(pitch[fr+1] - pitch[fr]));
		delta_psi = limit_angle(np.deg2rad(yaw[fr+1] - yaw[fr]));

		# convert them to rotations of world points around the camera's X, Y, Z axes:
		(Rx, Ry, Rz) = convertAnglesFromDroneToCamera(delta_phi, delta_theta, delta_psi);

		# get the rotation matrix:
		R = getRotationMatrix(Rx, Ry, Rz);

		# append the rotation matrix:
		Rot.append(R);

		# get the translation from the drone data:
		t = np.zeros([3,1]);
		t[0] = vx[fr] / 1000.0;
		t[1] = vy[fr] / 1000.0;
		t[2] = vz[fr] / 1000.0;
		t = t * snapshot_time_interval;
		
		# convert the drone translation to the translation of world points in the camera's view:
		t_camera = convertTranslationFromDroneToCamera(t);
			
		# append the translation vector:
		Transl.append(t_camera);

	# Put all translations / rotations in one frame:
	Rotations = [];
	Translations = [];
	
	# First camera is fixed:
	R = np.eye(3);
	transl = np.zeros([3,1]);
	
	if(graphics):
		# make a figure to plot positions and camera axes:
		pl.figure();
		pl.hold(True);
		camera_axis = np.array([0,0,0.1]);
		
	for fr in range(n_frames):

		# rotations and translations should still be made incremental:		
		
		# some nasty questions:
		# a) first translate and then rotate? If time delta is small enough it does not matter - but it's not so it does : )
		# b) do vx, vy, vz give velocities in body coordinates? Or in the inertial reference frame?
		
		# get the translation:
		transl = np.dot(np.linalg.inv(R), Transl[fr]) + transl;
		
		# rotation:	
		R = np.dot(Rot[fr], R);
		
		# add them to the Rotations / Translations structs:
		Rotations.append(R);
		Translations.append(transl);

		if(graphics):
			# plot camera position and camera axis in the X,Z-plane:
			# since translations / rotations are of world points, we have to negate them /
			# invert them to get the camera translations / rotations:
			pl.plot(-transl[0], -transl[2], 'x');
			cR = np.dot(np.linalg.inv(R), camera_axis)
			pl.plot([-transl[0], -transl[0]+cR[0]], [-transl[2], -transl[2]+cR[2]]);
			

	return (Rot, Transl, Rotations, Translations);

def convertAnglesFromDroneToCamera(delta_phi, delta_theta, delta_psi):
	""" Convert the angles from the drone to the rotation of the world points with respect to the camera.
	"""
	
	# There are three differences:
	# (1) the drone coordinate frame is right-handed while the camera is left-handed
	# (2) the axes have different names
	# (3) the angles are of the drone, while the "camera" angles actually indicate how the world points rotate (* -1 here)

	# This results in the following mapping:
	Rx = delta_theta; 
	Ry = delta_psi;
	Rz = delta_phi;

	return (Rx, Ry, Rz);

def convertTranslationFromDroneToCamera(t):
	""" Convert	the drone translation t to the translation of the world points in the camera.
	"""

	# There are two differences:
	# (1) The drone's axes are differently labeled x, y, z
	# (2) The world point motion is opposite of the drone's motion

	# This results in the following mapping:
	t_camera = np.zeros([3,1]);
	t_camera[0,0] = -t[0,0];
	t_camera[1,0] = -t[2,0];
	t_camera[2,0] = -t[1,0];

	return t_camera;

def getTriangulatedPoints(image_points1, image_points2, R2, t2, K, R1=[], t1=[]):

	""" Get the triangulated world points on the basis of corresponding image points and 
			a rotation matrix, displacement vector and calibration matrix.
			The rotation and translation of the first camera can optionally also be given.
	"""

	# 3D-reconstruction:
	ip1 = cv2.convertPointsToHomogeneous(image_points1.astype(np.float32));
	ip2 = cv2.convertPointsToHomogeneous(image_points2.astype(np.float32));

	# P1 is at the origin
	P1 = np.zeros([3, 4]);
	# default assumption is that the first camera is at the origin:
	if(len(R1) == 0):
		P1[:3,:3] = np.eye(3);
	else:
		P1[:3,:3] = R1;
	if(len(t1) != 0):
		P1[:3,3] = np.array([t1[0][0], t1[1][0], t1[2][0]]);
	P1 = np.dot(K, P1);

	# P2 is the displaced camera:
	P2_est = getProjectionMatrix(R2, t2, K);

	# reconstruct the points:
	X_est = triangulate(ip1, ip2, P1, P2_est);

	return X_est;


def cleanUpR(R):
	for i in range(3):
		if(R[i,i] < 0):
			for c in range(3):
				R[i,c] = -R[i,c];
	
	return R;

def flatten(X):
	Y = [x[0] for x in X];
	return Y;

def getRotationMatrix(Rx, Ry, Rz):
	""" Create rotation matrix R on the basis of Rx, Ry, Rz, in the left-handed camera coordinate system.
			Takes radians.
	"""
	
	# create rotation matrix:
	R_Rx = np.zeros([3,3]);
	R_Rx[0,0] = 1;
	R_Rx[1,1] = np.cos(Rx);
	R_Rx[1,2] = -np.sin(Rx);
	R_Rx[2,1] = np.sin(Rx);
	R_Rx[2,2] = np.cos(Rx);

	R_Ry = np.zeros([3,3]);
	R_Ry[1,1] = 1;
	R_Ry[0,0] = np.cos(Ry);
	R_Ry[2,0] = -np.sin(Ry);
	R_Ry[0,2] = np.sin(Ry);
	R_Ry[2,2] = np.cos(Ry);

	R_Rz = np.zeros([3,3]);
	R_Rz[0,0] = np.cos(Rz);
	R_Rz[0,1] = -np.sin(Rz);
	R_Rz[1,0] = np.sin(Rz);
	R_Rz[1,1] = np.cos(Rz);
	R_Rz[2,2] = 1;

	# multiply the matrices:
	# first x, then y, then z:
	R = np.dot(R_Rz, np.dot(R_Ry, R_Rx));

	return R;

def calculateReprojectionError(Rs, Ts, X, IPs, n_cameras, n_world_points, K):
	"""	calculateReprojectionError takes a number of rotation and translation matrices, 
			a matrix of world points, and the corresponding image coordinates of those points.

			It then projects the world points to the image planes and calculates how far off 
			they are from the observed locations.	
	"""

	distCoeffs = np.zeros([4]); # distortion coefficients

	# per camera, project the world points into the image and calculate the error with respect to the measured image points:
	total_error = 0;
	error_per_point = np.array([0.0]*n_world_points);

	for cam in range(n_cameras):

		# The measured image points:
		measured_image_points = IPs[cam];
			
		# project the world points in the cameras:
		t = Ts[cam];
		R = Rs[cam];
		rvec = cv2.Rodrigues(R);
		rvec = rvec[0];
		result = cv2.projectPoints(X, rvec, t, K, distCoeffs);
		image_points = result[0];

		# calculate the error for this camera:
		err = 0;
		for ip in range(n_world_points):
			# only determine error if the point is observed in the image:
			if(observed(measured_image_points[ip])):
				error_per_point[ip] += np.linalg.norm(image_points[ip] - measured_image_points[ip]);
			err += error_per_point[ip];

		# print 'Total error: %f' % total_error;
		total_error += err;

	return (total_error, error_per_point);

def observed(image_point):
	# unobserved points have image coordinate (-1,-1)
	if(image_point[0][0] == -1 or image_point[0][1] == -1):
		return False;
	else:
		return True;

def printRotationMatrix(R):

	rows = R.shape[0];
	columns = R.shape[1];

	for row in range(rows):
		print '';
		for col in range(columns):
			print '%f ' % (R[row, col]),
	print ''
	

def infeasibleP2(x1, x2, R1, t1, R2, t2, K):
	""" Triangulates a single point and checks whether it lies
			behind one of the cameras. If so, it returns 1, else 0. """

	infeasible = 0;

	#n = x1.shape[0]
	#if x2.shape[0] != n:
	#	raise ValueError("Number of points don't match.")
	n = 1;	

	# get P1, P2 for triangulation:
	P1 = getProjectionMatrix(R1, t1, K);
	P2 = getProjectionMatrix(R2, t2, K);

	# a point is behind the camera, if the angle between the optical axis and the world point
	# is larger than 90 degrees:
	# cos(theta) = (v1 * v2) / (norm(v1) * norm(v2))
	# abs(theta) = abs(acos(...) > 0.25 pi)

	#optical_axis = np.array(3);
	#optical_axis[2] = 1.0;
	#OA1 = np.dot(R1, optical_axis);
	#OA2 = np.dot(R2, optical_axis);
	
	IR1 = np.linalg.inv(R1);
	IR2 = np.linalg.inv(R2);

	# triangulating one point should suffice (in a noiseless world?)
	for i in range(n):
		# triangulate point to world coordinates:
		X_w = np.asarray(triangulate_point(x1[i,:],x2[i,:],P1,P2));
		# v1 is the vector from camera 1's optical center to the world point
		v1 = X_w[:3] - t1.T;
		# rotate the vector with the inverse rotation
		v1 = np.dot(IR1, v1.T);
		# check whether behind camera:
		if(v1[2] < 0):
			infeasible = 1;
			break;
		# v2 points from the second camera's optical center to the world point:
		v2 = X_w[:3] - t2.T;
		# camera 2's optical axis is rotated with R2, so here 
		v2 = np.dot(IR2, v2.T);
		if(v2[2] < 0):
			infeasible = 1;
			break;
	
	return infeasible;

	# in case (b) it can immediately return the triangulated points as well...

	


def getProjectionMatrix(R, t, K):
	""" On the basis of a rotation matrix R, a translation vector t,
			and camera calibration matrix K, determines the projection matrix P """
	P = np.zeros([3, 4]);
	P[:3,:3] = R;
	P[:,3] = t.transpose();
	P = np.dot(K, P);
	return P;


def triangulate_point(x1,x2,P1,P2):
	""" Point pair triangulation from 
			least squares solution. """
        
	M = np.zeros((6,6))
	M[:3,:4] = P1
	M[3:,:4] = P2
	M[:3,4] = -x1
	M[3:,5] = -x2

	U,S,V = np.linalg.svd(M)
	X = V[-1,:4]

	return X / X[3]


def triangulate(x1,x2,P1,P2):
	"""    Two-view triangulation of points in 
					x1,x2 (3*n homog. coordinates). """
        
	n = x1.shape[0]
	if x2.shape[0] != n:
		raise ValueError("Number of points don't match.")

	X = [ np.asarray(triangulate_point(x1[i,:],x2[i,:],P1,P2)) for i in range(n)]
	
	return X #np.array(X).T

def determineTransformation(points1, points2, K, W=[], H=[]):

	pdb.set_trace();

	DEBUG = False;

	# find fundamental matrix:
	OPENCV = True;
	if(OPENCV):
		# we need to pass image points going from 0 to W, 0 to H:
		# a small change in the fundamental matrix can make a big difference after the SVD... how to solve this?
		max_dist_el = 1.0;
		prob_sure = 0.9999;
		(F, inliers) = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS, max_dist_el, prob_sure);
		#res = np.dot(M1, np.dot(F, M2.transpose()));
	else:
		x1 = transformPointsSFM(points1);
		x2 = transformPointsSFM(points2);
		F = compute_fundamental_normalized(x1,x2);

	# make F rank 2:
	print 'F determinant: %f' % (np.linalg.det(F))
	(UF, SigmaF, VTF) = np.linalg.svd(F);
	D = np.zeros([3,3]);
	D[0,0] = SigmaF[0];
	D[1,1] = SigmaF[1];
	F = np.dot(UF, np.dot( D , VTF ) );
	print 'F determinant: %f' % (np.linalg.det(F))	
	
	if(DEBUG):
		print 'F = '
		printRotationMatrix(F);
		print 'F determinant: %f' % (np.linalg.det(F))
	
	# extract essential matrix:
	E = np.dot(K.transpose(), np.dot(F, K));

	if(DEBUG):
		print 'E = '
		printRotationMatrix(E);
	

	# extract R and t:
	(U, Sigma, VT) = np.linalg.svd(E);

	if(DEBUG):
		print 'U = '
		printRotationMatrix(U);
		#print 'Sigma = '
		#printRotationMatrix(Sigma);
		print 'VT = '
		printRotationMatrix(VT);


	W1 = np.zeros([3,3]);	
	W1[0,1] = -1;
	W1[1,0] = 1;
	W1[2,2] = 1;
	W2 = W1.transpose();
	
	# method to directly retrieve the matrix / vector:
	R1 = np.dot(U, np.dot(W1, VT));
	R2 = np.dot(U, np.dot(W2, VT));
	t1 = U[:,2];
	t2 = -t1;

	return (R1, R2, t1, t2)

def transformPointsSFM(points):
	n_points = points.shape[0];
	x = np.array([[0.0]*n_points, [0.0]*n_points, [0.0]*n_points]);
	for p in range(n_points):
		x[0][p] = points[p,0,0];
		x[1][p] = points[p,0,1];
		x[2][p] = 1.0;
	return x;

def compute_fundamental(x1,x2):
	"""    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
  
	n = x1.shape[1]
	if x2.shape[1] != n:
		raise ValueError("Number of points don't match.")
    
	# build matrix for equations
	A = np.zeros((n,9))
	for i in range(n):
		A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
	# compute linear least square solution
	U,S,V = np.linalg.svd(A)
	F = V[-1].reshape(3,3)
	# constrain F
	# make rank 2 by zeroing out last singular value
	U,S,V = np.linalg.svd(F)
	S[2] = 0
	F = np.dot(U,np.dot(np.diag(S),V))
    
	return F/F[2,2]

def compute_fundamental_normalized(x1,x2):
	"""    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the normalized 8 point algorithm. """

	n = x1.shape[1]
	if x2.shape[1] != n:
		raise ValueError("Number of points don't match.")

	# normalize image coordinates
	x1 = x1 / x1[2]
	mean_1 = np.mean(x1[:2],axis=1)
	S1 = np.sqrt(2) / np.std(x1[:2])
	T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
	x1 = np.dot(T1,x1)
    
	x2 = x2 / x2[2]
	mean_2 = np.mean(x2[:2],axis=1)
	S2 = np.sqrt(2) / np.std(x2[:2])
	T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
	x2 = np.dot(T2,x2)

	# compute F with the normalized coordinates
	F = compute_fundamental(x1,x2)

	# reverse normalization
	F = np.dot(T1.T,np.dot(F,T2))

	return F/F[2,2]


def wrongSolution(R1, R2):
	# wrong solutions due to small variations in the fundamental matrix have a negative diagonal element
	# this simple method detects such solutions
	pdb.set_trace();

	neg_diag = 0;
	for d in range(3):
		if(R1[d,d] < 0):
			neg_diag += 1;
			break;

	for d in range(3):
		if(R2[d,d] < 0):
			neg_diag += 1;
			break;
	
	if(neg_diag == 2):
		return 1;
	else:
		return 0;
	


	# other method:
	# tx = V W Sigma VT
  # R = U W-1 VT
	# or
	# t = V Z VT

	#Z = zeros([3,3]);
	#Z[0,1] = -1;
	#Z[1,0] = 1;
	#tx = np.dot(VT.transpose(), np.dot(Z, VT));

	# check:
	# image_coord = R * X_world + t

#cvFindFundamentalMat(const CvMat* points1, const CvMat* points2, CvMat* fundamentalMatrix, int method=CV_FM_RANSAC, double param1=1., double param2=0.99, CvMat* status=NULL)
#cvDecomposeProjectionMatrix(const CvMat *projMatrix, CvMat *cameraMatrix, CvMat *rotMatrix, CvMat *transVect, CvMat *rotMatrX=NULL, CvMat *rotMatrY=NULL, CvMat *rotMatrZ=NULL, CvPoint3D64f *eulerAngles=NULL)
# void cvProjectPoints2(const CvMat* objectPoints, const CvMat* rvec, const CvMat* tvec, const CvMat* cameraMatrix, const CvMat* distCoeffs, CvMat* imagePoints, CvMat* dpdrot=NULL, CvMat* dpdt=NULL, CvMat* dpdf=NULL, CvMat* dpdc=NULL, CvMat* dpddist=NULL)

# python photogrammetry toolbox:
# http://www.arc-team.homelinux.com/arcteam/ppt.php
# http://vai.uibk.ac.at/dadp/doku.php?id=ppt_loewen_en

# entire projective geometry
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FUSIELLO4/tutorial.html#x1-15002r24

#http://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
#http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
#http://en.wikipedia.org/wiki/Essential_matrix
#http://stackoverflow.com/questions/6211809/how-to-calculate-normalized-image-coordinates-from-pixel-coordinates
#http://stackoverflow.com/questions/15940663/correct-way-to-extract-translation-from-essential-matrix-through-svd
#http://stackoverflow.com/questions/15157756/where-do-i-add-a-scale-factor-to-the-essential-matrix-to-produce-a-real-world-tr
#http://stackoverflow.com/questions/5533856/how-to-calculate-the-fundamental-matrix-for-stereo-vision
#http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
#http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html
#http://stackoverflow.com/questions/3678317/t-and-r-estimation-from-essential-matrix
#http://stackoverflow.com/questions/10187582/opencv-python-fundamental-and-essential-matrix-do-not-agree



# introduction to matplotlib:
# http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/scientific-python-lectures/master/Lecture-4-Matplotlib.ipynb

#	# p2 = points2.tolist();
#	p2 = points2;
#	for p in p2:
#		#p = p[0];
#		p[0] /= W
#		p[1] /= H
#		p[0] -= 1/2
#		p[1] -= 1/2
#		#p.append(1);
#p1 = np.asarray(p1);
#p2 = np.asarray(p2);
#M1 = np.matrix(p1);
#M2 = np.matrix(p2);



# to homogeneous coordinates:
	# p2 = points2.tolist();
#	p2 = points2;
	#for p in p2:
		#p = p[0];
		#p[0,0] /= W
		#p[0,1] /= H
		#p[0,0] -= 0.5
		#p[0,1] -= 0.5
		#p.append(1);
	#p1 = points1.tolist();
#	p1 = points1;
#	for p in p1:
		#p = p[0];
#		p[0,0] /= W
	#	p[0,1] /= H
		#p[0,0] -= 0.5
		#p[0,1] -= 0.5
		#p.append(1);
	#p1 = np.asarray(p1);
	#p2 = np.asarray(p2);
	#M1 = np.matrix(p1);
	#M2 = np.matrix(p2);


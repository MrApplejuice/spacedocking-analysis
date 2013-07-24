#!/usr/bin/env python3


# low-dimensional representation with t-sne:
import sys
sys.path.insert(0, './py2/tsne_python/')
sys.path.insert(0, './py2/calc_tsne/')
sys.path.insert(0, './py3/')
#from tsne import *
#from calc_tsne import *
import scipy.io

from readdata import *
from Kohonen import *
import os
import pdb
import cv2
import numpy as np;
from matplotlib import pyplot as pl;
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid import AxesGrid
from VisualOdometry import *
import pylab as Plot
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from analyzeDistanceVariation import *
from visualizeFeaturePositions import *
# size(result) -> total number of samples
# size(result[i]['frames']) == 5
# result[i]['frames'][j]
# result[i]['image_features']['flat_descriptor_values']
# result[i]['frames'][j]['features']['features'][fnr]
# result[i]['frames'][j]['features']['features'][fnr]['descriptor']

roll_index = 0;
yaw_index = 1;
pitch_index = 2;
vx_index = 0;
vz_index = 1;
vy_index = 2;
x_index = 1;
y_index = 0;
z_index = 2;

snapshot_time_interval = 0.25;

def initializeFeatures(features, time_step):
	n_features = len(features);
	# Copy the features into a struct used for matching:
	FTS = [];
	for f in range(n_features):
		feature = {'sizes': [], 'descriptors': [], 'time_steps': [], 'n_not_observed': 0, 'x': [], 'y': []};
		FTS.append(feature);
		FTS[f]['sizes'].append( features[f]['size'] );
		FTS[f]['time_steps'].append( time_step ); # goes from 1 to 4???
		FTS[f]['descriptors'].append( features[f]['descriptor'] );
		FTS[f]['x'].append( features[f]['x'] );
		FTS[f]['y'].append( features[f]['y'] );
	return FTS;

def addMatchedFeature(FT, feature, time_step):
	# add the info on a matched feature
	FT['sizes'].append(feature['size']);
	FT['time_steps'].append(time_step);
	FT['descriptors'].append(feature['descriptor']);
	FT['x'].append(feature['x']);
	FT['y'].append(feature['y']);

def getMatchedFeatures(sample, graphics=False):
	"""	Given a sample from the database (consisting of 5 frames), matches features from one frame to the next.
			
			input:
			- sample from the database 
			- graphics: whether to show all kinds of plots

			output:
			- TimeToContact estimate at the last image
			- n_features_used_for_estimate
			- TTC_estimates of all the selected features in the last image
			- FTS_USED: the features used for the estimate
			- ALL_FTS: all features encountered in the images
	"""

	# whether to show graphics.
	ttc_graphics = True;

	# nearest neighbor threshold for matching:
	NN_THRESHOLD = 0.75;
	
	# maximum number of time steps a feature can go unobserved:
	max_n_not_observed = 1;
	
	# minimum number of observations of a feature to be considered for the estimate:
	min_memory = 4; # 4 means that the features are present in all images: is that true?
	
	# number of frames in a stored sequence:
	n_frames = len(sample['frames']);
	
	# first show velocities, angles:
	
	#roll_index = 0;
	#yaw_index = 1;
	#pitch_index = 2;
	#vx_index = 0;
	#vy_index = 2;
	#x_index = 0;
	#y_index = 1;
	#z_index = 2;
	roll = np.array([0.0] * n_frames);
	yaw = np.array([0.0] * n_frames);
	pitch = np.array([0.0] * n_frames);
	vx = np.array([0.0] * n_frames);
	vy = np.array([0.0] * n_frames);
	x = np.array([0.0] * n_frames);
	y = np.array([0.0] * n_frames);
	z = np.array([0.0] * n_frames);
	for fr in range(n_frames):
		roll[fr] = sample['frames'][fr]['euler_angles'][roll_index];
		yaw[fr] = sample['frames'][fr]['euler_angles'][yaw_index];
		pitch[fr] = sample['frames'][fr]['euler_angles'][pitch_index];
		vx[fr] = sample['frames'][fr]['velocities'][vx_index]; # velocities, not ground_speed!!!
		vy[fr] = sample['frames'][fr]['velocities'][vy_index];
		x[fr] = sample['frames'][fr]['position'][x_index];
		y[fr] = sample['frames'][fr]['position'][y_index];
		z[fr] = sample['frames'][fr]['position'][z_index];
	
	if(graphics):
		pl.figure();
		pl.hold(True);
		pl.plot(roll);
		pl.plot(pitch);
		pl.plot(yaw);
		pl.legend(('roll','pitch','yaw'), 'upper right')
		pl.show();
		
		pl.figure();
		pl.hold(True);
		pl.plot(vx);
		pl.plot(vy);
		pl.legend(('vx','vy'), 'upper right')
		pl.show();
		
		pl.figure();
		pl.plot(z);
		pl.legend(('z'), 'upper right')
		pl.show();
		
		pl.figure();
		pl.hold(True);
		pl.plot(x);
		pl.plot(y);
		pl.legend(('x','y'), 'upper right')
		pl.show();
	
	# will contain the time to contact estimates:
	TimeToContact = np.array([1E4]*(n_frames-1));
	
	# FTS will contain all current features
	frame1 = sample['frames'][0];
	FTS = initializeFeatures(frame1['features']['features'], 0);
	
	for fr in range(n_frames-1):
		
		n_features1 = len(FTS);
		
		# get the features from the next frame:
		frame2 = sample['frames'][fr+1];
		n_features2 = len(frame2['features']['features']);
		matched = np.array([0] * n_features2);
		
		
		if(graphics):
			pl.figure();
			pl.hold(True);
		
		#tau = [];
		
		for ft1 in range(n_features1):
		
			# determine the distances to the features in the second frame:
			distances = np.array([0.0] * n_features2);
			for ft2 in range(n_features2):
				# use the last added descriptor:
				distances[ft2] = np.linalg.norm(np.array(FTS[ft1]['descriptors'][-1]) - np.array(frame2['features']['features'][ft2]['descriptor']));
			
			# sort the distances:
			sindices = np.argsort(distances);
			
			# the second nearest neighbor has to be sufficiently far for a match:
			if(len(distances) > 1 and distances[sindices[0]] / distances[sindices[1]] < NN_THRESHOLD):
				# we have a match:
				addMatchedFeature(FTS[ft1], frame2['features']['features'][sindices[0]], fr+1);
				matched[sindices[0]] = 1;
				FTS[ft1]['n_not_observed'] = 0;
				
				#delta_size = frame2['features']['features'][sindices[0]]['size'] - frame1['features']['features'][ft1]['size'];
				#if(np.abs(delta_size) > 1E-4):
				#	ttc = frame1['features']['features'][ft1]['size'] / delta_size;
				#else:
				#	ttc = np.sign(delta_size) * 1E4;
				#	
				#tau.append(ttc);
				
				if(graphics):
					x1 = frame1['features']['features'][ft1]['x'];
					y1 = frame1['features']['features'][ft1]['y'];
					x2 = frame2['features']['features'][sindices[0]]['x'];
					y2 = frame2['features']['features'][sindices[0]]['y'];
					pl.plot([x1, x2], [y1,y2]);
			else:
				# feature remained unmatched:
				FTS[ft1]['n_not_observed'] += 1;
		
		if(graphics):
			pl.title('t = %d' % (fr));
			pl.show();
		
		
		# housekeeping:

		# remove features that have gone unobserved for too long:
		print 'len(FTS) before = %d' % len(FTS);
		FTS = [ft for ft in FTS if ft['n_not_observed'] <= max_n_not_observed];
		print 'len(FTS) after = %d' % len(FTS);
		
		# initialize a new group of (unmatched) features:
		new_features = [frame2['features']['features'][ft2] for ft2 in range(n_features2) if matched[ft2] == 0];
		NEW_FTS = initializeFeatures(new_features, fr+1);
		print 'n new features = %d' % len(NEW_FTS)
		
		# append the new features:
		FTS = FTS + NEW_FTS;
		
		#if(ttc_graphics):
		#	pl.figure();
		#	pl.hist(tau)
		#
		#if(len(tau) > 0):
		#	TimeToContact[fr] = np.median(tau);
		#else:
		#	TimeToContact[fr] = 10;
		#	
		#print 'TTC frame %d = %f' % (fr, np.median(tau))
	
	n_features = len(FTS);
	# only determine time-to-contact at the end:
	memory_distribution = np.array([0] * n_features);
	TTC_estimates = [];
	FTS_USED = [];
	n_features_used_for_estimate = 0;
	for ft in range(n_features):
		memory_size = len(FTS[ft]['sizes'])
		memory_distribution[ft] = memory_size;
		if(memory_size >= min_memory):
			TTC_estimates.append(determineTTCLinearFit(FTS[ft]));
			FTS_USED = FTS_USED + [FTS[ft]];
			n_features_used_for_estimate += 1;
			
	if(ttc_graphics):
		if(n_features > 0):
			pl.figure();
			pl.hist(memory_distribution);
			pl.title('mem dist');
			pl.show();

		if(len(TTC_estimates) > 0):
			pl.figure();
			pl.hist(TTC_estimates);
			pl.title('TTC ests')
			pl.show();
	
	if(len(TTC_estimates) > 0):
		TimeToContact = np.median(TTC_estimates);
	else:
		TimeToContact = 1E3;
		
	TimeToContact *= snapshot_time_interval;
	
	# also return all features present in the last frame:
	ALL_FTS = [];
	for ft2 in range(n_features2):
		# use the last added descriptor:
		ALL_FTS = ALL_FTS + [frame2['features']['features'][ft2]['descriptor']];
	
	return (TimeToContact, n_features_used_for_estimate, TTC_estimates, FTS_USED, ALL_FTS);
	

def determineTTCLinearFit(feature):
	
	# perform a linear fit of the sizes:
	n_ts = len(feature['time_steps']);
	A = np.ones([n_ts, 2]);
	A[:,1] = feature['time_steps'];
	# to sqrt or not to sqrt?
	b = np.array(feature['sizes']);
	(x, residuals, rnk, s) = np.linalg.lstsq(A,b);
	
	# calculate TTC:
	size_slope = x[1];
	if(abs(size_slope) > 1E-3):
		TTC = feature['sizes'][-1] / size_slope;
	else:
		TTC = np.sign(size_slope) * 1E3;
		
	#pl.figure();
	#pl.plot(feature['time_steps'], feature['sizes'], 'x', color=(0.0,0.0,0.0));
	#pl.hold = True;
	#y = np.array([0.0]* n_ts);
	#ii = 0;
	#for t in feature['time_steps']:
	#	y[ii] = x[0] + x[1] * t;
	#	ii += 1;
	#pl.plot(feature['time_steps'], y, color=(0.0, 1.0, 0.0), linewidth=2);
	#pl.hold=False;
		
	return TTC;

def matchTwoImages(test_dir, image_name1, image_name2, NN_THRESHOLD = 0.9):

	# extract the SURF features from the images:
	(keypoints1, descriptors1, img1, img1_gray) = extractSURFfeaturesFromImage(test_dir + "/" + image_name1);
	(keypoints2, descriptors2, img2, img2_gray) = extractSURFfeaturesFromImage(test_dir + "/" + image_name2);
	n_features1 = len(keypoints1);
	n_features2 = len(keypoints2);
	
	# make the figure:
	#pl.figure();
	#pl.hold(True);
	
	cv2.namedWindow("matches", cv2.cv.CV_WINDOW_NORMAL)
	h1, w1 = img1_gray.shape[:2]
	h2, w2 = img2_gray.shape[:2]
	vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
	vis[:h1, :w1] = img1_gray
	vis[:h2, w1:w1+w2] = img2_gray
	vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

	points1 = [];
	points2 = [];

	# match the features:
	for ft1 in range(n_features1):
	
		# determine the distances to the features in the second frame:
		distances = np.array([0.0] * n_features2);
		for ft2 in range(n_features2):
			distances[ft2] = np.linalg.norm(np.array(descriptors1[ft1]) - np.array(descriptors2[ft2]));
		
		# sort the distances:
		sindices = np.argsort(distances);
		
		# the second nearest neighbor has to be sufficiently far for a match:
		if(len(distances) > 1 and distances[sindices[0]] / distances[sindices[1]] < NN_THRESHOLD):
			print 'match'
			# we have a match:
			# properties of key points:
			# pt.x, pt.y
			# size
			# angle
			# response
			# octave
			x1 = keypoints1[ft1].pt[0];
			y1 = keypoints1[ft1].pt[1];
			points1.append(np.array([x1, y1]));
			x2 = keypoints2[sindices[0]].pt[0];
			y2 = keypoints2[sindices[0]].pt[1];
			points2.append(np.array([x2, y2]));
			#cv2.circle(img1, (int(x1), int(y1)), 4, (0.0,255.0,0.0), -1)
			#cv2.circle(img2, (int(x2), int(y2)), 4, (0.0,255.0,0.0), -1)
			cv2.line(vis, (int(x1), int(y1)), (int(x2+w1), int(y2)), (0.0, 255.0, 0.0), 1)
			#pl.plot([x1, x2], [y1,y2]);
	
	cv2.imshow('matches', vis)
	#cv2.namedWindow("img1", cv2.cv.CV_WINDOW_NORMAL)
	#cv2.imshow('img1',img1)
	#cv2.namedWindow("img2", cv2.cv.CV_WINDOW_NORMAL)
	#cv2.imshow('img2',img2)

	# get dummy camera calibration matrix:
	K = getK(w1, h1);

	# 3D reconstruction:
	# we should subtract (W/2, H/2) from the image points!
	(R, t, X, errors_per_point) = performStructureFromMotion(np.array(points1), np.array(points2), K, w1, h1);

	# show 3D reconstruction:
	fig = pl.figure()
	ax = fig.gca(projection='3d')
	M_est = np.matrix(X);
	x = np.array(M_est[:,0]); y = np.array(M_est[:,1]); z = np.array(M_est[:,2]);
	x = flatten(x); y = flatten(y); z = flatten(z);
	cm = pl.cm.get_cmap('hot')
	ax.scatter(x, y, z, '*', c=errors_per_point, cmap=cm);
	pl.show();



def contains(L, element):
	""" Determines whether a list contains a certain element.
	"""
	n_elements = len(L);

	for el in range(n_elements):
		if(L[el] == element):
			return True;
	
	return False;

def getImagePoints(FTS, frame):
	""" getImagePoints takes the features as determined by the matching, and one frame.
			It returns the image points of the features present in the frame, with (-1,-1) as coordinate for unobserved features.
	"""	

	n_features = len(FTS);
	image_points = [];
	for ft in range(n_features):
		try:
			index_frame = FTS[ft]['time_steps'].index(frame);
			# point is visible in this frame:
			image_points.append([FTS[ft]['x'][index_frame], FTS[ft]['y'][index_frame]]);
		except ValueError:
			# point not visible in this frame:
			image_points.append([-1, -1]);			

	return (np.array(image_points));


def getImagePointsTwoFrames(FTS, first_frame, second_frame):
	""" getImagePointsTwoFrames takes the features as determined by the matching, and two frames
			It returns the image points of the features present in both these frames, with the indices
	"""	

	# FTS[ind]['time_steps'] can go from 0 to 4, but if it is not observed for one of the time steps, this one will be missing from the list

	n_features = len(FTS);
	
	# make a list of features that have both time steps
	#selected_features = [FTS[ind] for ind in range(n_features) if (contains(FTS[ind]['time_steps'], first_frame) and contains(FTS[ind]['time_steps'], second_frame))];
	indices = [];
	selected_features = [];
	for ind in range(n_features):
		if(contains(FTS[ind]['time_steps'], first_frame) and contains(FTS[ind]['time_steps'], second_frame)):
			selected_features.append(FTS[ind]);
			indices.append(ind);

	# make a list of the corresponding image points in image 1 and 2:
	n_features = len(selected_features);
	image_points1 = [];
	image_points2 = [];
	for ft in range(n_features):
		#image_points1.append(np.array([selected_features[ft]['x'][first_frame], selected_features[ft]['y'][first_frame]]));
		#image_points2.append(np.array([selected_features[ft]['x'][second_frame], selected_features[ft]['y'][second_frame]]));
		index_first_frame = selected_features[ft]['time_steps'].index(first_frame);
		image_points1.append([selected_features[ft]['x'][index_first_frame], selected_features[ft]['y'][index_first_frame]]);
		index_second_frame = selected_features[ft]['time_steps'].index(second_frame);
		image_points2.append([selected_features[ft]['x'][index_second_frame], selected_features[ft]['y'][index_second_frame]]);

	return (np.array(image_points1), np.array(image_points2), indices);
	
def getFeaturesWithDistance(sample):
	""" Determines the distances to features that persist throughout the sequence.
			It also returns the scale-TTC-based distances of these features.
	"""
	
	# 1) Get the features that persist throughout the sequence with corresponding TTC estimates (map these to distances with the help of the velocity)
	# 2) Reconstruct from frame to frame the camera positions and attitudes, and world points, scaling them with the help of the drone velocity
	# 3) Perform some kind of bundle adjustment
	# 4) Determine a distance per feature in the last frame

	# if True, the translation and rotation are used and step 2 above skipped:
	use_drone_info = True;

	# 1) Get the features that persist throughout the sequence with corresponding TTC estimates (map these to distances with the help of the velocity)
	(TimeToContact, n_features_used_for_estimate, TTC_estimates, FTS_USED, ALL_FTS) = getMatchedFeatures(sample);

	# check if there are enough features to go by.
	# although with our implementation 8 is the absolute minimum, we require more points for reliability
	# not all points are visible in each image!
	n_features = len(FTS_USED);
	if(n_features < 20):
		print 'Not enough features for state reconstruction';
		distsReconstruction = []; distsTTC = [];
		return (distsReconstruction, distsTTC);

	# 2) Reconstruct from frame to frame the camera positions and attitudes, and world points, scaling them with the help of the drone velocity
	
	# number of frames in a stored sequence:
	n_frames = len(sample['frames']);

	# width and height of the image, should be determined with sample:
	device_version = sample['device_version'];
	if(isDrone1(device_version)):
		# W = 320; H = 240;
		K = getKdrone1();
	else:
		# W = 640; H = 360;
		K = getKdrone2();

	# reconstruct the camera movement and world points from frame to frame:
	points_world = [[]];
	for f in range(n_features-1):
		points_world.append([]);
	Rotations = [];
	Translations = [];
	vx = np.array([0.0] * (n_frames-1));
	vy = np.array([0.0] * (n_frames-1));
	vz = np.array([0.0] * (n_frames-1));
	# IPs will contain the image_points per image, with coordinate (-1, -1) for invisible points:
	IPs = [];

	# get state info:
	roll = np.array([0.0] * n_frames);
	yaw = np.array([0.0] * n_frames);
	pitch = np.array([0.0] * n_frames);
	for fr in range(n_frames):
		roll[fr] = sample['frames'][fr]['euler_angles'][roll_index];
		yaw[fr] = sample['frames'][fr]['euler_angles'][yaw_index];
		pitch[fr] = sample['frames'][fr]['euler_angles'][pitch_index];
		# IPs contains the image_points per frame, with (-1, -1) for the unobserved ones:
		IPs.append(getImagePoints(FTS_USED, fr));

	# get all frame-to-frame reconstructions:
	phis = np.array([0.0]*(n_frames-1));
	thetas = np.array([0.0]*(n_frames-1));
	psis = np.array([0.0]*(n_frames-1));
	for fr in range(n_frames-1):
		
		# get the image points that have the right size of memory, and occur in the two currently relevant images (fr, fr+1)
		# indices are necessary, since triangulation between these frames only gives info on world points with these indices:
		pdb.set_trace();
		(image_points1, image_points2, indices) = getImagePointsTwoFrames(FTS_USED, fr, fr+1);
		n_matches = len(indices);

		if(use_drone_info):

			# This assumes the rotation of the drone to be equal to the rotation of the camera matrix, but this should probably be inversed.
			# To match the camera convention, y of the drone should become z of the camera, x stays x and z becomes y.
			# What should happen to roll, pitch, yaw? roll drone -> rotation around z-axis (psi), pitch drone -> rotation around x (roll), yaw drone -> rotation around y-axis (theta)

			# get the rotation from the drone data: 
			delta_phi = limit_angle(np.deg2rad(roll[fr+1] - roll[fr]));
			delta_theta = limit_angle(np.deg2rad(pitch[fr+1] - pitch[fr]));
			delta_psi = limit_angle(np.deg2rad(yaw[fr+1] - yaw[fr]));
			# This assumes the camera to be on the center-of-gravity => change it:
			R = getRotationMatrix(delta_phi, delta_theta, delta_psi);
			
			phis[fr] = delta_phi;
			thetas[fr] = delta_theta;
			psis[fr] = delta_psi;
			# determine the translation with the help of drone data:
			
			vx[fr] = sample['frames'][fr]['velocities'][vx_index] / 1000.0;
			vy[fr] = sample['frames'][fr]['velocities'][vy_index] / 1000.0;
			vz[fr] = sample['frames'][fr]['velocities'][vz_index] / 1000.0; # is always zero... but Z is not... and can be used here, since absolute height does not matter
			t = np.zeros([3,1]);
			t[0] = vx[fr];
			t[1] = vy[fr];
			t[2] = vz[fr];
			t = t * snapshot_time_interval;
			speed = np.linalg.norm(np.array([vx[fr], vy[fr], vz[fr]]));
			
			# it is questionable that X is really necessary, one could not send parameters for X and optimize the R, ts so that single triangulations fit as good as possible with all image points:
			# also, R and t should be translated to a camera definition:
			X = getTriangulatedPoints(image_points1, image_points2, R, t, K);
			
			# like this it is impossible to determine what rotation to apply to the points
			# so should we rotate them immediately? Or include more info? Or add empty elements to the vector for non-matched points?
			for f in range(n_features):
				try:
					ii = indices.index(f);
					points_world[f].append(X[ii][:3]);
				except ValueError:
					points_world[f].append(np.array([]));

			# append rotation:
			Rotations.append(R);
			# scale the translation and world points, but don't rotate them yet.
			Translations.append(t);

		else:
			# 3D reconstruction without scale:
			(R, t, X, errors_per_point) = performStructureFromMotion(image_points1, image_points2, K, W, H);

			# determine the scale on the basis of the velocity:
			vx[fr] = sample['frames'][fr]['velocities'][vx_index];
			vy[fr] = sample['frames'][fr]['velocities'][vy_index];
			vz[fr] = sample['frames'][fr]['velocities'][vz_index];
			speed = np.linalg.norm(np.array([vx[fr], vy[fr], vz[fr]]));
			scale = snapshot_time_interval * speed;

			# append rotation:
			Rotations.append(R);
			# scale the translation and world points, but don't rotate them yet.
			Translations.append(scale * t);
			# points_world.append(scale * X);
			for m in range(n_matches):
				points_world[indices[m]].append(scale * X[m,:]);


	# 3) Perform some kind of bundle adjustment
	
	# Representation to optimize: let's start with something traditional to allow for straightforward
	# Levenberg-Marcquardt:
	# [R2, ..., Rm, t2, ..., tm, X1, Y1, Z1, ..., Xn, Yn, Zn]
	# Later we can think of a more compact representation with angles for the rotation matrices, direction
	# and speed for the translations, and inverse coordinates for X, Y, and Z.

	# First we need to transform the translations, rotations, and world points
	# to a common frame of reference, that of the first camera view.

	# the first elements in these vectors are already in the frame-of-reference of the first camera:
	pdb.set_trace();
	Rs = [Rotations[0]];
	Ts = [Translations[0]];
	pw = [points_world[0]];

	for fr in range(n_frames-2):
		matrix_ind = fr+1;
		# first determine the rotation up until the first camera of the current pair:
		Rotation_cam1 = np.eye(3);
		for ff in range(matrix_ind):
			Rotation_cam1 = np.dot(Rotation_cam1, Rs[ff]);
		# the translation is expressed with respect to the first camera of the current pair;
		translation_cam1 = np.dot(Rotation_cam1, Translations[matrix_ind]);
		Ts.append(translation_cam1);
		# the same goes for the world points:
		n_world_points = len(X);
		X_cam1 = [];
		for wp in range(n_world_points):
			# rotate the triangulated point if it was matched in the current views:
			if(len(points_world[matrix_ind][wp]) > 0):
				X_cam1.append(np.dot(Rotation_cam1, points_world[matrix_ind][wp]));
			else:
				X_cam1.append(np.array([]));
		pw.append(X_cam1);
		# then multiply the rotation with the rotation to the second camera of the pair:
		Rotation_cam1 = np.dot(Rotation_cam1, Rotations[matrix_ind]);
		# add it to the list of rotations:
		Rs = Rs.append(Rotation_cam1);
	
	# Here we choose one X per point. We can also seed an evolution-like algorithm with all possible triangulations.
	# First we determine what world point coordinates to start with:
	X = determineWorldPoints(pw); # it is not clear yet how to rotate each point, since now there is uncertainty on when each point was observed.
	
	# Then concatenate everything into a large vector:
	#genome = transformMatricesToGenome(Rs, Ts, X);
	# And optimize:
	# optimizeReconstructionGenome(genome, IPs1, IPs2, n_frames-1, n_world_points);
	# transform the genome back to matrices:
	#(Rs, Ts, X) = transformGenomeToMatrices(genome, n_cameras, n_world_points);

	# determine the genome on the above information:
	genome = constructGenome(phis, thetas, psis, Ts, n_features, X);
	
	# Get rotations, translations, X_est:
	(Rs, Ts, X_est) = evolveReconstruction('test', 2, n_features, IPs, 3.0, 10.0, K, genome);

	# Now we have all rotations and translations in the frame of camera 1, 
	# and the world points. We now want to know the distances to the points
	# in the final image:
	distances = np.array([0.0]*n_world_points);	
	for p in range(n_world_points):
		distances[p] = np.linalg.norm(X[p] - Ts[-1]);

	# distances with TTC estimates:
	distances_TTC = TTC_estimates * speed;
	
	return (distances, distances_TTC);
		

def isDrone1(device_string):
	if(device_string.find('ARDrone 2') == -1):
		return True;
	else:
		return False;

#def transformMatricesToGenome(Rs, Ts, X):
#	""" Transforms lists of rotation matrices, translation vectors, and world points to a vector that can be used for optimization.
#	"""
#	genome = [];
#	n_mats = len(Rs);
#	# Rotations:
#	for m in range(n_mats):
#		R = Rs[m].tolist();
#		n_rows = len(R);
#		for r in range(n_rows):
#			for c in range(3):
#				genome.append(R[r][c]);

#	# Translations:
#	for m in range(n_mats):
#		t = Ts[m].tolist();
#		for ti in range(3):
#			genome.append(t[ti]);

#	# World points:
#	n_world_points = len(X);
#	for p in range(n_world_points):
#		genome.append(X[p].tolist());

#	return genome;

#def transformGenomeToMatrices(genome, n_cameras, n_world_points):
#	""" Transforms the optimized genome to rotation matrices, translation vectors, and world points.
#	"""

#	# Get rotation matrices:
#	Rs = [];
#	nR = 9;
#	for cam in range(n_cameras):
#		part = genome[cam*nR:(cam+1)*nR];
#		R = np.zeros([3,3]);
#		index = 0;
#		for r in range(3):
#			for c in range(3):
#				R[r][c] = part[index];
#				index += 1;
#		Rs.append(R);
#	
#	# Get translation vectors:
#	start_ind = n_cameras * nR;
#	Ts = [];
#	nT = 3;
#	for cam in range(n_cameras):
#		part = genome[start_ind + cam * nT : start_ind + (cam+1) * nT];
#		t = np.zeros([3,1]);
#		for ti in range(nT):
#				t[ti] = part[ti];
#		Ts.append(t);

#	# Get the world points:
#	start_ind = start_ind + n_cameras * nT;
#	pw = [];
#	nX = 3;
#	for p in range(n_world_points):
#		part = genome[start_ind + p * nX : start_ind + (p+1) * nX];	
#		point = np.array(part);
#		pw.append(point);

#	return (Rs, Ts, pw);
	
	
def determineWorldPoints(pw):
	""" Given multiple estimates of a point in the world, return the median position
	"""

	X = [];
	n_estimates = len(pw);
	n_points = len(pw[0]);
	# per point, put all estimates in a vector and then make a final estimate:
	for p in range(n_points):
		Xests = np.array([0.0] * n_estimates);
		Yests = np.array([0.0] * n_estimates);
		Zests = np.array([0.0] * n_estimates);
		for e in range(n_estimates):
			Xests[e] = pw[e][p][0];
			Yests[e] = pw[e][p][1];
			Zests[e] = pw[e][p][2];
		# take the median as the final estimate:
		X.append(np.array([np.median(Xests), np.median(Yests), np.median(Zests)]));

	return X;
	
def investigateResponseValues(image_name):
	# extract keypoint features:
	(keypoints, descriptors, im2, im) = extractSURFfeaturesFromImage(image_name, IM_RESIZE=True);
	
	# put the response values in an array:
	n_keypoints = len(keypoints);
	responses = np.array([0.0] * n_keypoints);
	for kp in range(n_keypoints):
		responses[kp] = keypoints[kp].response;
		
	# show a histogram of the responses:
	pl.figure();
	pl.hist(responses, 30);
	pl.title('Response values ' + image_name);

def extractSURFfeaturesFromImage(image_name, IM_RESIZE=False, W=640, H=360):
	im2 = cv2.imread(image_name);
	
	#cv2.namedWindow("img", cv2.cv.CV_WINDOW_NORMAL)
	#cv2.imshow('img',im2)
	
	im = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	
	if(IM_RESIZE):
		im = cv2.resize(im, (W, H));
	
	#im = cv2.resize(im, (im.shape[1] / 2, im.shape[0] / 2))
	
	#surfDetector = cv2.SURF();
	#surfDetector.hessianThreshold = 2500;
	#surfDetector.nOctaves = 4;
	#surfDetector.nOctaveLayers = 2;
	#mask = np.ones(im.shape, dtype=np.uint8)
	#keypoints = surfDetector.detect(im, mask);
	
	##surfDetector = cv2.FeatureDetector_create("SURF")
	##keypoints = surfDetector.detect(im)
	
	# surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
	# (keypoints, descriptors) = surfDescriptorExtractor.compute(im,keypoints)
	
	hessianThreshold = 2500;
	nOctaves = 4;
	nOctaveLayers = 2;
	surfDetector = cv2.SURF(hessianThreshold, nOctaves, nOctaveLayers, True, False)
	keypoints, descriptors = surfDetector.detectAndCompute(im, None)
	
	return (keypoints, descriptors, im2, im);

def transformFeaturesToCVFormat(frame_features, H=360):
	kp_list = [];
	desc_list = [];
	for f in frame_features:
		# cv2.KeyPoint([x, y, _size[, _angle[, _response[, _octave[, _class_id]]]]])
		kp_list.append(cv2.KeyPoint(f['x'], H-f['y'], f['size'], f['angle'], f['response'], f['octave']));
		desc_list.append(np.array(f['descriptor']))
		
	return (kp_list, desc_list);

def drawFeature(plot, fpos, size, rotation, initTransform=Affine2D(), col='y'):
    vertices = ((0, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1))
    drawPairs = [(0, 1), (2, 3), (3, 4), (4, 5), (5, 2)]
    transform = Affine2D().scale(size / 2.0).rotate_deg(rotation).translate(*fpos)
    vertices = initTransform.transform(transform.transform(vertices))

    for i, j in drawPairs:
      plot.plot((vertices[i][0], vertices[j][0]), (vertices[i][1], vertices[j][1]), color=col)

def plotSURFFeatures(keypoints, plot, offset = 0):
	for k in keypoints:
		if(offset == 0):
			drawFeature(plot, k.pt, k.size, k.angle);
		else:
			pt = (k.pt[0]+offset, k.pt[1]);
			drawFeature(plot, pt, k.size, k.angle);

#def onclick(event):
#    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
#        event.button, event.x, event.y, event.xdata, event.ydata)

def investigateFeatureDistances(keypoints, descriptors, keypoints_db, descriptors_db, im):

	n_im = len(keypoints);
	n_db = len(keypoints_db);

	# make a distance matrix to be used later on:
	Distances = np.zeros((n_im, n_db));
	for ft1 in range(n_im):
		for ft2 in range(n_db):
			Distances[ft1, ft2] = np.linalg.norm(descriptors[ft1] - descriptors_db[ft2]);
	

	# make a big image:
	H = im.shape[0];
	W = im.shape[1];
	large_im = 255 * np.ones((H, 2*W, 3), np.uint8);
	large_im[0:H, 0:W, 0] = im; 
	large_im[0:H, 0:W, 1] = im; 
	large_im[0:H, 0:W, 2] = im; 
	
	matchFigure = figure(figsize=figaspect(0.5))
	matchFigure.suptitle("Image - image matches")
	ag = AxesGrid(matchFigure, nrows_ncols=[1, 1], rect=[0.05, 0.05, 0.9, 0.8])
	matchPlot = ag[0];
	matchPlot.hold(True)
	matchPlot.imshow(large_im);
		
	# plot the features with an orientation:
	plotSURFFeatures(keypoints, matchPlot);
	plotSURFFeatures(keypoints_db, matchPlot, offset = W);
	matchFigure.show()
	#cid = matchFigure.canvas.mpl_connect('button_press_event', onclick)
	coord = ginput(1);
	
	while len(coord) > 0:
		
		pl.close();
		matchFigure = figure(figsize=figaspect(0.5))
		matchFigure.suptitle("Image - image matches")
		ag = AxesGrid(matchFigure, nrows_ncols=[1, 1], rect=[0.05, 0.05, 0.9, 0.8])
		matchPlot = ag[0];
		matchPlot.hold(True)
		matchPlot.imshow(large_im);
	
		# get closest point:
		if(coord[0][0] > W):
			# database feature:
			x = coord[0][0] - float(W);
			y = coord[0][1];
			coord = np.array((x,y));
			# find closest database feature:
			distances = np.array([0.0]*n_db);
			for kp in range(n_db):
				distances[kp] = np.linalg.norm(coord - np.array(keypoints_db[kp].pt));
			closest_ind = np.argmin(distances);
			pt = (keypoints_db[closest_ind].pt[0]+W, keypoints_db[closest_ind].pt[1]);
			
			plotSURFFeatures(keypoints_db, matchPlot, offset = W);
			drawFeature(matchPlot, pt, keypoints_db[closest_ind].size, keypoints_db[closest_ind].angle, col='r');
			
			# redraw image features according to distance:
			min_dist = np.min(Distances[:, closest_ind]);
			max_dist = np.max(Distances[:, closest_ind] - min_dist);
			for kp in range(n_im):
				pt = (keypoints[kp].pt[0], keypoints[kp].pt[1]);
				if(kp != closest_ind):
					col2 = (((Distances[kp, closest_ind] - min_dist)/max_dist), ((Distances[kp, closest_ind] - min_dist)/max_dist), 0);
					drawFeature(matchPlot, pt, keypoints[kp].size, keypoints[kp].angle, col=col2);
				else:
					drawFeature(matchPlot, pt, keypoints[kp].size, keypoints[kp].angle, col='g');
					
			pt = (keypoints[closest_ind].pt[0], keypoints[closest_ind].pt[1]);
			drawFeature(matchPlot, pt, keypoints[closest_ind].size, keypoints[closest_ind].angle, col='g');
		else:
			# image feature:
			x = coord[0][0];
			y = coord[0][1];
			coord = np.array((x,y));
			# find closest database feature:
			distances = np.array([0.0]*n_im);
			for kp in range(n_im):
				distances[kp] = np.linalg.norm(coord - np.array(keypoints[kp].pt));
			closest_ind = np.argmin(distances);
			pt = (keypoints[closest_ind].pt[0], keypoints[closest_ind].pt[1]);
			
			plotSURFFeatures(keypoints, matchPlot);
			drawFeature(matchPlot, pt, keypoints[closest_ind].size, keypoints[closest_ind].angle, col='r');
			
			# redraw image features according to distance:
			min_dist = np.min(Distances[closest_ind, :]);
			max_dist = np.max(Distances[closest_ind, :] - min_dist);
			for kp in range(n_db):
				pt = (keypoints_db[kp].pt[0]+W, keypoints_db[kp].pt[1]);
				if(kp != closest_ind):
					col2 = (((Distances[closest_ind, kp] - min_dist)/max_dist), ((Distances[closest_ind, kp] - min_dist)/max_dist), 0);
					drawFeature(matchPlot, pt, keypoints_db[kp].size, keypoints_db[kp].angle, col=col2);
				else:
					drawFeature(matchPlot, pt, keypoints_db[kp].size, keypoints_db[kp].angle, col='g');
			
			pt = (keypoints_db[closest_ind].pt[0]+W, keypoints_db[closest_ind].pt[1]);
			drawFeature(matchPlot, pt, keypoints_db[closest_ind].size, keypoints_db[closest_ind].angle, col='g');

		#matchPlot.draw();
		#matchFigure.canvas.draw();
		matchFigure.show();
		
		# get new point:
		coord = ginput(1);
		
	pl.close();



def testExperimentalSetup(test_dir="../data_GDC", target_name="video_CocaCola", data_name="output_GDC.txt", DATABASE = True, VIDEO = True, histogram_sizes = False, INVESTIGATE_SINGLE_IMAGE=False):
	
	# the data will serve as a database for matching with the image(s):

	if(DATABASE):
		# read the data from the database:
		result = loadData(test_dir + "/" + data_name);
		# iterate over the data:
		n_samples = len(result);
	else:
		# get data from an image:
		(kp_data, desc_data, im2_data, im_data) = extractSURFfeaturesFromImage(test_dir + "/" + data_name, IM_RESIZE=True);
		result = [desc_data];
		n_samples = 1;

	# definition of a match:
	NN_THRESHOLD = 0.75;
	
	if(not(VIDEO)):
		# target is a single image:
		image_names = [test_dir + "/" + target_name];
	else:
		# get image names from directory:
		image_names = os.listdir(test_dir + "/" + target_name);
	
	# number of images:
	n_images = len(image_names);
	
	Distances = np.zeros([n_images, n_samples]);
	Matches = np.zeros([n_images, n_samples]);
	
	imn = 0;
	for iname in image_names:
	
		print 'Image %d' % imn
		
		if(VIDEO):
			image_name = test_dir + "/" + "/" + target_name + "/" + iname;
		else:
			image_name = test_dir + "/" + "/" + iname;
			
		(keypoints, descriptors, im2, im) = extractSURFfeaturesFromImage(image_name, IM_RESIZE=True);
		
		n_features_image = len(descriptors);
		# print 'number of features in image: %d' % n_features_image
		if(histogram_sizes):
			descriptor_magnitudes_image = np.array([0.0] * n_features_image);
			for fi in range(n_features_image):
				magn = np.linalg.norm(descriptors[fi]);
				#print 'fi = %d, magn = %f' % (fi, magn);
				descriptor_magnitudes_image[fi] = magn;
		
		sam = 0;
		for sample in result:
			if(DATABASE):
				# number of frames in sample:
				n_frames = len(sample['frames']);
			else:
				n_frames = 1;
			
			print 'Sample %d' % sam;
			
			for fr in range(n_frames):
			
				n_matches = 0;
			
				if(DATABASE):
					# define current frame:
					frame = sample['frames'][fr];
					n_features_frame = len(frame['features']['features']);
					# print 'number of features in frame %d = %d' % (fr, n_features_frame); 
				else:
					n_features_frame = len(sample);

				if(INVESTIGATE_SINGLE_IMAGE):
					(kp_list, desc_list) = transformFeaturesToCVFormat(frame['features']['features']);
					investigateFeatureDistances(keypoints, descriptors, kp_list, desc_list, im);

				if(histogram_sizes and DATABASE):
					# compare descriptor magnitude with the ones from the image:
					descriptor_magnitudes_frame = np.array([0.0] * n_features_frame);
					for ff in range(n_features_frame):
						descriptor_magnitudes_frame[ff] = np.linalg.norm(frame['features']['features'][ff]['descriptor']);
					
					# plot histograms descriptor magnitudes image / frame for comparison:
					pl.figure(facecolor='white', edgecolor='white');
					pl.hist([np.asarray(descriptor_magnitudes_image), np.asarray(descriptor_magnitudes_frame)], 30, color=[(0.2,0.2,0.2), (1.0,1.0,1.0)]);
				
				closest_distances = np.array([0.0] * n_features_image);
				for ft1 in range(n_features_image):
				
					step = int(round(n_features_image / 10));
					#if(ft1 % step == 0):
					#	print '.',

					# determine the distances to the features in the frame:
					distances = np.array([0.0] * n_features_frame);
					for ft2 in range(n_features_frame):
						if(DATABASE):
							distances[ft2] = np.linalg.norm(np.array(descriptors[ft1]) - np.array(frame['features']['features'][ft2]['descriptor']));
						else:
							distances[ft2] = np.linalg.norm(np.array(descriptors[ft1]) - np.array(sample[ft2]));
				
					# sort the distances:
					sindices = np.argsort(distances);
					
					# store the closest distance:
					closest_distances[ft1] = distances[sindices[0]];
					
					# the second nearest neighbor has to be sufficiently far for a match:
					if(distances[sindices[0]] / distances[sindices[1]] < NN_THRESHOLD):
						n_matches += 1;
				
				# gather statistics:
				#pdb.set_trace();
				print 'im = %d, sam = %d' % (imn, sam)
				Matches[imn][sam] += n_matches;
				Distances[imn][sam] += np.mean(closest_distances);
				
				print '\nFrame %d, number of matches = %d, average closest match (dist in feature space) = %f\n' % (fr, n_matches, np.mean(closest_distances));
			
			# to keep the average distance:
			Distances[imn][sam] /= n_frames;
			
			sam += 1;
		imn += 1;
	
	np.savetxt('Distances.txt', Distances);
	np.savetxt('Matches.txt', Matches);
	
	colors = [(1,0,0), (0,1,0), (0,0,1)];
	pl.figure();
	pl.hold = True;
	for sam in range(n_samples):
		pl.plot(Distances[:,sam], color=colors[mod(sam, 3)]);
		
		
def tSNEDatabase(test_dir="../data", data_name="output.txt", selectSubset=True, n_selected_samples = 10):
	"""Runs t-SNE low dimension embedding on the AstroDrone database """
	
	# load the database, and put the features in the right format:
	result = loadData(test_dir + "/" + data_name);
	
	if(selectSubset):
		# select a number of random samples:
		n_samples = len(result);
		rand_inds = np.random.permutation(range(n_samples));
		result = [result[i] for i in rand_inds[0:n_selected_samples].tolist()];
		
	# iterate over the data:
	n_samples = len(result);
	n_frames = len(result[0]['frames']);
	# X will contain the feature descriptor data:
	X = [];
	# the following arrays will contain feature properties that can be seen as labels.
	responses = [];
	sizes = [];
	orientations = [];
	ys = [];
	zs = [];
	distances = [];
	n_features = [];
	for sample in result:
		for f in range(n_frames):
			# get frame:
			frame = sample['frames'][f];
			
			# get position information for labelling:
			position = frame['position'];
			x = position[x_index];
			y = position[y_index];
			z = position[z_index];
			distance_to_marker = np.linalg.norm(position);
			nf = len(frame['features']['features']);
			
			# process features:
			for ft in frame['features']['features']:
				X.append(ft['descriptor']);
				responses.append(ft['response']);
				sizes.append(ft['size']);
				orientations.append(ft['angle']);
				ys.append(y);
				zs.append(z);
				distances.append(distance_to_marker);
				n_features.append(nf);
	
	# run t-SNE on the feature database:
	scipy.io.savemat('X.mat', mdict={'X': X});
	scipy.io.savemat('responses.mat', mdict={'responses': responses});
	scipy.io.savemat('orientations.mat', mdict={'orientations': orientations});
	scipy.io.savemat('sizes.mat', mdict={'sizes': sizes});
	scipy.io.savemat('distances.mat', mdict={'distances': distances});
	scipy.io.savemat('ys.mat', mdict={'ys': ys});
	scipy.io.savemat('zs.mat', mdict={'zs': zs});
	scipy.io.savemat('n_features.mat', mdict={'n_features': n_features});
	#np.savetxt('X.txt', X);
	#np.savetxt('responses.txt', responses);
	#np.savetxt('sizes.txt', sizes);
	#np.savetxt('orientations.txt', orientations);
	
	## transform X to a Math array:
	#XM = Math.array(X);
	#
	## perform t-SNE:
	## Y = calc_tsne(XM);
	#Y = tsne(XM, 2, 50, 5.0);
	#
	## plot the results:
	#pl.figure();
	#Plot.scatter(Y[:,0], Y[:,1], 20, responses);
	#pl.title('responses');
	#pl.figure();
	#Plot.scatter(Y[:,0], Y[:,1], 20, sizes);
	#pl.title('sizes');
	#pl.figure();
	#Plot.scatter(Y[:,0], Y[:,1], 20, orientations);
	#pl.title('orientations');

def evaluateDistances(test_dir="../data", data_name="output.txt", selectSubset=True):
	""" Uses 3D reconstruction to estimate distances per point and compares these with scale-based estimates.
	"""

	# read the data from the database:
	result = loadData(test_dir + "/" + data_name);
	
	if(selectSubset):
		# select a number of random samples:
		n_selected_samples = 10;
		n_samples = len(result);
		rand_inds = np.random.permutation(range(n_samples));
		result = [result[i] for i in rand_inds[0:n_selected_samples].tolist()];
		
	# iterate over the data:
	n_samples = len(result);
	n_frames = len(result[0]['frames']);

	D = []; D_TTC = [];
	for sample in result:
		# determine the distances to all features that persist through the entire sequence:
		(distances, distances_TTC) = getFeaturesWithDistance(sample);
		if(len(distances) > 0):
			D.append(distances.tolist());
			D_TTC.append(distances_TTC.tolist());
		
	# compare the distances:
	pl.figure();
	pl.plot(D, D_TTC, 'x');
		
def plotDatabaseStatistics(test_dir="../data", data_name="output.txt", selectSubset=True, filterData = False, analyze_TTC=False, storeKohonenHistograms=False, KohonenFile='Kohonen.txt'):
	""" Plots simple statistics from the database such as where the photos were taken, etc.
		If analyze_TTC = True, it also tracks features over multiple frames and assigns Time-To-Contacs to them.
		If selectSubset = True, only 10 samples from the database are processed.
	"""

	# read the data from the database:
	result = loadData(test_dir + "/" + data_name);
	
	# filter so that only trajectories going toward the marker remain:
	if(filterData):
		result = filterDataForApproachingSamples(result);
	
	if(selectSubset):
		# select a number of random samples:
		n_selected_samples = 10;
		n_samples = len(result);
		rand_inds = np.random.permutation(range(n_samples));
		result = [result[i] for i in rand_inds[0:n_selected_samples].tolist()];
		
	# iterate over the data:
	n_samples = len(result);
	n_frames = len(result[0]['frames']);
	
	# statistics to gather:
	n_data_points = n_samples * n_frames;
	distances_to_marker = np.array([0.0] * n_data_points);
	X = np.array([0.0] * n_data_points);
	Y = np.array([0.0] * n_data_points);
	Z = np.array([0.0] * n_data_points);
	speeds = np.array([0.0] * n_data_points);
	VX = np.array([0.0] * n_data_points);
	VY = np.array([0.0] * n_data_points);
	VZ = np.array([0.0] * n_data_points);
	pitch = np.array([0.0] * n_data_points);
	roll = np.array([0.0] * n_data_points);
	yaw = np.array([0.0] * n_data_points);
	responses = [];
	sizes = [];
	n_features_frame = [];
	
	# loop over all samples:
	dp = 0;
	sp = 0;
	
	if(analyze_TTC):
		TTC = np.array([0.0] * n_samples); #np.zeros([n_samples, n_frames-1]);
		NF = np.array([0.0] * n_samples); 
		GT_TTC = np.array([0.0] * n_samples); #np.zeros([n_samples, n_frames-1]);
		epsi = 1E-3;#1E-4 * np.ones([1, n_frames-1]);
		R = np.zeros([2,2]);
		# the following lists contain the information per individual feature:
		TTC_ests = [];
		FTS_descr = [];
		Dists_fts = [];
		if(storeKohonenHistograms):
			Dists_hists = [];
			KohonenHistograms = [];
			Kohonen = np.loadtxt(KohonenFile);
			n_clusters = len(Kohonen);
		
	drone1 = 0.0;
	drone2 = 0.0;
	
	marker_detected = 0.0;
	marker_not_detected = 0.0;
	
	for sample in result:
	
		device_version = sample['device_version'];
		if(isDrone1(device_version)):
			drone1 += 1.0;
		else:
			drone2 += 1.0;
	
		for f in range(n_frames):
			
			# get frame:
			frame = sample['frames'][f];
		
			if(frame['marker_detected']):
				marker_detected += 1.0;
			else:
				marker_not_detected += 1.0;
		
			# get distance to marker:
			position = frame['position'];
			X[dp] = position[x_index];
			Y[dp] = position[y_index];
			Z[dp] = position[z_index];
			distances_to_marker[dp] = np.linalg.norm(position);
			
			# get speed:
			velocities = frame['velocities'];
			VX[dp] = velocities[vx_index] / 1000.0;
			VY[dp] = velocities[vy_index] / 1000.0;
			VZ[dp] = velocities[vz_index] / 1000.0;
			speeds[dp] = np.linalg.norm(velocities) / 1000.0;
			
			angles = frame['euler_angles'];
			pitch[dp] = angles[pitch_index];
			roll[dp] = angles[roll_index];
			yaw[dp] = angles[yaw_index];
			
			n_features_frame.append(len(frame['features']['features']));
			
			# get statistics at feature level:
			# n_features = len(frame['features']['features']);
			for ft in frame['features']['features']:
				responses.append(ft['response']);
				sizes.append(ft['size']);
			
			dp += 1;
		
		
		if(analyze_TTC):
			# time to contact estimated with feature sizes:
			(TTC[sp], NF[sp], TTC_estimates, FTS_USED, ALL_FTS) = getMatchedFeatures(sample);
			TTC_ests = TTC_ests + TTC_estimates;
			n_fts = len(FTS_USED);
			for ft in range(n_fts):
				# we add the last descriptor:
				FTS_descr = FTS_descr + [FTS_USED[ft]['descriptors'][-1]];
				Dists_fts = Dists_fts + [TTC_estimates[ft] * 0.25 * VY[dp-1]];
				# Dists_fts = Dists_fts + [TTC[sp] * VY[dp-1]];
			
			# store a Kohonen histogram (bag of words) for the image:
			if(storeKohonenHistograms):
				n_all_fts = len(ALL_FTS);
				image_histogram = np.array([0.0] * n_clusters);
				distances = np.array([0.0] * n_clusters);
				for ft in range(n_all_fts):
					sample = np.array(ALL_FTS[ft]);
					# find closest cluster:
					for i in range(n_clusters):
						distances[i] = np.linalg.norm(Kohonen[i] - sample);
					min_ind = np.argmin(distances);
					image_histogram[min_ind] += 1;
				Dists_hists = Dists_hists + [TTC[sp] * VY[dp-1]];	
				KohonenHistograms = KohonenHistograms + [image_histogram];
			
			
			# first rotate the 2D body velocities to obtain world velocities:
			# heading was not logged well!
			#hd = np.radians(yaw[dp-1]);
			#R[0,0] = np.cos(hd);
			#R[1,1] = R[0,0];
			#R[0,1] = -np.sin(hd);
			#R[1,0] = - R[0,1];
			#v_body = np.zeros([2,1]);
			#v_body[0] = VX[dp-1];
			#v_body[1] = VY[dp-1];
			#v_world = np.dot(R, v_body);
			
			v_world = np.zeros([2,1]); 
			v_world[0] = VX[dp-1];
			v_world[1] = VY[dp-1];
			
			# assume a wall at Y = 0: 
			dist = Y[dp-1];
			vel = -v_world[1];
			
			# determine "ground truth" TTC:
			if(abs(vel) > epsi):
				GT_TTC[sp] = dist / vel;
			else:
				GT_TTC[sp] = 1 / epsi;
			
		sp += 1;
	
	# Parrot AR drone 1 / 2:
	# The slices will be ordered and plotted counter-clockwise.
	total = drone1 + drone2;
	perc1 = (drone1 / total) * 100.0;
	perc2 = (drone2 / total) * 100.0;
	labels = 'AR drone 1', 'AR drone 2'
	fracs = [perc1, perc2]
	#explode=(0, 0.05, 0, 0)
	#pie(fracs, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
	pl.figure(facecolor='white', edgecolor='white');
	#pl.pie(fracs, labels = labels, autopct='%1.1f%%', colors=((37.0/255.0,222.0/255.0,211.0/255.0), (37.0/255.0,222.0/255.0,37.0/255.0)), shadow=True);
	pl.pie(fracs, labels = labels, autopct='%1.1f%%', colors=((246.0/255.0,103.0/255.0,47.0/255.0), (246.0/255.0,183.0/255.0,47.0/255.0)), shadow=True);
	
	# number of features:
	pl.figure(facecolor='white', edgecolor='white');
	pl.hist(n_features_frame, 60,normed=True);
	pl.xlabel('Number of features')
	pl.ylabel('Number of occurrences');
	
	n_bins = 5;
	max_features = 125;
	(nft_hist, bin_edges) = np.histogram(n_features_frame, range=[0.0,float(max_features)], bins=n_bins, density=False);
	nft_hist = nft_hist.astype('float');
	nft_hist /= float(np.sum(nft_hist));
	fracs = nft_hist * 100;
	labels = [];
	for be in range(len(bin_edges)-1):
		labels += ['(%1.1f, %1.1f)' % (bin_edges[be], bin_edges[be+1])];
	pl.figure(facecolor='white', edgecolor='white');
	#pl.pie(fracs, labels = labels, autopct='%1.1f%%', colors=((37.0/255.0,222.0/255.0,211.0/255.0), (37.0/255.0,222.0/255.0,37.0/255.0)), shadow=True);
	my_norm = mpl.colors.Normalize(0, 1); # maps your data to the range [0, 1]
	my_cmap = mpl.cm.get_cmap('Greens'); # can pick your color map
	color_vals = np.cumsum(fracs) / 100.0;
	pl.pie(fracs, labels = labels, autopct='%1.1f%%', shadow=True, colors=my_cmap(my_norm(color_vals)));
	pl.title('Number of features per frame');
	
	# marker detection:
	total = marker_detected + marker_not_detected;
	perc1 = (marker_detected / total) * 100.0;
	perc2 = (marker_not_detected / total) * 100.0;
	labels = 'Marker detected', 'Marker not detected'
	fracs = [perc1, perc2]
	#explode=(0, 0.05, 0, 0)
	#pie(fracs, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
	pl.figure(facecolor='white', edgecolor='white');
	#pl.pie(fracs, labels = labels, autopct='%1.1f%%', colors=((37.0/255.0,222.0/255.0,211.0/255.0), (37.0/255.0,222.0/255.0,37.0/255.0)), shadow=True);
	pl.pie(fracs, labels = labels, autopct='%1.1f%%', colors=((62.0/255.0,229.0/255.0,40.0/255.0), (238.0/255.0,229.0/255.0,40.0/255.0)), shadow=True);
	
	n_bins = 5;
	(dist_hist, bin_edges) = np.histogram(distances_to_marker, range=[0.0,float(n_bins)], bins=n_bins, density=False);
	outliers = n_samples * n_frames - np.sum(dist_hist);
	ldh = dist_hist.tolist();
	ldh += [outliers];
	dist_hist = np.asarray(ldh);
	dist_hist = dist_hist.astype('float');
	dist_hist /= float(np.sum(dist_hist));
	fracs = dist_hist * 100;
	labels = [];
	for be in range(len(bin_edges)-1):
		labels += ['(%1.1f, %1.1f)' % (bin_edges[be], bin_edges[be+1])];
	labels += ['Other'];
	pl.figure(facecolor='white', edgecolor='white');
	#pl.pie(fracs, labels = labels, autopct='%1.1f%%', colors=((37.0/255.0,222.0/255.0,211.0/255.0), (37.0/255.0,222.0/255.0,37.0/255.0)), shadow=True);
	my_norm = mpl.colors.Normalize(0, 1); # maps your data to the range [0, 1]
	my_cmap = mpl.cm.get_cmap('Blues'); # can pick your color map
	color_vals = np.cumsum(fracs) / 100.0;
	pl.pie(fracs, labels = labels, autopct='%1.1f%%', shadow=True, colors=my_cmap(my_norm(color_vals)));
	pl.title('Estimated distance to marker');
	
	# show a histogram of the distances:
	pl.figure(facecolor='white', edgecolor='white');
	pl.hist(distances_to_marker, 60);
	pl.title('Distances at which photos are taken');
	
	pl.figure(facecolor='white', edgecolor='white');
	pl.plot(X, Y, 'x');
	pl.title('Photo positions');
	
	# show a histogram of the speeds:
	pl.figure(facecolor='white', edgecolor='white');
	pl.hist(speeds, 60);
	pl.title('Speeds which photos are taken');
	
	# show a histogram of the feature response values:
	pl.figure(facecolor='white', edgecolor='white');
	pl.hist(np.array(responses), 60);
	pl.title('Feature response value distribution in database');
	
	# show a histogram of the feature sizes:
	pl.figure(facecolor='white', edgecolor='white');
	pl.hist(np.array(sizes), 60);
	pl.title('Feature size distribution in database');
	
	# show a histogram of yaw angles:
	pl.figure(facecolor='white', edgecolor='white');
	pl.hist(yaw, 60);
	pl.title('Histogram of yaw angles');
	
	if(analyze_TTC):
	
		# save the matrix of descriptors with corresponding TTC values for further analysis:
		scipy.io.savemat('FTS_descr.mat', mdict={'FTS_descr': FTS_descr});
		scipy.io.savemat('TTC_ests.mat', mdict={'TTC_ests': TTC_ests});
		scipy.io.savemat('Dists_fts.mat', mdict={'Dists_fts': Dists_fts});
		np.savetxt('FTS_descr.txt', FTS_descr);
		np.savetxt('TTC_ests.txt', TTC_ests);
		np.savetxt('Dists_fts.txt', Dists_fts);
		if(storeKohonenHistograms):
			scipy.io.savemat('KohonenHistograms.mat', mdict={'KohonenHistograms': KohonenHistograms});
			np.savetxt('KohonenHistograms.txt', KohonenHistograms);
			scipy.io.savemat('Dists_hists.mat', mdict={'Dists_hists': Dists_hists});
			np.savetxt('Dists_hists.txt', Dists_hists);
		
		# plot TTC and GT_TTC in the same figure:
		pl.figure();
		pl.plot(TTC, GT_TTC, 'x');
		pl.hold=True;
		min_TTC = np.min(TTC);
		max_TTC = np.max(TTC);
		pl.plot([min_TTC, max_TTC], [min_TTC, max_TTC], color=(0.0,1.0,0.0));
		pl.title('TTC vs ground truth TTC')
	
		pl.figure();
		pl.hist(NF, 30);
		pl.title('Number of features used for estimate');
	
		## show the TTCs
		#pl.figure();
		#pl.hold = True;
		#time_steps = range(n_frames-1)
		#for s in range(n_samples):
		#	pl.plot(time_steps, TTC[s,:], color=(0.7,0.7,0.7));
		#pl.plot(time_steps, np.median(TTC, axis=0), color=(1.0,0.0,0.0), linewidth=2);
		#pl.hold=False;
		#pl.title('TTC estimates');
		#
		## ground truth:
		#pl.figure();
		#pl.hold = True;
		#for s in range(n_samples):
		#	pl.plot(time_steps, GT_TTC[s,:], color=(0.7,0.7,0.7));
		#pl.plot(time_steps, np.median(GT_TTC, axis=0), color=(0.0,0.0,1.0), linewidth=2);
		#pl.title('Ground truth TTC values');
		#pdb.set_trace();
		
def distanceVariationAnalysis():
	""" Select only trajectories of which we are pretty sure that they are well estimated.
		Then evaluate the number of features during an approach.
	"""
	data = readdata.loadData("../data/output.txt")
	filtered_data = filterDataForApproachingSamples(data);
	#plotFlownPaths(filtered_data);
	
	# plot for drone 2:
	extractFeaturesFromFile("../data/output.txt", resampledResolution=(640, 360));
	# plot for drone 1:
	extractFeaturesFromFile("../data/output.txt");
	
def plotFeaturePositions():
	""" Show image coordinates of features in the data set.
	"""
	
	data = readdata.loadData("../data/output.txt");
	showFeaturePositions(filter(lambda x: "Drone 1" in x["device_version"], data), title="AR Drone 1")
	#showFeaturePositions(filter(lambda x: "Drone 1" in x["device_version"], data), title="AR Drone 2")

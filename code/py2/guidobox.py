#!/usr/bin/env python3

# low-dimensional representation with t-sne:
import sys
sys.path.insert(0, './py2/tsne_python/')
sys.path.insert(0, './py2/calc_tsne/')
from tsne import *
from calc_tsne import *
import scipy.io

from readdata import *
#from clustering import *
import os
import pdb
import cv2
import numpy as np;
from matplotlib import pyplot as pl;
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid import AxesGrid

import pylab as Plot

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
		feature = {'sizes': [], 'descriptors': [], 'time_steps': [], 'n_not_observed': 0};
		FTS.append(feature);
		FTS[f]['sizes'].append( features[f]['size'] );
		FTS[f]['time_steps'].append( time_step );
		FTS[f]['descriptors'].append( features[f]['descriptor'] );
	
	return FTS;

def addMatchedFeature(FT, feature, time_step):
	# add the info on a matched feature
	FT['sizes'].append(feature['size']);
	FT['time_steps'].append(time_step);
	FT['descriptors'].append(feature['descriptor']);
	
	
# def getMatchedFeatures(sample):
#
# Given a sample from the database (consisting of 5 frames), matches features from one frame to the next.
#
# input:
# - sample from the database 
# - graphics: whether to show all kinds of plots
#
# output:
# - TTC
def getMatchedFeatures(sample, graphics=False):

	ttc_graphics = False;

	# nearest neighbor threshold for matching:
	NN_THRESHOLD = 0.75;
	
	# maximum number of time steps a feature can go unobserved:
	max_n_not_observed = 1;
	
	# minimum number of observations of a feature to be considered for the estimate:
	min_memory = 4;
	
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
	n_features_used_for_estimate = 0;
	for ft in range(n_features):
		memory_size = len(FTS[ft]['sizes'])
		memory_distribution[ft] = memory_size;
		if(memory_size >= min_memory):
			TTC_estimates.append(determineTTCLinearFit(FTS[ft]));
			n_features_used_for_estimate += 1;
			
	if(ttc_graphics):
		if(n_features > 0):
			pl.figure();
			pl.hist(memory_distribution);
			pl.title('mem dist')

		if(len(TTC_estimates) > 0):
			pl.figure();
			pl.hist(TTC_estimates);
			pl.title('TTC ests')
	
	if(len(TTC_estimates) > 0):
		TimeToContact = np.median(TTC_estimates);
	else:
		TimeToContact = 1E3;
		
	TimeToContact *= snapshot_time_interval;
	
	return (TimeToContact, n_features_used_for_estimate);
	

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
		TTC = sign(size_slope) * 1E3;
		
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
			x2 = keypoints2[sindices[0]].pt[0];
			y2 = keypoints2[sindices[0]].pt[1];
			#cv2.circle(img1, (int(x1), int(y1)), 4, (0.0,255.0,0.0), -1)
			#cv2.circle(img2, (int(x2), int(y2)), 4, (0.0,255.0,0.0), -1)
			cv2.line(vis, (int(x1), int(y1)), (int(x2+w1), int(y2)), (0.0, 255.0, 0.0), 1)
			#pl.plot([x1, x2], [y1,y2]);
	
	cv2.imshow('matches', vis)
	#cv2.namedWindow("img1", cv2.cv.CV_WINDOW_NORMAL)
	#cv2.imshow('img1',img1)
	#cv2.namedWindow("img2", cv2.cv.CV_WINDOW_NORMAL)
	#cv2.imshow('img2',img2)

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
	for sample in result:
		for f in range(n_frames):
			# get frame:
			frame = sample['frames'][f];
			
			# process features:
			for ft in frame['features']['features']:
				X.append(ft['descriptor']);
				responses.append(ft['response']);
				sizes.append(ft['size']);
				orientations.append(ft['angle']);
	
	# run t-SNE on the feature database:
	scipy.io.savemat('X.mat', mdict={'X': X});
	scipy.io.savemat('responses.mat', mdict={'responses': responses});
	scipy.io.savemat('orientations.mat', mdict={'orientations': orientations});
	scipy.io.savemat('sizes.mat', mdict={'sizes': sizes});
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
		
def plotDatabaseStatistics(test_dir="../data", data_name="output.txt", selectSubset=True, analyze_TTC=True):
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
	n_frames = size(result[0]['frames']);
	
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
	# loop over all samples:
	dp = 0;
	sp = 0;
	
	if(analyze_TTC):
		TTC = np.array([0.0] * n_samples); #np.zeros([n_samples, n_frames-1]);
		NF = np.array([0.0] * n_samples); 
		GT_TTC = np.array([0.0] * n_samples); #np.zeros([n_samples, n_frames-1]);
		epsi = 1E-3;#1E-4 * np.ones([1, n_frames-1]);
		R = np.zeros([2,2]);
		
	for sample in result:
	
		for f in range(n_frames):
			
			# get frame:
			frame = sample['frames'][f];
		
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
			
			# get statistics at feature level:
			# n_features = len(frame['features']['features']);
			for ft in frame['features']['features']:
				responses.append(ft['response']);
				sizes.append(ft['size']);
			
			dp += 1;
		
		
		if(analyze_TTC):
			# time to contact estimated with feature sizes:
			(TTC[sp], NF[sp]) = getMatchedFeatures(sample);
			
			# Ground truth TTC:
			
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
	
	# show a histogram of the distances:
	pl.figure();
	pl.hist(distances_to_marker, 60);
	pl.title('Distances at which photos are taken');
	
	pl.figure();
	pl.plot(X, Y, 'x');
	pl.title('Photo positions');
	
	# show a histogram of the speeds:
	pl.figure();
	pl.hist(speeds, 60);
	pl.title('Speeds which photos are taken');
	
	# show a histogram of the feature response values:
	pl.figure();
	pl.hist(np.array(responses), 60);
	pl.title('Feature response value distribution in database');
	
	# show a histogram of the feature sizes:
	pl.figure();
	pl.hist(np.array(sizes), 60);
	pl.title('Feature size distribution in database');
	
	# show a histogram of yaw angles:
	pl.figure();
	pl.hist(yaw, 60);
	pl.title('Histogram of yaw angles');
	
	if(analyze_TTC):
	
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
		
		
#!/usr/bin/env python3

from readdata import *
from clustering import *
import os
import pdb
import cv2
import numpy as np;
from matplotlib import pyplot as pl;

# size(result) -> total number of samples
# size(result[i]['frames']) == 5
# result[i]['frames'][j]
# result[i]['image_features']['flat_descriptor_values']
# result[i]['frames'][j]['features']['features'][fnr]
# result[i]['frames'][j]['features']['features'][fnr]['descriptor']

def getMatchedFeatures(sample):
	NN_THRESHOLD = 0.75;
	
	# number of frames in a stored sequence:
	n_frames = len(sample['frames']);
	
	# first show velocities, angles:
	
	roll_index = 0;
	yaw_index = 1;
	pitch_index = 2;
	vx_index = 0;
	vy_index = 2;
	x_index = 0;
	y_index = 1;
	z_index = 2;
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
	
	
	for fr in range(n_frames-1):
	
		# define current and next frame:
		frame1 = sample['frames'][fr];
		n_features1 = len(frame1['features']['features']);
		frame2 = sample['frames'][fr+1];
		n_features2 = len(frame2['features']['features']);
		
		pl.figure();
		pl.hold(True);
		for ft1 in range(n_features1):
			# determine the distances to the features in the second frame:
			distances = np.array([0.0] * n_features2);
			for ft2 in range(n_features2):
				distances[ft2] = np.linalg.norm(np.array(frame1['features']['features'][ft1]['descriptor']) - np.array(frame2['features']['features'][ft2]['descriptor']));
			
			# sort the distances:
			sindices = np.argsort(distances);
			
			# the second nearest neighbor has to be sufficiently far for a match:
			if(distances[sindices[0]] / distances[sindices[1]] < NN_THRESHOLD):
				# we have a match:
				x1 = frame1['features']['features'][ft1]['x'];
				y1 = frame1['features']['features'][ft1]['y'];
				x2 = frame2['features']['features'][sindices[0]]['x'];
				y2 = frame2['features']['features'][sindices[0]]['y'];
				pl.plot([x1, x2], [y1,y2]);
		
		pl.title('t = %d' % (fr));
		pl.show();
		

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
		if(distances[sindices[0]] / distances[sindices[1]] < NN_THRESHOLD):
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

def extractSURFfeaturesFromImage(image_name, IM_RESIZE=False, W=640, H=360):
	im2 = cv2.imread(image_name);
	
	#cv2.namedWindow("img", cv2.cv.CV_WINDOW_NORMAL)
	#cv2.imshow('img',im2)
	
	im = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	
	if(IM_RESIZE):
		im = cv2.resize(im, (W, H));
	
	#im = cv2.resize(im, (im.shape[1] / 2, im.shape[0] / 2))
	
	surfDetector = cv2.SURF();
	surfDetector.hessianThreshold = 2500;
	surfDetector.nOctaves = 4;
	surfDetector.nOctaveLayers = 2;
	mask = np.ones(im.shape, dtype=np.uint8)
	keypoints = surfDetector.detect(im, mask);
	
	#surfDetector = cv2.FeatureDetector_create("SURF")
	#keypoints = surfDetector.detect(im)
	
	surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
	(keypoints, descriptors) = surfDescriptorExtractor.compute(im,keypoints)
	return (keypoints, descriptors, im2, im);



def testExperimentalSetup(test_dir="../data_GDC", target_name="video_CocaCola", data_name="output_GDC.txt", VIDEO = True, histogram_sizes = False):
	
	# the data will serve as a database for matching with the image(s):

	# read the data:
	result = loadData(test_dir + "/" + data_name);

	# definition of a match:
	NN_THRESHOLD = 0.75;
	
	# iterate over the data:
	n_samples = len(result);


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
			
		(keypoints, descriptors, im2, im) = extractSURFfeaturesFromImage(image_name);
		
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
			# number of frames in sample:
			n_frames = len(sample['frames']);
			
			print 'Sample %d' % sam;
			
			for fr in range(n_frames):
			
				n_matches = 0;
			
				# define current frame:
				frame = sample['frames'][fr];
				n_features_frame = len(frame['features']['features']);
				# print 'number of features in frame %d = %d' % (fr, n_features_frame); 

				if(histogram_sizes):
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
						#pdb.set_trace();
						distances[ft2] = np.linalg.norm(np.array(descriptors[ft1]) - np.array(frame['features']['features'][ft2]['descriptor']));
						# print 'distances[%d] = %f' % (ft2, distances[ft2]);
				
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
	
	pdb.set_trace();
	colors = [(1,0,0), (0,1,0), (0,0,1)];
	pl.figure();
	pl.hold = True;
	for sam in range(n_samples):
		pl.plot(Distances[:][sam], color=colors[mod(sam, 3)]);
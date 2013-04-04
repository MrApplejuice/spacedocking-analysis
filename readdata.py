#!/usr/bin/env python3

import json
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

def loadData(filename):
  result = []
  f = open(filename, 'r')
  max_samples = 200;
  sample = 0;
  for line in f:
	if(sample < max_samples): 
		line = line.strip()
		if (len(line) > 0) and (line[0] != '#'):
		  result.append(json.loads(line))
		sample += 1;
  f.close()
  
  return result
  

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
		
  
def testDescriptor(image_name):
	# extract surf features:
	im2 = cv2.imread(image_name);
	im = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	surfDetector = cv2.FeatureDetector_create("SURF")
	surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
	keypoints = surfDetector.detect(im)
	(keypoints, descriptors) = surfDescriptorExtractor.compute(im,keypoints)
	pdb.set_trace();
  
  
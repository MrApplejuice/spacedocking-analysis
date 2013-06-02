import pdb
import cv2
import numpy as np

def testVisualOdometry(n_points=100):

	# location camera 1:
	l1 = np.zeros([3,1]);
	R1 = np.eye(3);
	rvec1 = cv2.Rodrigues(R1);
	rvec1 = rvec1[0];

	# translation vector
	t = np.zeros([3,1]);#np.random.rand(3,1);
	t[2] = 0;
	t[1] = 3;
	print 't = %f, %f, %f' % (t[0], t[1], t[2]);
	scale = np.linalg.norm(t);
	print 'scale = %f' % scale;
	l2 = l1 + t;

	# Rotation matrix:
	phi = 0.1 * np.pi;#0.001*(np.random.random(1)*2-1) * np.pi;
	theta = 0.0;#0.001*(np.random.random(1)*2-1) * np.pi;
	psi = 0.0;#0.001*(np.random.random(1)*2-1) * np.pi;

	R_phi = np.zeros([3,3]);
	R_phi[0,0] = 1;
	R_phi[1,1] = np.cos(phi);
	R_phi[1,2] = np.sin(phi);
	R_phi[2,1] = -np.sin(phi);
	R_phi[2,2] = np.cos(phi);

	R_theta = np.zeros([3,3]);
	R_theta[1,1] = 1;
	R_theta[0,0] = np.cos(phi);
	R_theta[2,0] = -np.sin(phi);
	R_theta[0,2] = np.sin(phi);
	R_theta[2,2] = np.cos(phi);

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
	distance = 8;
	transl = np.zeros([1,3]);
	transl[0,2] = distance;
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

	# determine the rotation and translation between the two views:
	(R21, R22, t21, t22) = determineTransformation(image_points1, image_points2, K, W, H);
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

	scales = np.array([0.0] * n_points);
	for i in range(n_points):
		scales[i] = X_est[i][0] / points_world[i][0];


	print 't_est = %f, %f, %f' % (t2_est[0], t2_est[1], t2_est[2]);
	print 'R = '
	printRotationMatrix(R2_est);
	print 'Scale = %f, Mean scale = %f' % (scale, np.mean(scales));


	# now we have R2, t2, and X, which we return:
	return (R2_est, t2_est, X_est);	

def printRotationMatrix(R):

		for row in range(3):
			print '%f, %f, %f' % (R[row, 0], R[row, 1], R[row, 2])

	

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

def determineTransformation(points1, points2, K, W, H):

	# find fundamental matrix:
	# we need to pass normal image points:
	(F, inliers) = cv2.findFundamentalMat(points1, points2);
	#res = np.dot(M1, np.dot(F, M2.transpose()));
	
	
	# extract essential matrix:
	E = np.dot(K.transpose(), np.dot(F, K));

	# extract R and t:
	(U, Sigma, VT) = np.linalg.svd(E);
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


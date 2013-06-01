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
	t[1] = 1;
	print 't = %f, %f, %f' % (t[0], t[1], t[2]);
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
	for row in range(3):
		print '%f, %f, %f' % (R2[row, 0], R2[row, 1], R2[row, 2])

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
	# (R, t) = determineTransformation(image_points1, image_points2, K, W, H);

	# reproject the points into the 3D-world:

	# P1 is at the origin
	P1 = np.zeros([3, 4]);
	P1[:3,:3] = np.eye(3);

	# P2 is rotated and translated with respect to camera 1:
	P2 = np.zeros([3, 4]);
	P2[:3,:3] = R2;
	pdb.set_trace();
	P2[:,3] = t.transpose();

	ip1 = cv2.convertPointsToHomogeneous(image_points1);
	ip2 = cv2.convertPointsToHomogeneous(image_points2);

	pdb.set_trace();

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
	pdb.set_trace();
	R = R1; 
	t = t1;

	return (R, t)

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


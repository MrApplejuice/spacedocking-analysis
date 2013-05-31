import pdb
import cv2
import numpy as np

def testVisualOdometry(n_points=30):

	# location camera 1:
	l1 = np.zeros([3,1]);
	R1 = np.eye(3);
	rvec1 = cv2.Rodrigues(R1);
	rvec1 = rvec1[0];

	# translation vector
	t = np.random.rand(3,1)* 0.001;
	l2 = l1 + t;

	# Rotation matrix:
	phi = 0.001*(np.random.random(1)*2-1) * np.pi;
	theta = 0.001*(np.random.random(1)*2-1) * np.pi;
	psi = 0.001*(np.random.random(1)*2-1) * np.pi;

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

	rvec2 = cv2.Rodrigues(R2);
	rvec2 = rvec2[0];

	# create X, Y, Z points:
	size = 3;
	distance = 4;
	transl = np.zeros([1,3]);
	transl[0,2] = distance;
	points_world = np.zeros([n_points, 3]);
	for p in range(n_points):
		points_world[p, :] = size * (np.random.rand(1,3)*2-np.ones([1,3])) + transl;

	# camera calibration matrix:
	K = np.zeros([3,3]);
	K[0,0] = 320.0;
	K[1,1] = 240.0;
	K[2,2] = 1.0;
	
	distCoeffs = np.zeros([4]); # no clue what this means

	pdb.set_trace();
	result = cv2.projectPoints(points_world, rvec1, np.array([0.0]*3), K, distCoeffs);	
	image_points1 = result[0];

	result = cv2.projectPoints(points_world, rvec2, t, K, distCoeffs);
	image_points2 = result[0];


def determineEssentialMatrix():
	print 'a';

#cvFindFundamentalMat(const CvMat* points1, const CvMat* points2, CvMat* fundamentalMatrix, int method=CV_FM_RANSAC, double param1=1., double param2=0.99, CvMat* status=NULL)
#cvDecomposeProjectionMatrix(const CvMat *projMatrix, CvMat *cameraMatrix, CvMat *rotMatrix, CvMat *transVect, CvMat *rotMatrX=NULL, CvMat *rotMatrY=NULL, CvMat *rotMatrZ=NULL, CvPoint3D64f *eulerAngles=NULL)
# void cvProjectPoints2(const CvMat* objectPoints, const CvMat* rvec, const CvMat* tvec, const CvMat* cameraMatrix, const CvMat* distCoeffs, CvMat* imagePoints, CvMat* dpdrot=NULL, CvMat* dpdt=NULL, CvMat* dpdf=NULL, CvMat* dpdc=NULL, CvMat* dpddist=NULL)


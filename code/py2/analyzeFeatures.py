from pylab import *
import cv, cv2
import re
import pylab as pl
import sys

def extractSURFfeaturesFromImage(image, hessianThreshold=2500, nOctaves=4, nOctaveLayers=2):
  im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  #Legacy version only comput
  #keypoints, descriptors = cv.ExtractSURF(cv.fromarray(im), None, cv.CreateMemStorage(), (1, hessianThreshold, nOctaves, nOctaveLayers))
  #Result style for keypoints: 
  #  (xy-tuple, class_id, size, angle, hessianResponse)
  
  surfDetector = cv2.SURF(hessianThreshold, nOctaves, nOctaveLayers, True, False)
  keypoints, descriptors = surfDetector.detectAndCompute(im, None)

  if len(keypoints) == 0:
    return []
  else:
    return zip([(k.pt, k.class_id, k.size, k.angle, k.response) for k in keypoints], descriptors, keypoints)

def extractFeaturesFromFile(filename, resampledResolution=(320, 240), showLiveImage=True):
  cv2.namedWindow("video")
  
  targetFilename = None
  if type(showLiveImage) is str:
    targetFilename = showLiveImage
    showLiveImage = False
  
  if re.match('^.*\\.avi$', filename):
    video = cv2.VideoCapture(filename)

    if showLiveImage:
      pl.ion()

    fig = pl.figure()
    plt1 = fig.add_subplot(3, 1, 1)
    plt2 = fig.add_subplot(3, 1, 2)
    plt3 = fig.add_subplot(3, 1, 3)
    
    result = []
    retval = True
    frameCounter = 0
    while retval:
      retval, frame = video.read()
      if retval:
        frame = cv2.resize(frame, resampledResolution)
        matchData = extractSURFfeaturesFromImage(frame)
        result.append(matchData)
        
        cv2.drawKeypoints(frame, [s[2] for s in matchData], frame, (128, 128, 128), 4)
        
        if len(result) > 1:
          plt1.plot(range(len(result) - 2, len(result)), list(map(len, result[-2:])), '-k')
          plt2.plot(range(len(result) - 2, len(result)), list(map(sum, [[x[0][4] for x in l] for l in result[-2:]])), '-k')
          plt3.plot(range(len(result) - 2, len(result)), map(lambda x, y: x / y, list(map(sum, [[x[0][4] for x in l] for l in result[-2:]])), list(map(len, result[-2:]))), '-k')
          
          plt1.set_xlim(0, frameCounter)
          plt2.set_xlim(0, frameCounter)
          plt3.set_xlim(0, frameCounter)
        if showLiveImage:
          draw()
          
          cv2.imshow("video", frame)
          cv2.waitKey(10)
      
      frameCounter += 1

    if showLiveImage:
      pl.ioff()
      pl.show()
      
    if not targetFilename is None:
      fig.savefig(targetFilename, dpi=180)
  else:
    raise Exception("Unsupported filename")

def doStuff():
  #data = readdata.loadData("data_GDC/output_GDC.txt")
  #data = readdata.loadData("../data/data-2013.06.11-thesis.txt")
  
  resolutions = ((320, 240), (640, 360))
  for filename in sys.argv[1:]:
    for resolution in resolutions:
      print "Processing", filename, "at", "x".join(map(str, resolution))
      features = extractFeaturesFromFile(filename, resampledResolution=resolution, showLiveImage=filename + "-" + "x".join(map(str, resolution)) + ".png")

doStuff()

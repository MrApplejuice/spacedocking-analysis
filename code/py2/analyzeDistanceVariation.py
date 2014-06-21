import sys
import os

sys.path += [os.path.join(os.path.realpath("."), os.path.dirname(sys.argv[0]), p) for p in ["../universal"]]

from pylab import *
#import cv, cv2
import cv2
import random
import re
import pylab as pl
import readdata
from flightpaths import plotFlownPaths
#import numpy as np

class TriplePlot:
  def __init__(self):
    self.fig = pl.figure(facecolor='white', edgecolor='white')
    self.plt1 = self.fig.add_subplot(1, 1, 1);
    #self.plt1 = self.fig.add_subplot(2, 1, 1)
    #self.plt2 = self.fig.add_subplot(2, 1, 2)
    #self.plt3 = self.fig.add_subplot(3, 1, 3)
    
    self.plt1.hold(True)
    #self.plt2.hold(True)
    #self.plt3.hold(True)
    
    self.plt1.set_ylabel("Number of features [-]", fontsize=16)
    self.plt1.set_xlabel("Distance to marker [m]", fontsize=16)
    #self.plt2.set_ylabel("Summed feature strengths")
    #self.plt3.ylabel("")

    self.currentMaxCounter = 0
    
    self.prevData = None

  def record(self, frameCounter, data, pair=True):
    self.currentMaxCounter = max(self.currentMaxCounter, frameCounter)
    
    if not pair:
      self.prevData = None
    
    featureCount = len(data)
    summedFeatureStrength = sum([x[0][4] for x in data])
    
    if featureCount > 0:
      if not self.prevData is None:
        self.plt1.plot([self.prevData["frame counter"], frameCounter], [self.prevData["feature count"], featureCount], '-k')
        #self.plt2.plot([self.prevData["frame counter"], frameCounter], [self.prevData["summed feature strength"], summedFeatureStrength], '-k')
        #self.plt3.plot([self.prevData["frame counter"], frameCounter], map(lambda x, y: x / y, [self.prevData["summed feature strength"], summedFeatureStrength], [self.prevData["feature count"], featureCount]), '-k')

      self.plt1.plot([frameCounter], [featureCount], 'k') #ok
      #self.plt2.plot([frameCounter], [summedFeatureStrength], 'xk')
      #self.plt3.plot([frameCounter], [summedFeatureStrength / featureCount], 'xk')
        
    self.plt1.set_xlim(0, self.currentMaxCounter)
    #self.plt2.set_xlim(0, self.currentMaxCounter)
    #self.plt3.set_xlim(0, self.currentMaxCounter)
      
    if featureCount == 0:
      self.prevData = None
    else:
      self.prevData = {"feature count": featureCount, "summed feature strength": summedFeatureStrength, "frame counter": frameCounter}

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

def filterDataForApproachingSamples(data):
  def getFlightPath(sample):
    return [pl.array(frame["position"][:2]) for frame in sample["frames"]]
    
  filteredData = data

  # filter based on marker detection
  filteredData = filter(lambda x: any([f["marker_detected"] for f in x["frames"]]), filteredData)

  # filter data by flight direction and flight distance
  def checkFlightLength(sample):
    path = getFlightPath(sample)
    length = 0.0
    for i in range(len(path) - 1):
      length += pl.norm(path[i + 1] - path[i])
    return length > 0.3 # Must have flown more than 30 cm total
    
  def checkFlightDirection(sample):
    path = getFlightPath(sample)
    
    if path[0][0] >= 0:
      return False # May not start behind marker
    
    if path[-1][0] >= 0:
      return False # May not start behind marker

    direction = path[-1] - path[0]
    
    if direction[0] > 0:
      projectedYDeviation = path[0][1] - direction[1] * path[0][0] / direction[0]
      return pl.fabs(projectedYDeviation) < 0.5 # Only if aiming at +-50 cm area around marker
    return False

    #direction = direction / pl.norm(direction)
    #return pl.arccos(direction[0]) < pl.radians(30)
    
  filteredData = filter(lambda x: checkFlightLength(x) and checkFlightDirection(x), filteredData)
  
  # filter on the basis of absolute distance to marker
  def checkDistanceToMarker(sample):
    path = getFlightPath(sample)
    
    return all(map(lambda x: pl.norm(x) < 5.0, path)) # All flight path points must be < 5m away from 0,0
  
  filteredData = filter(lambda x: checkDistanceToMarker(x), filteredData)
  
  return filteredData

def calculateCorrelationStatistics(data):
  def bootstrapConfidenceInterval(sample, count=1000, ratio=0.05):
    sampledQuantities = [0] * count
    for i in range(count):
      bs_sample = [random.choice(sample) for s in sample]
      sampledQuantities[i] = pl.mean(bs_sample)
    sampledQuantities.sort()
    return sampledQuantities[count * 500 / 1000], \
            (sampledQuantities[count * int(round(1000 * ratio / 2)) / 1000], sampledQuantities[count * int(round(1000 * (1 - ratio / 2))) / 1000]),\
            (sampledQuantities[count * int(round(1000 * ratio)) / 1000], sampledQuantities[count * int(round(1000 * (1 - ratio))) / 1000])

  def computeSampleSlope(sample, yfunc):
    xs = [pl.norm(frame["position"][:2]) for frame in sample["frames"]]
    ys = [yfunc(frame) for frame in sample["frames"]]
    covarianceMat = pl.cov(xs, ys)
    return covarianceMat[1,0] / covarianceMat[0,0]
    
  def zeroedDivide(x, y):
    if y == 0:
      return 0
    return x / y

  def bootstrapConfidenceIntervalPercentage(sample, count=1000, ratio=0.05):
    sampledQuantities = [0] * count
    for i in range(count):
      bs_sample = [random.choice(sample) for s in sample]; positive_inds = [x for x in bs_sample if x >= 0];
      sampledQuantities[i] = (100.0 * len(positive_inds) / len(sample));
    sampledQuantities.sort()
    return sampledQuantities[count * 500 / 1000], \
            (sampledQuantities[count * int(round(1000 * ratio / 2)) / 1000], sampledQuantities[count * int(round(1000 * (1 - ratio / 2))) / 1000]),\
            (sampledQuantities[count * int(round(1000 * ratio)) / 1000], sampledQuantities[count * int(round(1000 * (1 - ratio))) / 1000])

  featureCountSlopes = [computeSampleSlope(s, lambda x: len(x["features"]["features"])) for s in data]
  featureResponseStrengthSlopes = [computeSampleSlope(s, lambda x: sum([f["response"] for f in x["features"]["features"]])) for s in data]
  averageFeatureResponseStrengthSlopes = [computeSampleSlope(s, lambda x: zeroedDivide(sum([f["response"] for f in x["features"]["features"]]), len(x["features"]["features"]))) for s in data]
  
  pl.figure(facecolor='white', edgecolor='white');
  pl.hist(featureCountSlopes, 30, normed=True);
  pl.xlabel('Slope');
  pl.show();
  positive_inds = [x for x in featureCountSlopes if x > 0];
  zero_inds = [x for x in featureCountSlopes if x == 0];
  print 'Percentage positive slopes: %f, percentage 0: %f' % (100.0 * len(positive_inds) / len(featureCountSlopes), 100.0 * len(zero_inds) / len(featureCountSlopes));
  print 'Percentage of slopes >= 0: %f' % (100.0 * (len(positive_inds) + len(zero_inds)) / len(featureCountSlopes))
  
  mean, tsconfidenceInterval, osconfidenceInterval = bootstrapConfidenceInterval(featureCountSlopes, count=100000)
  print "Feature count slope data, mean, one-sided confidence interval, two-sided confidence interval: ", mean, osconfidenceInterval, tsconfidenceInterval
  
  mean, tsconfidenceInterval, osconfidenceInterval = bootstrapConfidenceIntervalPercentage(featureCountSlopes, count=100000)
  print "Feature count slope data, percentage test: ", mean, osconfidenceInterval, tsconfidenceInterval

  mean, tsconfidenceInterval, osconfidenceInterval = bootstrapConfidenceInterval(featureResponseStrengthSlopes, count=100000)
  print "Total feature response strength slope data: ", mean, osconfidenceInterval, tsconfidenceInterval

  mean, tsconfidenceInterval, osconfidenceInterval = bootstrapConfidenceInterval(averageFeatureResponseStrengthSlopes, count=100000)
  print "Average feature response strength slope data: ", mean, osconfidenceInterval, tsconfidenceInterval

def getFilteredData(data, deviceString = "ARDrone 2"):
	filteredData = data;
	print "Read", len(filteredData), "samples";
	# filter data by device version
	deviceString = deviceString.lower();
	filteredData = filter(lambda x: deviceString in x["device_version"].lower(), filteredData);
	print "After device filtering:", len(filteredData);
	return filteredData;

def extractFeaturesFromFile(filename, resampledResolution=(320, 240), showLiveImage=True):
  cv2.namedWindow("video")
  
  targetFilename = None
  waitForEndKey = True
  if type(showLiveImage) is str:
    targetFilename = showLiveImage
    showLiveImage = True
    waitForEndKey = False
  
  if re.match('^.*\\.avi$', filename):
    video = cv2.VideoCapture(filename)

    if showLiveImage:
      pl.ion()

    triPlot = TriplePlot()
    
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
          triPlot.record(frameCounter, matchData)
        if showLiveImage:
          draw()
          
          cv2.imshow("video", frame)
          cv2.waitKey(10)
      
      frameCounter += 1

    if showLiveImage:
      pl.ioff()
      if waitForEndKey:
        pl.show()
      
    if not targetFilename is None:
      triPlot.fig.savefig(targetFilename, dpi=180)
  elif re.match('^.*\\.txt$', filename):
    data = readdata.loadData(filename)
    filteredData = data
    print "Read", len(filteredData), "samples"
    
    # filter data by device version
    if resampledResolution == (320, 240):
      deviceString = "ARDrone 1"
    else:
      deviceString = "ARDrone 2"
    
    deviceString = deviceString.lower()
    filteredData = filter(lambda x: deviceString in x["device_version"].lower(), filteredData)
    print "After device filtering:", len(filteredData)

    filteredData = filterDataForApproachingSamples(filteredData)
    print "Final used dataset:", len(filteredData)

    calculateCorrelationStatistics(filteredData)

    plotFlownPaths(filteredData, doShow=targetFilename)
    
    triPlot = TriplePlot()
    for sample in filteredData:
      first = True
      for frame in sample["frames"]:
        triPlot.record(pl.norm(frame["position"][:2]), [(((feature["x"], feature["y"]), -1, feature["size"], feature["angle"], feature["response"]),) for feature in frame["features"]["features"]], pair=not first)
        first = False

    triPlot.fig.show()

    if showLiveImage:
      if waitForEndKey:
        pl.show()

    if not targetFilename is None:
      triPlot.fig.savefig(targetFilename, dpi=180)
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

#doStuff()



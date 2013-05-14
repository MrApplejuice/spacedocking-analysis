from pylab import *
from mpl_toolkits.axes_grid import AxesGrid
from matplotlib.transforms import Affine2D
import cv, cv2
import readdata
import re
import pyopencl as cl

def generateOpenCLContext():
  result = None
  
  allDevices = []
  for platform in cl.get_platforms():
    try:
      allDevices += platform.get_devices(cl.device_type.GPU)
    except:
      pass
      
  if len(allDevices) <= 0:
    for platform in cl.get_platforms():
      try:
        allDevices += platform.get_devices(cl.device_type.ALL)
      except:
        pass
        
  if len(allDevices) > 0:
    result = cl.Context(allDevices)
  
  return result

def cvLoadImage(filename):
  im2 = cv2.imread(filename);
  return im2

def extractSURFfeaturesFromImage(image, hessianThreshold=2500, nOctaves=4, nOctaveLayers=2):
  im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  #Legacy version only comput
  #keypoints, descriptors = cv.ExtractSURF(cv.fromarray(im), None, cv.CreateMemStorage(), (1, hessianThreshold, nOctaves, nOctaveLayers))
  #Result style for keypoints: 
  #  (xy-tuple, class_id, size, angle, hessianResponse)
  
  surfDetector = cv2.SURF(hessianThreshold, nOctaves, nOctaveLayers, True, False)
  keypoints, descriptors = surfDetector.detectAndCompute(im, None)
  return zip([(k.pt, k.class_id, k.size, k.angle, k.response) for k in keypoints], descriptors)

def normDescriptor(descriptor):
  descriptor = array(descriptor)
  return descriptor / norm(descriptor)

def matchDescriptors(desc, descList, listKeyFunction=lambda x: x, distanceFunction=dot):
  distances = [None] * len(descList)
  for i, d in enumerate(descList):
    distances[i] = (distanceFunction(desc, listKeyFunction(d)), i)
  
  distances.sort(key=lambda x: x[0])
  
  return (distances[0][1], distances[0][0] / distances[1][0], distances[0][0], distances[1][0]) 

def bestFirstMatcher(descriptors1, descriptors2, maxMatches=1, distanceFunction=dot, differenceFactorThreshold=0.33):
  """
    Horrible version of a best first matcher... will not find the best among all pairs
    but only the best match between stimuli sequentially... fast but sloppy
  """
  
  flip = False
  if len(descriptors1) > len(descriptors2):
    flip = True
    descriptors1, descriptors2 = descriptors2, descriptors1
  
  class CompareDescriptor:
    def __init__(self, i, desc):
      self.index = i
      self.desc = desc
      self.matches = 0
  
  descriptors2 = [CompareDescriptor(i, d) for i, d in enumerate(descriptors2)]
  
  result_pairs = []
  for desc1_i, desc1 in enumerate(descriptors1):
    matchedBestDistanceInfo = matchDescriptors(desc1, descriptors2, listKeyFunction=lambda x: x.desc, distanceFunction=distanceFunction)
    if matchedBestDistanceInfo[1] < differenceFactorThreshold:
      # Add matched pair and remove old descriptor2 (best first match, no feature is matched twice)
      result_pairs.append((desc1_i, descriptors2[matchedBestDistanceInfo[0]].index))
      descriptors2[matchedBestDistanceInfo[0]].matches += 1
      if descriptors2[matchedBestDistanceInfo[0]].matches >= maxMatches:
        descriptors2.pop(matchedBestDistanceInfo[0])
    else:
      pass # No conclusive data for result pair

  if flip:
    result_pairs = [(j, i) for i, j in result_pairs]
    
  return result_pairs

def euclid(x, y):
  return norm(array(x) - array(y))

def dbFeaturesToData(features):
  return [(((f['x'], f['y']), f['size'], f['angle']), f['descriptor']) for f in features]
  
def selectDbSample(data, sample, frame=None):
  if not frame is None:
    return selectDbSample(data, sample)['frames'][frame]
  else:
    return data[sample]

def plotFile(fn, plotType='x'):
	f = open(fn, 'r')
	values = [[float(s) for s in re.split('\\s+', l.strip())] for l in f]
	f.close()

	maxValueCount = max(map(len, values))

	fig = figure()
	plot = fig.add_subplot(1, 1, 1)
	plot.hold(True)

	colors = "bgrcy"
	for vi in range(maxValueCount):	
		xy = [(i + 1, v[vi]) for i, v in enumerate(values) if vi < len(v)]
		plot.plot(map(lambda x: x[0], xy), map(lambda x: x[1], xy), plotType + colors[vi % len(colors)])

	fig.show()

###################################################################
# Test - Trying to match data aginst features computed from images
###################################################################

def matchImageWithData():
  data = readdata.loadData("data_GDC/output_GDC.txt")
  img = cvLoadImage("data_GDC/IMG_1062.JPG")
  img_dimensions = [len(img[0]), len(img)]
  img = cv2.resize(img, (640, 640 * img_dimensions[1] / img_dimensions[0]))
  #img = cv2.flip(img, flipCode=0)
  img_features = extractSURFfeaturesFromImage(img)#, nOctaves=16, nOctaveLayers=4)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Make a "normal" RGB image
  img_dimensions = [len(img[0]), len(img)]
  
  print "Number of image features:", len(img_features)

  for sample_i, sample in enumerate(data):
    averageFitHistory = []
    print "Sample", sample_i + 1
    for frame_i, frame in enumerate(sample['frames']):
      frame_desc = []
      print "Frame", frame_i + 1
      for feature in frame["features"]["features"]:
        frame_desc.append(array(feature["descriptor"]))

      distanceFunction = lambda x, y: arccos(dot(x, y) / norm(x) / norm(y))
      matchingPairs = bestFirstMatcher([x[1] for x in img_features], frame_desc, maxMatches=1, distanceFunction=distanceFunction, differenceFactorThreshold=0.85)
      
      fit = None
      print "Matched points:", len(matchingPairs)
      
      if len(matchingPairs) >= 5:
        src_points = array([list(img_features[i][0][0]) for i, j in matchingPairs])
        dest_points = array([[frame["features"]["features"][j]["x"], frame["features"]["features"][j]["y"]] for i, j in matchingPairs])

        homographyMat = cv.CreateMat(3, 3, cv.CV_32F)
        mask = cv.CreateMat(len(matchingPairs), 1, cv.CV_8U)
        cv.FindHomography(cv.fromarray(dest_points), cv.fromarray(src_points), homographyMat, method=cv.CV_RANSAC, ransacReprojThreshold=max(img_dimensions[0], img_dimensions[1]) * 0.1, status=mask)
        homographyMat = matrix(homographyMat)
        mask = array(matrix(mask).T)[0]

        print "Filtered matched points:", sum(mask)
        if sum(mask) < 5:
          print "(Not enough points contribute to the homography for a succesful matching)"
        else:
          matchFigure = figure()
          matchFigure.suptitle("Sample " + str(sample_i + 1) + " Frame " + str(frame_i + 1))
          matchPlot = matchFigure.add_subplot(1, 1, 1)
          
          matchPlot.hold(True)
          
          matchPlot.imshow(img)
          
          if len(matchingPairs) > 0:
            fit = sum([distanceFunction(img_features[i][1], frame_desc[j]) for i, j in matchingPairs]) / len(matchingPairs)
            
          homo_dest_points = [array((homographyMat * vstack((matrix(p).T, [1]))).T)[0] for p in dest_points]
          homo_dest_points = [p[:2] / p[2] for p in homo_dest_points]
          
          print "distance without homography =", sum([norm(i - j) for i, j in zip(src_points, dest_points)])
          print "distance with homography    =", sum([norm(i - j) for i, j in zip(src_points, homo_dest_points)])
          
          print "Geometric match:",float(sum(mask)) / float(len(mask))
        
          matchFigure.show()
          
          for sp, dp, m in zip(src_points, homo_dest_points, mask):
            if m == 1:
              matchPlot.plot([sp[0], dp[0]], [sp[1], dp[1]], '-m')
              matchPlot.plot([dp[0]], [dp[1]], 'om')
              matchPlot.plot([sp[0]], [sp[1]], 'xm')
      
      print "Average fit for frame is:", fit

      if not fit is None:
        averageFitHistory.append(fit)

    print "Average best fit:", 
    if len(averageFitHistory) == 0:
      print None
    else:
      print sum(averageFitHistory) / len(averageFitHistory)
    
  show()

#matchImageWithData()

#####################################################
# Test - Automapping of features in a image sequence
#####################################################

def autoMapDataSequence():
  data = readdata.loadData("data_GDC/output_GDC.txt")

  for sample_i, sample in enumerate(data):
    fig = figure()
    fig.suptitle("Sample " + str(sample_i))
    plot = fig.add_subplot(1, 1, 1)
    plot.hold(True)
    
    f1 = sample["frames"][0]["features"]["features"]
    f2 = sample["frames"][1]["features"]["features"]
    distanceFunction = lambda x, y: 1.001 - dot(x, y)
    #print([norm(f["descriptor"]) for f in f1])
    #print([norm(normDescriptor(f["descriptor"])) for f in f1])
    matches = bestFirstMatcher([normDescriptor(f["descriptor"]) for f in f1], [normDescriptor(f["descriptor"]) for f in f2], distanceFunction=distanceFunction, differenceFactorThreshold=0.5)
    print "Matches:",len(matches)
    for i, j in matches:
      plot.plot([f1[i]["x"], f2[j]["x"]], [f1[i]["y"], f2[j]["y"]], '-b')
    
    plot.invert_yaxis()
    fig.show()
  show()

#autoMapDataSequence()

#############################
# Test - Compare descriptors
#############################

def testCompareDescriptors():
  def showDescriptorDifferences(title, desc1, desc2):
    #distanceFunction = lambda x, y: arccos(dot(x, y) / norm(x) / norm(y))
    distanceFunction = euclid
    distances = array([[distanceFunction(img_f, frm_f) for frm_f in desc2] for img_f in desc1])

    fig = figure()
    fig.suptitle(title)
    
    plot = fig.add_subplot(1, 1, 1)
    plot.hold(True)

    for i in range(min(5, distances.shape[0])):
      hist, binEdges = histogram(distances[i], bins=20)
      binCenters = (binEdges[:-1] + binEdges[1:]) / 2.0
      plot.plot(binCenters, hist, '-')
      plot.plot(binCenters, hist, 'x')
    
    fig.show()
    
  def showMaxDescriptorCorrelation(title, desc1, desc2):
    fig = figure()
    fig.suptitle(title)
    
    plot = fig.add_subplot(1, 1, 1)
    plot.hold(True)
    
    plot.plot(desc1[0], desc2[0], ',k')
    
    fig.show()
  
  data = readdata.loadData("data_GDC/output_GDC.txt")

  img = cvLoadImage("data_GDC/IMG_1067.JPG")
  img_dimensions = [len(img[0]), len(img)]
  img = cv2.resize(img, (640, 640 * img_dimensions[1] / img_dimensions[0]))
  img_features = extractSURFfeaturesFromImage(img)# nOctaves=16, nOctaveLayers=4)

  img2 = cvLoadImage("data_GDC/IMG_1068.JPG")
  img2_dimensions = [len(img2[0]), len(img2)]
  img2 = cv2.resize(img2, (640, 640 * img2_dimensions[1] / img2_dimensions[0]))
  img2_features = extractSURFfeaturesFromImage(img2)# nOctaves=16, nOctaveLayers=4)

#  showDescriptorDifferences("Img 1 -> Img 2", [f[1] for f in img_features], [f[1] for f in img2_features])
  showMaxDescriptorCorrelation("Img 1 -> Img 2", [f[1] for f in img_features], [f[1] for f in img2_features])
  
  for sample_i, sample in enumerate(data):
    averageFitHistory = []
    print "Sample", sample_i + 1
    for frame_i, frame in enumerate(sample['frames']):
      frame_desc = []
      print "Frame", frame_i + 1
      for feature in frame["features"]["features"]:
        frame_desc.append(array(feature["descriptor"]))
       
      showMaxDescriptorCorrelation("Sample %d - Frame %d" % (sample_i + 1, frame_i + 1), [f[1] for f in img_features], frame_desc)
      #showDescriptorDifferences("Sample %d - Frame %d" % (sample_i + 1, frame_i + 1), [f[1] for f in img_features], frame_desc)


#testCompareDescriptors()

############################################
# Match features from image or feature data
# Uses data1 as reference and tries to 
# overlay data2
############################################

def reverseMatches(matches, f2count):
  result = map(list, [[]] * f2count)
  
  for ii, item in enumerate(matches):
    for match in item:
      result[match].append(ii)
  
  return result
 
def unifyMatches(matches1, matches2):
  result = [[]] * len(matches1)
  for i, (m1, m2) in enumerate(zip(matches1, matches2)):
    elements = m1 + m2
    result[i] = [e for j, e in enumerate(elements) if not e in elements[:j]]
  return result
 
def genMatchPairs(matches):
  result = []
  for i, m in enumerate(matches):
    result += [(i, j) for j in m]
  return result

def filterMatches1(features1, features2, distanceMatrix, clContext, matches=None):
  def flatten(l, depth=None):
    result = []
    if iterable(l) and ((depth > 0) or (depth is None)):
      nextDepth = None
      if not depth is None:
        nextDepth = depth - 1
      for i in l:
        result += flatten(i, depth=nextDepth)
    else:
      result.append(l)
    return result

  # Little helper for analyzing if we are on the right way
  def avgDistance(fs1, fs2, pairs, transform=Affine2D()):
    pt1 = array([transform.transform(f[0][0]) for f in fs1])
    pt2 = array([f[0][0] for f in fs2])

    print "Avg distance:", sum([norm(pt1[i] - pt2[j]) for i, j in pairs]) / len(pairs)
  
  def getDivergenceScore(matches, debug=False):
    result = 0.0
    count = len(matches)
    fmatches = filter(lambda x: len(x) > 0, matches)
    for fi1, fi2 in fmatches:
      if debug:
        print "d", fi1, "->", fi2, "=", distanceMatrix[fi1][fi2]
      result += distanceMatrix[fi1][fi2]
    return result * count / len(fmatches) * count / len(fmatches)
  
  def checkConsitency(pairs):
    def checkUniqueness(l):
      l = list(l)
      l.sort()
      return all(map(lambda i: l[i] != l[i + 1], range(len(l) - 1)))
    return all(map(lambda i: checkUniqueness(map(lambda x: x[i], pairs)), range(2)))
    
  def bruteforce(matches):
    # Limit number of elements over which we iterate
    ELEMENT_LIMIT = 6
    rmatches = reverseMatches(matches, len(features2))
    
    def limitMatches(matches):
      count = 0
      for i in range(len(matches)):
        if len(matches[i]) > 0:
          count += 1
        if count >= ELEMENT_LIMIT:
          boolToInt = {True: 0, False: 1}
          return [m[boolToInt[j <= i]] for j, m in enumerate(zip(matches, [[]] * len(matches)))]
      return matches
      
    matches = limitMatches(matches)
    rmatches = limitMatches(rmatches)
    rrmatches = reverseMatches(rmatches, len(features1))
    matches = unifyMatches(matches, rrmatches)
    
    limitsVector = [len(m) for m in matches]
    indexVector = [0 for l in limitsVector]
    
    # Calculate and show number of combinations
    combs = 1
    for l in limitsVector:
      combs *= l + 1
    print "Iterating over", combs, "combinations"
    
    def incrementIndexVector(subIndex=0):
      if subIndex >= len(indexVector):
        return False
        
      indexVector[subIndex] += 1
      if indexVector[subIndex] > limitsVector[subIndex]:
        indexVector[subIndex] = 0
        return incrementIndexVector(subIndex + 1)
      return True

    # Brute force
    maxMatchCount = sum(map(lambda x: len(x) > 0, matches))
    bestScore = None
    best = []
    while incrementIndexVector():
      # Generate matches
      candidatePairs = [(i, cm[j - 1]) for i, (cm, j) in enumerate(zip(matches, indexVector)) if j > 0]
      
      if not checkConsitency(candidatePairs):
        continue
      divergenceScore = getDivergenceScore(candidatePairs + [[]] * (maxMatchCount - len(candidatePairs)))
      #print divergenceScore, len(candidatePairs), len(candidatePairs + [[]] * (maxMatchCount - len(candidatePairs)))
      if (divergenceScore < bestScore) or (bestScore is None):
        print divergenceScore
        bestScore = divergenceScore
        best = candidatePairs
      
    print getDivergenceScore(best, True)
    return best
  
  if matches is None:
    matches = flatten([[(i, j) for i in range(len(features1))] for j in range(len(features2))], depth=2)
  rmatches = reverseMatches(matches, len(features2))

  avgDistance(features1, features2, genMatchPairs(matches))

  matchPairs = genMatchPairs(matches)
  matchPairs = bruteforce(matches)
  
  if not checkConsitency(matchPairs):
    print "WARNING!","Inconsistent match pairs"
    
  return matchPairs
  
  

def matchData(data1, data2, showOnlyMatchedFeatures=True):
  clContext = generateOpenCLContext()
  clQueue = cl.CommandQueue(clContext)
  
  def genFeatures(data, hardLimit=None):
    if type(data) is str:
      img = cvLoadImage(data)
      img_dimensions = [len(img[0]), len(img)]
      img = cv2.resize(img, (640, 640 * img_dimensions[1] / img_dimensions[0]))
      img_features = extractSURFfeaturesFromImage(img, nOctaves=4, nOctaveLayers=2)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Make a "normal" RGB image
      img_dimensions = [len(img[0]), len(img)]
      
      print "Features in", data, ": ", len(img_features)

      # Result: ((position, size, rotation), image-data)      
      return ([((f[0][0], f[0][2], f[0][3]), f[1]) for i, f in enumerate(img_features) if (i < hardLimit) or (hardLimit is None)], (img_dimensions, img))
    else:
      return (data, None)
    
  hardLimit = None
  data1 = genFeatures(data1, hardLimit=hardLimit)
  data2 = genFeatures(data2, hardLimit=hardLimit)
  
  # Calculate all distances between features
  print "Calculating distances..."
  distances = array([[None for y in data2[0]] for x in data1[0]])

  # Python or opencl subroutine?  
  def calcDistances(fs1, fs2, distances):
    calcDistance = lambda x, y: arccos(dot(array(x), array(y)) / norm(array(x)) / norm(array(y)))

    #calcDistance = euclid
    for i, f1 in enumerate(fs1):
      d1 = f1[1]
      for j, f2 in enumerate(fs2):
        d2 = f2[1]
        distances[i][j] = calcDistance(d1, d2)
  def clCalcDistances(fs1, fs2, distances):
    f1descs = array([x[1] for x in fs1]).astype(float32)
    f2descs = array([x[1] for x in fs2]).astype(float32)
    
    f1buf = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f1descs.flatten())
    f2buf = cl.Buffer(clContext, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=f2descs.flatten())
    
    output = zeros(distances.shape).astype(float32)
    outputBuffer = cl.Buffer(clContext, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=output)
    
    progCode = """
    __kernel void calculateDistance(int matWidth, __global float d1[], __global float d2[], __global float out[]) {
      const int d1i = get_global_id(0);
      const int d2i = get_global_id(1);
      
      float sum = 0.0f;
      float norm1 = 0.0f;
      float norm2 = 0.0f;
      for (int i = 0; i < 128; i++) {
        norm1 += pow(d1[d1i * 128 + i], 2);
        norm2 += pow(d2[d2i * 128 + i], 2);
        sum += d1[d1i * 128 + i] * d2[d2i * 128 + i];
      }
      out[d2i + d1i * matWidth] = acos(sum / sqrt(norm1) / sqrt(norm2));
    }
    """
    
    prog = cl.Program(clContext, progCode).build()
    prog.calculateDistance(clQueue, [len(f1descs), len(f2descs)], None, int32(len(fs2)), f1buf, f2buf, outputBuffer)
    cl.enqueue_copy(clQueue, output, outputBuffer)
    distances[:,:] = output
    
    
  clCalcDistances(data1[0], data2[0], distances)
  
  # Find all candidate matches under a given threshold
  print "Calculating matches..."
  THRESHOLD = 25 * pi / 180.0
  matches = [[i for i, d in enumerate(row) if d < THRESHOLD] for row in distances]
  
  # Filter matches
  pairs = filterMatches1(data1[0], data2[0], distances, clContext, matches)
  matches = [[]] * len(matches)
  for p in pairs:
    matches[p[0]] = [p[1]];

  # Draw match plot
  imagePairOffset = (700, 0)
  def drawFeature(plot, fpos, size, rotation, initTransform=Affine2D(), color='y'):
    vertices = ((0, 0), (0, 1), (1, 1), (-1, 1), (-1, -1), (1, -1))
    drawPairs = [(0, 1), (2, 3), (3, 4), (4, 5), (5, 2)]
    transform = Affine2D().scale(size / 2.0).rotate_deg(rotation).translate(*fpos)
    vertices = initTransform.transform(transform.transform(vertices))

    for i, j in drawPairs:
      plot.plot((vertices[i][0], vertices[j][0]), (vertices[i][1], vertices[j][1]), '-' + color)
  
  def showMatchPlot(data1, data2, matches):
    matchFigure = figure(figsize=figaspect(0.5))
    matchFigure.suptitle("Image - image matches")
    
    ag = AxesGrid(matchFigure, nrows_ncols=[1, 1], rect=[0.05, 0.05, 0.9, 0.8])
    #matchPlot = matchFigure.add_subplot(1, 1, 1)
    matchPlot = ag[0]
    matchPlot.hold(True)

    # Draw plots next to each other
    ymax = 100
    if not data1[1] is None:
      matchPlot.imshow(data1[1][1], extent=[0, data1[1][0][0], data1[1][0][1], 0])
      ymax = max(ymax, data1[1][0][1])
    if not data2[1] is None:
      matchPlot.imshow(data2[1][1], extent=[imagePairOffset[0], imagePairOffset[0] + data2[1][0][0], imagePairOffset[1] + data2[1][0][1], imagePairOffset[1]])
      ymax = max(ymax, data2[1][0][1])

    # Draw features matchings
    featurePairs = genMatchPairs(matches)
    
    if not showOnlyMatchedFeatures:
      # Draw all features
      for f in map(lambda x: x[0], data1[0]):
        drawFeature(matchPlot, f[0], f[1], f[2], initTransform=Affine2D())
      for f in map(lambda x: x[0], data2[0]):
        drawFeature(matchPlot, f[0], f[1], f[2], initTransform=Affine2D().translate(*imagePairOffset))
    else:
      # Draw only features that are in matching
      matchedItems = map(lambda x: x[0], featurePairs)
      for i, f in filter(lambda x: x[0] in matchedItems, enumerate(map(lambda x: x[0], data1[0]))):
        drawFeature(matchPlot, f[0], f[1], f[2], initTransform=Affine2D())
      matchedItems = map(lambda x: x[1], featurePairs)
      for i, f in filter(lambda x: x[0] in matchedItems, enumerate(map(lambda x: x[0], data2[0]))):
        drawFeature(matchPlot, f[0], f[1], f[2], initTransform=Affine2D().translate(*imagePairOffset))
    
    for i, j in featurePairs:
      d1f = data1[0][i][0]
      d2f = data2[0][j][0]
      matchPlot.plot([d1f[0][0], d2f[0][0] + imagePairOffset[0]], [d1f[0][1], d2f[0][1] + imagePairOffset[1]], ':y')

    matchPlot.set_xlim(0, imagePairOffset[0] * 2);
    matchPlot.set_ylim(ymax, 0);

    matchPlot.set_xticklabels([])

    matchFigure.show()
    
  showMatchPlot(data1, data2, matches)

def doStuff():  
  data = readdata.loadData("data_GDC/output_GDC.txt")

  matchData("data_PKG/feature-rich-scenes/window-close.jpg", "data_PKG/feature-rich-scenes/dishes.jpg")   
  #matchData("data_PKG/feature-rich-scenes/window-far.jpg", "data_PKG/feature-rich-scenes/window-close.jpg")   
  #matchData("data_PKG/feature-rich-scenes/map1.jpg", "data_PKG/feature-rich-scenes/map2.jpg")   
  #matchData("data_PKG/feature-rich-scenes/map1.jpg", "data_PKG/feature-rich-scenes/map1.jpg")   
  #matchData("data_PKG/feature-rich-scenes/photowall-far.jpg", "data_PKG/feature-rich-scenes/photowall-near.jpg")   
  #matchData("data_GDC/IMG_1068.JPG", "data_GDC/IMG_1069.JPG")

  #selectedSample = selectDbSample(data, 0, 1);
  #sampleData = dbFeaturesToData(selectedSample['features']['features'])
  #matchData(sampleData, "data_GDC/IMG_1067.JPG", showOnlyMatchedFeatures=True)

  #selectedSample = selectDbSample(data, 1, 4);
  #sampleData = dbFeaturesToData(selectedSample['features']['features'])
  #matchData(sampleData, "data_GDC/IMG_1063.JPG", showOnlyMatchedFeatures=False)


from pylab import *
import cv, cv2

def cvLoadImage(filename):
  im2 = cv2.imread(filename);
  return im2

def extractSURFfeaturesFromImage(image, hessianThreshold=2500, nOctaves=4, nOctaveLayers=2):
  im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  keypoints, descriptors = cv.ExtractSURF(cv.fromarray(im), None, cv.CreateMemStorage(), (1, hessianThreshold, nOctaves, nOctaveLayers))
  return zip(keypoints, descriptors)

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

import readdata

###################################################################
# Test - Trying to match data aginst features computed from images
###################################################################

def matchImageWithData():
  data = readdata.loadData("data_GDC/output_GDC.txt")
  img = cvLoadImage("data_GDC/IMG_1068.JPG")
  img_dimensions = [len(img[0]), len(img)]
  img = cv2.resize(img, (640, 640 * img_dimensions[1] / img_dimensions[0]))
  img = cv2.flip(img, flipCode=0)
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
      matchingPairs = bestFirstMatcher([x[1] for x in img_features], frame_desc, maxMatches=1, distanceFunction=distanceFunction, differenceFactorThreshold=0.9)
      
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

matchImageWithData()

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
  
  data = readdata.loadData("data_GDC/output_GDC.txt")

  img = cvLoadImage("data_GDC/IMG_1067.JPG")
  img_dimensions = [len(img[0]), len(img)]
  img = cv2.resize(img, (640, 640 * img_dimensions[1] / img_dimensions[0]))
  img_features = extractSURFfeaturesFromImage(img)# nOctaves=16, nOctaveLayers=4)

  img2 = cvLoadImage("data_GDC/IMG_1068.JPG")
  img2_dimensions = [len(img2[0]), len(img2)]
  img2 = cv2.resize(img2, (640, 640 * img2_dimensions[1] / img2_dimensions[0]))
  img2_features = extractSURFfeaturesFromImage(img2)# nOctaves=16, nOctaveLayers=4)

  showDescriptorDifferences("Img 1 -> Img 2", [f[1] for f in img_features], [f[1] for f in img2_features])
  
  for sample_i, sample in enumerate(data):
    averageFitHistory = []
    print "Sample", sample_i + 1
    for frame_i, frame in enumerate(sample['frames']):
      frame_desc = []
      print "Frame", frame_i + 1
      for feature in frame["features"]["features"]:
        frame_desc.append(array(feature["descriptor"]))
        
      showDescriptorDifferences("Sample %d - Frame %d" % (sample_i + 1, frame_i + 1), [f[1] for f in img_features], frame_desc)


#testCompareDescriptors()

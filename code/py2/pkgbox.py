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

def bestFirstMatcher(descriptors1, descriptors2, distanceFunction=dot, differenceFactorThreshold=0.33):
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
  
  descriptors2 = [CompareDescriptor(i, d) for i, d in enumerate(descriptors2)]
  
  result_pairs = []
  for desc1_i, desc1 in enumerate(descriptors1):
    matchedBestDistanceInfo = matchDescriptors(desc1, descriptors2, listKeyFunction=lambda x: x.desc, distanceFunction=distanceFunction)
    if matchedBestDistanceInfo[1] < differenceFactorThreshold:
      # Add matched pair and remove old descriptor2 (best first match, no feature is matched twice)
      result_pairs.append((desc1_i, descriptors2.pop(matchedBestDistanceInfo[0]).index))
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
  img = cvLoadImage("data_GDC/IMG_1063.JPG")
  img_features = extractSURFfeaturesFromImage(img)

  print "Number of image features:", len(img_features)

  for sample_i, sample in enumerate(data):
    averageFitHistory = []
    print "Sample", sample_i + 1
    for frame_i, frame in enumerate(sample['frames']):
      frame_desc = []
      print "Frame", frame_i + 1
      for feature in frame["features"]["features"]:
        frame_desc.append(array(feature["descriptor"]))

      distanceFunction = lambda x, y: 1.001 - dot(x, y)
      matchingPairs = bestFirstMatcher([x[1] for x in img_features], frame_desc, distanceFunction=distanceFunction, differenceFactorThreshold=0.5)
      
      if len(matchingPairs) > 0:
        fit = sum([distanceFunction(img_features[i][1], frame_desc[j]) for i, j in matchingPairs]) / len(matchingPairs)
      else:
        fit = None
      
      print "Average fit for frame is:", fit
      print "Matched points:", len(matchingPairs)
      
      if not fit is None:
        averageFitHistory.append(fit)
    print "Average best fit:", sum(averageFitHistory) / len(averageFitHistory)

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

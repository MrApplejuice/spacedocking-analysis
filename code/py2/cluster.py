from clustering2 import pairer
from readdata import loadData

def cluster_main(args):
  if len(args) < 3:
    print "Two arguments required: datafile and target cluster file"
  else:
    data = loadData(args[1])
    
    features = []
    featuresSources = []
    for sample in data:
      for frame in sample["frames"]:
        for feature in frame["features"]["features"]:
          features.append(feature["descriptor"])
          featuresSources.append((sample, frame))
        
    print "pairing", len(features)
    pairs = pairer(features)
        
    f = file(args[2], 'w')
    f.write("Works!\n")
    f.write(str(pairs))
    f.write("\n")
    f.close()

if __name__ == "__main__":
  import sys
  cluster_main(sys.argv)

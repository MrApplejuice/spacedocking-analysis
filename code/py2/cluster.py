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

    # Assemble file lines
    lines = []
    def assembleFileLine(pair):
      if type(pair) is int:
        return '[' + str(pair) + ']'
      else:
        lines.append(str(pair[2]) + ' ' + str(assembleFileLine(pair[0])) + ' ' + str(assembleFileLine(pair[1])))
        return len(lines) - 1
    assembleFileLine(pairs)
    
    f = file(args[2], 'w')
    for l in lines:
      f.write(l + '\n')
    f.close()

if __name__ == "__main__":
  import sys
  cluster_main(sys.argv)

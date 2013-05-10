#include "Cluster.hpp"

#include <fstream>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace boost;

const Cluster::NodeRef* Cluster :: Node :: getChildren() const {
  return children;
}

bool Cluster :: Node :: contains(Cluster::NodeRef node) const {
  return (node == children[0]) or (node == children[1]);
}

Cluster::NodeRef Cluster :: Node :: getParent() const {
  return parent;
}

double Cluster :: Node :: getDistance() const {
  return distance;
}

Cluster :: Node :: Node() : distance(0) {
}


void Cluster :: EditableNode :: setChildren(double distance, NodeRef node1, NodeRef node2) {
  this->distance = distance;
  children[0] = node1;
  children[1] = node2;
}
  
void Cluster :: EditableNode :: setParent(const NodeRef& parent) {
  this->parent = parent;
}


static int parseNodeIndex(const string& sni) {
  try {
    return lexical_cast<int>(sni);
  }
  catch (bad_lexical_cast) {
    return -1;
  }
}

void Cluster :: readCluster(const std::string& filename) {
  allNodes.clear();
  
  fstream file(filename.c_str(), ios_base::in);
  
  int lineIndex = 0;
  while ((!file.fail()) && (!file.bad())) {
    lineIndex++;
    
    string line;
    getline(file, line);
    trim(line);

    if (line.length() > 0) {
      if (line[0] != '#') {
        vector<string> splits;
        split(splits, line, is_any_of(" "), token_compress_on);
        
        EditableNodeRef newNode;
        if (splits.size() != 3) {
          cerr << "Warning! Data from line " << lineIndex << " ignored: Wrong number of columns" << endl;
        } else {
          try {
            const float distance = lexical_cast<float>(splits[0]);
            const int node1Index = parseNodeIndex(splits[1]);
            const int node2Index = parseNodeIndex(splits[2]);
            
            EditableNodeRef children[2];
            if (node1Index >= 0) {
              if (node1Index >= allNodes.size()) {
                cerr << "Warning! Have to ignore one childnode - data in line " << lineIndex << " is faulty" << endl;
              } else {
                children[0] = allNodes[node1Index];
              }
            }
            if (node2Index >= 0) {
              if (node2Index >= allNodes.size()) {
                cerr << "Warning! Have to ignore one childnode - data in line " << lineIndex << " is faulty" << endl;
              } else {
                children[1] = allNodes[node2Index];
              }
            }
            
            newNode = EditableNodeRef(new EditableNode());
            newNode->setChildren(distance, children[0], children[1]);
            
            for (int i = 0; i < 2; i++) {
              if (children[i]) {
                children[i]->setParent(newNode);
              }
            }
          }
          catch (bad_lexical_cast) {
            cerr << "Warning! Data from line " << lineIndex << " ignored" << endl;
          }
        }
        
        allNodes.push_back(newNode);
      }
    }
  }
  if (!file.eof()) {
    cerr << "IO Error during read of cluster file" << endl;
  }
}

Cluster::NodeRef Cluster :: getRoot() const {
  if (allNodes.empty()) {
    return NodeRef();
  }
  return allNodes[allNodes.size() - 1];
}

size_t Cluster :: getNodeCount() const {
  return allNodes.size();
}

Cluster :: Cluster() {
}

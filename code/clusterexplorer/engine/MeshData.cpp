#include "MeshData.hpp"

#include <algorithm>

#include "local/Global.hpp"

namespace engine {

using namespace std;
using namespace boost;

void MeshData :: TriangleIndexes :: normalize() {
  // Find smallest index and swap to first place (without sequence reordering)
  unsigned int swap = 0;
  if (indexes[swap] > indexes[1]) {
    swap = 1;
  }
  if (indexes[swap] > indexes[2]) {
    swap = 2;
  }
  
  if (swap) {
    unsigned int tmp = indexes[0];
    indexes[0] = indexes[swap];
    indexes[swap] = indexes[(2 * swap) % 3];
    indexes[(2 * swap) % 3] = tmp; 
  }
}

bool MeshData :: TriangleIndexes :: operator<(const MeshData::TriangleIndexes& other) const {
  return  (indexes[0] <  other.indexes[0]) || 
          ((indexes[0] == other.indexes[0]) && (indexes[1] <  other.indexes[1])) || 
          ((indexes[0] == other.indexes[0]) && (indexes[1] == other.indexes[1]) && (indexes[2] < other.indexes[2]));
}

bool MeshData :: TriangleIndexes :: operator==(const MeshData::TriangleIndexes& other) const {
  return equal(indexes, indexes + 3, other.indexes);
}

static void sortFaces(MeshData::FaceGroup::TriangleVector& triangles) {
  sort(triangles.begin(), triangles.end());
}


void MeshData :: FaceGroup :: normalize() {
  foreach (MeshData::TriangleIndexes& face, faces) {
    face.normalize();
  }
  
  sortFaces(faces);
}

MeshData::FaceGroup MeshData :: FaceGroup :: operator*(const FaceGroup& other) const {
  MeshData::FaceGroup result;
  MeshData::FaceGroup::TriangleVector thisFaces = faces;
  MeshData::FaceGroup::TriangleVector otherFaces = other.faces;
  sortFaces(thisFaces);
  sortFaces(otherFaces);
  
  result.name = this->name + " * " + other.name;
  
  unsigned int ti = 0; // This index
  unsigned int oi = 0; // Other index
  while ((ti < thisFaces.size()) && (oi < otherFaces.size())) {
    MeshData::TriangleIndexes& tv = thisFaces[ti];
    MeshData::TriangleIndexes& ov = otherFaces[oi]; 
    
    if (tv == ov) {
      result.faces.push_back(tv);
      ti++;
      oi++;
    } else {
      if (tv < ov) {
        ti++;
      } else {
        oi++;
      }
    }
  }
  
  sortFaces(result.faces);
  return result;
}


MeshData::FaceGroup& MeshData :: FaceGroup :: operator+=(const MeshData::FaceGroup& other) {
  sortFaces(faces);

  MeshData::FaceGroup::TriangleVector otherFaces = other.faces;
  sortFaces(otherFaces);
  
  size_t thisFaceCount = faces.size();
  
  unsigned int ti = 0; // This index
  unsigned int oi = 0; // Other index
  while ((ti < thisFaceCount) && (oi < otherFaces.size())) {
    MeshData::TriangleIndexes& tv = faces[ti];
    MeshData::TriangleIndexes& ov = otherFaces[oi]; 
    
    if (tv == ov) {
      ti++;
      oi++;
    } else {
      if (tv < ov) {
        ti++;
      } else {
        faces.push_back(ov);
        oi++;
      }
    }
  }
  for (; oi < otherFaces.size(); oi++) {
    faces.push_back(otherFaces[oi]);
  }

  return *this;
}


} // namespace engine

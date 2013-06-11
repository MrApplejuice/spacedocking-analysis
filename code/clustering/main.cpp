#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <boost/utility.hpp>
#include <boost/foreach.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <json/json.h>

using std::cout;
using std::cerr;
using std::endl;

inline float minf(float a, float b) {
  return a < b ? a : b;
}

template <typename X>
void checkDiagonal(X mat, size_t size) {
  for (int i = 0; i < size; i++) {
    for (int j = i; j < size; j++) {
      if (mat[j]->getValue(i) != mat[i]->getValue(j)) {
        cout << "Not diagonal at " << i << "  " << j << endl;
        cout << "Values " << mat[j]->getValue(i) << " " << mat[i]->getValue(j) << endl;
        exit(10);
      }
    }
  }
  cout << "checked" << endl;
}

/**
 * Class representing all descriptors retrieved from the flattened list
 * of descriptors of an input file.
 * 
 * Non copyable to prevent memory leakage through excessive copying.
 */
class DescriptorMatrix : public boost::noncopyable {
  public:
    typedef std::vector<float> ValueVector;
    
    class Descriptor {
      private:
        unsigned int sample, frame, feature;
        ValueVector values;
      public:
        unsigned int getSampleIndex() const {
          return sample;
        }

        unsigned int getFrameIndex() const {
          return frame;
        }
        
        unsigned int getFeatureIndex() const {
          return feature;
        }
        
        const ValueVector& getValues() const {
          return values;
        }
        
        float operator[](int index) const {
          return values[index];
        }
        
        size_t size() const {
          return values.size();
        }
        
        Descriptor() {
          sample = 0;
          frame = 0;
          feature = 0;
        }
        
        Descriptor(unsigned int sample, unsigned int frame, unsigned int feature, const ValueVector& values) {
          this->sample = sample;
          this->frame = frame;
          this->feature = feature;
          this->values = values;
        }
    };
  private:
    typedef std::vector<Descriptor> DescriptorVector;
    DescriptorVector descriptors;
  public:
    size_t size() const {
      return descriptors.size();
    }
    
    const Descriptor& operator[](int index) {
      return descriptors[index];
    }
    
    template <typename Iterator>
    void addSamples(Iterator begin, Iterator end) {
      unsigned int sampleIndex = 0;
      for (Iterator it = begin; it != end; it++) {
        const Json::Value* sampleNode = &(*it);
        
        const Json::Value* currentNode = sampleNode;
        if (!currentNode->isMember("frames")) {
          throw std::string("missing frames section in sample");
        }
        currentNode = &((*currentNode)["frames"]);
        
        if (!currentNode->isArray()) {
          throw std::string("frames is not of type array");
        }
        
        const Json::Value* frameNode = currentNode;
        for (unsigned int frameIndex = 0; frameIndex < frameNode->size(); frameIndex++) {
          currentNode = &((*frameNode)[frameIndex]);
          
          if (!currentNode->isMember("features")) {
            throw std::string("missing features section in frame-sample");
          }
          currentNode = &((*currentNode)["features"]);

          if (!currentNode->isObject()) {
            throw std::string("features(1) is not of type object");
          }

          if (!currentNode->isMember("features")) {
            throw std::string("missing features section in frame-sample");
          }
          currentNode = &((*currentNode)["features"]);

          if (!currentNode->isArray()) {
            throw std::string("features(2) is not of type array");
          }

          const Json::Value* featuresNode = currentNode;
          for (unsigned int featureIndex = 0; featureIndex < featuresNode->size(); featureIndex++) {
            currentNode = &((*featuresNode)[featureIndex]);
            
            if (!currentNode->isObject()) {
              throw std::string("features array member is not of type object");
            }

            if (!currentNode->isMember("descriptor")) {
              throw std::string("missing descriptor section in frame-sample");
            }
            currentNode = &((*currentNode)["descriptor"]);
            
            if (!currentNode->isArray()) {
              throw std::string("descriptor is not of type array");
            }
            
            ValueVector descriptorValues;
            descriptorValues.reserve(currentNode->size());
            for (int i = 0; i < currentNode->size(); i++) {
              if (!(*currentNode)[i].isNumeric()) {
                throw std::string("descriptor value is not of type float");
              }
              
              descriptorValues.push_back((*currentNode)[i].asDouble());
            }
            descriptors.push_back(Descriptor(sampleIndex, frameIndex, featureIndex, descriptorValues));
          }
        }
        
        sampleIndex++;
      }
    }
      
    DescriptorMatrix() {
    }
};
typedef boost::shared_ptr<DescriptorMatrix> DescriptorMatrixRef;

class JsonParser {
  public:
    bool succeeded;
    std::string error;
    std::string line;
    Json::Value root;
    Json::Reader jsonReader;
    
    void operator()() {
      if (!jsonReader.parse(line, root)) {
        error = jsonReader.getFormatedErrorMessages();
        succeeded = false;
      } else {
        succeeded = true;
      }
    }
    
    JsonParser() {
    }
    
    JsonParser(const std::string& line) {
      succeeded = false;
      this->line = line;
    }
};

DescriptorMatrixRef loadDescriptors(std::fstream& file, unsigned int parallelCount=1) {
  DescriptorMatrixRef result(new DescriptorMatrix());
  std::vector<Json::Value> jsonSamples;
  
  assert(parallelCount > 0);
  
  JsonParser parsers[parallelCount];
  boost::thread loadThreads[parallelCount];
  
  std::string line;
  int parallelIndex = 0;
  while (file.good()) {
    std::getline(file, line);
    boost::trim(line);
    
    if (line.length() > 0) { 
      parsers[parallelIndex] = JsonParser(line);
      loadThreads[parallelIndex] = boost::thread(boost::ref(parsers[parallelIndex]));
      parallelIndex = (parallelIndex + 1) % parallelCount;
      
      if (parallelIndex == 0) {
        // Must wait for threads
        for (int i = 0; i < parallelCount; i++) {
          loadThreads[i].join();
          if (!parsers[i].succeeded) {
            throw parsers[i].error;
          }
          jsonSamples.push_back(parsers[i].root);
        }
        result->addSamples(jsonSamples.begin(), jsonSamples.end());
        jsonSamples.clear();
      }
    }
  }
  if (parallelIndex > 0) {
    // Must wait for threads at the end that have not finished yet
    for (int i = 0; i < parallelIndex; i++) {
      loadThreads[i].join();
      if (!parsers[i].succeeded) {
        throw parsers[i].error;
      }
      jsonSamples.push_back(parsers[i].root);
    }
    result->addSamples(jsonSamples.begin(), jsonSamples.end());
    jsonSamples.clear();
  }
  
  if (!file.eof() && file.bad()) {
    throw std::string("IO Error during reading of input file");
  }

  return result;
}

class DistanceComputer {
  public:
    unsigned int no;
    DescriptorMatrixRef descriptors;

    std::vector<float> values;
  
    template <typename F1, typename F2>
    static float euclidean(size_t count, F1 v1, F2 v2) {
      float result = 0.0f;
      for (unsigned int i = 0; i < count; i++) {
        result += (v1[i] - v2[i]) * (v1[i] - v2[i]);
      }
      return sqrt(result);
    }
  
    void operator()() {
      DescriptorMatrix::Descriptor descriptor = (*descriptors)[no];
      
      for (unsigned int i = 0; i < descriptors->size(); i++) {
        if (i == no) {
          values[i] = HUGE_VALF;
        } else {
          values[i] = euclidean(descriptor.size(), descriptor, (*descriptors)[i]);
        }
      }
    }
  
    DistanceComputer() {
    }
  
    DistanceComputer(DescriptorMatrixRef descriptors, int no) {
      this->descriptors = descriptors;
      this->no = no;
      values = std::vector<float>(descriptors->size(), 0.0f);
    }
};

class SerializedMinimumMatrix {
  public:
    class Column;
    typedef boost::shared_ptr<Column> ColumnRef;
    typedef std::vector<ColumnRef> ColumnRefVector;

    class ColumnMemoryParameters {
      private:
        size_t memory, matSize;
      public:
        size_t getMaxStoredModifications() const {
          //cout << "getMaxStoredModifications " << memory / 2 / (matSize * 20) << endl;
          size_t result = memory / 2 / (matSize * 20);
          if (result <= 0) {
            result = 1;
          }
          return result;
        }
        
        size_t getMaxStoredVectors() const {
          //cout << "getMaxStoredVectors " << memory / 2 / (matSize * sizeof(float)) << endl;
          size_t result = memory / 2 / (matSize * sizeof(float));
          if (result <= 1) {
            result = 2;
          }
          return result;
        }
        
        void setMemoryHint(size_t memory) {
          this->memory = memory;
        }

        void setMatrixSize(size_t matSize) {
          this->matSize = matSize;
        }
        
        ColumnMemoryParameters() {
          memory = 0;
          matSize = 1;
        }
        
        ColumnMemoryParameters(size_t memory) {
          this->memory = memory;
          matSize = 1;
        }
    };
    typedef boost::shared_ptr<ColumnMemoryParameters> ColumnMemoryParametersRef;
    
    class UnloadManager {
      private:
        ColumnMemoryParametersRef memParams;
        ColumnRefVector loadedVectors;
      public:
        void registerLoadedVector(ColumnRef colRef) {
          for (ColumnRefVector::iterator it = loadedVectors.begin(); it != loadedVectors.end(); it++) {
            if (it->get() == colRef.get()) {
              loadedVectors.erase(it);
              break;
            }
          }
          
          while (loadedVectors.size() > memParams->getMaxStoredVectors()) {
            ColumnRefVector::iterator lastIt = loadedVectors.end() - 1;
            ColumnRef& col = *lastIt;
            if (col) {
              col->save();
              col->unload();
            }
            loadedVectors.erase(lastIt);
          }

          loadedVectors.insert(loadedVectors.begin(), colRef);
        }
        
        UnloadManager(ColumnMemoryParametersRef memParams) {
          this->memParams = memParams;
        }
    };
    typedef boost::shared_ptr<UnloadManager> UnloadManagerRef;
  
    class ColumnDataLabel;
    typedef boost::shared_ptr<ColumnDataLabel> ColumnDataLabelRef;
    class ColumnDataLabel {
      private:
        bool number;
        int num;
        ColumnDataLabelRef children[2];
        
        ColumnDataLabel(int number) {
          this->number = true;
          num = number;
        }

        ColumnDataLabel(ColumnDataLabelRef c1, ColumnDataLabelRef c2) {
          this->number = false;
          children[0] = c1;
          children[1] = c2;
        }
      public:
        bool isNumber() const {
          return number;
        }
        
        bool isPair() const {
          return !number;
        }
        
        size_t size() const {
          if (isNumber()) {
            return 1;
          } else {
            return 2;
          }
        }
        
        int getNumber() const {
          if (!number) {
            throw std::string("ColumnDataLabel - Error: trying to get number from pair");
          }
          return num;
        }

        ColumnDataLabelRef getChild(int i) const {
          if (!number) {
            throw std::string("ColumnDataLabel - Error: trying to get child from number");
          }
          return children[i];
        }
        
        std::string asString() const {
          if (isPair()) {
            return std::string("[") + children[0]->asString() + std::string(",") + children[1]->asString() + std::string("]");
          } else {
            return std::string("[") + boost::lexical_cast<std::string>(num) + std::string("]");
          }
        }
        
        static ColumnDataLabelRef create(int number) {
          return ColumnDataLabelRef(new ColumnDataLabel(number));
        }
        
        static ColumnDataLabelRef create(ColumnDataLabelRef c1, ColumnDataLabelRef c2) {
          return ColumnDataLabelRef(new ColumnDataLabel(c1, c2));
        }
    };
  
    class Column : public virtual boost::enable_shared_from_this<Column> {
      private:
        class Modification {
          public:
            virtual size_t modifyLength(size_t size) const = 0;
            virtual int modifyIndex(int i) const = 0;
            virtual void modifyVector(size_t& size, float values[]) const = 0;
        };
        
        class SetValueModification : public virtual Modification {
          private:
            int si;
            float value;
          public:
            virtual size_t modifyLength(size_t size) const {
              return size;
            }
            
            virtual int modifyIndex(int i) const {
              return i;
            }
            
            virtual void modifyVector(size_t& size, float values[]) const {
              values[si] = value;
            }
            
            SetValueModification(int i, float v) {
              si = i;
              value = v;
            }
        };
        
        class DeleteModification : public virtual Modification {
          private:
            int di;
          public:
            virtual size_t modifyLength(size_t size) const {
              return size - 1;
            }
            
            virtual int modifyIndex(int i) const {
              if (i >= di) {
                return i + 1;
              }
              return i;
            }
            
            virtual void modifyVector(size_t& size, float values[]) const {
              for (int i = di; i < size - 1; i++) {
                values[i] = values[i + 1];
              }
              size--;
            }
            
            DeleteModification(int i) {
              di = i;
            }
        };
        
        typedef boost::shared_ptr<Modification> ModificationRef;
        typedef std::vector<ModificationRef> ModificationRefVector;
      
        ColumnMemoryParametersRef memParams;
        UnloadManagerRef unloadManager;

        ColumnDataLabelRef label;
        boost::filesystem::path tmpfile;
        
        size_t valueCount;
        boost::shared_ptr<float[]> values;

        struct {
          float value;
          int index;
          bool valid;
        } minValue;
        
        ModificationRefVector modifications;
        
        void applyModifications() {
          if (modifications.size() >= memParams->getMaxStoredModifications()) {
            if (!values) {
              load();
            }
          }
          
          if (values) {
            if (!modifications.empty()) {
              BOOST_FOREACH (const ModificationRef& mod, modifications) {
                mod->modifyVector(valueCount, values.get());
              }
              modifications.clear();
            }
          }
        }
        
        void calculateMinValue() {
          if (!minValue.valid) {
            if (!values) {
              load();
            }
            
            minValue.index = -1;
            for (int i = 0; i < valueCount; i++) {
              if ((minValue.index < 0) || (values[i] < minValue.value)) {
                minValue.index = i;
                minValue.value = values[i];
              }
            }
            minValue.valid = true;
          }
        }
      public:
        class ColumnValueAccessor {
          private:
            Column& col;
            int offset;
            
            ColumnValueAccessor(Column& col, int offset) : col(col), offset(offset) {}
          public:
            ColumnValueAccessor& operator=(float v) {
              col.setValue(offset, v);
            }
            
            ColumnValueAccessor& operator=(const ColumnValueAccessor& v) {
              col.setValue(offset, (float) v);
            }
            
            operator float() const {
              return col.getValue(offset);
            }
            
            friend class Column;
        };

        ColumnDataLabelRef getLabel() const {
          return label;
        }
        
        ColumnDataLabelRef setLabel(ColumnDataLabelRef newLabel) {
          label = newLabel;
        }
        
        ColumnValueAccessor operator[](int i) {
          return ColumnValueAccessor(*this, i);
        }
        
        float getValue(int i) {
          load();
          return values[i];
        }
        
        void setValue(int i, float v) {
          if (v <= minValue.value) {
            minValue.index = i;
            minValue.value = v;
            //if (minValue.valid) cout << "Adapted on set " << i << endl;
          } else if (i == minValue.index) {
            minValue.valid = false;
            //cout << "Invalidated set " << i << endl;
          }

          modifications.push_back(ModificationRef(new SetValueModification(i, v)));
          applyModifications();
        }
        
        size_t size() const {
          size_t s = valueCount;
          
          for (ModificationRefVector::const_iterator it = modifications.begin(); it != modifications.end(); it++) {
            s = (*it)->modifyLength(s);
          }
          
          return s;
        }
        
        void deleteRow(int i) {
          if (i == minValue.index) {
            minValue.valid = false;
            //cout << "Invalidated delete " << i << endl;
          } else if (i < minValue.index) {
            minValue.index--;
            //cout << "Decremented on delete " << i << "  now is " << minValue.index << endl;
          }
          
          modifications.push_back(ModificationRef(new DeleteModification(i)));
          applyModifications();
        }
        
        float getMinValue() {
          calculateMinValue();
          return minValue.value;
        }
        
        float getMinIndex() {
          calculateMinValue();
          return minValue.index;
        }
        
        void load() {
          if (!values) {
            std::fstream file(tmpfile.string().c_str(), std::ios_base::in);
            file.read((char*) &valueCount, sizeof(valueCount));
            values = boost::shared_ptr<float[]>(new float[valueCount]);
            file.read((char*) values.get(), valueCount * sizeof(*(values.get())));
            
            applyModifications();
            calculateMinValue();

            if (unloadManager) {
              unloadManager->registerLoadedVector(shared_from_this());
            }
          }
        }
        
        void save() {
          applyModifications();

          std::fstream file(tmpfile.string().c_str(), std::ios_base::out | std::ios_base::binary);
          file.write((char*) &valueCount, sizeof(valueCount));
          file.write((char*) values.get(), valueCount * sizeof(*(values.get())));
        }
        
        void unload() {
          values.reset();
        }
     
        void setUnloadManager(UnloadManagerRef um) {
          unloadManager = um;
        }
     
        template <typename Iterator>   
        Column(ColumnMemoryParametersRef memParameters, ColumnDataLabelRef label, boost::filesystem::path tmpfile, Iterator begin, Iterator end) {
          memParams = memParameters;
          
          this->label = label;
          this->tmpfile = tmpfile;
          
          valueCount = 0;
          for (Iterator it = begin; it != end; it++) {
            valueCount++;
          }
          
          values = boost::shared_ptr<float[]>(new float[valueCount]);
          std::copy(begin, end, values.get());
          
          minValue.valid = false;
          calculateMinValue();
          
          save();
          unload();
        }
    };
  private:
    ColumnMemoryParametersRef colParams;
    UnloadManagerRef unloadManager;
  
    DescriptorMatrixRef descriptorMatrix;
    boost::filesystem::path tmpDir;
    
    ColumnRefVector columns;
  public:
    void computeDistances(size_t threadCount=1) {
      columns.clear();
      
      boost::thread threads[threadCount];
      ColumnDataLabelRef labels[threadCount];
      DistanceComputer distanceComputers[threadCount];
      
      int threadOffset = 0;
      for (int i = 0; i < descriptorMatrix->size(); i++) {
        if (((i + 1) % 1000) == 0) {
          cout << "Computing distance: " << (i + 1) << endl;
        }
        
        distanceComputers[threadOffset] = DistanceComputer(descriptorMatrix, i);
        threads[threadOffset] = boost::thread(boost::ref(distanceComputers[threadOffset]));
        labels[threadOffset] = ColumnDataLabel::create(i);
        threadOffset = (threadOffset + 1) % threadCount;
        
        if (threadOffset == 0) {
          // Wait for threads to finish
          for (int i = 0; i < threadCount; i++) {
            threads[i].join();
            
            boost::filesystem::path tmpFilepath(tmpDir);
            tmpFilepath += std::string("c") + boost::lexical_cast<std::string>(distanceComputers[i].no);
            columns.push_back(ColumnRef(new Column(colParams, labels[i], tmpFilepath, distanceComputers[i].values.begin(), distanceComputers[i].values.end())));
          }
        }
      }
      if (threadOffset > 0) {
        for (int i = 0; i < threadOffset; i++) {
          threads[i].join();
          boost::filesystem::path tmpFilepath(tmpDir);
          tmpFilepath += std::string("c") + boost::lexical_cast<std::string>(distanceComputers[i].no);
          columns.push_back(ColumnRef(new Column(colParams, labels[i], tmpFilepath, distanceComputers[i].values.begin(), distanceComputers[i].values.end())));
        }
      }
      
      // Set unload manager for all columns
      BOOST_FOREACH (const ColumnRef& col, columns) {
        col->setUnloadManager(unloadManager);
      }
    }
    
    bool pair() {
      checkDiagonal(columns, columns.size());
      
      colParams->setMatrixSize(columns.size());
      
      if (columns.size() < 2) {
        return false;
      }
      
      int minIndex = -1;
      {
        float minValue = 0;
        for (int i = 0; i < columns.size(); i++) {
          if ((minIndex < 0) || (columns[i]->getMinValue() < minValue)) {
            minIndex = i;
            minValue = columns[minIndex]->getMinValue();
          }
        }
      }
      int otherIndex = columns[minIndex]->getMinIndex();
      
      // Make sure that minIndex < otherIndex
      if (otherIndex < minIndex) {
        int xchg = otherIndex;
        otherIndex = minIndex;
        minIndex = xchg;
      }
      
      Column& thisCol = *columns[minIndex];
      {
        Column& otherCol = *columns[otherIndex];
        cout << "Pairing " << minIndex << " with " << otherIndex << "  distance: " << columns[minIndex]->getValue(otherIndex) << endl;
        
        //if (otherIndex + 1 < columns[minIndex]->size()) cout << "+1 " << columns[minIndex]->getValue(otherIndex + 1) << endl;
        //if (otherIndex - 1 >= 0) cout << "-1 " << columns[minIndex]->getValue(otherIndex - 1) << endl;
        //if (minIndex + 1 < columns[otherIndex]->size()) cout << "+1 " << columns[otherIndex]->getValue(minIndex + 1) << endl;
        //if (minIndex - 1 >= 0) cout << "-1 " << columns[otherIndex]->getValue(minIndex - 1) << endl;
        
        cout << "Min indexes are " << thisCol.getMinIndex() << " vs " << otherCol.getMinIndex() << endl;
        cout << "Min values are " << thisCol.getMinValue() << " vs " << otherCol.getMinValue() << endl;
        if (thisCol.getMinValue() != otherCol.getMinValue()) {
          exit(10);
        }

        // Unify vectors
        for (int i = 0; i < thisCol.size(); i++) {
          if ((thisCol[i] >= HUGE_VALF) || (otherCol[i] >= HUGE_VALF)) {
            thisCol[i] = HUGE_VALF;
          } else {
            thisCol[i] = minf(thisCol[i], otherCol[i]);
          }
        }
        
        // Update thisCol label
        thisCol.setLabel(ColumnDataLabel::create(thisCol.getLabel(), otherCol.getLabel()));
        
        // Remove otherCol
        columns.erase(columns.begin() + otherIndex);
        // Important! otherCol INVALID after this!!!
      }

      // Remove otherCol-row and update row values
      for (int i = 0; i < columns.size(); i++) {
        columns[i]->deleteRow(otherIndex);
        columns[i]->setValue(minIndex, thisCol[i]);
      }
      
      return true;
    }
    
    size_t size() const {
      return columns.size();
    }
    
    SerializedMinimumMatrix(size_t memoryHint, DescriptorMatrixRef descriptorMatrix, const boost::filesystem::path& tmpDir) {
      this->colParams = ColumnMemoryParametersRef(new ColumnMemoryParameters(memoryHint));
      this->colParams->setMatrixSize(descriptorMatrix->size());
      this->descriptorMatrix = descriptorMatrix;
      this->tmpDir = tmpDir;
      
      unloadManager = UnloadManagerRef(new UnloadManager(colParams));
    }
};

int main(int argc, char** argv) {
  try {
    boost::program_options::options_description poDesc("Options");
    
    boost::program_options::options_description poPositionalDesc("Positional Arguments");
    boost::program_options::positional_options_description poPositional;
    
    poDesc.add_options()
      ("threads", boost::program_options::value<unsigned int>()->default_value(1), "Maximum number of threads to use for speeding up the computation")
      ("tmpdir,t", boost::program_options::value<std::string>()->default_value(std::string("/tmp")), "Temporary directory to use for swapping matrix columns")
      ("memory,m", boost::program_options::value<size_t>()->default_value(100000000), "Memory to use to buffer data vectors")
      ("help,h", "Displays this help");
    poPositionalDesc.add_options()
      ("input-filename", "The decoded json-data containing AstroDrone data samples")
      ("output-filename", "Cluster file that should be created");
    poPositional.add("input-filename", 1);
    poPositional.add("output-filename", 1);

    boost::program_options::variables_map poValueMap;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(boost::program_options::options_description().add(poPositionalDesc).add(poDesc)).positional(poPositional).run(), poValueMap);
    poValueMap.notify();
    
    if (!poValueMap["help"].empty()) {
      cout << "Custom binary clustering algorithm implemented by Paul Konstantin Gerke (2013)." << endl;
      cout << "" << endl;
      cout << "Syntax:" << endl;
      cout << "  cluster [options] input-filename output-filename" << endl;
      cout << "" << endl;
      cout << poPositionalDesc << endl;
      cout << poDesc << endl;
      return 1;
    }
    
    if (poValueMap["input-filename"].empty()) {
      cout << "Error: no input file specified" << endl;
      return 2;
    }
    
    if (poValueMap["output-filename"].empty()) {
      cout << "Error: no output file specified" << endl;
      return 2;
    }

    if ((poValueMap["threads"].as<unsigned int>() <= 0) || (poValueMap["threads"].as<unsigned int>() > 256)) {
      cout << "Error: number of threads must be >= 1 and <= 256" << endl;
      return 2;
    }

    if (poValueMap["memory"].as<size_t>() < 1024) {
      cout << "Error: must allow more than 1 KB to be used for buffering" << endl;
      return 2;
    }
    
    size_t memoryHint = poValueMap["memory"].as<size_t>();

    boost::filesystem::path tmpDir(poValueMap["tmpdir"].as<std::string>());
    if (!boost::filesystem::exists(tmpDir)) {
      cout << "Error: " << tmpDir << " does not exist" << endl;
      return 2;
    }
    if (!boost::filesystem::is_directory(tmpDir)) {
      cout << "Error: " << tmpDir << " does not exist" << endl;
      return 2;
    }

    std::fstream inputFile(poValueMap["input-filename"].as<std::string>().c_str(), std::ios_base::in);
    if (!inputFile.good()) {
      cout << "Error: Cannot open input file" << endl;
      return 3;
    }
    
    std::fstream outputFile(poValueMap["output-filename"].as<std::string>().c_str(), std::ios_base::out | std::ios_base::binary);
    if (!outputFile.good()) {
      cout << "Error: Cannot open output file" << endl;
      return 4;
    }
    
    const unsigned int threadCount = poValueMap["threads"].as<unsigned int>();
    
    cout << "Loading input file... " << std::flush;
    DescriptorMatrixRef descriptors;
    try {
      descriptors = loadDescriptors(inputFile, threadCount);
      inputFile.close();
      
      cout << "loaded " << descriptors->size() << " descriptors" << endl;
    }
    catch (const std::string& e) {
      cout << "Error during loading of input file: " << e << endl;
      return 5;
    }
    
    cout << "Computing distance matrix..." << endl;
    SerializedMinimumMatrix minimumMatrix(memoryHint, descriptors, tmpDir);
    minimumMatrix.computeDistances(threadCount);
    
    while (minimumMatrix.pair()) {
      cout << minimumMatrix.size() << " left" << endl;
    }
    
    return 0;
  }
  catch (boost::program_options::unknown_option e) {
    cout << "Error: " << e.what() << endl;
  }
  catch (boost::program_options::too_many_positional_options_error e) {
    cout << "Error: " << e.what() << endl;
  }
  catch (boost::program_options::invalid_option_value e) {
    cout << "Error: " << e.what() << endl;
  }
  return 2;
}

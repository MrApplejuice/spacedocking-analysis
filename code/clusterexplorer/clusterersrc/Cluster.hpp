#pragma once

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

class Cluster {
  public:
    class Node;
    typedef boost::shared_ptr<Node> NodeRef;
  
    class Node {
      protected:
        double distance;
        NodeRef parent;
        NodeRef children[2];
      public:
        virtual const NodeRef* getChildren() const;
        virtual bool contains(NodeRef node) const;
        virtual NodeRef getParent() const;
        virtual double getDistance() const;
        
        Node();
    };
  private:
    class EditableNode;
    typedef boost::shared_ptr<EditableNode> EditableNodeRef;
    class EditableNode : public virtual Node {
      private:
      public:
        virtual void setChildren(double distance, NodeRef node1, NodeRef node2);
        virtual void setParent(const NodeRef& parent);
    };
    
    std::vector<EditableNodeRef> allNodes;
  public:
    virtual void readCluster(const std::string& filename);
    
    virtual NodeRef getRoot() const;
    virtual size_t getNodeCount() const;
  
    Cluster();
};

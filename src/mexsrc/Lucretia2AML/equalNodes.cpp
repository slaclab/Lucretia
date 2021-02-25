#include "UAPUtilities.hpp"

bool equalNodes(UAPNode *NodeOne, UAPNode *NodeTwo) {
 /* Use this to compare two nodes.  Checks for equality of the name attribute. */
  bool equality;

  string AttribOne = NodeOne->getAttributeString("name");
  string AttribTwo = NodeTwo->getAttributeString("name");

  if (AttribOne==AttribTwo) equality = true;
  else equality = false;

  return equality;
}


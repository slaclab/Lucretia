#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"

using namespace std;

UAPNode* addS(UAPNode *EleNode, mxArray *Elemx) {
  mxArray *Smx = mxGetField(Elemx, 0, "S");
  double Sdoub = mxGetScalar(Smx);
  UAPNode *SNode = EleNode->addChild(ELEMENT_NODE, "s");
  bool ok;
  SNode->addAttribute("actual", BasicUtilities::double_to_string(Sdoub, ok), false);

  return EleNode;
}


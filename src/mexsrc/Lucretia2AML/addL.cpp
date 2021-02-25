#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"

using namespace std;

UAPNode* addL(UAPNode *EleNode, mxArray *Elemx) {
  bool ok;
  mxArray *Lmx = mxGetField(Elemx, 0, "L");
  double Ldoub = mxGetScalar(Lmx);
  UAPNode *LNode = EleNode->addChild(ELEMENT_NODE, "length");
  LNode->addAttribute("design", BasicUtilities::double_to_string(Ldoub, ok), false);

  return EleNode;
}


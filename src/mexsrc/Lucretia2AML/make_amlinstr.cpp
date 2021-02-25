#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void make_amlinstr(UAPNode *EleNode, mxArray *Elemx) {
  bool ok;
  UAPNode *InstrNode = EleNode->addChild(ELEMENT_NODE, "instrument");

  EleNode = addName(EleNode, Elemx);

  mxArray *Resmx = mxGetField(Elemx, 0, "Resolution");
  mxArray * = mxGetField(Elemx, 0, "");
  mxArray * = mxGetField(Elemx, 0, "");
}


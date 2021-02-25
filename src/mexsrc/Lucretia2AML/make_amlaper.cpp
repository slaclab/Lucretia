#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void make_amlaper(UAPNode *EleNode, mxArray *Elemx, string location) {
  bool ok;
  UAPNode *AperNode = EleNode->addChild(ELEMENT_NODE, "aperture");
  AperNode->addAttribute("at", location, false);
  AperNode->addAttribute("shape", "CIRCLE", false);
  AperNode->addAttribute("orientation_dependent", "TRUE", false);
  AperNode->addAttribute("side", "BOTH", false);

  UAPNode *LimNode = AperNode->addChild(ELEMENT_NODE, "xy_limit");
  double aper = mxGetScalar( mxGetField(Elemx, 0, "aper") );
  LimNode->addAttribute("design", BasicUtilities::double_to_string(aper, ok) );
}


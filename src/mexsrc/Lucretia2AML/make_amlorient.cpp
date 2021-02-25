#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;
      
void make_amlorient(UAPNode *EleNode, mxArray *Elemx){
 /* Extract the orientation information from the Offset and Tilt fields, and create a new node.*/
  bool ok;
  double x=0, xp=0, y=0, yp=0, s=0, tilt=0;
  mxArray *Tiltmx = mxGetField(Elemx, 0, "Tilt");
  if (Tiltmx) tilt = mxGetScalar(Tiltmx);

  mxArray *Orientmx = mxGetField(Elemx, 0, "Offset");
  double *orientptr = mxGetPr(Orientmx);
  x += *orientptr;
  xp += *(orientptr+1);
  y += *(orientptr+2);
  yp += *(orientptr+3);
  s += *(orientptr+4);
  tilt += *(orientptr+5);

  UAPNode *OrientNode = EleNode->addChild(ELEMENT_NODE, "orientation");
  OrientNode->addAttribute("origin", "CENTER", false);

  if (x!=0) {
    UAPNode *XNode = OrientNode->addChild(ELEMENT_NODE, "x_offset");
    XNode->addAttribute("design", BasicUtilities::double_to_string(x, ok), false);
  }
  if (xp!=0) {
    UAPNode *XpNode = OrientNode->addChild(ELEMENT_NODE, "x_pitch");
    XpNode->addAttribute("design", BasicUtilities::double_to_string(xp, ok), false);
  }
  if (y!=0) {
    UAPNode *YNode = OrientNode->addChild(ELEMENT_NODE, "y_offset");
    YNode->addAttribute("design", BasicUtilities::double_to_string(y, ok), false);
  }
  if (yp!=0) {
    UAPNode *YpNode = OrientNode->addChild(ELEMENT_NODE, "y_pitch");
    YpNode->addAttribute("design", BasicUtilities::double_to_string(yp, ok), false);
  }
  if (x!=0) {
    UAPNode *SNode = OrientNode->addChild(ELEMENT_NODE, "s_offset");
    SNode->addAttribute("design", BasicUtilities::double_to_string(s, ok), false);
  }
  if (tilt!=0) {
    UAPNode *TiltNode = OrientNode->addChild(ELEMENT_NODE, "tilt");
    TiltNode->addAttribute("design", BasicUtilities::double_to_string(tilt, ok), false);
  }
}


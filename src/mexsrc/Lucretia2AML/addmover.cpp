#include <iostream>
#include "UAPUtilities.hpp"
#include "AMLReader.hpp"
#include "AMLLatticeExpander.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void addmover (mxArray *GCell, UAPNode *GirderNode) {
  mxArray *MoverInds = mxGetField(GCell, 0, "Mover");
  if (!MoverInds) return;
  mxArray *MoverPoss = mxGetField(GCell, 0, "MoverPos");
  mxArray *MoverSteps = mxGetField(GCell, 0, "MoverStep");
  double *movernum, *moverpos, *moverstep;
  bool ok;

  movernum = mxGetPr(MoverInds);
  moverpos = mxGetPr(MoverPoss);
  moverstep = mxGetPr(MoverSteps);
  UAPNode *OrientNode = GirderNode->addChild("orientation");

  for (int i = 0 ; i < mxGetN(MoverInds) ; i++) {
    if (*movernum==1) {
      UAPNode *Node = OrientNode->addChild("x_offset");
      Node->addAttribute("design", BasicUtilities::double_to_string(*moverpos, ok));
      Node->addAttribute("step", BasicUtilities::double_to_string(*moverstep, ok));
    }
    else if (*movernum==2) {
      UAPNode *Node = OrientNode->addChild("x_pitch");
      Node->addAttribute("design", BasicUtilities::double_to_string(*moverpos, ok));
      Node->addAttribute("step", BasicUtilities::double_to_string(*moverstep, ok));
    }
    else if (*movernum==3) {
      UAPNode *Node = OrientNode->addChild("y_offset");
      Node->addAttribute("design", BasicUtilities::double_to_string(*moverpos, ok));
      Node->addAttribute("step", BasicUtilities::double_to_string(*moverstep, ok));
    }
    else if (*movernum==4) {
      UAPNode *Node = OrientNode->addChild("y_pitch");
      Node->addAttribute("design", BasicUtilities::double_to_string(*moverpos, ok));
      Node->addAttribute("step", BasicUtilities::double_to_string(*moverstep, ok));
    }
    else if (*movernum==5) {
      UAPNode *Node = OrientNode->addChild("s_offset");
      Node->addAttribute("design", BasicUtilities::double_to_string(*moverpos, ok));
      Node->addAttribute("step", BasicUtilities::double_to_string(*moverstep, ok));
    }
    else if (*movernum==6) {
      UAPNode *Node = OrientNode->addChild("tilt");
      Node->addAttribute("design", BasicUtilities::double_to_string(*moverpos, ok));
      Node->addAttribute("step", BasicUtilities::double_to_string(*moverstep, ok));
    }

    movernum++;
    moverpos++;
    moverstep++;
  }

  return;
}


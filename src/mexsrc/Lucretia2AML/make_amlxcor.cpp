#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void make_amlxcor(UAPNode *EleNode, mxArray *Elemx, UAPNode *AMLRepNode, mxArray* FLPSmx) {

  bool ok, hasPS=false;
  double PSnum;
  mxArray *PSstrucmx, *ElePSmx;
  mxArray *PSmx = mexGetVariable("global", "PS");

  UAPNode *XCORNode = EleNode->addChild(ELEMENT_NODE, "kicker");

  EleNode = addL(EleNode, Elemx);

  UAPNode *XKickNode = XCORNode->addChild(ELEMENT_NODE, "x_kick");
  XKickNode = addB(XKickNode, Elemx);
  XKickNode = addBerr(XKickNode, Elemx);

  make_amlorient(XCORNode, Elemx);

  PSstrucmx = mxGetField(Elemx, 0, "PS");
  if (PSstrucmx) {
    hasPS=true;
    PSnum = mxGetScalar(PSstrucmx);
  }

  UAPAttribute *kuAttrib = XKickNode->getAttribute("design");
  string kustr = kuAttrib->getValue();

  if ( hasPS ) {
    EleNode = addName(EleNode, Elemx);
    UAPAttribute *NameAttrib = EleNode->getAttribute("name");
    string namestr = NameAttrib->getValue();
    CreateAMLController(AMLRepNode, PSnum, "kicker:x_kick", PSmx, namestr,
                                           BasicUtilities::string_to_double(kustr, ok), FLPSmx);
    EleNode->removeAttribute("name");
  }

  return;
}


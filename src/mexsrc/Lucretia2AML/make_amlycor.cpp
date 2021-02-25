#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void make_amlycor(UAPNode *EleNode, mxArray *Elemx, UAPNode *AMLRepNode, mxArray *FLPSmx) {
  bool ok, hasPS=false;
  double PSnum;
  mxArray *PSstrucmx, *ElePSmx;
  mxArray *PSmx = mexGetVariable("global", "PS");

  UAPNode *YCORNode = EleNode->addChild(ELEMENT_NODE, "kicker");

  EleNode = addL(EleNode, Elemx);

  UAPNode *YKickNode = YCORNode->addChild(ELEMENT_NODE, "y_kick");
  YKickNode = addB(YKickNode, Elemx);
  YKickNode = addBerr(YKickNode, Elemx);

  make_amlorient(YCORNode, Elemx);

  PSstrucmx = mxGetField(Elemx, 0, "PS");
  if (PSstrucmx) {
    hasPS=true;
    PSnum = mxGetScalar(PSstrucmx);
  }

  UAPAttribute *kuAttrib = YKickNode->getAttribute("design");
  string kustr = kuAttrib->getValue();

  if ( hasPS ) {
    EleNode = addName(EleNode, Elemx);
    UAPAttribute *NameAttrib = EleNode->getAttribute("name");
    string namestr = NameAttrib->getValue();
    CreateAMLController(AMLRepNode, PSnum, "kicker:y_kick", PSmx, namestr,
                                           BasicUtilities::string_to_double(kustr, ok), FLPSmx);
    EleNode->removeAttribute("name");
  }

  return;
}


#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void make_amloct(UAPNode *EleNode, mxArray *Elemx, UAPNode *AMLRepNode, mxArray *FLPSmx) {
  bool ok, hasPS=false;
  double PSnum;
  mxArray *PSstrucmx, *ElePSmx;
  mxArray *PSmx = mexGetVariable("global", "PS");

  UAPNode *SextNode = EleNode->addChild(ELEMENT_NODE, "octupole");

  EleNode = addName(EleNode, Elemx);

  EleNode = addS(EleNode, Elemx);

  EleNode = addL(EleNode, Elemx);

  UAPNode *BNode = SextNode->addChild(ELEMENT_NODE, "k");
  BNode = addB(BNode, Elemx);
  BNode = addBerr(BNode, Elemx);

  PSstrucmx = mxGetField(Elemx, 0, "PS");
  if (PSstrucmx) {
    hasPS=true;
    PSnum = mxGetScalar(PSstrucmx);
  }

  UAPAttribute *kuAttrib = BNode->getAttribute("design");
  string kustr = kuAttrib->getValue();

  if ( hasPS ) {
    UAPAttribute *NameAttrib = EleNode->getAttribute("name");
    string namestr = NameAttrib->getValue(); 
    CreateAMLController(AMLRepNode, PSnum, "octupole:k", PSmx, namestr, 
                                           BasicUtilities::string_to_double(kustr, ok), FLPSmx);
  }

  return;
}


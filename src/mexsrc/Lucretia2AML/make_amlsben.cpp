#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void make_amlsben(UAPNode *EleNode, mxArray *Elemx, UAPNode *AMLRepNode, mxArray *FLPSmx) {
  bool ok, hasPS=false;
  double PSnum;
  mxArray *PSstrucmx, *ElePSmx;
  mxArray *PSmx = mexGetVariable("global", "PS");

 /* Add a bend node to the element.*/
  UAPNode *BendNode = EleNode->addChild(ELEMENT_NODE, "bend");

  EleNode = addName(EleNode, Elemx);

  EleNode = addL(EleNode, Elemx);

  UAPNode *BNode = BendNode->addChild(ELEMENT_NODE, "g");
  BNode = addB(BNode, Elemx);
  BNode = addBerr(BNode, Elemx);

  mxArray *EdgeAnglemx = mxGetField(Elemx, 0, "EdgeAngle");
  double *E1doub = mxGetPr(EdgeAnglemx);
  double *E2doub;
  if (mxGetN(EdgeAnglemx)>1) E2doub = E1doub+1;
  else E2doub = E1doub;
  UAPNode *E1Node = BendNode->addChild(ELEMENT_NODE, "e1");
  E1Node->addAttribute("design", BasicUtilities::double_to_string(*E1doub, ok), false);
  UAPNode *E2Node = BendNode->addChild(ELEMENT_NODE, "e2");
  E2Node->addAttribute("design", BasicUtilities::double_to_string(*E2doub, ok), false);

  mxArray *EdgeCurvaturemx = mxGetField(Elemx, 0, "EdgeCurvature");
  double *H1doub = mxGetPr(EdgeCurvaturemx);
  double *H2doub;
  if (mxGetN(EdgeCurvaturemx)>1) H2doub = H1doub+1;
  else H2doub = H1doub;
  UAPNode *H1Node = BendNode->addChild(ELEMENT_NODE, "h1");
  H1Node->addAttribute("design", BasicUtilities::double_to_string(*H1doub, ok), false);
  UAPNode *H2Node = BendNode->addChild(ELEMENT_NODE, "h2");
  H2Node->addAttribute("design", BasicUtilities::double_to_string(*H2doub, ok), false);

  mxArray *HGAPmx = mxGetField(Elemx, 0, "HGAP");
  double *HGAP1doub = mxGetPr(HGAPmx);
  double *HGAP2doub;
  if (mxGetN(HGAPmx)>1) HGAP2doub = HGAP1doub+1;
  else HGAP2doub = HGAP1doub;
  UAPNode *HGAP1Node = BendNode->addChild(ELEMENT_NODE, "h_gap1");
  HGAP1Node->addAttribute("design", BasicUtilities::double_to_string(*HGAP1doub, ok), false);
  UAPNode *HGAP2Node = BendNode->addChild(ELEMENT_NODE, "h_gap2");
  HGAP2Node->addAttribute("design", BasicUtilities::double_to_string(*HGAP2doub, ok), false);

  mxArray *FINTmx = mxGetField(Elemx, 0, "FINT");
  double *FINT1doub = mxGetPr(FINTmx);
  double *FINT2doub;
  if (mxGetN(FINTmx)>1) FINT2doub = FINT1doub+1;
  else FINT2doub = FINT1doub;
  UAPNode *FINT1Node = BendNode->addChild(ELEMENT_NODE, "f_int1");
  FINT1Node->addAttribute("design", BasicUtilities::double_to_string(*FINT1doub, ok), false);
  UAPNode *FINT2Node = BendNode->addChild(ELEMENT_NODE, "f_int2");
  FINT2Node->addAttribute("design", BasicUtilities::double_to_string(*FINT2doub, ok), false);

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
    CreateAMLController(AMLRepNode, PSnum, "bend:g", PSmx, namestr,
                                           BasicUtilities::string_to_double(kustr, ok), FLPSmx);
  }

  return;
}


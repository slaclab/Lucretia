#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

UAPNode* LEle2UAPNode(int i, UAPNode* AMLRepNode, mxArray* Elemx, mxArray *FLPSmx, 
                                            mxArray *INSTRmx, mxArray *FLINSTRmx) {
 /* Get the "Class" of this element structure.*/
  mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
 /* Reserve some dynamic memory for the Class string.*/
  char *ClassType;
  int strsize = mxGetN(ClassVal)+1;
  ClassType = new char[strsize];
  mxGetString(ClassVal, ClassType, strsize);

  UAPNode *EleNode = AMLRepNode->addChild(ELEMENT_NODE, "element");

 /* Determine the element class, and call the appropriate routine.*/
  if ( !strcmp(ClassType, (char*)"MARK")){
    make_amlmarker(EleNode, Elemx);
  }
  else if (!strcmp(ClassType, (char*)"DRIF")) {
    make_amldrift(EleNode, Elemx);
  }
  else if (!strcmp(ClassType, (char*)"QUAD")) {
    make_amlquad(EleNode, Elemx, AMLRepNode, FLPSmx);
  }
  else if (!strcmp(ClassType, (char*)"SEXT")) {
    make_amlsext(EleNode, Elemx, AMLRepNode, FLPSmx);
    EleNode = addName(EleNode, Elemx);
  }
  else if (!strcmp(ClassType, (char*)"OCTU")) {
    make_amloct(EleNode, Elemx, AMLRepNode, FLPSmx);
    EleNode = addName(EleNode, Elemx);
  }
  else if (!strcmp(ClassType, (char*)"MULT")) {
  }
  else if (!strcmp(ClassType, (char*)"SBEN")) {
    make_amlsben(EleNode, Elemx, AMLRepNode, FLPSmx);
  }
  else if (!strcmp(ClassType, (char*)"SOLENOID")) {
  }
  else if (!strcmp(ClassType, (char*)"LCAV")) {
  }
  else if (!strcmp(ClassType, (char*)"TCAV")) {
  }
  else if (!strcmp(ClassType, (char*)"XCOR")) {
    make_amlxcor(EleNode, Elemx, AMLRepNode, FLPSmx);
    EleNode = addName(EleNode, Elemx);
  }
  else if (!strcmp(ClassType, (char*)"YCOR")) {
    make_amlycor(EleNode, Elemx, AMLRepNode, FLPSmx);
    EleNode = addName(EleNode, Elemx);
  }
  else if (!strcmp(ClassType, (char*)"XYCOR")) {
  }
  else if (!strcmp(ClassType, (char*)"MONI") ||
    !strcmp(ClassType, (char*)"HMON" ) || !strcmp(ClassType, (char*)"VMON") ||
    !strcmp(ClassType, (char*)"INST") || !strcmp(ClassType, (char*)"PROF") || 
    !strcmp(ClassType, (char*)"WIRE") || !strcmp(ClassType, (char*)"BLMO") ||
    !strcmp(ClassType, (char*)"SLMO") || !strcmp(ClassType, (char*)"IMON")) {
    make_amlbpm(i, EleNode, Elemx, INSTRmx, FLINSTRmx);
  }
  else if (!strcmp(ClassType, (char*)"COLL")) {
  }

 /* Delete the dynamic memory for the Class string.*/
  delete ClassType;

  return EleNode;
}


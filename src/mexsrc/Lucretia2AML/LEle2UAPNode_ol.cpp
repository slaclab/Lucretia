#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

UAPNode* LEle2UAPNode(int i, UAPNode* AMLRepNode, mxArrayList LucEleList, mxArray *FLPSmx, 
                                                          mxArray *INSTRmx, mxArray *FLINSTRmx) {
  UAPNode *EleNode = AMLRepNode->addChild(ELEMENT_NODE, "element");

 /* Loop around the contents of the list.*/
  string prim_type;
  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
   /* Get the "Class" of this element structure.*/
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

    if ( !strcmp(ClassType, (char*)"QUAD") || !strcmp(ClassType, (char*)"OCTU") || 
         !strcmp(ClassType, (char*)"SBEN") || !strcmp(ClassType, (char*)"SOLENOID") ||
         !strcmp(ClassType, (char*)"LCAV") || !strcmp(ClassType, (char*)"TCAV") ||
         !strcmp(ClassType, (char*)"XCOR") || !strcmp(ClassType, (char*)"YCOR") ||
         !strcmp(ClassType, (char*)"SEXT") || !strcmp(ClassType, (char*)"MONI") ||
         !strcmp(ClassType, (char*)"HMON") || !strcmp(ClassType, (char*)"VMON") ||
         !strcmp(ClassType, (char*)"INST") || !strcmp(ClassType, (char*)"PROF") ||
         !strcmp(ClassType, (char*)"WIRE") || !strcmp(ClassType, (char*)"BLMO") ||
         !strcmp(ClassType, (char*)"SLMO") || !strcmp(ClassType, (char*)"IMON") ) {
      prim_type = ClassType;
    }

   /* Delete the dynamic memory for the Class string.*/
    delete ClassType;
  }

  if (!strcmp((char*)"QUAD", prim_type.c_str())) {
    unsplitquad(EleNode, LucEleList, AMLRepNode, FLPSmx);
  }
  else if (!strcmp((char*)"XCOR", prim_type.c_str())) {
    unsplitxcor(EleNode, LucEleList, AMLRepNode, FLPSmx);
  }
  else if (!strcmp((char*)"YCOR", prim_type.c_str())) {
    unsplitycor(EleNode, LucEleList, AMLRepNode, FLPSmx);
  }
  else if (!strcmp((char*)"MONI", prim_type.c_str()) || 
    !strcmp((char*)"HMON", prim_type.c_str()) || !strcmp((char*)"VMON", prim_type.c_str()) ||
    !strcmp((char*)"INST", prim_type.c_str()) || !strcmp((char*)"PROF", prim_type.c_str()) ||
    !strcmp((char*)"WIRE", prim_type.c_str()) || !strcmp((char*)"BLMO", prim_type.c_str()) ||
    !strcmp((char*)"SLMO", prim_type.c_str()) || !strcmp((char*)"IMON", prim_type.c_str()) ) {
    unsplitbpm(i, EleNode, LucEleList, AMLRepNode, INSTRmx, FLINSTRmx);
  }
  else if (!strcmp((char*)"OCTU", prim_type.c_str())) {
  }
  else if (!strcmp((char*)"SBEN", prim_type.c_str())) {
    unsplitsben(EleNode, LucEleList, AMLRepNode, FLPSmx);
  }
  else if (!strcmp((char*)"SOLENOID", prim_type.c_str())) {
  }
  else if (!strcmp((char*)"LCAV", prim_type.c_str())) {
  }
  else if (!strcmp((char*)"TCAV", prim_type.c_str())) {
  }
  else if (!strcmp((char*)"SEXT", prim_type.c_str())) {
    unsplitsext(EleNode, LucEleList, AMLRepNode, FLPSmx);
  }

  return EleNode;

}


#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void unsplitbpm(int i, UAPNode *EleNode, mxArrayList LucEleList, UAPNode *AMLRepNode, 
                                                           mxArray *INSTRmx, mxArray *FLINSTRmx) {
  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
   /* Get the "Class" of this element structure.*/
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

    if ( !strcmp(ClassType, (char*)"MONI") ||
    !strcmp(ClassType, (char*)"HMON" ) || !strcmp(ClassType, (char*)"VMON") ||
    !strcmp(ClassType, (char*)"INST") || !strcmp(ClassType, (char*)"PROF") || 
    !strcmp(ClassType, (char*)"WIRE") || !strcmp(ClassType, (char*)"BLMO") ||
    !strcmp(ClassType, (char*)"SLMO") || !strcmp(ClassType, (char*)"IMON")) {
      make_amlbpm(i, EleNode, Elemx, INSTRmx, FLINSTRmx);
    }
    /*else if ( !strcmp(ClassType, (char*)"MARK") ) {
      make_amlmarker(EleNode, Elemx);
    }*/
  }

  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
   /* Get the "Class" of this element structure.*/
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

    if ( !strcmp(ClassType, (char*)"MONI") ||
    !strcmp(ClassType, (char*)"HMON" ) || !strcmp(ClassType, (char*)"VMON") ||
    !strcmp(ClassType, (char*)"INST") || !strcmp(ClassType, (char*)"PROF") || 
    !strcmp(ClassType, (char*)"WIRE") || !strcmp(ClassType, (char*)"BLMO") ||
    !strcmp(ClassType, (char*)"SLMO") || !strcmp(ClassType, (char*)"IMON")) {
      addName(EleNode, Elemx);
    }
  }

  return;
}


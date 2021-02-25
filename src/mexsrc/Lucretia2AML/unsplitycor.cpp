#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void unsplitycor(UAPNode *EleNode, mxArrayList LucEleList, UAPNode *AMLRepNode, mxArray *FLPSmx) {
  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
   /* Get the "Class" of this element structure.*/
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

    if ( !strcmp(ClassType, (char*)"YCOR") ) {
      make_amlycor(EleNode, Elemx, AMLRepNode, FLPSmx);
      EleNode = addName(EleNode, Elemx);
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

    if ( !strcmp(ClassType, (char*)"YCOR") ) {
      addName(EleNode, Elemx);
    }
  }

  return;
}


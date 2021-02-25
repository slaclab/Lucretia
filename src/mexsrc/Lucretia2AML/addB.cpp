#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"

using namespace std;

UAPNode* addB(UAPNode *BNode, mxArray *Elemx) {

  double c_light = 0.299792458;	
  bool ok;
  double Ldoub = mxGetScalar( mxGetField(Elemx, 0, "L") );

  mxArray *Bmx = mxGetField(Elemx, 0, "B");
  mxArray *Pmx = mxGetField(Elemx, 0, "P");
  double Bdoub = mxGetScalar(Bmx);
  double Pdoub = mxGetScalar(Pmx);
  Bdoub /= (Pdoub/c_light);

  mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
  char *ClassType;
  int strsize = mxGetN(ClassVal)+1;
  ClassType = new char[strsize];
  mxGetString(ClassVal, ClassType, strsize);

  if (!strcmp(ClassType, (char*)"XCOR") || !strcmp(ClassType, (char*)"YCOR") || 
          !strcmp(ClassType, (char*)"XYCOR")) {
    BNode->addAttribute("design", BasicUtilities::double_to_string(Bdoub, ok), false);
  }
  else {
    BNode->addAttribute("design", BasicUtilities::double_to_string(Bdoub/Ldoub, ok), false);
  }

  return BNode;

}


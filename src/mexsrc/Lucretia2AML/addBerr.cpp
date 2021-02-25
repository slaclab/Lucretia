#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"

using namespace std;

UAPNode* addBerr(UAPNode *BNode, mxArray *Elemx) {
  double c_light = 0.299792458;
  bool ok;
  double Ldoub = mxGetScalar( mxGetField(Elemx, 0, "L") );

  mxArray *dBmx = mxGetField(Elemx, 0, "dB");
  mxArray *Pmx = mxGetField(Elemx, 0, "P");
  double dBdoub = mxGetScalar(dBmx);
  double Pdoub = mxGetScalar(Pmx);
  dBdoub /= (Pdoub/c_light);

  mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
  char *ClassType;
  int strsize = mxGetN(ClassVal)+1;
  ClassType = new char[strsize];
  mxGetString(ClassVal, ClassType, strsize);

  if (!strcmp(ClassType, (char*)"XCOR") || !strcmp(ClassType, (char*)"YCOR") || 
          !strcmp(ClassType, (char*)"XYCOR")) {
    BNode->addAttribute("err", BasicUtilities::double_to_string(dBdoub, ok), false);
  }
  else {
    BNode->addAttribute("err", BasicUtilities::double_to_string(dBdoub/Ldoub, ok), false);
  }

  return BNode;
}


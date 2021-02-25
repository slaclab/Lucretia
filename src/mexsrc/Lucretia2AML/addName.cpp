#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"

using namespace std;

UAPNode* addName(UAPNode *EleNode, mxArray *Elemx) {
  mxArray *Namemx = mxGetField(Elemx, 0, "Name");
  int Namelength = mxGetN(Namemx);
  char *Namechar;
  Namechar = new char[Namelength+1];
  mxGetString(Namemx, Namechar, Namelength+1);
  string Namestr(Namechar);
  EleNode->addAttribute("name", Namestr, false);

  delete Namechar;

  return EleNode;
}



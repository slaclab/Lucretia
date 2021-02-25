#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"

using namespace std;

void make_amlmarker(UAPNode *EleNode, mxArray *Elemx) {
  UAPNode *MarkNode = EleNode->addChild(ELEMENT_NODE, "marker");

  mxArray *Namemx = mxGetField(Elemx, 0, "Name");
  int Namelength = mxGetN(Namemx);
  char *Namechar;
  Namechar = new char[Namelength+1];
  mxGetString(Namemx, Namechar, Namelength+1);
  string Namestr(Namechar);
  EleNode->addAttribute("name", Namestr, false);

  delete Namechar;
}


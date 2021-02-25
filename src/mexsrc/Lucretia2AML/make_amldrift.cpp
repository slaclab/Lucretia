#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"

using namespace std;

void make_amldrift(UAPNode *EleNode, mxArray *Elemx) {
 /* EleNode is a pointer to the UAPNode for this element.
  * Elemx is a pointer to the Matlab data structure.*/
  bool ok;

 /* Extract the elements name from the Matlab structure.*/
  mxArray *Namemx = mxGetField(Elemx, 0, "Name");
  int Namelength = mxGetN(Namemx);
  char *Namechar;
  Namechar = new char[Namelength+1];
  mxGetString(Namemx, Namechar, Namelength+1);
  string Namestr(Namechar);
 /* Add the "name" attribute with the string "Namestr".*/
  EleNode->addAttribute("name", Namestr, false);
 /* Don't cause a memory leak.*/
  delete Namechar;

 /* Get the length field, and convert it to a C++ double*/
  double Ldoub = mxGetScalar( mxGetField(Elemx, 0, "L") );
 /* Add a child node called "length" to EleNode.*/
  UAPNode *LengthNode = EleNode->addChild(ELEMENT_NODE, "length");
 /* Add the design attribute with a value of Ldoub.*/
  LengthNode->addAttribute("design", BasicUtilities::double_to_string(Ldoub, ok), false);
}


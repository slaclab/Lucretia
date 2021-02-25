#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "AML/AMLReader.hpp"
#include "AML/AMLLatticeExpander.hpp"
#include "mex.h"
#include "matrix.h"

#include "AML2Lucretia.hpp"

using namespace std;

mxArray* make_marker(UAPNode* EleNode, beamdef beamparams){
 /* Store the element's name in "Name"*/
  UAPAttribute* EleNameAttrib = EleNode->getAttribute("name");
  string Name = EleNameAttrib->getValue();

  mxArray* MarkerStruc;
  mxArray* rhs[1];
  mxArray* Namemx = mxCreateString(Name.c_str());
  rhs[0] = Namemx;
  if (mexCallMATLAB(1,&MarkerStruc,1,rhs,"MarkerStruc")) cout << "Didn't work :(" << endl;
  mxDestroyArray(Namemx);

  return MarkerStruc;
}


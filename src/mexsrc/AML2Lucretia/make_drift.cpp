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

mxArray* make_drift(UAPNode* EleNode, beamdef beamparams){
  bool ok;
 /* Class string will always be "SEXT" for sextupoles. */
  string Class = "SEXT";

 /* Store the element's name in "Name"*/
  UAPAttribute* EleNameAttrib = EleNode->getAttribute("name");
  string Name = EleNameAttrib->getValue();

 /* Find the design and error length: Length_design & Length_err
 *   * Extract them as strings from their UAPAttributes*/
  UAPNode* LengthNode = EleNode->getChildByName("length");
  UAPAttribute* LDesAttrib = LengthNode->getAttribute("design");
  string Length_design_str = LDesAttrib->getValue();
 /* Now use BasicUtilities to convert these to doubles*/
  double Length_design = BasicUtilities::string_to_double(Length_design_str,ok);

  mxArray* DriftStruc;
  mxArray* Length_designmx = mxCreateDoubleScalar(Length_design);
  mxArray* rhs[2];
  mxArray* Namemx = mxCreateString(Name.c_str());
  rhs[0] = Length_designmx;
  rhs[1] = Namemx;
  if (mexCallMATLAB(1,&DriftStruc,2,rhs,"DrifStruc")) cout << "Didn't work :(" << endl;

  mxDestroyArray(Length_designmx);
  mxDestroyArray(Namemx);

  return DriftStruc;
}


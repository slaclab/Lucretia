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

mxArray* make_quad(UAPNode* EleNode, beamdef beamparams){
 /* Class string will always be "QUAD" for quadrupoles. */
  string Class = "QUAD";

  magmake quadstruc = make_magnet(EleNode, beamparams);

 /* We now have enough info to call SextStruc to make the quadupole structure.*/
  mxArray* QuadStruc;
  mxArray* rhs[5];
  mxArray* Length_designmx = mxCreateDoubleScalar(quadstruc.L);
  mxArray* Bmx = mxCreateDoubleScalar(quadstruc.B);
  mxArray* Tiltmx = mxCreateDoubleScalar(quadstruc.Tilt);
  mxArray* Apermx = mxCreateDoubleScalar(quadstruc.Aper);
  mxArray* Namemx = mxCreateString(quadstruc.Name);
  rhs[0] = Length_designmx;
  rhs[1] = Bmx;
  rhs[2] = Tiltmx;
  rhs[3] = Apermx;
  rhs[4] = Namemx;
  mexCallMATLAB(0,NULL,1,&rhs[4],"disp");
  if (mexCallMATLAB(1,&QuadStruc,5,rhs,"QuadStruc")) cout << "Didn't work :(" << endl;

  mxArray* dBmx = mxCreateDoubleScalar(quadstruc.dB);
  mxSetField(QuadStruc, 0, "dB", dBmx);

  mxDestroyArray(Length_designmx);
  mxDestroyArray(Bmx);
  mxDestroyArray(Tiltmx);
  mxDestroyArray(Apermx);
  mxDestroyArray(Namemx);

  return QuadStruc;
}


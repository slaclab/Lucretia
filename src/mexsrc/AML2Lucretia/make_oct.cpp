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

mxArray* make_oct(UAPNode* EleNode, beamdef beamparams){
 /* Class string will always be "QUAD" for octrupoles. */
  string Class = "QUAD";

  magmake octstruc = make_magnet(EleNode, beamparams);

 /* We now have enough info to call QuadStruc to make the octupole structure.*/
  mxArray* OctuStruc;
  mxArray* rhs[5];
  mxArray* Length_designmx = mxCreateDoubleScalar(octstruc.L);
  mxArray* Bmx = mxCreateDoubleScalar(octstruc.B);
  mxArray* Tiltmx = mxCreateDoubleScalar(octstruc.Tilt);
  mxArray* Apermx = mxCreateDoubleScalar(octstruc.Aper);
  mxArray* Namemx = mxCreateString(octstruc.Name);
  rhs[0] = Length_designmx;
  rhs[1] = Bmx;
  rhs[2] = Tiltmx;
  rhs[3] = Apermx;
  rhs[4] = Namemx;
  mexCallMATLAB(0,NULL,1,&rhs[4],"disp");
  if (mexCallMATLAB(1,&OctuStruc,5,rhs,"OctuStruc")) cout << "Didn't work :(" << endl;

  mxArray* dBmx = mxCreateDoubleScalar(octstruc.dB);
  mxSetField(OctuStruc, 0, "dB", dBmx);

  mxDestroyArray(Length_designmx);
  mxDestroyArray(Bmx);
  mxDestroyArray(Tiltmx);
  mxDestroyArray(Apermx);
  mxDestroyArray(Namemx);

  return OctuStruc;
}


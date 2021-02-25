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

mxArray* make_sext(UAPNode* EleNode, beamdef beamparams){
 /* Class string will always be "SEXT" for sextupoles. */
  string Class = "SEXT";

  magmake sextstruc = make_magnet(EleNode, beamparams);

 /* We now have enough info to call SextStruc to make the sextupole structure.*/
  mxArray* SextStruc;
  mxArray* rhs[5];
  mxArray* Length_designmx = mxCreateDoubleScalar(sextstruc.L);
  mxArray* Bmx = mxCreateDoubleScalar(sextstruc.B);
  mxArray* Tiltmx = mxCreateDoubleScalar(sextstruc.Tilt);
  mxArray* Apermx = mxCreateDoubleScalar(sextstruc.Aper);
  mxArray* Namemx = mxCreateString(sextstruc.Name);
  rhs[0] = Length_designmx;
  rhs[1] = Bmx;
  rhs[2] = Tiltmx;
  rhs[3] = Apermx;
  rhs[4] = Namemx;
  mexCallMATLAB(0,NULL,1,&rhs[4],"disp");
  if (mexCallMATLAB(1,&SextStruc,5,rhs,"SextStruc")) cout << "Didn't work :(" << endl;

  mxArray* dBmx = mxCreateDoubleScalar(sextstruc.dB);
  mxSetField(SextStruc, 0, "dB", dBmx);

  mxDestroyArray(Length_designmx);
  mxDestroyArray(Bmx);
  mxDestroyArray(Tiltmx);
  mxDestroyArray(Apermx);
  mxDestroyArray(Namemx);

  return SextStruc;
}


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

mxArray* UAPNode2Lucretia (UAPNode* EleNode, beamdef beamparams) {
 /* This function accepts three UAPNode* as inputs:
  *      EleNode     -- The lattice element node.
  *      ControlNode -- The node containing the PS, KLYSTRON, information.
  *      GirderNode  -- The GIRDER node. */

  mxArray* EleStruc;

  if (EleNode->getChildByName("beambeam")) {
    EleStruc = make_beambeam(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("bend")) {
    EleStruc = make_bend(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("custom")) {
    EleStruc = make_drift(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("electric_kicker")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else if (EleNode->getChildByName("kicker")) {
    EleStruc = make_kicker(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("marker")) {
    EleStruc = make_marker(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("match")) {
    EleStruc = make_marker(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("multipole")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else if (EleNode->getChildByName("octupole")) {
    EleStruc = make_oct(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("patch")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else if (EleNode->getChildByName("quadrupole")) {
    EleStruc = make_quad(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("rf_cavity")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else if (EleNode->getChildByName("sextupole")) {
    EleStruc = make_sext(EleNode, beamparams);
  }

  else if (EleNode->getChildByName("solenoid")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else if (EleNode->getChildByName("taylor_map")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else if (EleNode->getChildByName("thin_multipole")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else if (EleNode->getChildByName("wiggler")) {
    const char* fname = "tempfield";
    EleStruc = mxCreateStructMatrix(1,1,1,&fname);
    mxSetField(EleStruc,1,"tempfield",mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL));
  }

  else {
    EleStruc = make_drift(EleNode, beamparams);
  }

  return EleStruc;

}


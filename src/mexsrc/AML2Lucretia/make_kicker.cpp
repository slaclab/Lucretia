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

mxArray* make_kicker(UAPNode* EleNode, beamdef beamparams){
  double c_light = 299792458;
  double xkickdes = 0;
  double xkickerr = 0;
  double ykickdes = 0;
  double ykickerr = 0;
  double lengthdes = 0;
  double lengtherr = 0;
  double tilt = 0;
  double type = 0;
  bool ok, is_x = false, is_y = false;
  mxArray* CorrStruc;

  UAPNode *lengthNode, *xkickNode, *x_ukickNode, *ykickNode;
  UAPNode *y_ukickNode, *xdispNode, *ydispNode, *orientnode, *kickerNode;

  kickerNode = EleNode->getChildByName("kicker");
  lengthNode = kickerNode->getChildByName("length");
  xkickNode = kickerNode->getChildByName("x_kick");
  x_ukickNode = kickerNode->getChildByName("x_kick_u");
  ykickNode = kickerNode->getChildByName("y_kick");
  y_ukickNode = kickerNode->getChildByName("y_kick_u");
  xdispNode = kickerNode->getChildByName("x_displacement");
  ydispNode = kickerNode->getChildByName("y_displacement");
  orientnode = kickerNode->getChildByName("orientation");
  
  if (xdispNode) {
    cout << "AML kicker specifies \"x_displacement\", ";
    cout << "which cannot be understood by Lucretia." << endl;
    cout << "Ignoring." << endl;
  }
  if (ydispNode) {
    cout << "AML kicker specifies \"y_displacement\", ";
    cout << "which cannot be understood by Lucretia." << endl;
    cout << "Ignoring." << endl;
  }
  if (xkickNode && x_ukickNode) {
    cout << "AML file specifies x_kick and x_kick_u.  Only parsing x_kick_u." << endl;
    xkickNode = NULL;
  }
  if (ykickNode && y_ukickNode) {
    cout << "AML file specifies y_kick and y_kick_u.  Only parsing y_kick_u." << endl;
    ykickNode = NULL;
  }

  if (xkickNode) {
    xkickdes = getUAPinfo(xkickNode,"design") * (beamparams.DesignBeamP / (c_light / 1e9));
    xkickerr = getUAPinfo(xkickNode,"err") * (beamparams.DesignBeamP / (c_light / 1e9));
    is_x = true;
  }

  if (ykickNode) {
    ykickdes = getUAPinfo(ykickNode,"design") * (beamparams.DesignBeamP / (c_light / 1e9));
    ykickerr = getUAPinfo(ykickNode,"err") * (beamparams.DesignBeamP / (c_light / 1e9));
    is_y = true;
  }

  if (x_ukickNode) {
    xkickdes = getUAPinfo(x_ukickNode,"design");
    xkickerr = getUAPinfo(x_ukickNode,"err");
    is_x = true;
  }

  if (y_ukickNode) {
    ykickdes = getUAPinfo(y_ukickNode,"design");
    ykickerr = getUAPinfo(y_ukickNode,"err");
    is_y = true;
  }

  if (lengthNode) {
    lengthdes = getUAPinfo(lengthNode,"design");
    lengtherr = getUAPinfo(lengthNode,"err");
  }

  if (orientnode) {
    if (UAPNode* tiltNode = orientnode->getChildByName("tilt")) {
      UAPAttribute* tiltDesAttrib = tiltNode->getAttribute("design");
      string tiltDesstr = tiltDesAttrib->getValue();
      tilt = BasicUtilities::string_to_double(tiltDesstr,ok);
    }
  }

  UAPAttribute* NameAttrib = EleNode->getAttribute("name");
  string name = NameAttrib->getValue();

  if (is_x && !is_y) {
    type = 1;
    mxArray* rhs[5];
    rhs[0] = mxCreateDoubleScalar(lengthdes);
    rhs[1] = mxCreateDoubleScalar(xkickdes);
    rhs[2] = mxCreateDoubleScalar(tilt);
    rhs[3] = mxCreateDoubleScalar(type);
    rhs[4] = mxCreateString(name.c_str());
    mexCallMATLAB(1,&CorrStruc,5,rhs,"CorrectorStruc");
  }
  else if (!is_x && is_y) {
    type = 2;
    mxArray* rhs[5];
    rhs[0] = mxCreateDoubleScalar(lengthdes);
    rhs[1] = mxCreateDoubleScalar(xkickdes);
    rhs[2] = mxCreateDoubleScalar(tilt);
    rhs[3] = mxCreateDoubleScalar(type);
    rhs[4] = mxCreateString(name.c_str());
    mexCallMATLAB(1,&CorrStruc,5,rhs,"CorrectorStruc");
  }
  else if (is_x && is_y) {
    type = 3;
    mxArray* rhs[5];
    rhs[0] = mxCreateDoubleScalar(lengthdes);
    rhs[1] = mxCreateDoubleScalar(xkickdes);
    rhs[2] = mxCreateDoubleScalar(tilt);
    rhs[3] = mxCreateDoubleScalar(type);
    rhs[4] = mxCreateString(name.c_str());
    mexCallMATLAB(1,&CorrStruc,5,rhs,"CorrectorStruc");
  }
  else {
    mxArray* Struc = make_marker(EleNode, beamparams);
    return Struc;
  }

  return CorrStruc;

}


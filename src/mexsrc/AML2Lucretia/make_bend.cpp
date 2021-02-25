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

mxArray* make_bend(UAPNode* EleNode, beamdef beamparams) {
 /* In order to calculate all the required bend fields, we need to know the
  * beam momentum at the entrance to this element.
  * Call some Lucretia things to figure this out.
  *
  * This function still needs a little work.  Specifically making the parameters
  * E, H, Hgap, and Fint, 1x2 arrays.*/
  bool ok;
  double c_light = 299792458;
  mxArray * BEAMLINEmx = mexGetVariable("global", "BEAMLINE");
  mxArray * BLLengthmx;
  mexCallMATLAB(1,&BLLengthmx,1,&BEAMLINEmx,"length");
  double P, BLLength = mxGetScalar(BLLengthmx);
  int i, lastelenum;
  for (i=0; i<BLLength; i++) {
    mxArray* BCell = mxGetCell(BEAMLINEmx, i);
    if (!BCell) { lastelenum = i; break; }
  }

  string Name;
  UAPAttribute* NameAttrib = EleNode->getAttribute("name");
  if (NameAttrib) { Name = NameAttrib->getValue(); }
  else { Name = "sbend"; }

  mxArray* rhs[3];
  rhs[0] = mxCreateDoubleScalar(1);
  rhs[1] = mxCreateDoubleScalar(lastelenum);
  rhs[2] = mxCreateDoubleScalar(0);
  if (mexCallMATLAB(0,NULL,3,rhs,"SetSPositions")) cout << "Didn't work :(" << endl;
  mxDestroyArray(*rhs);

  mxArray* rhs1[5];
  rhs1[0] = mxCreateDoubleScalar(1);
  rhs1[1] = mxCreateDoubleScalar(lastelenum);
  rhs1[2] = mxCreateDoubleScalar(beamparams.BunchPopulation);
  rhs1[3] = mxCreateDoubleScalar(beamparams.DesignBeamP);
  rhs1[4] = mxCreateDoubleScalar(0);
  mxArray* lhs;
  if (mexCallMATLAB(1,&lhs,5,rhs1,"UpdateMomentumProfile")) cout << "Didn't work :(" << endl;
  mxDestroyArray(*rhs1);

  BEAMLINEmx = mexGetVariable("global", "BEAMLINE");
  mxArray* LastStruc = mxGetCell(BEAMLINEmx, lastelenum-1);
  mxArray* Pmx = mxGetField(LastStruc, 0, "P");
  if (!Pmx) {cout << "ERROR!" << endl;}
  else { P = mxGetScalar(Pmx); }

  UAPNode* LNode = EleNode->getChildByName("length");
  UAPNode* BendNode = EleNode->getChildByName("bend");
  UAPNode* BNode = BendNode->getChildByName("g");
  UAPNode* BuNode = BendNode->getChildByName("g_u");
  UAPNode* OrientNode = BendNode->getChildByName("orientation");
  UAPNode* E1Node = BendNode->getChildByName("e1");
  UAPNode* E2Node = BendNode->getChildByName("e2");
  UAPNode* H1Node = BendNode->getChildByName("h1");
  UAPNode* H2Node = BendNode->getChildByName("h2");
  UAPNode* F_int1Node = BendNode->getChildByName("f_int1");
  UAPNode* F_int2Node = BendNode->getChildByName("f_int2");
  UAPNode* H_gap1Node = BendNode->getChildByName("h_gap1");
  UAPNode* H_gap2Node = BendNode->getChildByName("h_gap2");

  if (!BNode && !BuNode) {
    cout << "No magnetic field element.  Expanding as a drift." << endl;
    mxArray* EleStruc = make_drift(EleNode, beamparams);
    return EleStruc;
  }

  if (!LNode) {
    cout << "No length element.  Expanding as a marker." << endl;
    mxArray* EleStruc = make_marker(EleNode, beamparams);
    return EleStruc;
  }

  double gDes, gErr, LDes, LErr, E1Des, E2Des, H1Des, H2Des;
  double F_int1Des, F_int2Des, H_gap1Des, H_gap2Des;
  double B, dB, EDes, HDes, F_intDes, H_gapDes, Angle, Tilt;
 /* Extract the length from LNode. */
  LDes = getUAPinfo(LNode, "design");
  LErr = getUAPinfo(LNode, "err");
  if (!LErr) { LErr = 0; }

 /* Determine tilt of the bend.  If not present, set to zero.*/
  if (OrientNode) {
    if (UAPNode* tiltNode = OrientNode->getChildByName("tilt")) {
      UAPAttribute* tiltDesAttrib = tiltNode->getAttribute("design");
      string tiltDesstr = tiltDesAttrib->getValue();
      Tilt = BasicUtilities::string_to_double(tiltDesstr,ok);
    }
  }
  else { Tilt = 0; }

  if (BNode) {
   /* Extract the field from BNode. */
    gDes = getUAPinfo(BNode, "design") * (beamparams.DesignBeamP / (c_light / 1e9));
    gErr = getUAPinfo(BNode, "err") * (beamparams.DesignBeamP / (c_light / 1e9));
    if (!gErr) { gErr = 0; }
  }
  else if (BuNode) {
   /* Extract the field from BNode. */
    gDes = getUAPinfo(BuNode, "design");
    gErr = getUAPinfo(BuNode, "err");
    if (!gErr) { gErr = 0; }
  }
  B = gDes * LDes;
  dB = ((gDes + gErr) * (LDes * LErr)) - B;
  Angle = B * ((c_light / 1e9) / beamparams.DesignBeamP);

 /* Now get e1 and e2. */
  E1Des = getUAPinfo(E1Node, "design");
  if (!E1Des) {E1Des = 0;}
  E2Des = getUAPinfo(E2Node, "design");
  if (!E2Des) {E2Des = 0;}
  EDes = (E1Des + E2Des) / 2;

 /* h1 and h2. Then find the average.*/
  H1Des = getUAPinfo(H1Node, "design");
  if (!H1Des) {H1Des = 0;}
  H2Des = getUAPinfo(H2Node, "design");
  if (!H2Des) {H2Des = 0;}
  HDes = (H1Des + H2Des) / 2;
  
 /* f_int1 and f_int2. Find the average.*/
  F_int1Des = getUAPinfo(F_int1Node, "design");
  if (!F_int1Des) {F_int1Des = 0;}
  F_int2Des = getUAPinfo(F_int2Node, "design");
  if (!F_int2Des) {F_int2Des = 0;}
  F_intDes = (F_int1Des + F_int2Des) / 2;

 /* h_gap1 and h_gap2. And average.*/
  H_gap1Des = getUAPinfo(H_gap1Node, "design");
  if (!H_gap1Des) {H_gap1Des = 0;}
  H_gap2Des = getUAPinfo(H_gap2Node, "design");
  if (!H_gap2Des) {H_gap2Des = 0;}
  H_gapDes = (H_gap1Des + H_gap2Des) / 2;

 /* We now have enough info to call SextStruc to make the quadupole structure.*/
  mxArray* SBendStruc;
  mxArray* rhs2[9];
  mxArray* Length_designmx = mxCreateDoubleScalar(LDes);
  mxArray* Bmx = mxCreateDoubleScalar(B);
  mxArray* Anglemx = mxCreateDoubleScalar(Angle);
  mxArray* Emx = mxCreateDoubleScalar(EDes);
  mxArray* Hmx = mxCreateDoubleScalar(HDes);
  mxArray* Hgapmx = mxCreateDoubleScalar(H_gapDes);
  mxArray* Fintmx = mxCreateDoubleScalar(F_intDes);
  mxArray* Tiltmx = mxCreateDoubleScalar(Tilt);
  mxArray* Namemx = mxCreateString(Name.c_str());
  rhs2[0] = Length_designmx;
  rhs2[1] = Bmx;
  rhs2[2] = Anglemx;
  rhs2[3] = Emx;
  rhs2[4] = Hmx;
  rhs2[5] = Hgapmx;
  rhs2[6] = Fintmx;
  rhs2[7] = Tiltmx;
  rhs2[8] = Namemx;
  if (mexCallMATLAB(1,&SBendStruc,9,rhs2,"SBendStruc")) {
    cout << "Didn't work :(" << endl;
    return NULL;
  }

  return SBendStruc;
  
}


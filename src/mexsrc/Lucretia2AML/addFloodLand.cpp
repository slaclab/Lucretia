#include <iostream>
#include "UAPUtilities.hpp"
#include "AMLReader.hpp"
#include "AMLLatticeExpander.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void addFloodLand(UAPNode *CtrlSysNode, mxArray *FLGIRDSmx, double GirderNum) {

  bool ok;
  mxArray *protocol = mxGetField(FLGIRDSmx, (int)GirderNum-1, "protocol");
  if (protocol) {
    UAPNode *ProtoNode = CtrlSysNode->addChild("protocol");
    ProtoNode->addAttribute("value", mxArrayToString(protocol));
  }

  mxArray *preCommandmx = mxGetField(FLGIRDSmx, (int)GirderNum-1, "preCommand");
  if (preCommandmx) {
    UAPNode *PreNode = CtrlSysNode->addChild("precommands");
    mxArray *preComRmx = mxGetCell(preCommandmx, 0);
    mxArray *preComWmx = mxGetCell(preCommandmx, 1);
    if (preComRmx) {
      UAPNode *PreReadNode = PreNode->addChild("read");
      int preComRmxsize = mxGetN(preComRmx);
      for (int i=0; i<preComRmxsize; i++) {
        mxArray *W1 = mxGetCell(preComRmx, i);
        string W1str;
        if (mxIsChar(W1)) mxArrayToString(W1);
        else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
        UAPNode *ValNode = PreReadNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
        ValNode->addAttribute("value", W1str);
      }
    }
    if (preComWmx) {
      UAPNode *PreWriteNode = PreNode->addChild("write");
      int preComWmxsize = mxGetN(preComWmx);
      for (int i=0; i<preComWmxsize; i++) {
        mxArray *W1 = mxGetCell(preComWmx, i);
        string W1str;
        if (mxIsChar(W1)) mxArrayToString(W1);
        else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
        UAPNode *ValNode = PreWriteNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
        ValNode->addAttribute("value", W1str);
      }
    }
  }

  mxArray *postCommandmx = mxGetField(FLGIRDSmx, (int)GirderNum-1, "postCommand");
  if (postCommandmx) {
    UAPNode *PostNode = CtrlSysNode->addChild("postcommands");
    mxArray *postComRmx = mxGetCell(postCommandmx, 0);
    mxArray *postComWmx = mxGetCell(postCommandmx, 1);
    if (postComRmx) {
      UAPNode *PostReadNode = PostNode->addChild("read");
      int postComRmxsize = mxGetN(postComRmx);
      for (int i=0; i<postComRmxsize ; i++) {
        mxArray *W1 = mxGetCell(postComRmx, i);
        string W1str;
        if (mxIsChar(W1)) W1str = mxArrayToString(W1);
        else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
        UAPNode *ValNode = PostReadNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
        ValNode->addAttribute("value", W1str);
      }
    }
    if (postComWmx) {
      UAPNode *PostWriteNode = PostNode->addChild("write");
      int postComWsize = mxGetN(postComWmx);
      for (int i=0; i<postComWsize; i++) {
        mxArray *W1 = mxGetCell(postComWmx, i);
        string W1str;
        if (mxIsChar(W1)) W1str = mxArrayToString(W1);
        else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
        UAPNode *ValNode = PostWriteNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
        ValNode->addAttribute("value", W1str);
      }
    }
  }

  mxArray *pvname = mxGetField(FLGIRDSmx, (int)GirderNum-1, "pvname");
  if (pvname) {
    UAPNode *PVNameNode = CtrlSysNode->addChild("pvname");
    UAPNode *PVNameReadNode = PVNameNode->addChild("read");
    UAPNode *PVNameWriteNode = PVNameNode->addChild("write");
    mxArray *pvnameRmx = mxGetCell(pvname, 0);
    mxArray *pvnameWmx = mxGetCell(pvname, 1);

    for (int i=0; i<mxGetN(pvnameRmx) ; i++) {
      string dofstr = "DOF" + BasicUtilities::double_to_string((double)i+1, ok);
      UAPNode *PVNameReadDOF = PVNameReadNode->addChild(dofstr);
      PVNameReadDOF->addAttribute("value", mxArrayToString(mxGetCell(pvnameRmx, i)));
    }

    for (int i=0; i<mxGetN(pvnameWmx) ; i++) {
      string dofstr = "DOF" + BasicUtilities::double_to_string((double)i+1, ok);
      UAPNode *PVNameWriteDOF = PVNameWriteNode->addChild(dofstr);
      PVNameWriteDOF->addAttribute("value", mxArrayToString(mxGetCell(pvnameWmx, i)));
    }
  }

  mxArray *units = mxGetField(FLGIRDSmx, (int)GirderNum-1, "units");
  if (units) {
    UAPNode *UnitsNode = CtrlSysNode->addChild("units");
    UAPNode *UnitsReadNode = UnitsNode->addChild("read");
    UAPNode *UnitsWriteNode = UnitsNode->addChild("write");
    UnitsReadNode->addAttribute("value", mxArrayToString(mxGetCell(units, 0)));
    UnitsWriteNode->addAttribute("value", mxArrayToString(mxGetCell(units, 1)));
  }

}


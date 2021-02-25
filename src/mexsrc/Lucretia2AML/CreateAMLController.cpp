#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "math.h"
#include "Lucretia2AML.hpp"

using namespace std;

void CreateAMLController(UAPNode *LabNode, double PSnum, string defattrib, 
                                  mxArray *ElePSmx, string targetname, double EleB, mxArray *FLPSmx) {
 /* Adds a controller node to the current AML node, and populates it with PS info.*/
  bool ok;

 /* Add the controller node to the main AML root.*/
  UAPNode *ContNode = LabNode->addChild(ELEMENT_NODE, "controller");

 /* Figure out the basic properties of this PS.*/
  double Step = mxGetScalar( mxGetField(ElePSmx, (int)PSnum-1, "Step") );
  double Design = mxGetScalar( mxGetField(ElePSmx, (int)PSnum-1, "Ampl") );
  string namestr = "PS" + BasicUtilities::double_to_string(PSnum, ok);
  string targetexp = BasicUtilities::double_to_string(EleB, ok) + " * " + namestr + "[@actual]";

  ContNode->addAttribute("name", namestr, false);
  ContNode->addAttribute("variation", "ABSOLUTE", false);
  ContNode->addAttribute("design", BasicUtilities::double_to_string(Design, ok) , false);
  ContNode->addAttribute("step", BasicUtilities::double_to_string(Step, ok), false);
  ContNode->addAttribute("default_attribute", defattrib, false);

  UAPNode *SlaveNode = ContNode->addChild(ELEMENT_NODE, "slave");
  SlaveNode->addAttribute("target", targetname, false);
  SlaveNode->addAttribute("expression", targetexp, false);

  if (FLPSmx && mxGetN(FLPSmx)>0) {
    UAPNode *CtrlNode = ContNode->addChild("control_sys");
    mxArray *Unitsmx = mxGetField(FLPSmx, (int)PSnum-1, "units");
    if (Unitsmx && mxGetN(Unitsmx)>0) {
      UAPNode *UnitsNode = CtrlNode->addChild("units");
      UAPNode *ReadNode = UnitsNode->addChild("read");
      UAPNode *WriteNode = UnitsNode->addChild("write");
      int Unitsize = mxGetN(Unitsmx);
      if (Unitsize==1) {
        ReadNode->addAttribute("value", mxArrayToString(mxGetCell(Unitsmx, 0)));
        WriteNode->addAttribute("value", mxArrayToString(mxGetCell(Unitsmx, 0)));
      }
      else if (Unitsize==2) {
        ReadNode->addAttribute("value", mxArrayToString(mxGetCell(Unitsmx, 0)));
        WriteNode->addAttribute("value", mxArrayToString(mxGetCell(Unitsmx, 1)));
      }
    }

    mxArray *PVNamemx = mxGetField(FLPSmx, (int)PSnum-1, "pvname");
    if (PVNamemx && mxGetN(PVNamemx)>0) {
      UAPNode *PVNameNode = CtrlNode->addChild("pvname");
      UAPNode *ReadNode = PVNameNode->addChild("read");
      UAPNode *WriteNode = PVNameNode->addChild("write");
      mxArray *ReadCell = mxGetCell(PVNamemx, 0);
      mxArray *WriteCell = mxGetCell(PVNamemx, 1);
      ReadNode->addAttribute("value", mxArrayToString(mxGetCell(ReadCell, 0)));
      WriteNode->addAttribute("value", mxArrayToString(mxGetCell(WriteCell, 0)));
    }

    mxArray *Postmx = mxGetField(FLPSmx, (int)PSnum-1, "postCommand");
    if (Postmx && mxGetN(Postmx)>0) {
      UAPNode *PostNode = CtrlNode->addChild("postCommand");
      mxArray *Readmx = mxGetCell(Postmx, 0);
      if (Readmx && mxGetN(Readmx)>0){
        UAPNode *ReadNode = PostNode->addChild("read");
        int readsize = mxGetN(Readmx);
        for (int i=0; i<readsize; i++) {
          mxArray *W1 = mxGetCell(Readmx, i);
          string W1str;
          if (mxIsChar(W1)) W1str = mxArrayToString(W1);
          else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
          UAPNode *ValNode = ReadNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
          ValNode->addAttribute("value", W1str);
        }
      }
      mxArray *Writemx = mxGetCell(Postmx, 1);
      if (Writemx && mxGetN(Writemx)>0){
        UAPNode *WriteNode = PostNode->addChild("write");
        int writesize = mxGetN(Writemx);
        for (int i=0; i<writesize; i++) {
          mxArray *W1 = mxGetCell(Writemx, i);
          string W1str;
          if (mxIsChar(W1)) W1str = mxArrayToString(W1);
          else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
          UAPNode *ValNode = WriteNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
          ValNode->addAttribute("value", W1str);
        }
      }
    }

    mxArray *Premx = mxGetField(FLPSmx, (int)PSnum-1, "preCommand");
    if (Premx && mxGetN(Premx)>0) {
      UAPNode *PreNode = CtrlNode->addChild("preCommand");
      mxArray *Readmx = mxGetCell(Premx, 0);
      if (Readmx && mxGetN(Readmx)>0){
        UAPNode *ReadNode = PreNode->addChild("read");
        int readsize = mxGetN(Readmx);
        for (int i=0; i<readsize; i++) {
          mxArray *W1 = mxGetCell(Readmx, i);
          string W1str;
          if (mxIsChar(W1)) W1str = mxArrayToString(W1);
          else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
          UAPNode *ValNode = ReadNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
          ValNode->addAttribute("value", W1str);
        }
      }
      mxArray *Writemx = mxGetCell(Premx, 1);
      if (Writemx && mxGetN(Writemx)>0){
        UAPNode *WriteNode = PreNode->addChild("write");
        int writesize = mxGetN(Writemx);
        for (int i=0; i<writesize; i++) {
          mxArray *W1 = mxGetCell(Writemx, i);
          string W1str;
          if (mxIsChar(W1)) W1str = mxArrayToString(W1);
          else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
          UAPNode *ValNode = WriteNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
          ValNode->addAttribute("value", W1str);
        }
      }
    }

    mxArray *Initmx = mxGetField(FLPSmx, (int)PSnum-1, "init");
    if (Initmx && mxGetN(Initmx)>0) {
      UAPNode *InitNode = CtrlNode->addChild("init");
      mxArray *Initcellmx = mxGetCell(Initmx, 0);
      int Initsize = mxGetN(Initcellmx);
      for (int i=0; i<Initsize; i++) {
        mxArray *W1 = mxGetCell(Initcellmx, i);
        string W1str;
        if (mxIsChar(W1)) W1str = mxArrayToString(W1);
        else if (mxIsDouble(W1)) W1str = BasicUtilities::double_to_string(mxGetScalar(W1), ok);
        UAPNode *ValNode = InitNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
        ValNode->addAttribute("value", W1str);
      }
    }

    mxArray *Offmx = mxGetField(FLPSmx, (int)PSnum-1, "off");
    if (Offmx && mxGetN(Offmx)>0) {
      UAPNode *OffNode = CtrlNode->addChild("off");
      int Offsize = mxGetN(Offmx);
      string W1str;
      for (int i=0; i<Offsize; i++) {
        mxArray *Offcellmx = mxGetCell(Offmx, i);
        if (mxIsChar(Offcellmx)) W1str = mxArrayToString(Offcellmx);
	else if (mxIsDouble(Offcellmx)) W1str = BasicUtilities::double_to_string(mxGetScalar(Offcellmx), ok);
        UAPNode *ValNode = OffNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
        ValNode->addAttribute("value", W1str);
      }
    }

    mxArray *Onmx = mxGetField(FLPSmx, (int)PSnum-1, "on");
    if (Onmx && mxGetN(Onmx)>0) {
      UAPNode *OnNode = CtrlNode->addChild("on");
      int Onsize = mxGetN(Onmx);
      string W1str;
      for (int i=0; i<Onsize; i++) {
        mxArray *Oncellmx = mxGetCell(Onmx, i);
        if (mxIsChar(Oncellmx)) W1str = mxArrayToString(Oncellmx);
	else if (mxIsDouble(Oncellmx)) W1str = BasicUtilities::double_to_string(mxGetScalar(Oncellmx), ok);
        UAPNode *ValNode = OnNode->addChild("Str" + BasicUtilities::double_to_string(i+1, ok));
        ValNode->addAttribute("value", W1str);
      }
    }

    mxArray *Protomx = mxGetField(FLPSmx, (int)PSnum-1, "protocol");
    if (Protomx && mxGetN(Protomx)>0) {
      UAPNode *ProtoNode = CtrlNode->addChild("protocol");
      ProtoNode->addAttribute("value", mxArrayToString(Protomx));
    }

    mxArray *Unipolarmx = mxGetField(FLPSmx, (int)PSnum-1, "unipolar");
    if (Unipolarmx && mxIsLogicalScalar(Unipolarmx)) {
      UAPNode *UnipolarNode = CtrlNode->addChild("unipolar");
      if (mxIsLogicalScalarTrue(Unipolarmx)) UnipolarNode->addAttribute("value","true");
      else UnipolarNode->addAttribute("value","false");
    }

    mxArray *nt_ratiomx = mxGetField(FLPSmx, (int)PSnum-1, "nt_ratio");
    if (nt_ratiomx && mxIsNumeric(nt_ratiomx)) {
      UAPNode *nt_ratioNode = CtrlNode->addChild("nt_ratio");
      nt_ratioNode->addAttribute("value", BasicUtilities::double_to_string(mxGetScalar(nt_ratiomx), ok));
    }
  }

  return;

}


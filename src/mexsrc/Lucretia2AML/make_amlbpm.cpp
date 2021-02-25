#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void make_amlbpm(int i, UAPNode *EleNode, mxArray *Elemx, mxArray *INSTRmx, mxArray *FLINSTRmx) {
  bool ok;
  UAPNode *InstrNode = EleNode->addChild(ELEMENT_NODE, "instrument");

 /* Get the "Class" of this element structure.*/
  mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
 /* Reserve some dynamic memory for the Class string.*/
  char *ClassType;
  int strsize = mxGetN(ClassVal)+1;
  ClassType = new char[strsize];
  mxGetString(ClassVal, ClassType, strsize);

 /* Add the ClassType as the "type" of monitor.*/
  InstrNode->addAttribute("type", ClassType, false);
  delete ClassType;

 /* Add its name*/
  EleNode = addName(EleNode, Elemx);

 /* Figure out its orientation.*/
  make_amlorient(EleNode, Elemx);

 /* Determine the electrical offset if necessary.*/
  double exoffsdoub = 0, eyoffsdoub = 0;
  mxArray *ElecOffsmx = mxGetField(Elemx, 0, "ElecOffset");
  if (ElecOffsmx) {
    double *ElecOffsptr = mxGetPr(ElecOffsmx);
    exoffsdoub = *ElecOffsptr;
    eyoffsdoub = *(ElecOffsptr+1);
  }

 /* Get the orientation in order to add the elec offset to it.*/
  UAPNode *xoffsNode, *yoffsNode;
  UAPNode *OrientNode = EleNode->getChildByName("orientation");
  if (OrientNode) {
    xoffsNode= OrientNode->getChildByName("x_offset");
    yoffsNode = OrientNode->getChildByName("y_offset");
  }

 /* Add the offsets, and correct x_offset and y_offset*/
  double xoffsdoub = 0, yoffsdoub = 0;
  if (xoffsNode) {
    UAPAttribute *xoffsAttrib = xoffsNode->getAttribute("design");
    string xoffsstr = xoffsAttrib->getValue();
    xoffsdoub = BasicUtilities::string_to_double(xoffsstr, ok);
    xoffsAttrib->setValue(BasicUtilities::double_to_string(xoffsdoub+exoffsdoub, ok));
  }
  if (yoffsNode) {
    UAPAttribute *yoffsAttrib = yoffsNode->getAttribute("design");
    string yoffsstr = yoffsAttrib->getValue();
    yoffsdoub = BasicUtilities::string_to_double(yoffsstr, ok);
    yoffsAttrib->setValue(BasicUtilities::double_to_string(yoffsdoub+eyoffsdoub, ok));
  }

 /* Figure out if there's any Floodland info that needs to be used*/
  mxArray *InstrStruc;
  if (INSTRmx && FLINSTRmx && mxGetN(INSTRmx)>0 && mxGetN(FLINSTRmx)>0) {
    for (int n=0; n<mxGetN(INSTRmx); n++) {
      InstrStruc = mxGetCell(INSTRmx, n);
      if (i==(int)mxGetScalar(mxGetField(InstrStruc, 0, "Index"))-1) {
        UAPNode *CtrlNode = EleNode->addChild("control_sys");

        mxArray *Unitsmx = mxGetField(FLINSTRmx, n, "units");
        if (Unitsmx) {
          UAPNode *UnitsNode = CtrlNode->addChild("units");
          UnitsNode->addAttribute("value", mxArrayToString(Unitsmx));
        }

        mxArray *PVmx = mxGetField(FLINSTRmx, n, "pvname");
        if (PVmx) {
          UAPNode *PVNode = CtrlNode->addChild("pvname");
          PVNode->addAttribute("value", mxArrayToString(mxGetCell(mxGetCell(PVmx, 0), 0)));
        }

        mxArray *Protomx = mxGetField(FLINSTRmx, n, "protocol");
        if (Protomx) {
          UAPNode *ProtoNode = CtrlNode->addChild("protocol");
          ProtoNode->addAttribute("value", mxArrayToString(Protomx));
        }
        break;
      }
    }
  }

  return;
}


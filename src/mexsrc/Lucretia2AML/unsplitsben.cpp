#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "math.h"
#include "Lucretia2AML.hpp"

using namespace std;

void unsplitsben(UAPNode *EleNode, mxArrayList LucEleList, UAPNode *AMLRepNode, mxArray *FLPSmx) {
 /* This routine is given the <element/> AML node, and a list of the Lucretia
  * elements it should combine into that node.  This will represent a split sbend
  * with an instrument or marker in the middle, plus (perhaps) some higher order
  * field perturbations.*/
  bool ok, named=false, oriented=false, mult_exist = false, hasPS=false, energy=false;
  double Ldoub = 0, Bdoub = 0, Berrdoub = 0, EA1 = 0, EA2 = 0, HGap1 = 0, HGap2 = 0, BLBdoub=0, Bquaddoub=0;
  double FInt1 = 0, FInt2 = 0, EdgeCurv1 = 0, EdgeCurv2 = 0, Pdoub, c_light=0.299792458;
  double *EA1ptr, *EA2ptr, *HGap1ptr, *HGap2ptr, *FInt1ptr, *FInt2ptr, *EdgeCurv1ptr, *EdgeCurv2ptr;
  double PSnum, *a, *b, *order;
  mxArray *PSstrucmx, *ElePSmx, *Bmx;
  double *Bptr;
  int num_poles=0;
  mxArray *PSmx = mexGetVariable("global", "PS");

  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
   /* Get the "Class" of this element structure.*/
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

   /* If this element is a sben, then we want to examine it.*/
    if ( !strcmp(ClassType, (char*)"SBEN") ) {
     /* If we haven't named the element yet, we want to do that now.*/
      if (!named) {
        EleNode = addName(EleNode, Elemx);
        named=true;
      }

     /* If we haven't figured out the beam energy, do that now.*/
      if (!energy){
        Pdoub = mxGetScalar( mxGetField(Elemx, 0, "P") );
        energy = true;
      }

      PSstrucmx = mxGetField(Elemx, 0, "PS");
      double PSmult = 1;
      if (PSstrucmx) {
        hasPS=true;
        PSnum = mxGetScalar(PSstrucmx);
        PSmult = mxGetScalar( mxGetField(PSmx, (int)PSnum-1, "SetPt") );
      }

     /* Add the length, B, and dB of this element to the sum.*/
      Ldoub += mxGetScalar( mxGetField(Elemx, 0, "L") );
      Bmx = mxGetField(Elemx, 0, "B");
      Bptr = mxGetPr(Bmx);
      Bdoub += (PSmult * mxGetScalar(Bmx));
      BLBdoub += mxGetScalar(Bmx);
      Berrdoub += (PSmult * mxGetScalar( mxGetField(Elemx, 0, "dB") ));
      if (mxGetN(Bmx)>1 & *(Bptr+1)!=0){
        Bquaddoub += *(Bptr+1);
      }
  
      if ( mxArray *EdgeAngmx = mxGetField(Elemx, 0, "EdgeAngle") ) {
        EA1ptr = mxGetPr(EdgeAngmx);
        EA2ptr = EA1ptr+1;
        EA1 += *EA1ptr;
        EA2 += *EA2ptr;
      }
  
      if ( mxArray *HGapmx = mxGetField(Elemx, 0, "HGAP") ) {
        HGap1ptr = mxGetPr(HGapmx);
        HGap2ptr = HGap1ptr+1;
        HGap1 = *HGap1ptr;
        HGap2 = *HGap2ptr;
      }
  
      if ( mxArray *FIntmx = mxGetField(Elemx, 0, "FINT") ) {
        FInt1ptr = mxGetPr(FIntmx);
        FInt2ptr = FInt1ptr+1;
        FInt1 += *FInt1ptr;
        FInt2 += *FInt2ptr;
      }
  
      if ( mxArray *EdgeCurvmx = mxGetField(Elemx, 0, "EdgeCurvature") ) {
        EdgeCurv1ptr = mxGetPr(EdgeCurvmx);
        EdgeCurv2ptr = EdgeCurv1ptr+1;
        EdgeCurv1 += *EdgeCurv1ptr;
        EdgeCurv2 += *EdgeCurv2ptr;
      }
    }

    if ( !strcmp(ClassType, (char*)"MULT") ) {
      double *PoleIndPtr = mxGetPr( mxGetField(Elemx, 0, "PoleIndex") );
      double *TiltIndPtr = mxGetPr( mxGetField(Elemx, 0, "Tilt") );
      double *BIndPtr = mxGetPr( mxGetField(Elemx, 0, "B") );
      num_poles = mxGetN( mxGetField(Elemx, 0, "PoleIndex") );
      if (!mult_exist) {
        a = new double[num_poles];
        b = new double[num_poles];
        order = new double[num_poles];
        mult_exist = true;
        for (int polenum=0; polenum<num_poles; polenum++) {
          a[polenum] = 0;
          b[polenum] = 0;
        }
      }

      for (int polenum=0; polenum<num_poles; polenum++) {
        order[polenum] = (*PoleIndPtr) + 1;
        a[polenum] += (sin( *PoleIndPtr * *TiltIndPtr)) * *BIndPtr;
        b[polenum] += (cos( *PoleIndPtr * *TiltIndPtr)) * *BIndPtr;
        PoleIndPtr++;
        TiltIndPtr++;
        BIndPtr++;
      }
    }

    delete ClassType;
  }

  Bdoub /= (Pdoub/c_light);
  BLBdoub /= (Pdoub/c_light);

  UAPNode* BendNode = EleNode->addChild(ELEMENT_NODE, "bend");

  UAPNode* BNode = BendNode->addChild(ELEMENT_NODE, "g");
  BNode->addAttribute("design", BasicUtilities::double_to_string(Bdoub/Ldoub, ok), false);
  BNode->addAttribute("err", BasicUtilities::double_to_string(Berrdoub/Ldoub, ok), false);

  if (Bquaddoub!=0) {
    UAPNode* BquadNode = BendNode->addChild(ELEMENT_NODE, "multipole");
    Bquaddoub /= (Pdoub/c_light);
    UAPNode *B_coefNode = BquadNode->addChild(ELEMENT_NODE, "b_coef");
    B_coefNode->addAttribute("n", BasicUtilities::double_to_string(2, ok), false);
    B_coefNode->addAttribute("design", BasicUtilities::double_to_string(Bquaddoub/Ldoub, ok), false);
  }

  UAPNode *LengthNode = EleNode->addChild(ELEMENT_NODE, "length");
  LengthNode->addAttribute("design", BasicUtilities::double_to_string(Ldoub, ok), false);

  UAPNode *EA1Node = BendNode->addChild(ELEMENT_NODE, "e1");
  EA1Node->addAttribute("design", BasicUtilities::double_to_string(EA1, ok), false);

  UAPNode *EA2Node = BendNode->addChild(ELEMENT_NODE, "e2");
  EA2Node->addAttribute("design", BasicUtilities::double_to_string(EA2, ok), false);

  UAPNode *HGap1Node = BendNode->addChild(ELEMENT_NODE, "h_gap1");
  HGap1Node->addAttribute("design", BasicUtilities::double_to_string(HGap1, ok), false);

  UAPNode *HGap2Node = BendNode->addChild(ELEMENT_NODE, "h_gap2");
  HGap2Node->addAttribute("design", BasicUtilities::double_to_string(HGap2, ok), false);

  UAPNode *FInt1Node = BendNode->addChild(ELEMENT_NODE, "f_int1");
  FInt1Node->addAttribute("design", BasicUtilities::double_to_string(FInt1, ok), false);

  UAPNode *FInt2Node = BendNode->addChild(ELEMENT_NODE, "f_int2");
  FInt2Node->addAttribute("design", BasicUtilities::double_to_string(FInt2, ok), false);

  UAPNode *EdgeCurv1Node = BendNode->addChild(ELEMENT_NODE, "h1");
  EdgeCurv1Node->addAttribute("design", BasicUtilities::double_to_string(EdgeCurv1, ok), false);

  UAPNode *EdgeCurv2Node = BendNode->addChild(ELEMENT_NODE, "h2");
  EdgeCurv2Node->addAttribute("design", BasicUtilities::double_to_string(EdgeCurv2, ok), false);

  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
   /* Get the "Class" of this element structure.*/
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

    if (!strcmp(ClassType, (char*)"SBEN") && !oriented) {
      make_amlorient(BendNode, Elemx);
      oriented = true;
    }

    /*if (!strcmp(ClassType, (char*)"MARK")) {
      UAPNode *MarkNode = EleNode->addChild(ELEMENT_NODE, "marker");
      mxArray *Namemx = mxGetField(Elemx, 0, "Name");
      int Namelength = mxGetN(Namemx);
      char *Namechar;
      Namechar = new char[Namelength+1];
      mxGetString(Namemx, Namechar, Namelength+1);
      string Namestr(Namechar);
      MarkNode->addAttribute("name", Namestr, false);
      delete Namechar;
    }*/

    if (!strcmp(ClassType, (char*)"XCOR")) {
      
    }
  }

 /* If we found a power supply, we'd better create it!*/
  if ( hasPS ) {
    UAPAttribute *NameAttrib = EleNode->getAttribute("name");
    string namestr = NameAttrib->getValue(); 
    CreateAMLController(AMLRepNode, PSnum, "bend:g", PSmx, namestr, BLBdoub/Ldoub, FLPSmx);
  }

  return;
}


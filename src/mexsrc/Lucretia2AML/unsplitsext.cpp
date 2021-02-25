#include <iostream>
#include <string>
#include <cstring>
#include "UAP/UAPUtilities.hpp"
#include "mex.h"
#include "math.h"
#include "Lucretia2AML.hpp"

using namespace std;

void unsplitsext(UAPNode *EleNode, mxArrayList LucEleList, UAPNode *AMLRepNode, mxArray *FLPSmx) {
 /* This routine is given the <element/> AML node, and a list of the Lucretia
  * elements it should combine into that node.  This will represent a split sext
  * with an instrument or marker in the middle, plus (perhaps) some higher order
  * field perturbations.*/
  bool ok, named=false, oriented=false, mult_exist = false, hasPS=false, energy=false;
  double Ldoub = 0, Bdoub = 0, Berrdoub = 0, *a, *b, *order, PSnum, BLBdoub=0;
  double Pdoub, c_light=0.299792458;
  int sextctr=0, num_poles=0;
  mxArray *PSstrucmx, *ElePSmx;
  mxArray *PSmx = mexGetVariable("global", "PS");

 /* Loop around the elements in the list to find and sum the sext fields and lengths.*/
  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
   /* Get the "Class" of this element structure.*/
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

   /* If this element is a sext, then we want to examine it.*/
    if ( !strcmp(ClassType, (char*)"SEXT") ) {
      sextctr++;
     /* If we haven't named the element yet, we want to do that now.*/
      if (!named) {
        EleNode = addName(EleNode, Elemx);
        named=true;
      }

     /* If we haven't figured out the energy, do it now.*/
      if (!energy){
        Pdoub = mxGetScalar( mxGetField(Elemx, 0, "P") );
      }

     /* Find the PS (if any).*/
      PSstrucmx = mxGetField(Elemx, 0, "PS");
      double PSmult = 1;
      if (PSstrucmx) { 
        hasPS=true;
        PSnum = mxGetScalar(PSstrucmx);
        PSmult = mxGetScalar( mxGetField(PSmx, (int)PSnum-1, "SetPt") );
      }

     /* Add the length, B, and dB of this element to the sum.*/
      Ldoub += mxGetScalar( mxGetField(Elemx, 0, "L") );
      Bdoub += (PSmult * mxGetScalar( mxGetField(Elemx, 0, "B") ));
      BLBdoub += mxGetScalar( mxGetField(Elemx, 0, "B") );
      Berrdoub += (PSmult * mxGetScalar( mxGetField(Elemx, 0, "dB") ));
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

   /* Delete the dynamic memory for the Class string.*/
    delete ClassType;
  }

  Bdoub /= (Pdoub/c_light);
  BLBdoub /= (Pdoub/c_light);

 /* Add a sextupole node to the element, and assign the design and err nodes.*/
  UAPNode *SextNode = EleNode->addChild(ELEMENT_NODE, "sextupole");
  UAPNode *BNode = SextNode->addChild(ELEMENT_NODE, "k");
  BNode->addAttribute("design", BasicUtilities::double_to_string(Bdoub/Ldoub, ok), false);
  BNode->addAttribute("err", BasicUtilities::double_to_string(Berrdoub/Ldoub, ok), false);
  UAPNode *LengthNode = EleNode->addChild(ELEMENT_NODE, "length");
  LengthNode->addAttribute("design", BasicUtilities::double_to_string(Ldoub, ok), false);

  if (mult_exist) {
    UAPNode *MultNode = SextNode->addChild(ELEMENT_NODE, "multipole");
    for (int polenum=0; polenum<num_poles; polenum++) {
      UAPNode *ACoefNode = MultNode->addChild(ELEMENT_NODE, "a_coef");
      UAPNode *BCoefNode = MultNode->addChild(ELEMENT_NODE, "b_coef");

      ACoefNode->addAttribute("n", BasicUtilities::double_to_string(order[polenum], ok), false);
      BCoefNode->addAttribute("n", BasicUtilities::double_to_string(order[polenum], ok), false);
      ACoefNode->addAttribute("design", BasicUtilities::double_to_string(a[polenum]/Ldoub, ok), false);
      BCoefNode->addAttribute("design", BasicUtilities::double_to_string(b[polenum]/Ldoub, ok), false);
    }
    delete a, b, order;
  }

 /* Another loop around the contents of the list to extract orientation and aperture information.*/
  bool ups = true;
  for (mxArrayListIter it=LucEleList.begin(); it!=LucEleList.end(); it++) {
    mxArray *Elemx = *it;
    mxArray *ClassVal = mxGetField(Elemx, 0, "Class");
    char *ClassType;
    int strsize = mxGetN(ClassVal)+1;
    ClassType = new char[strsize];
    mxGetString(ClassVal, ClassType, strsize);

   /* Only extract orientation info if this is a sext, and we haven't looked at orientation already*/
    if ( !strcmp(ClassType, (char*)"SEXT") ) {
      if (!oriented){
        make_amlorient(EleNode, Elemx);
        oriented = true;
      }

      if (sextctr==(int)2) {
        if (ups) {
          make_amlaper(EleNode, Elemx, "ENTRANCE");
          ups = false;
        }
        else make_amlaper(EleNode, Elemx, "EXIT");
      }	
      else {
        if ( !strcmp(ClassType, (char*)"SEXT") ) {
          make_amlaper(EleNode, Elemx, "BOTH");
        }
      }
    }

    /*if ( !strcmp(ClassType, (char*)"MARK") ) {
      UAPNode *MarkNode = EleNode->addChild(ELEMENT_NODE, "marker");
      mxArray *Namemx = mxGetField(Elemx, 0, "Name");
      int Namelength = mxGetN(Namemx);
      char *Namechar;
      Namechar = new char[Namelength+1];
      mxGetString(Namemx, Namechar, Namelength+1);
      string Namestr(Namechar);
      MarkNode->addAttribute("name", Namestr, false);
    }*/

   /* Delete the dynamic memory for the Class string.*/
    delete ClassType;
  }

 /* If we found a power supply, we'd better create it!*/
  if ( hasPS ) {
    UAPAttribute *NameAttrib = EleNode->getAttribute("name");
    string namestr = NameAttrib->getValue();
    CreateAMLController(AMLRepNode, PSnum, "sextupole:k", PSmx, namestr, BLBdoub/Ldoub, FLPSmx);
  }

  return;
}


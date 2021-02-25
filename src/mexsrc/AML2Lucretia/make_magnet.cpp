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

magmake make_magnet(UAPNode* EleNode, beamdef beamparams){
  bool ok;
  double Length_design, Length_err;
  double kDesign, kErr;
  double e_charge = 1.6021892e-19;
  double c_light = 299792458;

 /* Store the element's name in "Name"*/
  UAPAttribute* EleNameAttrib = EleNode->getAttribute("name");
  string Name = EleNameAttrib->getValue();

 /* Find the design and error length: Length_design & Length_err
 *   * Extract them as strings from their UAPAttributes*/
  UAPNode* LengthNode = EleNode->getChildByName("length");
  UAPAttribute* LDesAttrib = LengthNode->getAttribute("design");
  string Length_design_str = LDesAttrib->getValue();
  string Length_err_str;
  if (UAPAttribute* LErrAttrib = LengthNode->getAttribute("err")) {
    Length_err_str = LErrAttrib->getValue();
  }
  else {
    Length_err_str = "0";
  }
 /* Now use BasicUtilities to convert these to doubles*/
  Length_design = BasicUtilities::string_to_double(Length_design_str,ok);
  Length_err = BasicUtilities::string_to_double(Length_err_str,ok);

 /* Determine if k is defined as normalised/unnormalise and scale correctly*/
  UAPNode* SextNode;
  UAPNode* kNode;
  string kDesign_str, kErr_str;
  SextNode = EleNode->getChildByName("sextupole");
  if (kNode = SextNode->getChildByName("k")) {
   /* k has been defined normalised to the beam energy*/

    UAPAttribute* kDesignAttrib = kNode->getAttribute("design");
    kDesign_str = kDesignAttrib->getValue();
    if (UAPAttribute* kErrAttrib = kNode->getAttribute("err")) {
      kErr_str = kErrAttrib->getValue();
    }
    else {
      kErr_str = "0";
    }
    kDesign = BasicUtilities::string_to_double(kDesign_str,ok) * (beamparams.DesignBeamP / (c_light / 1e9));
    kErr = BasicUtilities::string_to_double(kErr_str,ok) * (beamparams.DesignBeamP / (c_light / 1e9));
  }

  else if (kNode = SextNode->getChildByName("k_u")) {
   /* k has been defined unnormalised to the beam energy*/

    UAPAttribute* kDesignAttrib = kNode->getAttribute("design");
    kDesign_str = kDesignAttrib->getValue();
    if (UAPAttribute* kErrAttrib = kNode->getAttribute("err")) {
      kErr_str = kErrAttrib->getValue();
    }
    else {
      kErr_str = "0";
    }
    kDesign = BasicUtilities::string_to_double(kDesign_str,ok);
    kErr = BasicUtilities::string_to_double(kErr_str,ok);
  }

  double B = kDesign * Length_design;
  double dB = ((kDesign + kErr) * (Length_design + Length_err)) - B;

 /* Determine tilt of sext.  If not present, set to zero.*/
  double Tilt;
  if (UAPNode* OrientNode = EleNode->getChildByName("orientation")) {
    if (UAPNode* tiltNode = OrientNode->getChildByName("tilt")) {
      UAPAttribute* tiltDesAttrib = tiltNode->getAttribute("design");
      string tiltDesstr = tiltDesAttrib->getValue();
      Tilt = BasicUtilities::string_to_double(tiltDesstr,ok);
    }
  }
  else { Tilt = 0; }

 /* Determine aperture.  If not present, set to 100 m (big enough not to matter).*/
  cout << "Not calculating aperture yet.  All apers set to 100 m." << endl;
  double Aper;
  if (UAPNode* AperNode = EleNode->getChildByName("aperture")) { Aper = 100; }
  else { Aper = 100; }

 /* Now let's cram all this into a magmake structure.*/
  magmake magstruct;
  magstruct.Name = Name.c_str();
  magstruct.L = Length_design;
  magstruct.B = B;
  magstruct.dB = dB;
  magstruct.Tilt = Tilt;
  magstruct.PS = 0;
  magstruct.Girder = 0;
  for (int i=0;i<6;i++) { magstruct.Offset[i] = ( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 );}
  magstruct.Aper = Aper;

  return magstruct;
}


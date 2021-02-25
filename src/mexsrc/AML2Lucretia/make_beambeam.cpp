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

mxArray* make_beambeam(UAPNode* EleNode, beamdef beamparams){
  cout << "<beambeam> element not applicable to Lucretia beams." << endl;
  cout << "Expanding as a marker." << endl;

  mxArray* BeamBeamStruc = make_marker(EleNode, beamparams);

  return BeamBeamStruc;
}


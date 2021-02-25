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

mxArray* make_custom(UAPNode* EleNode, beamdef beamparams){
  cout << "The <custom> node is undefined.  Setting as a marker." << endl;

  mxArray* MarkerNode = make_marker(EleNode, beamparams);

  return MarkerNode;
}


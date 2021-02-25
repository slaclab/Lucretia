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

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  UAPNode* UAPModel;
  string filein;

  if (nrhs==1) {
    filein = mxArrayToString(prhs[0]);
  }
  else if (nrhs==0) {
    filein = "expand.aml";
  }

  // Read AML lattice file and create an AML representation subtree.
  cout << "Expanding file: " << filein << endl;
  AMLReader reader;
  UAPModel = reader.AMLFileToAMLRep (filein);
  if (!UAPModel) { cout << "Stopping here." << endl; return ; }

  // Expand the AML representation subtree.
  try {
    AMLLatticeExpander LE;
    LE.AMLExpandLattice(UAPModel);
  } catch (UAPException err) {
    cerr << err << endl;
    return ;
  }

  // Check the tree for problems
  //UAPModel->checkTree(); 

  return ;
}


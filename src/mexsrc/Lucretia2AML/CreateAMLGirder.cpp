#include <iostream>
#include "UAP/UAPUtilities.hpp"
#include "AML/AMLReader.hpp"
#include "AML/AMLLatticeExpander.hpp"
#include "mex.h"
#include "Lucretia2AML.hpp"

using namespace std;

void CreateAMLGirder(UAPNode *AMLRepNode, UAPNode *MachineNode, mxArray *GIRDERmx,
                                                          int GirderNum, int EleNum) {
 /* Called if it's necessary to make a GIRDER.
  * Should first figure out if this girder has already been created.  If so, then it's only
  * necessary to add the relevant element(s) to the relevant sector.
  * It can be assumed that, if EleNum is the first element on the girder, then the girder
  * hasn't been created yet.  So then it should be created (duh!).
  * If it's *not* the first element, then we have to search for the right sector to add
  * the elements to.*/

  

  return;
}


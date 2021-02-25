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

double getUAPinfo(UAPNode* EleNode, string term) {
  if (!EleNode) {
    return NULL;
  }
  bool ok;
  double TermDes;
  UAPAttribute* TermDesAttrib = EleNode->getAttribute(term);

  if (!TermDesAttrib) {
    string TermName = EleNode->getName();
    return NULL;
  }
  string TermDesstr = TermDesAttrib->getValue();
  TermDes = BasicUtilities::string_to_double(TermDesstr,ok);

  return TermDes;
}


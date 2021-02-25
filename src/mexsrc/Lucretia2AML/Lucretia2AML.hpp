typedef list<mxArray*>                 mxArrayList;
typedef list<mxArray*>::iterator       mxArrayListIter;
typedef list<mxArray*>::const_iterator mxArrayListCIter;

UAPNode* LEle2UAPNode(int, UAPNode*, mxArray*, mxArray*, mxArray*, mxArray*);
UAPNode* LEle2UAPNode(int, UAPNode*, mxArrayList, mxArray*, mxArray*, mxArray*);

void make_amlmarker(UAPNode*, mxArray*);
void make_amldrift(UAPNode*, mxArray*);
void make_amlbpm(int, UAPNode*, mxArray*, mxArray*, mxArray*);
void make_amlxcor(UAPNode*, mxArray*, UAPNode*, mxArray*);
void make_amlycor(UAPNode*, mxArray*, UAPNode*, mxArray*);
void make_amlsben(UAPNode*, mxArray*, UAPNode*, mxArray*);
void make_amlquad(UAPNode*, mxArray*, UAPNode*, mxArray*);
void make_amlsext(UAPNode*, mxArray*, UAPNode*, mxArray*);
void make_amloct(UAPNode*, mxArray*, UAPNode*, mxArray*);
void make_amlinstr(UAPNode*, mxArray*, string);

UAPNode* addName(UAPNode*, mxArray*);
UAPNode* addS(UAPNode*, mxArray*);
UAPNode* addL(UAPNode*, mxArray*);
UAPNode* addB(UAPNode*, mxArray*);
UAPNode* addBerr(UAPNode*, mxArray*);

void unsplitquad(UAPNode*, mxArrayList, UAPNode*, mxArray*);
void unsplitsext(UAPNode*, mxArrayList, UAPNode*, mxArray*);
void unsplitsben(UAPNode*, mxArrayList, UAPNode*, mxArray*);
void unsplitxcor(UAPNode*, mxArrayList, UAPNode*, mxArray*);
void unsplitycor(UAPNode*, mxArrayList, UAPNode*, mxArray*);
void unsplitbpm(int, UAPNode*, mxArrayList, UAPNode*, mxArray*, mxArray*);
void make_amlaper(UAPNode*, mxArray*, string);
void make_amlorient(UAPNode*, mxArray*);

void CreateAMLController(UAPNode*, double, string, mxArray*, string, double, mxArray*);
void CreateAMLGirder(UAPNode*, UAPNode*, mxArray*, int, int);
void addmover (mxArray*, UAPNode*);
void addFloodLand(UAPNode*, mxArray*, double);

bool equalNodes(UAPNode*, UAPNode*);


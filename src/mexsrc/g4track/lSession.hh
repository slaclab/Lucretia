#ifndef lSession_hh
#define lSession_hh
#include "G4UIsession.hh"
#include <iostream>
#include <fstream>
#ifndef LUCRETIA_MANAGER
#include "lucretiaManager.hh"
#endif
using namespace std;

class lSession : public G4UIsession {
public:
	lSession(lucretiaManager* lman);
	~lSession();
	G4UIsession* SessionStart();
	G4int ReceiveG4cout(const G4String&);
	G4int ReceiveG4cerr(const G4String&);
private:
	lucretiaManager* fLman;
};
#endif



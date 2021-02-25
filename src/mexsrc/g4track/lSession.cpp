#include "lSession.hh"
#include "G4UImanager.hh"
#include "mex.h"
lSession::lSession(lucretiaManager* lman)
  : fLman(lman)
{
  G4UImanager* UI = G4UImanager::GetUIpointer();
    UI->SetCoutDestination(this);
}
lSession::~lSession(){
  G4UImanager* UI = G4UImanager::GetUIpointer();
  UI->SetCoutDestination(0);
}
G4UIsession* lSession::SessionStart(){
  return NULL;
}
G4int lSession::ReceiveG4cout(const G4String& output){
  if (fLman->Verbose>0)
    mexPrintf(output) ;
  return 0;
}
G4int lSession::ReceiveG4cerr(const G4String& err){
  mexErrMsgIdAndTxt("ExtG4Process (GEANT4) Error", err);
  return 0;
}


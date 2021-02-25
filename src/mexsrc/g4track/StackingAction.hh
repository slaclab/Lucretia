#ifndef StackingAction_h
#define StackingAction_h 1

#include "G4UserStackingAction.hh"
#include "globals.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class StackingAction : public G4UserStackingAction
{
public:
  StackingAction(lucretiaManager* lman);
  ~StackingAction();

  virtual G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track*);

private:
  lucretiaManager* fLman ;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

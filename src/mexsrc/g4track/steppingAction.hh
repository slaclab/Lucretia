#include "G4UserSteppingAction.hh"
#include "G4UnitsTable.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

class steppingAction : public G4UserSteppingAction {

  public:
  steppingAction(lucretiaManager* lman);
  virtual ~steppingAction() {};

  virtual void  UserSteppingAction(const G4Step*);

private:
  lucretiaManager* fLman ;
};

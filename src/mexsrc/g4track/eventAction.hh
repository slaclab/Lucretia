#include "G4UserEventAction.hh"
#include "globals.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

/// Event action class

class eventAction : public G4UserEventAction
{
  public:
    eventAction(lucretiaManager* lman);
    virtual ~eventAction();

    virtual void  BeginOfEventAction(const G4Event* );
    virtual void    EndOfEventAction(const G4Event* );
  private:
    lucretiaManager* fLman ;
};



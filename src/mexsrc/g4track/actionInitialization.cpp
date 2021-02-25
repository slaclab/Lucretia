#include "actionInitialization.hh"
#include "primaryGeneratorAction.hh"
#include "runAction.hh"
//#include "eventAction.hh"
#include "trackingAction.hh"
#include "steppingAction.hh"
#include "StackingAction.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

actionInitialization::actionInitialization(lucretiaManager* lman)
  : G4VUserActionInitialization(),
    fLman(lman)
{
}

actionInitialization::~actionInitialization()
{}


void actionInitialization::BuildForMaster() const
{
  SetUserAction(new runAction);
}

void actionInitialization::Build() const
{
  primaryGeneratorAction* prim = new primaryGeneratorAction(fLman) ;
  SetUserAction(prim);
  SetUserAction(new runAction);
  //SetUserAction(new eventAction(fLman));
  SetUserAction(new StackingAction(fLman));
  SetUserAction(new trackingAction(fLman));
  SetUserAction(new steppingAction(fLman));
}  


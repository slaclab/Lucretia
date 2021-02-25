#include "runAction.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"

runAction::runAction()
 : G4UserRunAction()
{ 
  // set printing event number per each 100 events
  //G4RunManager::GetRunManager()->SetPrintProgress(10);     
}

runAction::~runAction()
{}

void runAction::BeginOfRunAction(const G4Run*)
{ 
  //inform the runManager to save random number seed
  //G4RunManager::GetRunManager()->SetRandomNumberStore(false);
}

void runAction::EndOfRunAction(const G4Run* )
{
  /*  G4int nevent = run->GetNumberOfEvent() ;
  if (nevent == 0) return ;
  std::cout << "NEVENTS:" << nevent << "\n";*/
}


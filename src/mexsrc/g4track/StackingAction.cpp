#include "StackingAction.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "G4UnitsTable.hh"
#include "TrackInformation.hh"
#include "../LucretiaMatlab.h"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

StackingAction::StackingAction(lucretiaManager* lman)
  : G4UserStackingAction(),
    fLman(lman)
{

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

StackingAction::~StackingAction()
{

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ClassificationOfNewTrack
StackingAction::ClassifyNewTrack(const G4Track* aTrack)
{
  static const double qe=1.60217662e-19;
  //stack or delete secondaries
  G4ClassificationOfNewTrack status = fUrgent;

  // keep primary particle
  if (aTrack->GetParentID() == 0) { return status; }
  
  // kill secondary if requested or doesn't pass energy cut, but take care of accounting for energy
  if ( fLman->fMaxSecondaryParticles == 0 ||  aTrack->GetDynamicParticle()->GetKineticEnergy()/CLHEP::GeV < fLman->Ecut ) {
    double nelec;
    double* Qvec;
    TrackInformation* info = (TrackInformation*)(aTrack->GetUserInformation());
    G4int primaryParentID = fLman->fPrimIndex[info->GetOriginalTrackID()-1];
    Qvec=fLman->fBunch->Q;
    nelec = Qvec[primaryParentID-1] / qe;
    fLman->SumEdep += (aTrack->GetDynamicParticle()->GetKineticEnergy()/CLHEP::joule) * nelec;
    status = fKill;   
  }
    
 
  return status;
}

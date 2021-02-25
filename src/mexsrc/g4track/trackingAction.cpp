#include "trackingAction.hh"
#include "G4RunManager.hh"
#include "G4Track.hh"
#include "G4UnitsTable.hh"
#include "TrackInformation.hh"
#include "G4TrackingManager.hh"
#include "G4TrackVector.hh"
#include "../LucretiaMatlab.h"
#include <math.h>
#include <iostream>
#include "CLHEP/Units/SystemOfUnits.h"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

trackingAction::trackingAction(lucretiaManager* lman)
 :G4UserTrackingAction(),
  fLman(lman),
  fLastTrackStoreCounter(0)
{ }


void trackingAction::PreUserTrackingAction(const G4Track* track)
{
  if(track->GetParentID()==0 && track->GetUserInformation()==0)
  {
    TrackInformation* anInfo = new TrackInformation(track);
    G4Track* theTrack = (G4Track*)track;
    theTrack->SetUserInformation(anInfo);
  }
}

void trackingAction::PostUserTrackingAction(const G4Track* track)
{
  static const double qe=1.60217662e-19;
  G4TrackVector* secondaries = fpTrackingManager->GimmeSecondaries();
  if(secondaries)
  {
    TrackInformation* info = (TrackInformation*)(track->GetUserInformation());
    size_t nSeco = secondaries->size();
    if(nSeco>0)
    {
      for(size_t i=0;i<nSeco;i++)
      { 
        TrackInformation* infoNew = new TrackInformation(info);
        (*secondaries)[i]->SetUserInformation(infoNew);
      }
    }
  }
  G4ThreeVector pos = track->GetPosition() ;
  G4double x = track->GetPosition().x()/CLHEP::m ;
  G4double y = track->GetPosition().y()/CLHEP::m ;
  G4double z = track->GetPosition().z()/CLHEP::m ;
  G4double e = track->GetDynamicParticle()->GetKineticEnergy() / CLHEP::GeV ;
  G4double momx = track->GetMomentum().x()/CLHEP::GeV ;
  G4double momy = track->GetMomentum().y()/CLHEP::GeV ;
  G4double momz = track->GetMomentum().z()/CLHEP::GeV ;
  G4double gT = track->GetGlobalTime() ;
  G4int parentID = track->GetParentID() ;
  double xLucretia[6] ;
  int passCuts=0 ;
  int dosecondaries = 0;
  static int hitsPerParentCounter ;
  static int lastPrimaryID ;
  // Track energy
  TrackInformation* info = (TrackInformation*)(track->GetUserInformation());
  G4int primaryParentID = fLman->fPrimIndex[info->GetOriginalTrackID()-1];
  double* Qvec;
  double nelec, trackEnergy;
  Qvec=fLman->fBunch->Q;
  nelec = Qvec[primaryParentID-1] / qe;
  trackEnergy = (track->GetDynamicParticle()->GetTotalEnergy() / CLHEP::joule) * nelec ;
  //cout << "PNAME: " << track->GetDynamicParticle()->GetDefinition()->GetParticleName() << "\n" ;
  if ( fLman->fMaxSecondaryParticles>0 && fLman->fMaxSecondaryParticlesPerPrimary>0)
    dosecondaries = 1 ;
  if (parentID==0 || dosecondaries ) { 
    xLucretia[0] = x; xLucretia[2] = y; xLucretia[4] = z; // z doesn't get copied back to Lucretia bunch for primaries
    xLucretia[1] = atan(momx/momz) ; xLucretia[3] = atan(momy/momz) ;
    xLucretia[5] = e ;
    if (e>=fLman->Ecut && z>=fLman->Lcut) // Primaries get put back into Lucretia tracking if E>Ecut and tracks to world right edge
      passCuts=1;
    // Secondaries need E>Ecut or override
    if (dosecondaries && (fLman->fSecondaryStorageCuts==0 || e>=fLman->Ecut)  && fLman->fSecondariesCounter < fLman->fMaxSecondaryParticles ) 
      dosecondaries = 1 ;
    else
      dosecondaries = 0 ;
  }
  if (parentID!=0) { // Actions for secondary particles
    fLman->fTrackStoreCounter=fLastTrackStoreCounter; // Kludge to ensure only primary track points stored in fLman->fTrackStoreData_*
  }
  if (parentID==0) { // Actions for primary particles
    //printf("TrackID: %d\n",track->GetTrackID());
    hitsPerParentCounter=0;
    fLman->SetNextX(xLucretia, track->GetTrackID()-1, passCuts, trackEnergy, gT) ; // Write new tracked particle back to Lucretia Matlab Bunch
    lastPrimaryID=track->GetTrackID()-1;
    // Store tracked vector
    if (fLman->fMaxTrackStore>0) {
      fLman->fTrackStorePointer[fLman->fPrimIndex[track->GetTrackID()-1]]=fLman->fTrackStoreCounter-1;
      fLastTrackStoreCounter=fLman->fTrackStoreCounter-1;
    }
  }
  else if (dosecondaries && (uint32_T)hitsPerParentCounter<fLman->fMaxSecondaryParticlesPerPrimary) { // Process secondaries
    hitsPerParentCounter++;
    if ( fLman->SetNextSecondary(xLucretia, lastPrimaryID, track->GetDynamicParticle()->GetDefinition()->GetParticleName(),
			    track->GetCreatorProcess()->GetProcessName(), track->GetTrackStatus(), gT) )
      fLman->SumEdep += trackEnergy ; // Add secondary track to energy lost in this volume if not passing it back to Lucretia
  }
  else { // Add secondary track to energy lost in this volume if not passing it back to Lucretia
    fLman->SumEdep += trackEnergy ;
  }
}

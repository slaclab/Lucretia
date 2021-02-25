#include "primaryGeneratorAction.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "math.h"
#include "../LucretiaMatlab.h"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif
#include <iostream>

primaryGeneratorAction::primaryGeneratorAction(lucretiaManager* lman)
  : G4VUserPrimaryGeneratorAction(),
    fLman(lman)
{
  // Input particles type required
  G4int nofParticles = 1;
    fParticleGun = new G4ParticleGun(nofParticles);
  G4ParticleDefinition* particleDefinition 
    = G4ParticleTable::GetParticleTable()->FindParticle(lman->PrimaryType);
  fParticleGun->SetParticleDefinition(particleDefinition);
}

primaryGeneratorAction::~primaryGeneratorAction()
{
  delete fParticleGun;
}

void primaryGeneratorAction::GeneratePrimaries(G4Event* Event)
{
  // This function is called at the begining of event
  // Generate primary particles from Lucretia input (stopped) beam structure
  int xPtr ;
  double Px, Py, Pz, P;
  //cout << "+-+-+-+-+-+-+- primaryGeneratorAction +-+-+-+-+-+-+- (ele= " << *fLman->fEle << ")\n" ;
  fParticleGun->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle(fLman->PrimaryType));
  xPtr = fLman->GetNextX() ;
  while ( xPtr>=0 ) { // Loop through all stopped Lucretia macro-particles
    // X/Y co-ordinates from Lucretia Bunch, place Z at start of geometry or if SecondaryAsPrimary set, use given z location
    if (fLman->SecondaryAsPrimary)
      fParticleGun->SetParticlePosition(G4ThreeVector(fLman->fBunch->x[xPtr*6]*m, fLman->fBunch->x[xPtr*6+2]*m, fLman->fBunch->x[xPtr*6+4]*m));
    else
      fParticleGun->SetParticlePosition(G4ThreeVector(fLman->fBunch->x[xPtr*6]*m, fLman->fBunch->x[xPtr*6+2]*m, -fLman->Lcut*m));
    P=fLman->fBunch->x[xPtr*6+5]; // Kinetic Energy in GeV
    Px=P*sin(fLman->fBunch->x[xPtr*6+1]);
    Py=P*sin(fLman->fBunch->x[xPtr*6+3]);
    Pz=sqrt(P*P-(Px*Px+Py*Py));
    // Momentum directions and energy from Lucretia Bunch
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(Px*GeV,Py*GeV,Pz*GeV));
    fParticleGun->SetParticleEnergy(P*GeV); // Kinetic energy of particle
    // Create a GEANT4 primary vertex from this macro-particle
    fParticleGun->GeneratePrimaryVertex(Event);
    //cout << "X/Y/Z : " << fLman->fBunch->x[xPtr*6]*m << " / " << fLman->fBunch->x[xPtr*6+2]*m << " / " << -fLman->Lcut << " Px/Py/Pz : " << Px << " / " << Py << " / " << Pz << " P : " << P << "\n" ;
    // Next stopped Lucretia macro-particle
    xPtr = fLman->GetNextX() ;
  }
}


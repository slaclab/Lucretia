#include "steppingAction.hh"
#include "G4RunManager.hh"
#include "G4Track.hh"
#include "G4UnitsTable.hh"
#include "globals.hh"
#include <math.h>
#include <iostream>
#include <string>
#include "../LucretiaMatlab.h"
#include "TrackInformation.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "G4SystemOfUnits.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

steppingAction::steppingAction(lucretiaManager* lman)
 :G4UserSteppingAction(),
  fLman(lman)
{ }


void steppingAction::UserSteppingAction(const G4Step* step)
{
  static const double qe=1.60217662e-19;
  
  // Get track pointer
  const G4Track* track = step->GetTrack();
  
  // Get Primary parent ID and assign charge weight
  TrackInformation* info = (TrackInformation*)(track->GetUserInformation());
  G4int primaryParentID = fLman->fPrimIndex[info->GetOriginalTrackID()-1];
  double* Qvec;
  double nelec;
  Qvec=fLman->fBunch->Q;
  nelec = Qvec[primaryParentID-1] / qe;
  
  // Increment total energy deposit counter
  fLman->SumEdep += (step->GetTotalEnergyDeposit() / CLHEP::joule) * nelec ; 
  
  // Get step point locations in global coordinate system
  G4StepPoint* p1 = step->GetPreStepPoint();
  G4StepPoint* p2 = step->GetPostStepPoint();
  G4ThreeVector pos1 = p1->GetPosition();
  G4ThreeVector pos2 = p2->GetPosition();
  
//   // Store track point if requested
//   if (fLman->fMaxTrackStore>0)
//     fLman->WritePrimaryTrackData(pos2.x()/m, pos2.y()/m, pos2.z()/m);
  
  // If required, store step in mesh data
  if ( fLman->fMeshEntries==0 )
    return;

  // Get Particle name
  string pname = track->GetDynamicParticle()->GetDefinition()->GetParticleName();
  
  // Store step in mesh data
  // List of particle types for comparison from Lucretia ExtMesh class:
  const string plist[] ={"e-", "mu-", "nu_e", "nu_mu", "nu_tau", "tau-", "e+", "mu+", "anti_nu_e", "anti_nu_mu", "anti_nu_tau", "tau+",
			 "pi0", "pi+", "pi-", "kaon0", "anti_kaon0", "kaon+", "kaon-",
			 "neutron", "ani_neutron", "proton", "anti-proton", "gamma","all"} ;
  
  // - Cycle through each mesh definition in turn
  int imesh,ip;
  for (imesh=0; imesh<fLman->fMeshEntries; imesh++) {
    // Check if particle type applies to this mesh
    int pbit=31;
    for (ip=0;ip<25;ip++) {
      if (plist[24-ip].compare(pname) == 0 ) {
	      pbit=ip;
	      break;
      }
    }
    if ( !(fLman->fMeshPtypes[imesh] & (1 << 0)) && !(fLman->fMeshPtypes[imesh] & (1 << pbit)))
      return; // return if not 'all' selected or this particle type not in requested list for this mesh
    // Get weight value to use
    double weightVal=1 ;
    bool scaleByLen=0 ;
    //double stepLen= sqrt( pow(pos2.x()-pos1.x(),2) + pow(pos2.y()-pos1.y(),2) + pow(pos2.z()-pos1.z(),2) ) ;
    if (fLman->fMeshWeights[imesh] == 8) // k.e.
      weightVal=track->GetKineticEnergy() / CLHEP::GeV ;
    else if (fLman->fMeshWeights[imesh] == 4 ) { // momentum
      G4ThreeVector mom = track->GetMomentum() ;
      weightVal=sqrt(mom.x()*mom.x() + mom.y()*mom.y() + mom.z()*mom.z()) / CLHEP::GeV ;
    }
    else if (fLman->fMeshWeights[imesh] == 2 ) { // Edep
      weightVal=step->GetTotalEnergyDeposit() / CLHEP::joule ; 
      scaleByLen=1;
    }
    weightVal=weightVal*nelec ; // Weight by number of electrons
    
    // Get number of grid points in each dimension
    unsigned int n1=fLman->fMeshN1[imesh];
    unsigned int n2=fLman->fMeshN2[imesh];
    unsigned int n3=fLman->fMeshN3[imesh];
    
    // Mesh data pointer
    double* mdata = fLman->fMeshData[imesh];
    
    // --- Deal with weights where value needs to be scaled by fraction of track step in each mesh box --- //
    int idim;
      
    // Get starting & ending mesh boxes and step lengths
    int nmi[3], nmf[3], bind[3], bind_next[3];
    double dstep[3], ipos[3], fpos[3], wdim[3];
    wdim[0]=fLman->worldDimensions.x()/m;
    wdim[1]=fLman->worldDimensions.y()/m;
    wdim[2]=fLman->worldDimensions.z()/m;
    dstep[0] = (2*fLman->worldDimensions.x()/m) / n1;
    dstep[1] = (2*fLman->worldDimensions.y()/m) / n2;
    dstep[2] = (2*fLman->worldDimensions.z()/m) / n3;
    nmi[0] = (int) floor( (pos1.x()/m+fLman->worldDimensions.x()/m) / dstep[0] ) ;
    nmi[1] = (int) floor( (pos1.y()/m+fLman->worldDimensions.y()/m) / dstep[1] ) ;
    nmi[2] = (int) floor( (pos1.z()/m+fLman->worldDimensions.z()/m) / dstep[2] ) ;
    nmf[0] = (int) floor( (pos2.x()/m+fLman->worldDimensions.x()/m) / dstep[0] ) ;
    nmf[1] = (int) floor( (pos2.y()/m+fLman->worldDimensions.y()/m) / dstep[1] ) ;
    nmf[2] = (int) floor( (pos2.z()/m+fLman->worldDimensions.z()/m) / dstep[2] ) ;
    ipos[0] = pos1.x()/m+wdim[0]; fpos[0] = pos2.x()/m+wdim[0];
    ipos[1] = pos1.y()/m+wdim[1]; fpos[1] = pos2.y()/m+wdim[1];
    ipos[2] = pos1.z()/m+wdim[2]; fpos[2] = pos2.z()/m+wdim[2];
    
    // - Get 3-d line parameterization of step
    // x = at + x1 ; y = bt + y1 ; z = ct + z1
    //xpos=a*t + pos1.x();
    //ypos=b*t + pos1.y();
    //zpos=c*t + pos1.z();
    double a[3],t[3] ;
    double thist, tlast ;
    for (idim=0;idim<3;idim++)
      a[idim]=fpos[idim]-ipos[idim] ;
    
    // Get direction to travel in each dimension
    int dirtrav[3] = {1,1,1} ;
    for (idim=0; idim<3; idim++) {
      if ( nmf[idim] < nmi[idim] )
        dirtrav[idim]=-1;
    }
    
    // Set current mesh bin co-ordinate as starting position
    int dimnext;
    for (idim=0;idim<3;idim++) {
      bind[idim]=nmi[idim];
    }
    thist=0; tlast=0;
    
    while (thist<1) { // iterate through mesh boxes until in the final one
      // Get t value for each dimension boundary along step path, choose to step into closest
      for (idim=0;idim<3;idim++) {
        bind_next[idim]=bind[idim]+dirtrav[idim];
        // t coord to boundary in this dimension
        if (dirtrav[idim]<0)
          t[idim]=((dstep[idim]*bind[idim])-ipos[idim])/a[idim] ;
        else
          t[idim]=((dstep[idim]*bind_next[idim])-ipos[idim])/a[idim] ;
        if (t[idim]>1 || t[idim]<=0) // if final boundary beyond final pos, just go to final pos
          t[idim]=1;
      }
      if (t[0]<=t[1] && t[0]<=t[2])
        dimnext=0;
      else if (t[1]<t[0] && t[1]<=t[2])
        dimnext=1;
      else
        dimnext=2;
      // Throw an exception if invalid t
      if ( (t[0]<0 || t[0]>1) && (t[1]<0 || t[1]>1) && (t[2]<0 || t[2]>1) ) {
        printf("!!INVALID MESH CALCULATION PARAMETER!!\n");
        G4ExceptionDescription msg;
        msg << "Mesh calculation error: t out of bounds"; 
        G4Exception("steppingAction::userSteppingAction(const G4Step*)", "MyCode0", FatalException, msg);
      }
      thist=t[dimnext];
      if (scaleByLen) // Need to get fraction of whole step in this segment
        mdata[bind[0]+bind[1]*n1+bind[2]*n1*n2] += weightVal * (thist-tlast) ;
      else
        mdata[bind[0]+bind[1]*n1+bind[2]*n1*n2] += weightVal ;
      // Go to next mesh box
      bind[dimnext]=bind_next[dimnext];
      tlast=thist;
    }
  }
}

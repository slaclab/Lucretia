#include "eventAction.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Trajectory.hh"
#include "G4ios.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

eventAction::eventAction(lucretiaManager* lman)
: G4UserEventAction(),
  fLman(lman)
{}

eventAction::~eventAction()
{}


void eventAction::BeginOfEventAction(const G4Event*)
{}

void eventAction::EndOfEventAction(const G4Event* event)
{
  // Write trajectory data to lucretia data structures if requested
  /*cout << ">>IN EVENT<<" << G4endl;
  if (fLman->fMaxTrackStore==0)
    return ;
  G4int eventID = event->GetEventID();
  cout << ">>> Event: " << eventID  << G4endl;
  G4TrajectoryContainer* trajectoryContainer = event->GetTrajectoryContainer();
  G4int n_trajectories = 0;
  G4Trajectory* trj ;
  G4int id, parentID, n_point;
  G4ThreeVector pos ;
  if (trajectoryContainer) n_trajectories = trajectoryContainer->entries();
  cout << "TC: " << n_trajectories << G4endl ;
  if ( trajectoryContainer ) {
    for(G4int i=0; i<n_trajectories; i++) {
      trj = (G4Trajectory *)((*(event->GetTrajectoryContainer()))[i]);
      id = trj->GetTrackID();
      parentID = trj->GetParentID() ;
      n_point = trj->GetPointEntries();
      cout << "id: " << id << " parentID: " << parentID << " n_point: " << n_point << G4endl;
      if (parentID==0) { // store trajectories of primary particles 
        for(G4int j=0; j<n_point; j++) {
          pos = ((G4TrajectoryPoint*) (trj->GetPoint(j)))->GetPosition();
          cout << "x: " << pos[0] << " y: " << pos[1] << " z: " << pos[2] << G4endl;
          fLman->WritePrimaryTrackData(pos[0], pos[1], pos[2],id);
        }
      }
    } 
    }*/
  
}  

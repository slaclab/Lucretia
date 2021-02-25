#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "geomConstruction.hh"
#include "actionInitialization.hh"
#include "FTFP_BERT.hh"
#include "G4StepLimiterPhysics.hh"
#include "CLHEP/Units/SystemOfUnits.h"
int main()
{
  using namespace std;
  // construct the default run manager
  G4RunManager* runManager = new G4RunManager;
  // G4MTRunManager for multithreaded || G4RunManager for single threaded
  
  // Collimator geometry definition (apertures and lengths are half-lengths in meters)
  G4double length = 0.1 ;
  // Cuts to determine which particles get returned to Lucretia tracking after geant tracking through the geometry
  G4double ecut = 1.4; // GeV
  G4double zcut = length ; // m
  // collType ("Rectangle" or "Ellipse"), materialName, aper_x, aper_y, thickness, length
  geomConstruction* thisGeomConstruction = new geomConstruction("Rectangle", "G4_Sm", 0.0001, 0.0001, 0.01, length);
  runManager->SetUserInitialization(thisGeomConstruction);

  // Setup physics processes we wish to use
  G4VModularPhysicsList* physicsList = new FTFP_BERT; // Default ALL physics process list
  //physicsList->RegisterPhysics(new G4StepLimiterPhysics());
  runManager->SetUserInitialization(physicsList);
  
  // Initialize action routines (to set primary particles and routine to store tracking results)
  runManager->SetUserInitialization(new actionInitialization(length,ecut,zcut)); //

  // initialize G4 kernel
  runManager->Initialize();
  
  // get the pointer to the UI manager and set verbosities
  G4UImanager* UI = G4UImanager::GetUIpointer();
  UI->ApplyCommand("/run/verbose 0");
  UI->ApplyCommand("/event/verbose 0");
  UI->ApplyCommand("/tracking/verbose 0");
  
  // start a run
  runManager->BeamOn(1);

  // clean up and return
  delete runManager;
  return 0;
}

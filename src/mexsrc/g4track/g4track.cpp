#include "../LucretiaCommon.h"
#include "g4track.hh"
#ifndef LUCRETIA_MANAGER
#include "lucretiaManager.hh"
#endif
//#include "G4MTRunManager.hh" // -> this for the multi-threaded version
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4GeometryManager.hh"
#include "geomConstruction.hh"
#include "actionInitialization.hh"
#include "lSession.hh"
#include "PhysicsList.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include <string>
#include <stdlib.h>

using namespace std ;

// Interface from Lucretia to GEANT4 tracking routines
// - returns number of Lucretia macro-particles un-stopped or -1 if error parsing Lucretia data
int g4track(int* blele, int* bunchno, struct Beam* TheBeam, double* L, double* ElemOffset)
{                     
  struct Bunch* ThisBunch = TheBeam->bunches[*bunchno] ;
  static int runno=1; // run number starts at 1
  // construct the run manager, geometry, physics, UI and action objects (the first time this function is called only)
  //static G4MTRunManager* runManager; // -> this for the multi-threaded version
  static G4RunManager* runManager ;
  static G4UImanager* UI ;
  static geomConstruction* thisGeomConstruction ;
  static actionInitialization* thisAction ;
  static lucretiaManager* lman ;
  static int firstCall=1;
  // Geometry definition - apertures and lengths are half-lengths in meters
  G4double length = *L/2 ;
  // printf(">>>>>GEANT4 Run BL ELE %d\n",*blele);
  if (runManager == NULL) {
    // lucretiaManager manages interaction with Lucretia bunch structure and interfaces with Matlab data structures
    //printf("LMAN INIT 1...\n");
    lman = new lucretiaManager(blele, bunchno, ThisBunch, length, ElemOffset) ;
    //printf("done.\n");
    // return with error if no associated EXT Process on this BEAMLINE or there is a problem with one of the EXT Process object's properties
    if (lman->Status!=0)
      return -1 ;
    runManager = new G4RunManager;
    // get the pointer to the UI manager
    UI = G4UImanager::GetUIpointer();
    // Setup physics processes
    //G4VModularPhysicsList* physicsList = new FTFP_BERT_HP;
    PhysicsList* physicsList = new PhysicsList ;
    runManager->SetUserInitialization(physicsList);
    thisGeomConstruction = new geomConstruction(lman, length);
    if (thisGeomConstruction == NULL) {
      printf("GEANT4 GEOM Construction Failed!\n");
      return -2 ;
    }
    runManager->SetUserInitialization(thisGeomConstruction);
    // Initialize action routines (to set primary particles and routine to store tracking results)
    thisAction = new actionInitialization(lman) ;
    runManager->SetUserInitialization(thisAction); //
    //printf("Done Init\n");
  }
  else { // If this function already called once, just re-initialise the Lucretia pointers to the new element and geometry
    // printf("LMAN INIT...\n");
    lman->Initialize(blele, bunchno, ThisBunch, length, ElemOffset) ;
    if (lman->Status!=0)
      return lman->Status ;
    //
    //printf("reinitgeom...\n");
    thisGeomConstruction->SetGeomParameters(lman) ; // Set the new geometry parameters to be built on initialize command
  }
  // Define Lucreia (Matlab) console status and error output streams
  lSession* thisSession = new lSession(lman);
  UI->SetCoutDestination(thisSession);
  // Set Random number
  CLHEP::HepRandom::setTheEngine(new CLHEP::RanecuEngine);
  CLHEP::HepRandom::setTheSeed(lman->RandSeed);
  // initialize G4 kernel
  // printf("runno>> %d\n",runno);
  if (firstCall==0) {
    UI->ApplyCommand("/persistency/gdml/clear");
    runManager->ReinitializeGeometry(true); // Force new run to set new geometry and delete old
    runManager->Initialize();
  }
  else {
    firstCall = 0;
    UI->ApplyCommand("/cuts/setLowEdge 1 eV");
    UI->ApplyCommand("/cuts/setMaxCutEnergy 1000");
    runManager->Initialize();
    //UI->ApplyCommand("/score/open boxMesh_1");
  }
  // set verbosities
  if (lman->Verbose==1) {
    UI->ApplyCommand("/run/verbose 1");
    UI->ApplyCommand("/event/verbose 0");
    UI->ApplyCommand("/tracking/verbose 0");
    UI->ApplyCommand("/control/verbose 2");
    UI->ApplyCommand("/run/dumpRegion") ;
    UI->ApplyCommand("/process/verbose 1");
    UI->ApplyCommand("/run/particle/verbose 1");
    UI->ApplyCommand("/process/setVerbose 1 all");
    UI->ApplyCommand("/process/list all");
    UI->ApplyCommand("/particle/list all");
    UI->ApplyCommand("/material/g4/printMaterial User1");
    UI->ApplyCommand("/material/g4/printMaterial User2");
    UI->ApplyCommand("/material/g4/printMaterial User3");
  }
  else if (lman->Verbose==2 || lman->Verbose==3) {
    UI->ApplyCommand("/run/verbose 2");
    UI->ApplyCommand("/event/verbose 1");
    if (lman->Verbose==2)
      UI->ApplyCommand("/tracking/verbose 1");
    else
      UI->ApplyCommand("/tracking/verbose 3");
    UI->ApplyCommand("/control/verbose 2");
    UI->ApplyCommand("/run/dumpRegion") ;
    UI->ApplyCommand("/process/verbose 1");
    UI->ApplyCommand("/run/particle/verbose 2");
    UI->ApplyCommand("/process/setVerbose 2 all");
    UI->ApplyCommand("/process/list all");
    UI->ApplyCommand("/particle/list all");
    UI->ApplyCommand("/material/verbose 2");
    UI->ApplyCommand("/material/g4/printMaterial User1");
    UI->ApplyCommand("/material/g4/printMaterial User2");
    UI->ApplyCommand("/material/g4/printMaterial User3");
  }
  else {
    UI->ApplyCommand("/run/verbose 0");
    UI->ApplyCommand("/event/verbose 0");
    UI->ApplyCommand("/tracking/verbose 0");
    UI->ApplyCommand("/control/verbose 0");
    UI->ApplyCommand("/process/verbose 0");
    UI->ApplyCommand("/run/particle/verbose 0");
    UI->ApplyCommand("/process/setVerbose 0 all");
  }
//   printf("Apply Run Cuts...\n");
  lman->ApplyRunCuts(UI); // Apply user cuts on particles and processes specified from Lucretia ExtProcess object
  //UI->ApplyCommand("/process/list");
  // start a run
  // printf("Starting run...\n");
  runManager->BeamOn(runno);
  // printf("Finished run.\n");
  runno++;

  // return
  // - return variables to Matlab workspace
//   printf("Sending data to Matlab...\n");
  lman->SetLucretiaData() ;
//   printf("Free mem...\n");
  lman->freeMem() ;
//   printf("Deleting session...\n");
  delete thisSession ;
//   printf("Done.\n");
  return lman->fNumRaysResumed ;
}


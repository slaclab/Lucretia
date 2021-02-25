#include "FieldSetup.hh"
#include "G4UniformElectricField.hh"
#include "G4UniformMagField.hh"
#include "G4MagneticField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4EquationOfMotion.hh"
#include "G4EqMagElectricField.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4MagIntegratorDriver.hh"
#include "G4ChordFinder.hh"

#include "G4ExplicitEuler.hh"
#include "G4ImplicitEuler.hh"
#include "G4SimpleRunge.hh"
#include "G4SimpleHeum.hh"
#include "G4ClassicalRK4.hh"
#include "G4HelixExplicitEuler.hh"
#include "G4HelixImplicitEuler.hh"
#include "G4HelixSimpleRunge.hh"
#include "G4CashKarpRKF45.hh"
#include "G4RKG3_Stepper.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include <iostream>
//  Constructors:

FieldSetup::FieldSetup(lucretiaManager* lman)
 : fFieldManager(0),
   fChordFinder(0),
   fEquation(0),
   fElFieldValue(),
   fStepper(0),
   fIntgrDriver(0),
   fMinStep(lman->EMStepSize*m)
{
  fUniform=lman->EMisUniform;
  if (fUniform==1) {
    double magField[3]={0,0,0};
    lman->GetUniformField(magField);
    fMfield = new G4UniformMagField(G4ThreeVector(magField[0]*tesla,magField[1]*tesla,magField[2]*tesla));
    fEquation = new G4EqMagElectricField(fMfield);
  }
  else if (fUniform==2) {
    double eField[3]={0,0,0};
    lman->GetUniformField(eField);
    fEfield = new G4UniformElectricField(G4ThreeVector(eField[0]*kilovolt/m,eField[1]*kilovolt/m,eField[2]*kilovolt/m));
    fEquation = new G4EqMagElectricField(fEfield);
  }
  else {
    fEMfield = new GlobalField(lman); // Gets field data from LucretiaManager object
    fEquation = new G4EqMagElectricField(fEMfield);
  }
  fFieldManager = GetGlobalFieldManager();
  UpdateField(lman);
}

FieldSetup::~FieldSetup()
{
  delete fChordFinder;
  delete fStepper;
  delete fEquation;
  if (fUniform==1)
    delete fMfield;
  else if (fUniform==2)
    delete fEfield;
  else
    delete fEMfield;
}

void FieldSetup::UpdateField(lucretiaManager* lman)
{
// Register this field to 'global' Field Manager and
// Create Stepper and Chord Finder with requested type, minstep (resp.)

  SetStepper(lman);
  if (fUniform==1)
    fFieldManager->SetDetectorField(fMfield);
  else if (fUniform==2)
    fFieldManager->SetDetectorField(fEfield);
  else
    fFieldManager->SetDetectorField(fEMfield);
  if (fChordFinder) delete fChordFinder;
  fIntgrDriver = new G4MagInt_Driver(fMinStep,
                                     fStepper,
                                     fStepper->GetNumberOfVariables());
  fChordFinder = new G4ChordFinder(fIntgrDriver);
  if (lman->EMDeltaChord!=0) {
    fChordFinder->SetDeltaChord( lman->EMDeltaChord*m );
  }
  fFieldManager->SetChordFinder(fChordFinder);
  if (lman->EMDeltaOneStep!=0) {
    fFieldManager->SetAccuraciesWithDeltaOneStep(lman->EMDeltaOneStep*m);
  }
  if (lman->EMDeltaIntersection!=0) {
    fFieldManager->SetDeltaIntersection(lman->EMDeltaIntersection*m);
  }
  if (lman->EMEpsMin!=0) {
    G4TransportationManager::GetTransportationManager()->GetPropagatorInField()->SetMinimumEpsilonStep(lman->EMEpsMin*m);
  }
  if (lman->EMEpsMax!=0) {
    G4TransportationManager::GetTransportationManager()->GetPropagatorInField()->SetMaximumEpsilonStep(lman->EMEpsMax*m);
  }
}

void FieldSetup::SetStepper(lucretiaManager* lman)
{
// Set stepper according to the stepper type

  G4int nvar = 8;

  if (fStepper) delete fStepper;

  if (!strcmp(lman->EMStepperMethod,"ExplicitEuler"))
    fStepper = new G4ExplicitEuler( fEquation, nvar );
  else if (!strcmp(lman->EMStepperMethod,"ImplicitEuler"))
    fStepper = new G4ImplicitEuler( fEquation, nvar );
  else if (!strcmp(lman->EMStepperMethod,"SimpleRunge"))
    fStepper = new G4SimpleRunge( fEquation, nvar );
  else if (!strcmp(lman->EMStepperMethod,"SimpleHeum"))
    fStepper = new G4SimpleHeum( fEquation, nvar );
  else if (!strcmp(lman->EMStepperMethod,"ClassicalRK4"))
    fStepper = new G4ClassicalRK4( fEquation, nvar );
  else if (!strcmp(lman->EMStepperMethod,"CashKarpRKF45"))
    fStepper = new G4CashKarpRKF45( fEquation, nvar );
  else {
    G4cout<<"Unknown Stepper function, using default G4ClassicalRK4"<<G4endl;
    fStepper = new G4ClassicalRK4( fEquation, nvar );
  }
  
}


G4FieldManager* FieldSetup::GetGlobalFieldManager()
{
//  Utility method
  return G4TransportationManager::GetTransportationManager()->GetFieldManager();
}


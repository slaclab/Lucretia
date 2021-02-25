#include "G4ElectricField.hh"
#include "G4UniformElectricField.hh"
#include "G4UniformMagField.hh"
#include "GlobalField.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif
class G4FieldManager;
class G4ChordFinder;
class G4EquationOfMotion;
class G4Mag_EqRhs;
class G4EqMagElectricField;
class G4MagIntegratorStepper;
class G4MagInt_Driver;

class FieldSetup
{
public:
  FieldSetup(lucretiaManager* lman);

  virtual ~FieldSetup();

  void SetStepper(lucretiaManager* lman);

  void SetMinStep(G4double s) { fMinStep = s ; }

  void UpdateField(lucretiaManager* lman);

  GlobalField*            fEMfield;

protected:

  // Find the global Field Manager

  G4FieldManager*         GetGlobalFieldManager();

private:
  char fUniform;

  G4FieldManager*         fFieldManager;

  G4ChordFinder*          fChordFinder;

  G4EqMagElectricField*   fEquation;

  
  G4ElectricField*        fEfield;
  G4MagneticField*        fMfield;
 
  G4ThreeVector           fElFieldValue;

  G4MagIntegratorStepper* fStepper;
  G4MagInt_Driver*        fIntgrDriver;

  G4double                fMinStep;

};


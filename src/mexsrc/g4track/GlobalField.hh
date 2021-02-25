#include "G4FieldManager.hh"
#include "G4PropagatorInField.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4ChordFinder.hh"

#include "G4MagneticField.hh"
#include "G4ElectroMagneticField.hh"

#include "G4Mag_EqRhs.hh"
#include "G4Mag_SpinEqRhs.hh"

#include "G4EqMagElectricField.hh"
#include "G4EqEMFieldWithSpin.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

class GlobalField : public G4ElectroMagneticField {

public:
  GlobalField(lucretiaManager* lman);
  virtual ~GlobalField();

  /// GetFieldValue() returns the field value at a given point[].
  /// field is really field[6]: Bx,By,Bz,Ex,Ey,Ez.
  /// point[] is in global coordinates: x,y,z,t.
  virtual void GetFieldValue(const G4double* point, G4double* field) const;

  /// DoesFieldChangeEnergy() returns true.
  virtual G4bool DoesFieldChangeEnergy() const { return true; }

private:
  lucretiaManager* fLman;
};

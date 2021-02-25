#include <time.h>
#include "G4SystemOfUnits.hh"
#include "GlobalField.hh"
#include <iostream>

GlobalField::GlobalField(lucretiaManager* lman)
: G4ElectroMagneticField(),
        fLman(lman)
{}

GlobalField::~GlobalField()
{}

void GlobalField::GetFieldValue(const G4double* point, G4double* field) const
{
  
  field[0] = field[1] = field[2] = field[3] = field[4] = field[5] = 0.0;
  
  // protect against Geant4 bug that calls us with point[] NaN.
  if(point[0] != point[0]) return;
  int ifield;
  
  // Store track point if requested
  if (fLman->fMaxTrackStore>0)
    fLman->WritePrimaryTrackData(point[0]/m, point[1]/m, point[2]/m);
  
  // Interpolate provided field in lucretiaManager class
  // Assumes 3D field distributed over entire detector world volume
  if (fLman->EnableEM>0) {
    for (ifield=0;ifield<3;ifield++)
      field[ifield]=fLman->interpField(ifield,point)*tesla;
    for (ifield=3;ifield<6;ifield++)
      field[ifield]=fLman->interpField(ifield,point)*kilovolt/m;
  }
  //cout << "POINT: " << point[0]/m << " " << point[1]/m << " " << point[2]/m << " FIELD: " << field[0]/tesla << " " << field[1]/tesla << " " << field[2]/tesla << "\n" ;
  
}

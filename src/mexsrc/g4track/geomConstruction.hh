#include "G4VUserDetectorConstruction.hh"
#include "G4ThreeVector.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "G4EllipticalTube.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh" 
#include "G4Box.hh"
#include "G4Trd.hh"
#include "G4NistManager.hh"
#include "G4GDMLParser.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif
#include "FieldSetup.hh"

class G4LogicalVolume;
class G4Material;
class DetectorMessenger;

class geomConstruction : public G4VUserDetectorConstruction
{
  public:
    geomConstruction(
      lucretiaManager* lman,
      G4double dz = 1*CLHEP::m);
    ~geomConstruction();

  public:
    // methods from base class 
  virtual G4VPhysicalVolume* Construct();
    // others
  void SetGeomParameters(lucretiaManager* lman);

  virtual void ConstructSDandField();

  private:
  G4Material* ProcessMaterial(G4String materialName, int matID ) ;
  uint32_t hashL(char *key, size_t len) ;
  G4String fCollType;
  G4String fCollMaterialName;
  G4String fCollMaterialName2;
  G4double fCollAperX;
  G4double fCollAperY;
  G4double fCollThickness;
  G4double fCollLength;
  G4double fCollLength2;
  G4double fCollAperX2 ;
  G4double fCollAperY2 ;
  G4double fCollAperX3 ;
  G4double fCollAperY3 ;
  G4double fCollDX ;
  G4double fCollDY ;
  G4String fVacuumMaterial;
  G4NistManager* nistManager ;
  G4Box* sWorld;
  G4LogicalVolume* worldVolume;
  G4VPhysicalVolume* pWorld;
  G4Box* collBox;
  G4EllipticalTube* collTube ;
  G4LogicalVolume* collVolume;
  G4PVPlacement* collPlacement;
  G4Box* collInnerBox ;
  G4EllipticalTube* collInnerTube ;
  G4LogicalVolume* collVolumeInner ;
  G4PVPlacement* collInnerPlacement ;
  FieldSetup* fEmFieldSetup;
  lucretiaManager* fLman;
  G4Box* collTapBox ;
  G4Trd* collTapBox1 ;
  G4Box* collTapBox2 ;
  G4Box* collTapBox3 ;
  G4Trd* collTapBox4 ;
  G4PVPlacement* collInnerPlacement2 ;
  G4PVPlacement* collInnerPlacement3 ;
  G4PVPlacement* collInnerPlacement4 ;
  G4Material* Vacuum ;
  G4Material* collMaterial ;
  G4Material* collMaterial2 ;
  G4Material* tunnelMaterial ;
  G4Material* airMaterial ;
  G4Trap* mainTunnel ;
  G4Trap* servTunnel ;
  G4LogicalVolume* mainTunnelVolume ;
  G4LogicalVolume* servTunnelVolume ;
  G4PVPlacement* mainTunnelPlacement ;
  G4PVPlacement* servTunnelPlacement ;
};


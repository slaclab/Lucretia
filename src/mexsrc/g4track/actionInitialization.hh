#include "G4VUserActionInitialization.hh"
#include "G4UnitsTable.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

//class B4DetectorConstruction;

/// Action initialization class.
///

//class lucretiaManager;

class actionInitialization : public G4VUserActionInitialization
{
public:
  actionInitialization(lucretiaManager* lman );
  ~actionInitialization();
  virtual void BuildForMaster() const;
  virtual void Build() const;
private:
  lucretiaManager* fLman ;
};


    

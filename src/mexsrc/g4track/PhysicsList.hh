#ifndef PhysicsList_h
#define PhysicsList_h 1

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

//class PhysicsListMessenger;
class G4VPhysicsConstructor;

class PhysicsList: public G4VModularPhysicsList
{
public:
  PhysicsList();
 ~PhysicsList();

  // Construct particles
  virtual void ConstructParticle();

  virtual void SetCuts();
  void SetAnalyticSR(G4bool val) {fSRType = val;};

  // Construct processes and register them
  virtual void ConstructProcess();
  void ConstructGeneral();

private:
  G4bool                 fSRType;
  G4VPhysicsConstructor*  fParticleList;
  G4VPhysicsConstructor*  fHadPhysicsList;
  G4VPhysicsConstructor*  fPhysList1;
  //PhysicsListMessenger*  fMess;
};


#endif


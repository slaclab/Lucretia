#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"
#ifndef LUCRETIA_MANAGER
  #include "lucretiaManager.hh"
#endif

class G4ParticleGun;
class G4Event;

/// The primary generator action class with particle gum.
///
/// It defines a single particle which hits the Tracker 
/// perpendicular to the input face. The type of the particle
/// can be changed via the G4 build-in commands of G4ParticleGun class 
/// (see the macros provided with this example).

class primaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
  primaryGeneratorAction(lucretiaManager* lman);    
    virtual ~primaryGeneratorAction();

    virtual void GeneratePrimaries(G4Event* );

    G4ParticleGun* GetParticleGun() {return fParticleGun;}
  
    // Set methods
    void SetRandomFlag(G4bool );

  private:
    G4ParticleGun*          fParticleGun; // G4 particle gun
  lucretiaManager* fLman ;
};



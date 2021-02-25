#ifndef LUCRETIA_MANAGER
  #include "../LucretiaGlobalAccess.h"
#endif
//#include "../LucretiaMatlab.h"
#include <string>
#define LUCRETIA_MANAGER
#include "mex.h"
#include "matrix.h"
#include "G4UImanager.hh"
#include "G4TrackStatus.hh"
#include "G4UnitsTable.hh"
#include "G4ThreeVector.hh"

/* Interpolation code extracted and modified from ba_interp3.cpp from Mathworks File Exchange*/
/* ba_inter3 GPL License (c) 2008 Brian Amberg http://www.brian-amberg.de/ */

using namespace std;

class lucretiaManager
{
  public:
  lucretiaManager(int* blele, int* bunchno, struct Bunch* ThisBunch, double L, double* ElemOffset) ;
  ~lucretiaManager();
  void Initialize(int* blele, int* bunchno, struct Bunch* ThisBunch, double L, double* ElemOffset) ;
  struct Bunch* fBunch ;
  int GetNextX() ;
  void SetNextX(double x[6], int id, int doresume, double trackEnergy, double gTime) ;
  int SetNextSecondary(double x[6], int id, const char* type,G4String procName, G4TrackStatus trackStatus, double gTime) ;
  void SetLucretiaData() ;
  void freeMem() ;
  void WritePrimaryTrackData(double x, double y, double z) ;
  void GetUniformField(double uField[3]) ;
  double interpField(const int fieldno, const double* point) ;
  void ApplyRunCuts(G4UImanager* UI) ;
  int Status ; // 0=OK; 1=No external processes defined for this BEAMLINE element
  char* GeomType ; // Geometry type ("Ellipse" | "Rectangle" | "Tapered")
  char* Material ; 
  char* Material2 ; 
  char* VacuumMaterial ;
  uint8_T fMeshEntries ; // Number of scoring meshes defined
  float SumEdep ; // Total energy deposited in BEAMLINE element
  char* PrimaryType ; // Type of primary particle to track
  const mxArray *pBx,*pBy,*pBz,*pEx,*pEy,*pEz ; // EM field values
  char* EMStepperMethod ;
  double EMStepSize ;
  double EMDeltaOneStep ;
  double EMDeltaIntersection ;
  double EMDeltaChord ;
  double EMEpsMin ;
  double EMEpsMax ;
  char* EMInterpMethod ;
  char EnableEM ;
  char EMisUniform ;
  double AperX ; // Horizontal half-aperture / m
  double AperY ; // Vertical half-aperture / m
  double Ecut ; // Energy cut for storing tracks
  double Thickness ; // Material thickness / m
  int Verbose; // GEANT4 text output verbosity
  double Lcut;
  int fNumRaysResumed ; // Keep count of number of un-stopped rays
  uint32_T fMaxSecondaryParticles ; // Max number of secondaries to keep
  uint8_T fSecondaryStorageCuts ; // Control conditions under which secondary particles are stored (0= all 1=d/s face only)
  uint32_T fMaxSecondaryParticlesPerPrimary ; // Max secondaries to store per primary launched
  uint32_T fSecondariesCounter ;
  uint32_T RandSeed ;
  int* fSecondariesPerThisPrimary ;
  int* fEle ;
  double CollLen2 ;
  double AperX2 ;
  double AperY2 ;
  double AperX3 ;
  double AperY3 ;
  double CollDX ;
  double CollDY ;
  double MatP[2] ;
  double MatT[2] ;
  G4ThreeVector worldDimensions ; // half-width of world box
  struct UserElement {
    char* Name;
    char* Symbol;
    double Z;
    double A;
    double FractionMass;
  } ;
  struct {
    double density;
    double pressure;
    double temperature;
    char* state;
    size_t NumComponents;
    struct UserElement* element;
  } UserMaterial[3];
  uint32_T fMaxTrackStore ; // Max number of tracking points to store (per primary)
  unsigned int fTrackStoreCounter ; // Counter array for keeping track of stored points
  double* fTrackStoreData_x ; // pointer to stored tracking points
  double* fTrackStoreData_y ; // pointer to stored tracking points
  double* fTrackStoreData_z ; // pointer to stored tracking points
  unsigned int* fTrackStorePointer ; // array of pointers to last TrackStoreData for ray ID
  int* fPrimIndex ;
  uint32_T* fMeshWeights ; // Mesh weighting code
  uint32_T* fMeshPtypes ; // Mesh particle type codes
  uint32_T* fMeshN1 ;
  uint32_T* fMeshN2 ;
  uint32_T* fMeshN3 ;
  double** fMeshData ; // Pointer to storage for mesh data
  double TunnelGeom[2] ; // Width/height of beam tunnel
  char* TunnelMaterial ; // Material to fill tunnel with
  double TunnelMainGeom[12] ; // Main tunnel geometry description + offset
  double TunnelServiceGeom[12] ; // Service tunnel geometry description + offset
  double beamlineCoord[4] ; // Beam Pipe co-ordinates in tunnel reference frame [x,xang,y,yang] in m / degrees
  bool SecondaryAsPrimary ;
  double* fPrimaryRayTime ;
  double* fSecondaryRayTime ;
  G4String GDMLFile ;
  double* FieldBox ;

  private:
  const int fNPart; // number of particle types for physics processes supported
  const int fNProc; // number of supported physics processes
  string fPartList[5] ;
  string fProcList[20] ;
  int access(int M, int N, int O, int x, int y, int z) ;
  int access_unchecked(int M, int N, int x, int y, int z) ;
  void indices_linear(
        int &f000_i,
        int &f100_i,
        int &f010_i,
        int &f110_i,
        int &f001_i,
        int &f101_i,
        int &f011_i,
        int &f111_i,
        const int x, const int y, const int z,
        const mwSize &M, const mwSize &N, const mwSize &O) ;
  void indices_cubic(int f_i[64], const int x, const int y, const int z,
                                      const mwSize &M, const mwSize &N, const mwSize &O) ;
  void interpolate_nearest(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) ;
  void interpolate_linear(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) ;
  void interpolate_bicubic(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) ;
  mxArray* fTypeCellPtr ;
  uint32_T* fPrimaryRegenID ;
  uint32_T* fSecondaryPrimaryID ;
  uint8_T* fSecondaryProcType ;
  uint8_T* fSecondaryTrackStatus ;
  double* fSecondaryBunch_x ;
  int* fBunchNo ; // Bunch number
  uint32_T* fPrimOrder ; // Order in which to read in primary particles
  uint32_T fRayCount ;
  int fRayGetPtr ;
  uint32_T fMaxPrimaryParticles ; // Max number of primaries to generate from Lucretia beam
  bool fForceProcess ; // Force macro particles to be processed by ExtProcess regardless of aperture cuts
  bool* fProcessSelection ;
  double* fParticleCuts ;
} ;


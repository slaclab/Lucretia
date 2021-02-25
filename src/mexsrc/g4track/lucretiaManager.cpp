#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "lucretiaManager.hh"
//#include "../LucretiaGlobalAccess.h"
#include "../LucretiaMatlab.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include "mex.h"
#include "G4SystemOfUnits.hh"
//#include "PhysicalConstants.h"

using namespace std;

lucretiaManager::lucretiaManager(int* blele, int* bunchno, struct Bunch* ThisBunch, double len, double* ElemOffset):
  fMeshEntries(0), SumEdep(0), fNPart(5), fNProc(20)
{
  fPrimIndex=NULL;
  fSecondaryPrimaryID = NULL ;
  fSecondariesPerThisPrimary = NULL ;
  Material = NULL ;
  GeomType = NULL ;
  fSecondaryBunch_x = NULL ;
  fPrimaryRegenID = NULL ;
  fTrackStoreData_x = NULL ;
  fTrackStoreData_y = NULL ;
  fTrackStoreData_z = NULL ;
  fTrackStorePointer = NULL ;
  PrimaryType = NULL ;
  fPrimaryRayTime = NULL ;
  fSecondaryRayTime = NULL ;
  int imat;
  for (imat=0;imat<3;imat++)
    UserMaterial[imat].NumComponents=0;
  // Supported particles and processes
  fPartList[0] = "gamma";
  fPartList[1] = "e-";
  fPartList[2] = "e+";
  fPartList[3] = "mu-";
  fPartList[4] = "mu+";
  fProcList[0] = "msc";
  fProcList[1] = "eIoni";
  fProcList[2] = "eBrem";
  fProcList[3] = "annihil";
  fProcList[4] = "SynRad";
  fProcList[5] = "phot";
  fProcList[6] = "compt";
  fProcList[7] = "conv";
  fProcList[8] = "Rayl";
  fProcList[9] = "muIoni";
  fProcList[10] = "muBrems";
  fProcList[11] = "muPairProd";
  fProcList[12] = "AnnihiToMuPair";
  fProcList[13] = "GammaToMuPair";
  fProcList[14] = "ee2hadr";
  fProcList[15] = "electronNuclear";
  fProcList[16] = "positronNuclear";
  fProcList[17] = "photonNuclear";
  fProcList[18] = "muonNuclear";
  fProcList[19] = "Decay";
  Initialize(blele, bunchno, ThisBunch, len, ElemOffset) ;
}

lucretiaManager::~lucretiaManager()
{
  freeMem();
}

void lucretiaManager::freeMem()
{
  int imat,icomp;
  if (fMaxSecondaryParticles>0) {
    if (fSecondaryBunch_x!=NULL) {
      mxFree(fSecondaryPrimaryID);
      mxFree(fSecondaryProcType);
      mxFree(fSecondaryTrackStatus);
      mxFree(fSecondaryBunch_x);
    }
  }
  if ( fSecondariesPerThisPrimary != NULL )
    mxFree(fSecondariesPerThisPrimary) ;
  if (fPrimIndex!=NULL)
    mxFree(fPrimIndex) ;
  if (Material!=NULL) {
    mxFree(Material) ;
    mxFree(EMInterpMethod) ;
  }
  if (Material2!=NULL)
    mxFree(Material2) ;
  mxFree(EMStepperMethod) ;
  if (GeomType!=NULL)
    mxFree(GeomType) ;
  if (UserMaterial[0].NumComponents != 0)
    mxFree(VacuumMaterial);
  for (imat=0;imat<3;imat++) {
    if (UserMaterial[imat].NumComponents != 0) {
      mxFree(UserMaterial[imat].state) ;
      for (icomp=0;icomp<UserMaterial[imat].NumComponents;icomp++) {
        mxFree(UserMaterial[imat].element[icomp].Name) ;
        mxFree(UserMaterial[imat].element[icomp].Symbol) ;
      }
      mxFree(UserMaterial[imat].element) ;
    }
  }
  int imesh ;
  if (fMeshEntries>0) {
    mxFree(fMeshWeights);
    mxFree(fMeshN1); mxFree(fMeshN2); mxFree(fMeshN3);
    for (imesh=0;imesh<fMeshEntries;imesh++) {
      mxFree(fMeshData[imesh]) ;
    }
    mxFree(fMeshData);
    mxFree(fMeshPtypes);
  }
  if (PrimaryType!=NULL)
    mxFree(PrimaryType) ;
  if (fMaxTrackStore>0) {
    if (fTrackStoreData_x != NULL) {
      mxFree(fTrackStoreData_x) ;
      mxFree(fTrackStoreData_y) ;
      mxFree(fTrackStoreData_z) ;
      mxFree(fTrackStorePointer) ;
    } 
  }
  if (fPrimaryRegenID!=NULL)
    mxFree(fPrimaryRegenID) ;
  if (TunnelMaterial!=NULL)
    mxFree(TunnelMaterial) ;
  if (fPrimaryRayTime!=NULL)
    mxFree(fPrimaryRayTime);
  if (fSecondaryRayTime!=NULL)
    mxFree(fSecondaryRayTime);
}

void lucretiaManager::Initialize(int* blele, int* bunchno, struct Bunch* ThisBunch, double len, double* ElemOffset)
{
  uint32_T i ;
  fBunch=ThisBunch;
  fBunchNo=bunchno;
  Status=0;
  Lcut=len;
  fEle=blele;
  fRayCount=0;
  fRayGetPtr=0;
  fNumRaysResumed=0;
  fBunchNo=bunchno;
  size_t buflen=0 ;
  SecondaryAsPrimary=false;
  SumEdep=0;
  unsigned int imat,iele;
  int imesh ;
  // Get number of defined meshes for this element
  fMeshEntries = *(uint8_T*) mxGetData( GetExtProcessData(blele,"meshEntries") );
  // Allocate mesh memory
  if ( fMeshEntries>0 ) {
    fMeshWeights = (uint32_T*) mxMalloc(fMeshEntries*sizeof(uint32_T)) ;
    fMeshPtypes = (uint32_T*) mxMalloc(fMeshEntries*sizeof(uint32_T)) ;
    fMeshN1 = (uint32_T*) mxMalloc(fMeshEntries*sizeof(uint32_T)) ;
    fMeshN2 = (uint32_T*) mxMalloc(fMeshEntries*sizeof(uint32_T)) ;
    fMeshN3 = (uint32_T*) mxMalloc(fMeshEntries*sizeof(uint32_T)) ;
    fMeshData = (double**) mxMalloc(fMeshEntries*sizeof(double*)) ;
    // Read mesh info and initialize mesh data
    mxArray* pMeshInfo;
    //mxArray* pMeshData;
    double* meshtemp;
    int imeshv;
    pMeshInfo = GetExtProcessData(blele, "meshInfo") ;
    //pMeshData = GetExtProcessData(blele, "meshData") ;
    for (imesh=0; imesh<fMeshEntries; imesh++) {
      fMeshWeights[imesh] = *(uint32_T*) mxGetData(mxGetField( pMeshInfo, imesh, "weight")) ;
      fMeshPtypes[imesh] = *(uint32_T*) mxGetData(mxGetField( pMeshInfo, imesh, "ptypes")) ;
      fMeshN1[imesh] = *(uint32_T*) mxGetData(mxGetField(pMeshInfo, imesh, "n1"));
      fMeshN2[imesh] = *(uint32_T*) mxGetData(mxGetField(pMeshInfo, imesh, "n2"));
      fMeshN3[imesh] = *(uint32_T*) mxGetData(mxGetField(pMeshInfo, imesh, "n3"));
      // Allocate memory for data and initialize
      fMeshData[imesh] = (double*) mxMalloc(sizeof(double)*fMeshN1[imesh]*fMeshN2[imesh]*fMeshN3[imesh]);
      //*fMeshData[imesh] = *mxGetPr(mxGetCell(pMeshData, imesh));
      meshtemp = fMeshData[imesh];
      for (imeshv=0;imeshv<(fMeshN1[imesh]*fMeshN2[imesh]*fMeshN3[imesh]);imeshv++)
        meshtemp[imeshv]=0;
    }
  }
  // Tracking previous secondary particles as primaries? (start from somewhere else other than left edge of element)
  mxArray* pSecondaryAsPrimary = GetExtProcessData(blele,"TrackSecondaryAsPrimary") ;
  SecondaryAsPrimary = *(bool *) mxGetData(pSecondaryAsPrimary) ;
  // Get any user defined material definitions
  mxArray* pUserMaterial = GetExtProcessData(blele,"UserMaterial") ;
  size_t nmat = mxGetNumberOfElements(pUserMaterial) ;
  mxArray* pElement;
  mxArray* pEleName;
  mxArray* pEleSymbol;
  mxArray* pState;
  for (imat=0;imat<nmat;imat++) {
    UserMaterial[imat].density = *mxGetPr(mxGetField(pUserMaterial,imat,"Density")) ;
    UserMaterial[imat].pressure = *mxGetPr(mxGetField(pUserMaterial,imat,"Pressure")) ;
    UserMaterial[imat].temperature = *mxGetPr(mxGetField(pUserMaterial,imat,"Temperature")) ;
    pState = mxGetField(pUserMaterial,imat,"State") ;
    buflen = mxGetN(pState)*sizeof(mxChar)+1;
    UserMaterial[imat].state = (char*) mxMalloc(buflen) ;
    mxGetString(pState, UserMaterial[imat].state, buflen) ;
    UserMaterial[imat].NumComponents = *mxGetPr(mxGetField(pUserMaterial,imat,"NumComponents")) ;
    UserMaterial[imat].element = (struct UserElement*) mxMalloc( UserMaterial[imat].NumComponents * sizeof(struct UserElement) ) ;
    pElement = mxGetField(pUserMaterial,imat,"Element") ;
    for (iele=0;iele<UserMaterial[imat].NumComponents;iele++) {
      pEleName = mxGetField(pElement,iele,"Name");
      buflen = mxGetN(pEleName)*sizeof(mxChar)+1;
      UserMaterial[imat].element[iele].Name = (char*) mxMalloc(buflen);
      mxGetString(pEleName, UserMaterial[imat].element[iele].Name, buflen) ;
      pEleSymbol = mxGetField(pElement,iele,"Symbol");
      buflen = mxGetN(pEleSymbol)*sizeof(mxChar)+1;
      UserMaterial[imat].element[iele].Symbol = (char*) mxMalloc(buflen);
      mxGetString(pEleSymbol, UserMaterial[imat].element[iele].Symbol, buflen) ;
      UserMaterial[imat].element[iele].Z=*mxGetPr(mxGetField(pElement,iele,"Z"));
      UserMaterial[imat].element[iele].A=*mxGetPr(mxGetField(pElement,iele,"A"));
      UserMaterial[imat].element[iele].FractionMass=*mxGetPr(mxGetField(pElement,iele,"FractionMass"));
    }
  }
  mxArray* pVacuumMaterial = GetExtProcessData(blele,"VacuumMaterial") ;
  buflen = mxGetN(pVacuumMaterial)*sizeof(mxChar)+1;
  VacuumMaterial = (char*) mxMalloc(buflen);
  mxGetString(pVacuumMaterial,VacuumMaterial,buflen) ;
  /* ==== User Supplied GDML model ==== */
  mxArray* pGDMLFile = GetExtProcessData(blele,"GDMLFile") ;
  buflen = mxGetN(pGDMLFile)*sizeof(mxChar)+1;
  char* cGDMLFile = (char*) mxMalloc(buflen);
  mxGetString(pGDMLFile,cGDMLFile,buflen);
  GDMLFile = std::string(cGDMLFile);
  mxFree(cGDMLFile);
  /* ==== Primary particle type for tracking */
  mxArray* pPrimaryType = GetExtProcessData(blele,"PrimaryType") ;
  buflen = mxGetN(pPrimaryType)*sizeof(mxChar)+1;
  PrimaryType = (char*) mxMalloc(buflen);
  mxGetString(pPrimaryType,PrimaryType,buflen);
  /* ===== Get EM field data ===== */
  // B/E field values (scalar or 3D vectors)
  pBx = GetExtProcessData(blele,"Bx") ;
  pBy = GetExtProcessData(blele,"By") ;
  pBz = GetExtProcessData(blele,"Bz") ;
  pEx = GetExtProcessData(blele,"Ex") ;
  pEy = GetExtProcessData(blele,"Ey") ;
  pEz = GetExtProcessData(blele,"Ez") ;
  EnableEM = *(char*) mxGetData(GetExtProcessData(blele,"EnableEM")) ;
  EMisUniform = 0;
  if (mxGetNumberOfElements(pBx)==1 && mxGetNumberOfElements(pBy)==1 && mxGetNumberOfElements(pBz)==1 &&
          mxGetNumberOfElements(pEx)==1 && mxGetNumberOfElements(pEy)==1 && mxGetNumberOfElements(pEz)==1) {
    if ( (*mxGetPr(pBx)+*mxGetPr(pBy)+*mxGetPr(pBz))!=0  )
      EMisUniform=1;
    if ( (*mxGetPr(pEx)+*mxGetPr(pEy)+*mxGetPr(pEz))!=0  && EMisUniform==0)
      EMisUniform=2;
    else if ( (*mxGetPr(pEx)+*mxGetPr(pEy)+*mxGetPr(pEz))!=0  && EMisUniform==1)
      EMisUniform = 0;
  }
  // - EM Field Box size info
  mxArray* pFieldBox = GetExtProcessData(blele,"FieldBox") ;
  FieldBox = mxGetPr( pFieldBox ) ;
  // Force GlobalField map calculation, uniform field implementation causes calculation to hang for unknown reason...
  EMisUniform = 0;
  EMStepSize = *mxGetPr(GetExtProcessData(blele,"StepPrec")) ;
  EMDeltaOneStep = *mxGetPr(GetExtProcessData(blele,"DeltaOneStep")) ;
  EMDeltaIntersection = *mxGetPr(GetExtProcessData(blele,"DeltaIntersection")) ;
  EMDeltaChord = *mxGetPr(GetExtProcessData(blele,"DeltaChord")) ;
  EMEpsMin = *mxGetPr(GetExtProcessData(blele,"EpsMin")) ;
  EMEpsMax = *mxGetPr(GetExtProcessData(blele,"EpsMax")) ;
  mxArray* pIM = GetExtProcessData(blele,"Interpolator") ;
  if (pIM == NULL) {
    Status = 1;
    return;
  }
  buflen = mxGetN(pIM)*sizeof(mxChar)+1;
  EMInterpMethod = (char*) mxMalloc(buflen);
  mxGetString(pIM, EMInterpMethod, buflen) ;
  mxArray* pStep = GetExtProcessData(blele,"StepMethod") ;
  if (pStep == NULL) {
    Status = 1;
    return;
  }
  buflen = mxGetN(pStep)*sizeof(mxChar)+1;
  EMStepperMethod = (char*) mxMalloc(buflen);
  mxGetString(pStep, EMStepperMethod, buflen) ;
  // ====================================
  // - Geometry type (Ellipse or Rectangle)
  mxArray* pGeomType = GetExtProcessData(blele,"GeometryType") ;
  if (pGeomType == NULL) {
    Status = 1;
    return;
  }
  buflen = mxGetN(pGeomType)*sizeof(mxChar)+1;
  GeomType = (char*) mxMalloc(buflen);
  mxGetString(pGeomType, GeomType, buflen) ;
  // - Material
  mxArray* pMaterial = GetExtProcessData(blele,"Material") ;
  if (pMaterial == NULL) {
    Status = 1;
    return;
  }
  buflen = mxGetN(pMaterial)*sizeof(mxChar)+1;
  Material = (char*) mxMalloc(buflen);
  mxGetString(pMaterial, Material, buflen) ;
  //
  mxArray* pMaterial2 = GetExtProcessData(blele,"Material2") ;
  if (pMaterial2 == NULL) {
    Status = 1;
    return;
  }
  buflen = mxGetN(pMaterial2)*sizeof(mxChar)+1;
  Material2 = (char*) mxMalloc(buflen);
  mxGetString(pMaterial2, Material2, buflen) ;
  // - X and Y Apertures
  mxArray* pAperX = GetExtProcessData(blele,"AperX") ;
  if (pAperX == NULL) {
    Status = 1;
    return;
  }
  AperX = *mxGetPr( pAperX ) ;
  mxArray* pAperY = GetExtProcessData(blele,"AperY") ;
  if (pAperY == NULL) {
    Status = 1;
    return;
  }
  AperY = *mxGetPr( pAperY ) ;
  //
  mxArray* pAperX2 = GetExtProcessData(blele,"AperX2") ;
  if (pAperX2 == NULL) {
    Status = 1;
    return;
  }
  AperX2 = *mxGetPr( pAperX2 ) ;
  mxArray* pAperY2 = GetExtProcessData(blele,"AperY2") ;
  if (pAperY2 == NULL) {
    Status = 1;
    return;
  }
  AperY2 = *mxGetPr( pAperY2 ) ;
  //
  mxArray* pAperX3 = GetExtProcessData(blele,"AperX3") ;
  if (pAperX3 == NULL) {
    Status = 1;
    return;
  }
  AperX3 = *mxGetPr( pAperX3 ) ;
  mxArray* pAperY3 = GetExtProcessData(blele,"AperY3") ;
  if (pAperY3 == NULL) {
    Status = 1;
    return;
  }
  AperY3 = *mxGetPr( pAperY3 ) ;
  // Material pressures / temperatures
  mxArray* pMatP = GetExtProcessData(blele,"MaterialPressure") ;
  if (pMatP == NULL) {
    Status = 1;
    return;
  }
  MatP[0] = *mxGetPr(pMatP) ;
  mxArray* pMatP2 = GetExtProcessData(blele,"Material2Pressure") ;
  if (pMatP2 == NULL) {
    Status = 1;
    return;
  }
  MatP[1] = *mxGetPr(pMatP2) ;
  mxArray* pMatT = GetExtProcessData(blele,"MaterialTemperature") ;
  if (pMatT == NULL) {
    Status = 1;
    return;
  }
  MatT[0] = *mxGetPr(pMatT) ;
  mxArray* pMatT2 = GetExtProcessData(blele,"Material2Temperature") ;
  if (pMatT2 == NULL) {
    Status = 1;
    return;
  }
  MatT[1] = *mxGetPr(pMatT2) ;
  // - Energy cut for returning tracks
  mxArray* pEcut = GetExtProcessData(blele,"Ecut") ;
  if (pEcut == NULL) {
    Status = 1;
    return;
  }
  Ecut = *mxGetPr( pEcut ) ;
  // - Switch for forcing EXT process instead of using aperture cuts
  mxArray* pForceProcess = GetExtProcessData(blele,"ForceProcess") ;
  if (pForceProcess == NULL) {
    Status = 1;
    return;
  }
  fForceProcess = *(bool*)mxGetLogicals( pForceProcess ) ;
  // - Process selection list by particle
  mxArray* pProcessSelection = GetExtProcessData(blele,"processSelection") ;
  if (pProcessSelection == NULL) {
    Status = 1;
    return;
  }
  fProcessSelection = (bool*)mxGetLogicals( pProcessSelection ) ;
  // - Particle Cuts
  mxArray* pParticleCuts = GetExtProcessData(blele,"particleCuts") ;
  if (pParticleCuts == NULL) {
    Status = 1 ;
    return;
  }
  fParticleCuts = mxGetPr(pParticleCuts);
  // Random Number Seed
  mxArray* pRandSeed =  GetExtProcessData(blele,"RandSeed") ;
  if (pRandSeed == NULL) {
    Status = 1 ;
    return;
  }
  RandSeed = *(uint32_T*)mxGetData( pRandSeed ) ;
  // - Max Number of primary particles to track
  mxArray* pMax = GetExtProcessData(blele,"MaxPrimaryParticles") ;
  if (pMax == NULL) {
    Status = 1;
    return;
  }
  fMaxPrimaryParticles = *(uint32_T*)mxGetData( pMax ) ;
  fPrimIndex = (int*) mxMalloc(fMaxPrimaryParticles*sizeof(int)) ;
  // - Material thickness
  mxArray* pThickness = GetExtProcessData(blele,"Thickness") ;
  if (pThickness == NULL) {
    Status = 1;
    return;
  }
  Thickness = *mxGetPr( pThickness ) ;
  //
  mxArray* pCollDX = GetExtProcessData(blele,"CollDX") ;
  if (pCollDX == NULL) {
    Status = 1;
    return;
  }
  CollDX = *mxGetPr( pCollDX ) ;
  //
  mxArray* pCollDY = GetExtProcessData(blele,"CollDY") ;
  if (pCollDY == NULL) {
    Status = 1;
    return;
  }
  CollDY = *mxGetPr( pCollDY ) ;
  // - Secondary collimator length option
  mxArray* pCollLen2 = GetExtProcessData(blele,"CollLen2") ;
  if (pCollLen2 == NULL) {
    Status = 1;
    return;
  }
  CollLen2 = *mxGetPr( pCollLen2 ) ;
  // - Verbosity control
  mxArray* pVerbose = GetExtProcessData(blele,"Verbose") ;
  Verbose = 0 ;
  if (pVerbose != NULL) {
    Verbose = (int) *mxGetPr( pVerbose ) ;
  }
  // - Primaries ordering
  mxArray* pOrder = mxGetCell(GetExtProcessData(blele,"PrimarySampleOrder"), *bunchno) ;
  if (pOrder == NULL)
    fPrimOrder = NULL ;
  else if ( mxGetM(pOrder)*mxGetN(pOrder) < fMaxPrimaryParticles )
    fPrimOrder = NULL ;
  else {
    fPrimOrder = (uint32_T*) mxGetData(pOrder) ;
  }
  // - Primary ray time vectors (time since start of GEANT4 event)
  fPrimaryRayTime = (double*) mxMalloc(fBunch->nray*sizeof(double)) ;
  // - Tracking point data for primaries
  mxArray* pMaxTrackStore = GetExtProcessData(blele,"TrackStoreMax" ) ;
  fMaxTrackStore = 0 ;
  if (pMaxTrackStore != NULL)
    fMaxTrackStore = *(uint32_T*)mxGetData( pMaxTrackStore ) ;
  fTrackStoreCounter=0;
  if (fMaxTrackStore>0) {
    fTrackStoreData_x = (double*) mxMalloc(fMaxTrackStore*sizeof(double)) ;
    fTrackStoreData_y = (double*) mxMalloc(fMaxTrackStore*sizeof(double)) ;
    fTrackStoreData_z = (double*) mxMalloc(fMaxTrackStore*sizeof(double)) ;
    fTrackStorePointer = (unsigned int*) mxMalloc(fBunch->nray*sizeof(unsigned int)) ;
    for (unsigned int id=0;id<fMaxTrackStore;id++) {
      fTrackStoreData_x[id]=0;
      fTrackStoreData_y[id]=0;
      fTrackStoreData_z[id]=0;
    }
    for (int iray=0;iray<fBunch->nray;iray++)
      fTrackStorePointer[iray]=0;
  }
  // - vector of re-generated (unstopped) primary particles
  fPrimaryRegenID = (uint32_T*) mxMalloc(fMaxPrimaryParticles*sizeof(uint32_T)) ;
  for (i=0; i< fMaxPrimaryParticles; i++)
    fPrimaryRegenID[i]=0 ;
  // - Secondaries info
  fSecondariesCounter=0 ;
  //
  mxArray* pMaxSecondaries = GetExtProcessData(blele,"MaxSecondaryParticles") ;
  if ( pMaxSecondaries == NULL)
    fMaxSecondaryParticles = 0 ;
  else
    fMaxSecondaryParticles = *(uint32_T*)mxGetData( pMaxSecondaries ) ;
  //
  mxArray* pMaxSecondariesPerPrimary = GetExtProcessData(blele,"MaxSecondaryParticlesPerPrimary") ;
  if ( pMaxSecondariesPerPrimary == NULL)
    fMaxSecondaryParticlesPerPrimary = 0 ;
  else
    fMaxSecondaryParticlesPerPrimary = *(uint32_T*)mxGetData( pMaxSecondariesPerPrimary ) ;
  //
  fSecondariesPerThisPrimary = (int*) mxMalloc(fMaxPrimaryParticles*sizeof(int)) ;
  // - Control of when to store secondary particles (0= store all, 1= store only those which make it to d/s face of element
   mxArray* pSecondaryStorageCuts = GetExtProcessData(blele,"SecondaryStorageCuts") ;
  if ( pSecondaryStorageCuts == NULL)
    fSecondaryStorageCuts = 1 ;
  else
    fSecondaryStorageCuts = *(uint8_T*)mxGetData( pSecondaryStorageCuts ) ;
  // - Create secondary return bunch structure
  if (fMaxSecondaryParticles>0) {
    fSecondaryBunch_x = (double*) mxMalloc(sizeof(double)*fMaxSecondaryParticles*6) ;
    fSecondaryPrimaryID = (uint32_T*) mxMalloc(sizeof(uint32_T)*fMaxSecondaryParticles) ;
    fSecondaryProcType = (uint8_T*) mxMalloc(sizeof(uint8_T)*fMaxSecondaryParticles) ;
    fSecondaryTrackStatus = (uint8_T*) mxMalloc(sizeof(uint8_T)*fMaxSecondaryParticles) ;
  }
  fTypeCellPtr = mxCreateCellMatrix(1,fMaxSecondaryParticles) ;
  // - Secondary ray time vectors (time since start of GEANT4 event)
  fSecondaryRayTime = (double*) mxMalloc(fMaxSecondaryParticles*sizeof(double)) ;
  // Tunnel material and geometry
  mxArray* pTunnelMaterial = GetExtProcessData(blele,"TunnelMaterial") ;
  buflen = mxGetN(pTunnelMaterial)*sizeof(mxChar)+1;
  TunnelMaterial = (char*) mxMalloc(buflen);
  mxGetString(pTunnelMaterial,TunnelMaterial,buflen) ;
  TunnelGeom[0] = *mxGetPr(GetExtProcessData(blele,"TunnelWidth")) ;
  TunnelGeom[1] = *mxGetPr(GetExtProcessData(blele,"TunnelHeight")) ;
  if (TunnelGeom[0]>0 && TunnelGeom[1]>0) {
    double *tgPars ;
    int ipars ;
    tgPars = mxGetPr(GetExtProcessData(blele,"BeamTunnelGeom")) ;
    for (ipars=0;ipars<10;ipars++)
      TunnelMainGeom[ipars] = tgPars[ipars];
    tgPars = mxGetPr(GetExtProcessData(blele,"BeamTunnelPos")) ;
    TunnelMainGeom[10] = tgPars[0];
    TunnelMainGeom[11] = tgPars[1];
    tgPars = mxGetPr(GetExtProcessData(blele,"ServiceTunnelGeom")) ;
    for (ipars=0;ipars<10;ipars++)
      TunnelServiceGeom[ipars] = tgPars[ipars];
    tgPars = mxGetPr(GetExtProcessData(blele,"ServiceTunnelPos")) ;
    TunnelServiceGeom[10] = tgPars[0];
    TunnelServiceGeom[11] = tgPars[1];
  }
  // Beam pipe location in tunnel reference frame - adjust for user defined element offset
  double* pBeamlineCoord = mxGetPr(GetExtProcessData(blele,"beamlineCoord")) ;
  memcpy(beamlineCoord,pBeamlineCoord,sizeof(double)*4);
  if (ElemOffset != NULL) {
    beamlineCoord[0] += ElemOffset[0];
    beamlineCoord[1] += ElemOffset[1];
    beamlineCoord[2] += ElemOffset[2];
    beamlineCoord[3] += ElemOffset[3];
  }
}

void lucretiaManager::ApplyRunCuts(G4UImanager* UI)
{
  // Apply Process cuts by particle type
  string inactiveString = "/process/inactivate ";
  string activeString = "/process/activate ";
  string cutStringGamma = "/run/setCutForAGivenParticle gamma ";
  string cutStringElec = "/run/setCutForAGivenParticle e- ";
  string cutStringPosi = "/run/setCutForAGivenParticle e+ ";
  int ipart,iproc ;
  string buffer, gamCut, elecCut, posiCut ;
  ostringstream gamConvert,elecConvert,posiConvert;
  
  gamConvert << fParticleCuts[0];
  UI->ApplyCommand(cutStringGamma + gamConvert.str() + " mm") ; 
  elecConvert << fParticleCuts[1];
  UI->ApplyCommand(cutStringElec + elecConvert.str() + " mm") ;
  posiConvert << fParticleCuts[2];
  UI->ApplyCommand(cutStringPosi + posiConvert.str() + " mm") ;
  for (ipart=0; ipart<fNPart; ipart++) {
    for (iproc=0; iproc<fNProc; iproc++) {
      if (fProcessSelection[ipart+iproc*fNPart])
	      UI->ApplyCommand(activeString + fProcList[iproc] + " " + fPartList[ipart]) ; 
      else
	      UI->ApplyCommand(inactiveString + fProcList[iproc] + " " + fPartList[ipart]) ;
    }
  }
}

void lucretiaManager::GetUniformField(double uField[3])
{
  if (EMisUniform==1) {
    uField[0]=*mxGetPr(pBx);
    uField[1]=*mxGetPr(pBy);
    uField[2]=*mxGetPr(pBz);
  }
  else if (EMisUniform==2) {
    uField[0]=*mxGetPr(pEx);
    uField[1]=*mxGetPr(pEy);
    uField[2]=*mxGetPr(pEz);
  }
}

double lucretiaManager::interpField(const int fieldno, const double* point)
{
  const mxArray *F = NULL;
  char method = 0;
  double* magB; // Lucretia B & len fields of element
  double* magL;
  double BMag=0; // Bx or By field of Lucretia magnet
  char* ElemClass ;
  mwSize M,N,O ;
  
  // Which field are we looking up?
  switch(fieldno) {
    case 0:
      F = pBx;
      break;
    case 1:
      F = pBy;
      break;
    case 2:
      F = pBz;
      break;
    case 3:
      F = pEx;
      break;
    case 4:
      F = pEy;
      break;
    case 5:
      F = pEz;
      break;
  }
  
  // Get ray pointers
  double xray=point[0];
  double yray=point[1];
  
  // If Quadrupole or Sextupole or Bend and inside aperture, evaluate magnet field for Bx/By if not given another field map
  if (fieldno<2) {
    ElemClass = GetElemClass( *fEle ) ;
    if (mxGetNumberOfElements(F)<2 && (strcmp(ElemClass,"QUAD")==0 || strcmp(ElemClass,"SEXT")==0 || strcmp(ElemClass,"SBEN")==0) ) {
      magB=GetElemNumericPar(*fEle, "B", NULL) ;
      magL=GetElemNumericPar(*fEle,"L", NULL) ;
      if (magB!=NULL && magL!=NULL && fabs(xray)<AperX*1e3 && fabs(yray)<AperY*1e3) {
        if (fieldno==0) { // Bx
          if (strcmp(ElemClass,"QUAD")==0)
            BMag = -(*magB / *magL) * xray * 1e-3 ;
          else if (strcmp(ElemClass,"SEXT")==0)
            BMag = -(*magB / *magL) * xray * yray *1e-6; 
          else if (strcmp(ElemClass,"SBEN")==0)
            BMag = 0 ;
        }
        else if (fieldno==1) { // By
          if (strcmp(ElemClass,"QUAD")==0)
            BMag = -(*magB / *magL) * xray*1e-3 ;
          else if (strcmp(ElemClass,"SEXT")==0)
            BMag = -0.5 * (*magB / *magL) * (xray*xray*1e-6-yray*yray*1e-6) ;
          else if (strcmp(ElemClass,"SBEN")==0)
            BMag = -(magB[0] / *magL) ;
        }
      }
    }
  }
  
  // If just single field value then return that
  if (mxGetNumberOfElements(F)<2) {
    if (fieldno<3)
      return *mxGetPr(F)+BMag;
    else
      return *mxGetPr(F);
  }
  
  // Interpolation method (1,2,3 == nearest, linear, cubic)
  if (!strcmp(EMInterpMethod,"nearest"))
    method=0;
  else if (!strcmp(EMInterpMethod,"linear"))
    method=1;
  else if (!strcmp(EMInterpMethod,"cubic"))
    method=2;
  
  const mwSize *F_dims = mxGetDimensions(F);
  
  M=F_dims[0];
  N=F_dims[1];
  if (mxGetNumberOfDimensions(F)<3) // If only 2 dims, then z dim is length 1
    O=1;
  else
    O=F_dims[2];
  
  const double *pF = mxGetPr(F);
  const double *pX = &xray;
  const double *pY = &yray;
  const double *pZ = &point[2];
  double pO ;
  
  // Interpolation is based on x/y/z co-ordinates internal to G4 volume (X/Y/Z given in mm)
  double x_low, x_high, y_low, y_high, z_low, z_high, s_x, s_y, s_z, o_x, o_y, o_z ;
  if (FieldBox[0]==0 && FieldBox[1]==0) {
    x_low = -(AperX+Thickness)*1e3; x_high = (AperX+Thickness)*1e3;
    y_low = -(AperY+Thickness)*1e3; y_high = (AperY+Thickness)*1e3;
  }
  else {
    x_low = FieldBox[0]*1e3; x_high = FieldBox[1]*1e3;
    y_low = FieldBox[2]*1e3; y_high = FieldBox[3]*1e3;
  }
  z_low = -Lcut*1e3; z_high = Lcut*1e3;
  
  if (*pX<x_low || *pX>x_high || *pY<y_low || *pY>y_high || *pZ<z_low || *pZ>z_high)
   return 0;

  // If out of FieldBox defined range, return 0
  if (*pX<x_low || *pX>x_high || *pY<y_low || *pY>y_high)
    return 0 ;
  
  s_x = (double(1)-double(M))/(x_low - x_high);
  s_y = (double(1)-double(N))/(y_low - y_high);
  s_z = (double(1)-double(O))/(z_low - z_high);
  
  o_x = double(1)-x_low*s_x;
  o_y = double(1)-y_low*s_y;
  o_z = double(1)-z_low*s_z;
  
  // Do the interpolation
  switch(method) {
    case 0:
      interpolate_nearest(&pO, pF, pX, pY, pZ, 1, M, N, O, 1, s_x, o_x, s_y, o_y, s_z, o_z);
      break;
    case 1:
      interpolate_linear(&pO, pF, pX, pY, pZ, 1, M, N, O, 1, s_x, o_x, s_y, o_y, s_z, o_z);
      break;
    case 2:
      interpolate_bicubic(&pO, pF, pX, pY, pZ, 1, M, N, O, 1, s_x, o_x, s_y, o_y, s_z, o_z);
      break;
    default:
      mexErrMsgTxt("Unimplemented interpolation method.");
  }
  
  // Return the interpolated field value
  if (fieldno<3)
    return BMag+pO ;
  else
    return pO ;
}

/* ======== Interpolation Routines ========= */

int lucretiaManager::access(int M, int N, int O, int x, int y, int z) {
  if (x<0) x=0; else if (x>=N) x=N-1;
  if (y<0) y=0; else if (y>=M) y=M-1;
  if (z<0) z=0; else if (z>=O) z=O-1;
  return x+(y+z*N)*M ;
}

int lucretiaManager::access_unchecked(int M, int N, int x, int y, int z) {
  return x+(y+z*N)*M ;
}

void lucretiaManager::indices_linear(
        int &f000_i,
        int &f100_i,
        int &f010_i,
        int &f110_i,
        int &f001_i,
        int &f101_i,
        int &f011_i,
        int &f111_i,
        const int x, const int y, const int z,
        const mwSize &M, const mwSize &N, const mwSize &O) {
  if (x<=1 || y<=1 || z<=1 || x>=N-2 || y>=M-2 || z>=O-2) {
    f000_i = access(M,N,O, x,   y  , z);
    f100_i = access(M,N,O, x+1, y  , z);
    
    f010_i = access(M,N,O, x,   y+1, z);
    f110_i = access(M,N,O, x+1, y+1, z);
    
    f001_i = access(M,N,O, x,   y  , z+1);
    f101_i = access(M,N,O, x+1, y  , z+1);
    
    f011_i = access(M,N,O, x,   y+1, z+1);
    f111_i = access(M,N,O, x+1, y+1, z+1);
  } else {
    f000_i = access_unchecked(M,N, x,   y  , z);
    f100_i = access_unchecked(M,N, x+1, y  , z);
    
    f010_i = access_unchecked(M,N, x,   y+1, z);
    f110_i = access_unchecked(M,N, x+1, y+1, z);
    
    f001_i = access_unchecked(M,N, x,   y  , z+1);
    f101_i = access_unchecked(M,N, x+1, y  , z+1);
    
    f011_i = access_unchecked(M,N, x,   y+1, z+1);
    f111_i = access_unchecked(M,N, x+1, y+1, z+1);
  }
}

void lucretiaManager::indices_cubic(
        int f_i[64],
        const int x, const int y, const int z,
        const mwSize &M, const mwSize &N, const mwSize &O) {
  if (x<=2 || y<=2 || z<=2 || x>=N-3 || y>=M-3 || z>=O-3) {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          f_i[i+4*(j+4*k)] = access(M,N,O, x+i-1, y+j-1, z+k-1);
  } else {
    for (int i=0; i<4; ++i)
      for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
          f_i[i+4*(j+4*k)] = access_unchecked(M,N, x+i-1, y+j-1, z+k-1);
  }
}

void lucretiaManager::interpolate_nearest(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) {
  const mwSize LO = M*N*O;
  for (mwSize i=0; i<ND; ++i) {
    const double &x = pX[i];
    const double &y = pY[i];
    const double &z = pZ[i];
    
    const int x_round = int(round(s_x*x+o_x))-1;
    const int y_round = int(round(s_y*y+o_y))-1;
    const int z_round = int(round(s_z*z+o_z))-1;
    
    const int f00_i = access(M,N,O, x_round,y_round,z_round);
    for (mwSize j=0; j<P; ++j) {
      pO[i + j*ND] = pF[f00_i + j*LO];
    }
  }
}

void lucretiaManager::interpolate_linear(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z) {
  const mwSize LO = M*N*O;
  for (mwSize i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];
    
    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;
    
    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);
    
    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;
    
    const double wx0 = 1.0-dx;
    const double wx1 = dx;
    
    const double wy0 = 1.0-dy;
    const double wy1 = dy;
    
    const double wz0 = 1.0-dz;
    const double wz1 = dz;
    
    int f000_i, f100_i, f010_i, f110_i;
    int f001_i, f101_i, f011_i, f111_i;
    
    indices_linear(
            f000_i, f100_i, f010_i, f110_i,
            f001_i, f101_i, f011_i, f111_i,
            int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);
    
    for (mwSize j=0; j<P; ++j) {
      
      pO[i + j*ND] =
              wz0*(
              wy0*(wx0 * pF[f000_i + j*LO] + wx1 * pF[f100_i + j*LO]) +
              wy1*(wx0 * pF[f010_i + j*LO] + wx1 * pF[f110_i + j*LO])
              )+
              wz1*(
              wy0*(wx0 * pF[f001_i + j*LO] + wx1 * pF[f101_i + j*LO]) +
              wy1*(wx0 * pF[f011_i + j*LO] + wx1 * pF[f111_i + j*LO])
              );
    }
    
  }
}

void lucretiaManager::interpolate_bicubic(double *pO, const double *pF,
        const double *pX, const double *pY, const double *pZ,
        const mwSize ND, const mwSize M, const mwSize N, const mwSize O, const mwSize P,
        const double s_x, const double o_x,
        const double s_y, const double o_y,
        const double s_z, const double o_z)
{
  const mwSize LO = M*N*O;
  for (mwSize i=0; i<ND; ++i) {
    const double &x_ = pX[i];
    const double &y_ = pY[i];
    const double &z_ = pZ[i];
    
    const double x = s_x*x_+o_x;
    const double y = s_y*y_+o_y;
    const double z = s_z*z_+o_z;
    
    const double x_floor = floor(x);
    const double y_floor = floor(y);
    const double z_floor = floor(z);
    
    const double dx = x-x_floor;
    const double dy = y-y_floor;
    const double dz = z-z_floor;
    
    const double dxx = dx*dx;
    const double dxxx = dxx*dx;
    
    const double dyy = dy*dy;
    const double dyyy = dyy*dy;
    
    const double dzz = dz*dz;
    const double dzzz = dzz*dz;
    
    const double wx0 = 0.5 * (    - dx + 2.0*dxx -       dxxx);
    const double wx1 = 0.5 * (2.0      - 5.0*dxx + 3.0 * dxxx);
    const double wx2 = 0.5 * (      dx + 4.0*dxx - 3.0 * dxxx);
    const double wx3 = 0.5 * (         -     dxx +       dxxx);
    
    const double wy0 = 0.5 * (    - dy + 2.0*dyy -       dyyy);
    const double wy1 = 0.5 * (2.0      - 5.0*dyy + 3.0 * dyyy);
    const double wy2 = 0.5 * (      dy + 4.0*dyy - 3.0 * dyyy);
    const double wy3 = 0.5 * (         -     dyy +       dyyy);
    
    const double wz0 = 0.5 * (    - dz + 2.0*dzz -       dzzz);
    const double wz1 = 0.5 * (2.0      - 5.0*dzz + 3.0 * dzzz);
    const double wz2 = 0.5 * (      dz + 4.0*dzz - 3.0 * dzzz);
    const double wz3 = 0.5 * (         -     dzz +       dzzz);
    
    int f_i[64];
    
    indices_cubic(
            f_i,
            int(x_floor-1), int(y_floor-1), int(z_floor-1), M, N, O);
    
    for (mwSize j=0; j<P; ++j) {
      
      pO[i + j*ND] =
              wz0*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*0)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*0)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*0)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*0)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*0)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*0)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*0)] + j*LO])
              ) +
              wz1*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*1)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*1)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*1)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*1)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*1)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*1)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*1)] + j*LO])
              ) +
              wz2*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*2)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*2)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*2)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*2)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*2)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*2)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*2)] + j*LO])
              ) +
              wz3*(
              wy0*(wx0 * pF[f_i[0+4*(0+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(0+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(0+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(0+4*3)] + j*LO]) +
              wy1*(wx0 * pF[f_i[0+4*(1+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(1+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(1+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(1+4*3)] + j*LO]) +
              wy2*(wx0 * pF[f_i[0+4*(2+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(2+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(2+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(2+4*3)] + j*LO]) +
              wy3*(wx0 * pF[f_i[0+4*(3+4*3)] + j*LO] + wx1 * pF[f_i[1+4*(3+4*3)] + j*LO] +  wx2 * pF[f_i[2+4*(3+4*3)] + j*LO] + wx3 * pF[f_i[3+4*(3+4*3)] + j*LO])
              );
    }
    
  }
}

int lucretiaManager::GetNextX()
{
  // Return pointer to next 6D ray co-ordinates that has a stopped flag at this element #
  uint32_T iray;
  //printf("Getting Ray... %d\n",fRayGetPtr);
  int i;
  if (fRayGetPtr >= fBunch->nray || fRayCount >= fMaxPrimaryParticles )
    return -1 ;
  for (i=fRayGetPtr; i<fBunch->nray; i++) {
    if (fPrimOrder!=NULL)
      iray=fPrimOrder[i]-1 ;
    else
      iray=i ;
    //printf("iray: %d\n",iray);
    if ( fBunch->stop[iray] == *fEle+1 ) {
      fRayGetPtr = i+1 ;
      // return if ray doesn't pass energy cut
      fPrimIndex[fRayCount]=iray;
      fSecondariesPerThisPrimary[fRayCount]=0; // reset secondaries count
      fRayCount++;
      //printf("GetX: ele= %d iray= %d raycount= %d fPrimIndex= %d\n",*fEle,iray,fRayCount,fPrimIndex[fRayCount-1]);
      return iray ;
    }
  }
  return -1 ;
}

void lucretiaManager::SetLucretiaData()
{
  // Total energy deposited in this element
  mxSetProperty( GetExtProcessPtr(fEle), 0, "Edep",  mxCreateDoubleScalar(SumEdep) ) ;
  // Send object data back into Lucretia data structures
  if (fNumRaysResumed>0) {
    mxArray* pPrimaryRegenID = mxCreateNumericMatrix(1, fNumRaysResumed, mxUINT32_CLASS, mxREAL) ;
    uint32_T* dPrimaryRegenID = (uint32_T*) mxMalloc( sizeof(uint32_T)*fNumRaysResumed ) ;
    memcpy(dPrimaryRegenID, fPrimaryRegenID, sizeof(uint32_T)*fNumRaysResumed );
    mxSetData( pPrimaryRegenID, dPrimaryRegenID) ;
    mxSetProperty( GetExtProcessPrimariesData(fEle), *fBunchNo, "regeneratedID", pPrimaryRegenID ) ;
  }
  // Send ray time information back to Lucretia data structures
  mxArray* pPrimaryRayTime = mxCreateDoubleMatrix(fBunch->nray,1,mxREAL) ;
  double* dPrimaryRayTime = (double*) mxCalloc(fBunch->nray, sizeof(double)) ;
  memcpy(dPrimaryRayTime, fPrimaryRayTime, sizeof(double)*fBunch->nray) ;
  mxSetPr( pPrimaryRayTime, dPrimaryRayTime) ;
  mxSetProperty( GetExtProcessPrimariesData(fEle), *fBunchNo, "PrimaryTrackTime", pPrimaryRayTime ) ;
  //
  mxArray* pSecondaryRayTime = mxCreateDoubleMatrix(fSecondariesCounter,1,mxREAL) ;
  double* dSecondaryRayTime = (double*) mxCalloc(fSecondariesCounter, sizeof(double)) ;
  memcpy(dSecondaryRayTime, fSecondaryRayTime, sizeof(double)*fSecondariesCounter) ;
  mxSetPr( pSecondaryRayTime, dSecondaryRayTime) ;
  mxSetProperty( GetExtProcessSecondariesData(fEle), *fBunchNo, "SecondaryTrackTime", pSecondaryRayTime ) ;
  // Stored mesh data
  if (fMeshEntries>0) {
    mxArray* pMeshDataTemp ;
    double* dMeshDataTemp ;
    mxArray* pCellTemp = mxCreateCellMatrix(1,(mwSize)fMeshEntries) ;
    double* mdata;
    mwSize dims[3];
    for (int imesh=0; imesh<fMeshEntries; imesh++) {
      dims[0]=fMeshN1[imesh]; dims[1]=fMeshN2[imesh]; dims[2]=fMeshN3[imesh];
      pMeshDataTemp = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL) ;
      dMeshDataTemp = (double*) mxCalloc(fMeshN1[imesh]*fMeshN2[imesh]*fMeshN3[imesh],sizeof(double)) ;
      mdata = fMeshData[imesh];
      memcpy(dMeshDataTemp, mdata, sizeof(double)*fMeshN1[imesh]*fMeshN2[imesh]*fMeshN3[imesh]);
      mxSetData(pMeshDataTemp, dMeshDataTemp);
      mxSetCell(pCellTemp, imesh, pMeshDataTemp ) ;
    }
    //mxFree(mxGetData(mxGetProperty( GetExtProcessPtr(fEle), 0, "meshData")));
    mxSetProperty( GetExtProcessPtr(fEle), 0, "meshData", pCellTemp) ;
  }
  // Stored tracking data
  if (fMaxTrackStore>0) {
    unsigned int iMax=0;
    iMax=fTrackStoreCounter;
    if (iMax>0) {
      mxArray* pTrackStore_x = mxCreateDoubleMatrix(iMax,1,mxREAL) ;
      mxArray* pTrackStore_y = mxCreateDoubleMatrix(iMax,1,mxREAL) ;
      mxArray* pTrackStore_z = mxCreateDoubleMatrix(iMax,1,mxREAL) ;
      mxArray* pTrackStorePointer = mxCreateNumericMatrix(fBunch->nray, 1, mxUINT32_CLASS, mxREAL) ;
      double* dTrackStore_x = (double*) mxCalloc(iMax,sizeof(double)) ;
      double* dTrackStore_y = (double*) mxCalloc(iMax,sizeof(double)) ;
      double* dTrackStore_z = (double*) mxCalloc(iMax,sizeof(double)) ;
      uint32_T* dTrackStorePointer = (uint32_T*) mxMalloc( sizeof(uint32_T)*fBunch->nray) ;
      memcpy(dTrackStore_x, fTrackStoreData_x, sizeof(double)*iMax) ;
      memcpy(dTrackStore_y, fTrackStoreData_y, sizeof(double)*iMax) ;
      memcpy(dTrackStore_z, fTrackStoreData_z, sizeof(double)*iMax) ;
      memcpy(dTrackStorePointer, fTrackStorePointer, sizeof(uint32_T)*fBunch->nray) ;
      mxSetPr( pTrackStore_x, dTrackStore_x ) ;
      mxSetPr( pTrackStore_y, dTrackStore_y ) ;
      mxSetPr( pTrackStore_z, dTrackStore_z ) ;
      mxSetData( pTrackStorePointer, dTrackStorePointer ) ;
      mxSetProperty( GetExtProcessPrimariesData(fEle), *fBunchNo, "TrackingData_x", pTrackStore_x ) ;
      mxSetProperty( GetExtProcessPrimariesData(fEle), *fBunchNo, "TrackingData_y", pTrackStore_y ) ;
      mxSetProperty( GetExtProcessPrimariesData(fEle), *fBunchNo, "TrackingData_z", pTrackStore_z ) ;
      mxSetProperty( GetExtProcessPrimariesData(fEle), *fBunchNo, "TrackingDataPointer", pTrackStorePointer ) ;
    }
  }
  //
  if (fSecondariesCounter>0) {
    mxArray* pSecondaryBunch_x = mxCreateNumericMatrix(6, fSecondariesCounter, mxDOUBLE_CLASS, mxREAL) ;
    double* dSecondaryBunch_x = (double*) mxMalloc(fSecondariesCounter*sizeof(double)*6) ;
    memcpy( dSecondaryBunch_x, fSecondaryBunch_x, fSecondariesCounter*sizeof(double)*6) ;
    mxSetPr( pSecondaryBunch_x, dSecondaryBunch_x) ;
    mxSetProperty( GetExtProcessSecondariesData(fEle), *fBunchNo, "Pos", pSecondaryBunch_x) ;
    //
    mxArray* pSecondaryPrimaryID = mxCreateNumericMatrix(1, fSecondariesCounter, mxUINT32_CLASS, mxREAL) ;
    uint32_T* dSecondaryPrimaryID = (uint32_T*) mxMalloc(fSecondariesCounter*sizeof(uint32_T)) ;
    memcpy( dSecondaryPrimaryID, fSecondaryPrimaryID, fSecondariesCounter*sizeof(uint32_T)) ;
    mxSetData( pSecondaryPrimaryID, dSecondaryPrimaryID) ;
    mxSetProperty( GetExtProcessSecondariesData(fEle), *fBunchNo, "PrimaryID", pSecondaryPrimaryID) ;
    //
    mxArray* pSecondaryProcType = mxCreateNumericMatrix(1, fSecondariesCounter, mxUINT8_CLASS, mxREAL) ;
    uint8_T* dSecondaryProcType = (uint8_T*) mxMalloc(fSecondariesCounter*sizeof(uint8_T)) ;
    memcpy( dSecondaryProcType, fSecondaryProcType, fSecondariesCounter*sizeof(uint8_T)) ;
    mxSetData( pSecondaryProcType, dSecondaryProcType) ;
    mxSetProperty( GetExtProcessSecondariesData(fEle), *fBunchNo, "ProcType", pSecondaryProcType) ;
    //
    mxArray* pSecondaryTrackStatus = mxCreateNumericMatrix(1, fSecondariesCounter, mxUINT8_CLASS, mxREAL) ;
    uint8_T* dSecondaryTrackStatus = (uint8_T*) mxMalloc(fSecondariesCounter*sizeof(uint8_T)) ;
    memcpy( dSecondaryTrackStatus, fSecondaryTrackStatus, fSecondariesCounter*sizeof(uint8_T)) ;
    mxSetData( pSecondaryTrackStatus, dSecondaryTrackStatus) ;
    mxSetProperty( GetExtProcessSecondariesData(fEle), *fBunchNo, "TrackStatus", pSecondaryTrackStatus) ;
    //
    mxSetProperty( GetExtProcessSecondariesData(fEle), *fBunchNo, "ParticleType", fTypeCellPtr) ;
    //
    mxArray* pNumSec = mxCreateNumericMatrix(1, 1, mxUINT32_CLASS, mxREAL) ;
    uint32_T* dNumSec = (uint32_T*) mxMalloc(sizeof(uint32_T)) ;
    *dNumSec = fSecondariesCounter ;
    mxSetData( pNumSec, dNumSec );
    mxSetProperty( GetExtProcessSecondariesData(fEle), *fBunchNo, "NumStored", pNumSec) ;
  }
}

void lucretiaManager::WritePrimaryTrackData(double x, double y, double z)
{
  if (fMaxTrackStore>0 && fTrackStoreCounter<fMaxTrackStore && fabs(z)<Lcut) {
    fTrackStoreData_x[fTrackStoreCounter]=x;
    fTrackStoreData_y[fTrackStoreCounter]=y;
    fTrackStoreData_z[fTrackStoreCounter]=z;
    fTrackStoreCounter++;
  }
}

int lucretiaManager::SetNextSecondary(double x[6], int id, const char* type, G4String procName, G4TrackStatus trackStatus, double gTime)
{
  int icoord ;
  int iray=fPrimIndex[id] ;
  if (fMaxSecondaryParticles==0)
    return 1;
  // Store phase space of secondary particle and ID of primary which generated it
  if (fSecondaryBunch_x != NULL ) {
    //printf("Saving Secondary...\n");
    for (icoord=0; icoord<6; icoord++)
      fSecondaryBunch_x[fSecondariesCounter*6+icoord] = x[icoord] ;
    mxSetCell(fTypeCellPtr,fSecondariesCounter,mxCreateString(type)) ;
    fSecondaryPrimaryID[fSecondariesCounter] = iray+1 ;
    fSecondaryRayTime[fSecondariesCounter] = gTime ; // accumulated travel time / ns
  }
  // Save information on process generating secondary and the track object status ID
  fSecondaryProcType[fSecondariesCounter]=0;
  fSecondaryTrackStatus[fSecondariesCounter]=(uint8_T) trackStatus;
  for (int iproc=0; iproc<fNProc; iproc++) {
    if (fProcList[iproc].compare(procName) !=0 )
      fSecondaryProcType[fSecondariesCounter]++;
    else
      break;
  }
  fSecondaryProcType[fSecondariesCounter]++; // Matlab indexing
  fSecondariesCounter++ ;
  return 0 ;
}

void lucretiaManager::SetNextX(double x[6], int id, int doresume, double trackEnergy, double gTime)
{
  // Set next stopped particle processed here to GEANT tracked co-ordinates and reset stop (resume tracking) if requested
  int iray, icoord ;
  iray=fPrimIndex[id];
//   printf("SetNextX: id=%d iray= %d x=%g\n",id,iray,x[0]);
  for (icoord=0; icoord<6; icoord++) {
    if (icoord == 4)
      fPrimaryRayTime[iray] = gTime ; // accumulated travel time / ns
    if (icoord != 4) // Leave z co-ordinate alone
      fBunch->x[6*iray+icoord] = x[icoord] ;
  }
  if (doresume == 1) {
    fBunch->stop[iray] = 0 ;
    fPrimaryRegenID[fNumRaysResumed] = iray+1 ;
    fNumRaysResumed++;
  }
  else { // coordinate should be where particle stopped
    fBunch->x[6*iray+4] = x[4] ;
    SumEdep += trackEnergy ; // Register track energy stopped in this volume
  }
//   printf("Ele: %d Prim ID= %d resum= %d num_resum= %d\n",*fEle,iray+1,doresume,fNumRaysResumed);
  return ;
}


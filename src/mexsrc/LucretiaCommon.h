/* Definitions, prototypes, etc, related to LucretiaCommon.c. */

/* AUTH:  PT, 02-aug-2004 */
/* MOD:
			 02-aug-2007, PT:
				support for XYCOR elements.
			 08-mar-2006, PT:
			    support for coupled Twiss ("Wolski") parameter propagation
				 and solenoids.
			 09-dec-2005, PT:
			    move definition of SRFlagIndex from LucretiaDictionary
				 to here so that I can stop #include-ing LucretiaDictionary
				 in both LucretiaCommon and LucretiaPhysics (causes multiple
				 definition errors).
			 30-sep-2005, PT:
			    change TrackBunchThruLcav to TrackBunchThruRF, for support
				 of TCAV elements.
                          */

#define LUCRETIA_COMMON

/* Definitions and headers for CUDA code- need to call even if not using CUDA */
#include "LucretiaCuda.h"

/* some parameters related to whether a corrector is an XCOR, a YCOR, or an XYCOR */

#define XCOR  0
#define YCOR  1
#define XYCOR 2

/* defining upstream and downstream faces */

#define UPSTREAM 0
#define DOWNSTREAM 1

/* define frequency and timedomain integers */

#define TIMEDOMAIN 0
#define FREQDOMAIN 1
#define UNKNOWNDOMAIN 2

/* define collimator shape integers */

#define COLL_ELLIPSE   0
#define COLL_RECTANGLE 1
#define COLL_UNKNOWN   2

void AddMessage( const char*, int ) ;

/* data structure definitions: */

/* definition of a data structure for initial Twiss parameters */

struct beta0 {
	double betx, alfx, etax, etapx, nux ;
	double bety, alfy, etay, etapy, nuy ;
} ;

/* arguments for the Rmat operation */ 

struct RmatArgStruc {
	int start ;                 /* first entry */
	int end ;                   /* last entry */
	int ReturnEach ;            /* whether or not to return each matrix */
	int PropagateTwiss ;        /* whether or not to propagate twiss pars */
	struct beta0 InitialTwiss ; /* initial twiss (duh) */
	double *InitialWolski ;     /* initial values of coupled Twiss parameters */
	int nWolskiDimensions ;     /* number of planes in coupled calc (1 to 3) */
	int Version ;               /* did user want a version table? */
	int Status ;                /* are user args properly constructed? */
	int Backwards ;             /* are we doing Twiss in the reverse order ? */
} ;

/* type definition of an R-matrix */

typedef double Rmat[6][6] ;

/* structure definition for a set of R-matrices */

struct Rstruc { 
	int Nmatrix ;   /* number of Rmats */
	double* matrix ;  /* pointer to the Rmats */
} ;

/* structure definition for a collection of vectors with Twiss in them */

struct twiss {
	int nentry ;
	double *S ;
	double *E ;
	struct beta0* TwissPars ;
} ;

/* structure definition for a collection of coupled Twiss parameters */

struct Ctwiss {
	int nentry ; 
	double *S ;
	double *E ;
	double *Twiss ; 
} ;

/* generic structure for defining a numeric parameter */

struct LucretiaParameter {
	const char* name ;
	int Requirement[3] ;
	int LengthRequirement[3] ;
	int MinLength ;
	int MaxLength ;
	int Length ;
	double DefaultValue ;
	double* ValuePtr ;
} ;

/* complex number:  I know that many versions of C/C++ have a complex type
   already, but for multiplatform purposes I will use one of my own; this is
	OK since I really don't have very complicated requirements and I want it
	to work right on any platform I choose to build on, with any compiler */

struct LucretiaComplex {
	double Real ;
	double Imag ;
} ;

/* structure to represent the information on short-range wakefields after they
   have been convolved with a given bunch */

struct SRWF {
	int nbin ;                /* number of bins */
	int* binno ;              /* bin # for each ray */
	double* binQ ;            /* charge, xpos, ypos for each bin */
  unsigned short int* ptype ; /* Particle type ID for Q bins (0=e-, 1=e+) */
	double* binx ;
	double* biny ;
	double* binVx ;           /* deflecting voltages on each bin */
	double* binVy ;
	double* K ;               /* vector or matrix of wakes, by bin # */
} ;

/* structure to contain the information needed by a bunch about 
   frequency-mode LRWFs */

struct LRWFFreq {
	int nbin ;                /* number of bins */
	int* binno ;              /* bin # for each ray */
	double* binQ ;            /* charge, xpos, ypos for each bin */
	double* binx ;
	double* biny ;
	double* binVx ;           /* voltage applied to each bin (calculated */
	double* binVy ;           /* during tracking) */
/* the next 2 pointers point at the phase offset from the bin to the
   nominal t=0 bunch center */
	struct LucretiaComplex* xphase ;
	struct LucretiaComplex* yphase ;
/* the next 2 pointers cache the contribution from the current bin to
   the wakefield, referenced to t = 0 */
	struct LucretiaComplex* Wx ;
	struct LucretiaComplex* Wy ;
} ;

/* structure to contain the information needed by an element about its
   frequency-mode LRWF */

struct LRWFFreqKick {
	int LastBunch ;
	struct LucretiaComplex* xKick ;
	struct LucretiaComplex* yKick ;
} ;

/* structure to hold pointers to the Matlab data vectors for a 
   frequency-mode LRWF */

struct LRWFFreqData {
	int nModes ;
	double* Freq ;
	double* Q ;
	double* K ;
	double* Tilt ;
	double BinWidth ;
} ;

/* structure to represent one bunch */

struct Bunch {
	int nray ;                  /* number of rays in the bunch */
	int ngoodray ;              /* number unstopped rays in bunch */
	int StillTracking ;         /* are we still tracking the bunch? */
	double* x ;                 /* pointer to coords  */
#ifdef __CUDACC__  
	mxGPUArray* x_gpu ;             /* Matlab mxGPUArray */
  double* y ;                 /* second coordinate pointer in GPU memory*/
  mxGPUArray* Q_gpu ;                 /* Matlab mxGPUArray */
	mxGPUArray* stop_gpu ;              /* Matlab mxGPUArray */
  mxGPUArray* y_gpu ;              /* Matlab mxGPUArray */
  int* ngoodray_gpu ;
  unsigned short int* ptype_gpu ;
#else
        double* y ;                 /* second coordinate pointer*/
#endif
	double* Q ;                 /* pointer to charge vector */
	double* stop ;              /* pointer to stopping element, if any */
  unsigned short int* ptype ; /* Particle type ID (0=e-, 1=e+) */
	struct SRWF** ZSR ;         /* longitudinal wake information */
	struct SRWF** TSR ;         /* transverse wake information */
	struct LRWFFreq** TLRFreq ; /* LRWF_T frequency domain information */
	struct LRWFFreq** TLRErrFreq ; 
} ;

/* structure to represent the beam */

struct Beam {
	int nBunch ;                    /* number of bunches in the beam */
	double interval ;               /* bunch spacing in time (ie, seconds) */
	struct LucretiaComplex** TLRFreqDamping ; 
	                                /* how much to reduce wakefields per bunch
											     interval */
	struct LucretiaComplex** TLRFreqxPhase ;
	struct LucretiaComplex** TLRFreqyPhase ;
	                                /* how much to phase-advance wakefields
											     per bunch interval */
	struct LucretiaComplex** TLRErrFreqDamping ;
	struct LucretiaComplex** TLRErrFreqxPhase ;
	struct LucretiaComplex** TLRErrFreqyPhase ;
	struct Bunch** bunches ;        /* the bunches in the beam */
} ;

/* structure to contain data from a single BPM */

struct BPMdat {
	int indx ;                /* index into BEAMLINE array */
	int GetBeamPars ;         /* have we gotten beam parameters? */
	double S ;                /* s position */
	double Pmod ;             /* centroid momentum from model */
	double* xread ;           /* reading(s) in the xz plane */
	double* yread ;           /* reading(s) in the yz plane */
	double* sigma ;           /* sigma matrix(es)           */
	double* P ;               /* centroid momentum by bunch */
	double* z ;               /* centroid timing by bunch   */

/* the next few pointers are pieces of information that are needed during
   calculation of actual return variables, but which are not returned
	themselves in the present implementation. */

	double* Q ;               /* charge by bunch */
	double* sumpxq ;          /* sum(px*q) by bunch */
	double* sumpyq ;          /* sum(py*q) by bunch */

	int nbunchalloc ;         /* how many bunches do we have space for? */
	int nBunch ;              /* how many do we actually fill? */
} ;

/* structure to contain data from a single RF structure-s S-BPMs */

struct SBPMdat { 
	int indx ;        
	double *S ;               /* a single structure can have any # of BPMs */
	double *x ; 
	double *y ;
	double* Q ;               /* charge per bunch, not returned */
	int nbpmalloc ;           /* how many BPMs on this structure? */
	int nbpm ;
} ;

/* structure to contain data from a single instrument */

struct INSTdat {
	int indx ;
	double S ;
	double* x ;
	double* sig11 ;
	double* y ;
	double* sig33 ;
	double* sig13 ;
	double* z ;
	double* sig55 ;
	double* Q ;         /* charge per bunch, not returned */
	int nbunchalloc ;
	int nBunch ;
} ;

/* arguments for the tracking operation: */

struct TrackArgsStruc{ 
	int FirstElem ;             /* first element in BEAMLINE for tracking */
	int LastElem ;              /* last element in BEAMLINE for tracking  */
	int FirstBunch ;            /* first bunch in beam for tracking */
	int LastBunch ;             /* last bunch in beam for tracking */
	int nBunch ;                /* saves recalculating it all the time */
	int BunchwiseTracking ;     /* bunch-by-bunch or element-by-element tracking? */
        struct Beam* TheBeam ;      /* pointer to the beam data structure */
	struct BPMdat** bpmdata ;   /* pointer to array of BPM data */
	struct INSTdat** instdata ; /* pointer to array of instrument data */
	struct SBPMdat** sbpmdata ; /* pointer to array of SBPM data */
	int nBPM,nINST,nSBPM ;      /* counters for the above */
	int GetInstData ;           /* does the user want instrument data or not? */
	int GetVersion ;            /* was this a "version" call only? */
	int ClearLocal ;            /* was this a "clear" call only? */
	int Status ;                /* were the arguments well-conditioned? */
} ;

/* Translations of the synchrotron radiation flags */

enum SRFlagIndex{
		SR_None, SR_Mean, SR_AW, SR_HB
} ;



/*=================================================*/

/* Function prototypes */

/* Call the Rmat calculating engine, used to calculate a single,
   rmat-A-to-B matrix, or all of the Rij's from A to B, or the
   Twiss functions from A to B.  Since it does a bunch of things
   we need to return a typeless pointer which can be typecast in
   the calling procedure. */

void* RmatCalculate( struct RmatArgStruc* ) ; 

/* multiply two matrices, return as the third */

void RmatProduct(Rmat Rlate, Rmat Rearly, Rmat Rprod) ;

/* Copy one matrix into another */

void RmatCopy( Rmat Rsource, Rmat Rtarget ) ;

/* clear locally-allocated variables associated with tracking */

void ClearTrackingVars( ) ;

/* main procedure for the tracking loop */

void TrackThruMain( struct TrackArgsStruc* ) ;

/* tracking a bunch through various elements */
int TrackBunchThruDrift( int, int, struct TrackArgsStruc*, int*, double ) ;
int TrackBunchThruQSOS( int, int, struct TrackArgsStruc*, int*, int, double, double, int ) ;
int TrackBunchThruMult( int, int, struct TrackArgsStruc*, int*, double, double ) ;
int TrackBunchThruSBend( int, int, struct TrackArgsStruc*, int*, int, int, double, double  ) ;
int TrackBunchThruRF( int, int, struct TrackArgsStruc*, int*, int ) ;
int TrackBunchThruBPM( int, int, struct TrackArgsStruc*, int*, double ) ;
int TrackBunchThruInst( int, int, struct TrackArgsStruc*, int*, double ) ;
int TrackBunchThruCorrector( int, int, struct TrackArgsStruc*, int*, int, double, double ) ;				
int TrackBunchThruCollimator( int, int, struct TrackArgsStruc*, int*, double, double ) ;
int TrackBunchThruCoord( int, int, struct TrackArgsStruc*, int* ) ;
int TrackBunchThruTMap( int, int,  struct TrackArgsStruc*, int* ) ;
#ifdef __CUDACC__        
int ElemTracker(char* ElemClass,int* ElemLoop,int* BunchLoop,int* TFlag_gpu, int* TFlag,struct TrackArgsStruc* TrackArgs) ;
#else
int ElemTracker(char* ElemClass,int* ElemLoop,int* BunchLoop, int* TFlag, struct TrackArgsStruc* TrackArgs) ;
#endif  

/* return the tracking flags */

int* GetTrackingFlags( int ) ;

/* return complete offsets (upstream and downstream) for an element */

int GetTotalOffsetXfrms( double*, double*,
								 double*, double*,
								 double[6][2] ) ;

/* apply complete offsets (upstream and downstream) for an element */
CUDAHOSTDEVICEFUN void ApplyTotalXfrm( double[6][2], int, int*, double, double *x, double *px, double *y, double *py, double *z, double *p0 ) ;

/* check whether a particle needs to stop on aperture */
CUDAHOSTDEVICEFUN int CheckAperStopPart( double *x, double *y, double *stop, int *ngoodray, int, double*, int, int,
					   int*, double, int*, int ) ;

/* check whether a particle needs to stop for P0 <= 0 */
CUDAHOSTDEVICEFUN int CheckP0StopPart( double *stop, int *ngoodray, double *x, double *y, int, int, double, int, int* ) ;

/* check whether a particle needs to stop for |Pperp| >= 1 */
CUDAHOSTDEVICEFUN int CheckPperpStopPart( double*, int*, int, int, double*, double*, int* ) ;

/* initialization for BPM/INST operations */

void BPMInstIndexingSetup( int , int*, int*, int* ) ;

/* how many bunches needed on this BPM/inst, and which one are we on ? */

void BunchSlotSetup( int*, struct TrackArgsStruc*, int, int*, int*) ;

/* free memory and nullify the pointer */

void FreeAndNull( void** ) ;

/* Error message on ill-defined element */

void BadElementMessage( int ) ;

/* Error message on ill-defined power supply */

void BadPSMessage( int, int ) ;

/* Error message on ill-defined klystron */

void BadKlystronMessage( int, int ) ;

/* Error message on bad Twiss propagation */

void BadTwissMessage( int ) ;

/* Error message on failed attempt to do backwards Twiss propagation */

void BadInverseTwissMessage( int , int ) ;

/* additional error messages */

void BadTrackFlagMessage( int ) ;
void BadApertureMessage( int ) ;
void BadSROptionsMessage( int ) ; 
void BadParticleMomentumMessage( int, int, int ) ;
void BadParticlePperpMessage( int, int, int ) ;
void BadInitMomentumMessage( )  ;
void BadOffsetMessage( int ) ;
void BadBPMAllocMsg( int ) ;
void BadSliceAllocMessage( int ) ;
void BadSRWFAllocMsg( int, int ) ;
void BadSRWFMessage( int, int ) ;
void NonExistentLRWFmessage( int, int, int ) ;
void BadLRWFAllocMsg( int, int ) ;
void BadLRWFBunchOrder( int, int ) ;
void BunchStopMessage( int, int ) ;

/* Setting up the slicing of an LCAV */

int LcavSliceSetup( struct TrackArgsStruc*, int, int*, int*, int*, int*, 
					 int**, double**, int*) ;

/* perform SBPM initialization and setup */

int SBPMSetup( struct TrackArgsStruc*, int, int, int, int* ) ;

/* Set SBPM s positions */

void SBPMSetS( int, double, double, int, int*, double* ) ;

/* finish assembling readings on S-BPMs */

int ComputeSBPMReadings( int , int, double ) ;

/* deallocate one or more convolved SRWFs */

void ClearConvolvedSRWF( struct Bunch* , int , int ) ;

/* deallocate one or more bunch binnings for LRWFs */

void ClearBinnedLRWFFreq( struct Bunch*, int, int ) ;

/* look up an element's parameters based on its dictionary */

int GetDatabaseParameters( int, int, struct LucretiaParameter[], int, int ) ;

/* handle bend magnet parameters of variable length */

double GetSpecialSBendPar(struct LucretiaParameter*, int) ;

/* get the design Lorentz delay for an element, handling zero or null cases */

double GetDesignLorentzDelay( double* ) ;

/* exchange x and y coord vectors in a bunch */
CUDAHOSTFUN void XYExchange( struct Bunch* ) ;
CUDAHOSTFUN void postEleTrack( struct Beam*, int*, int*, double, double, int* ) ;

/* perform initial check of total and transverse momenta */
CUDAGLOBALFUN void InitialMomentumCheck( int*, double*, int*, double*, double*, int, int* ) ;

/* point local coords at desired entry in a data vector */
CUDAHOSTDEVICEFUN void GetLocalCoordPtrs( double[], int, double**, double**, double**, double**, double**, double** ) ;

/* do additional consistency checking of girder mover parameters */

int CheckGirderMoverPars( struct LucretiaParameter[7], int ) ;

/* Derefernce a double pointer, or return zero if pointer is null */

double GetDBValue( struct LucretiaParameter* ) ;

/* perform lattice verification */

void VerifyLattice( ) ;

/* perform parameter verification */

int VerifyParameters( int, int, struct LucretiaParameter[], 
							 int, int[], int[] ) ; 

/* look up an element's index number in a vector of element numbers */

int ElemIndexLookup( int, double*, int ) ;

/* get pointers to real data of a long range wakefield */

struct LRWFFreqData* UnpackLRWFFreqData( int, int, 
													  int* , int*  ) ;

/* compute bunch-to-bunch damping factor of the modes of a LRWF in the
   frequency domain */

struct LucretiaComplex* ComputeLRWFFreqDamping( double,
				double*, double*, int ) ;

/* compute complex phase factors needed to propagate LRWF modes from
   one time to another */

struct LucretiaComplex* ComputeLRWFFreqPhase( double,
				double*, int, int ) ;


/* do all preparations needed to allow an element tracker to access
   wakefield data */

int PrepareAllWF( double*, int, int*, int, int, 
					   struct Beam*, 
						int*, int*, int*, int*,
						            int*, int*,
						struct SRWF**, struct SRWF**, 
						struct LRWFFreq**,
						struct LRWFFreq**,
						struct LRWFFreqKick**,
						struct LRWFFreqKick**          ) ;

/* check out some memory for holding frequency-domain LRWF kicks */

struct LRWFFreqKick* GetThisElemTLRFreqKick( int, int ) ;

/* check the memory for the freq-domain LRWF kicks back in */

void PutThisElemTLRFreqKick( struct LRWFFreqKick**, int, 
									  int, int, int, int ) ;

/* calculate TSR kicks */

void ComputeTSRKicks( struct SRWF*, double ) ;

/* accumulate ray in WF bin */

void AccumulateWFBinPositions( double*, double*, int, 
										 double, double, double ) ;

/* clear out obsolete frequency-domain kick information */

void ClearOldLRWFFreqKicks( int ) ;

/* Deal with CSR and split track flags */
double GetCsrTrackFlags( int, int*, int*, char*, struct Bunch*, double*, double*, double ) ;

/* External process function calls (GEANT4) */
#ifdef LUCRETIA_G4TRACK
void ExtProcess(int*, struct Beam*, int*, double, double*, int*) ;
#endif

/* GEANT4 stuff */
#ifdef LUCRETIA_G4TRACK
#include "g4track/g4track.hh"
#endif

/* =================================================== */
/* GPU specific functions and tracking kernels         */
/* =================================================== */

/* CUDA-related kernels */
#ifdef __CUDACC__
__global__ void rngSetup_kernel(curandState *state, unsigned long long rSeed) ;
#endif

/* Tracking kernels */
#ifdef __CUDACC__
__global__ void TrackBunchThruDrift_kernel(double* Lfull, double* dZmod, double* yb, double* stop, int* TrackFlag, int N) ;
__global__ void TrackBunchThruQSOS_kernel(int nray, double* stop, double* yb, double* xb, int* TrackFlag, int* ngoodray,
        int elemno, double aper2, int nPoleFlag, double B, double L, double Tilt, int skew, double* pXfrms, double dZmod,
					  int* stp, double* PascalMatrix, double* Bang, double* MaxMultInd, unsigned long long rSeed, curandState *rState,
        unsigned short int* ptype) ;
__global__ void TrackBunchThruMult_kernel(double* MultAngleValue, double* MultBValue, double* MultTiltValue, double* MultPoleIndexValue,
        int MultPoleIndexLength, int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray,
        int elemno, double aper2, double L, double dB, double Tilt, double Lrad, double* pXfrms, double dZmod, double splitScale, int* stp,
        double* PascalMatrix, double* Bang, double* MaxMultInd, unsigned long long rSeed, curandState *rState,
        unsigned short int* ptype) ;
__global__ void TrackBunchThruSBend_kernel(int nray, double* xb, double* yb, double* stop, int* TrackFlag, double* pXfrms, double cTT,
        double sTT, double Tx, double Ty, double OffsetFromTiltError, double AngleFromTiltError, int* ngoodray, double hgap2,
        double intB, double intG, double L, int elemno, double E1, double H1, double hgap, double fint, double Theta, double E2,
					   double H2, double hgapx, double fintx, double hgapx2, int* stp, unsigned long long rSeed, curandState *rState,
        unsigned short int* ptype) ;
__global__ void TrackBunchThruBPM_kernel(int nray, float* pXfrms, int* TrackFlags, double* qb, double* yb, double* stop, float sintilt, float costilt,
        float* xread_ret, float* yread_ret, float* Q_ret, float* P_ret, float* pxq_ret, float* pyq_ret, float* z_ret, float* sigma_ret ) ;
__global__ void TrackBunchThruTMap_kernel(double *L, double* dZmod, double* xb, double* yb, double* stop, int* TrackFlag, int N,
        double* Offset, double* R, unsigned short T_size, unsigned short U_size, unsigned short V_size, unsigned short W_size,
        double* T, double* U, double* V, double* W, unsigned long* T_inds, unsigned long* U_inds, unsigned long* V_inds,
        unsigned long* W_inds, double Pmod) ;
__global__ void TrackBunchThruInst_kernel(int nray, float* pXfrms, int* TrackFlags, double* qb, double* yb, double* stop, float sintilt, float costilt,
        float* xread_ret, float* yread_ret, float* Q_ret, float* P_ret, float* sumpxq_ret,
        float* sumpyq_ret, float* z_ret, float* sigma_ret ) ;
__global__ void TrackBunchThruCorrector_kernel(int nray, double B, double B2, unsigned short int* ptype, double* stop, double Tilt, int* TrackFlag,
        double fabsB, double* xb, double* yb, double Lrad, int* ngoodray, int elemno, int* stp, double Lov2, curandState *rState) ;
__global__ void TrackBunchThruCollimator_kernel(int nray, int isupstream, double* xb, double* yb, double* stop, int* TrackFlag, int* ngoodray, int elemno,
        double aper2_0, double aper2_1, int shape, double Tilt, int* stp, double* pXfrms) ;
__global__ void TrackBunchThruCoord_kernel(int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray, int* stp, int* Stop, double* R_flat,
        double* dx, int elemno) ;
#else
void TrackBunchThruDrift_kernel(double* Lfull, double* dZmod, double* yb, double* stop, int* TrackFlag, int N) ;
void TrackBunchThruQSOS_kernel(int nray, double* stop, double* yb, double* xb, int* TrackFlag, int* ngoodray, int elemno,
        double aper2, int nPoleFlag, double B, double L, double Tilt, int skew, double Xfrms[6][2], double dZmod, int* stp,
        unsigned short int* ptype) ;
void TrackBunchThruMult_kernel(double* MultAngleValue, double* MultBValue, double* MultTiltValue, double* MultPoleIndexValue,
        int MultPoleIndexLength, int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray,
        int elemno, double aper2, double L, double dB, double Tilt, double Lrad, double Xfrms[6][2], double dZmod, double splitScale, int* stp,
        unsigned short int* ptype) ;
void TrackBunchThruSBend_kernel(int nray, double* xb, double* yb, double* stop, int* TrackFlag, double Xfrms[6][2], double cTT,
        double sTT, double Tx, double Ty, double OffsetFromTiltError, double AngleFromTiltError, int* ngoodray, double hgap2,
        double intB, double intG, double L, int elemno, double E1, double H1, double hgap, double fint, double Theta, double E2,
        double H2, double hgapx, double fintx, double hgapx2, int* stp, unsigned short int* ptype) ;
void TrackBunchThruBPM_kernel(int nray, double Xfrms[6][2], int* TrackFlags, double* qb, double* yb, double* stop, double sintilt, double costilt,
        double* xread_cnt, double* yread_cnt, double* Q_cnt, double* P_cnt, double* pxq_cnt, double* pyq_cnt, double* z_cnt, double sigma_cnt[36] ) ;
void TrackBunchThruTMap_kernel(double *L, double* dZmod, double* xb, double* yb, double* stop, int* TrackFlag, int N,
        double* Offset, double* R, unsigned short T_size, unsigned short U_size, unsigned short V_size, unsigned short W_size,
        double* T, double* U, double* V, double* W, unsigned long* T_inds, unsigned long* U_inds, unsigned long* V_inds,
        unsigned long* W_inds, double Pmod) ;
void TrackBunchThruInst_kernel(int nray, double Xfrms[6][2], int* TrackFlags, double* qb, double* yb, double* stop, double sintilt, double costilt,
        double* xread_cnt, double* yread_cnt, double* Q_cnt, double* P_cnt, double* sumpxq_cnt, double* sumpyq_cnt,
        double* z_cnt, double sigma_cnt[4] ) ;
void TrackBunchThruCorrector_kernel(int nray, double B, double B2, unsigned short int* ptype, double* stop, double Tilt, int* TrackFlag,
        double fabsB, double* xb, double* yb, double Lrad, int* ngoodray, int elemno, int* stp, double Lov2) ;
void TrackBunchThruCollimator_kernel(int nray, int isupstream, double* xb, double* yb, double* stop, int* TrackFlag, int* ngoodray, int elemno,
        double aper2_0, double aper2_1, int shape, double Tilt, int* stp, double Xfrms[6][2]) ;
void TrackBunchThruCoord_kernel(int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray, int* stp, Rmat R, double dx[6],
        int elemno, int bunchno) ;
#endif


/* LucretiaGlobalAccess.h:
   This is where data definitions and function prototypes are put which
   relate to accessing the Lucretia global data objects (notably the
   BEAMLINE, GIRDER, KLYSTRON, PS, WAKEFIELD objects).  The global access
   routines live in LucretiaMatlab.c or LucretiaOctave.c.  However, they
   need to be accessed by procedures in LucretiaCommon.c.  Therefore the
   information is stored here and not in LucretiaCommon.h, nor in either
   LucretiaMatlab.h nor LucretiaOctave.h. */

/* AUTH:  PT, 03-aug-2004. */
/* MOD:                  
        08-mar-2006, PT:
		     add GetMatrixNormalizer prototype.
                           */

#define LUCRETIA_GLOBAL_ACCESS

#ifdef __CUDACC__
  #include "curand_kernel.h"
#endif

/* Floating point precision to use everywhere */
#ifdef LUCRETIA_DPREC
  #define FPREC double
#else
  #define FPREC float
#endif

/* Return the random number seed from the Matlab caller workspace */

void getLucretiaRandSeed( unsigned long long *rseed );
unsigned int getLucretiaRandSeedC( );

/* define an enumeration type for klystron status */

enum KlystronStatus {ON, STANDBY, TRIPPED, MAKEUP, STANDBYTRIP} ;

/* function which returns the element class ("DRIF","QUAD",etc) */

char* GetElemClass( int ) ;

/* function to get a numeric parameter from a beamline element: */

double* GetElemNumericPar( int, const char*, int* ) ;

/* function to get the index of the power supply for a magnet */

int GetPS( int ) ;

/* get a numeric parameter from a power supply */

double* GetPSNumericPar( int, const char*, int* ) ;

/* function to get the index of the girder for an element */

int GetElemGirder( int ) ;

/* Get a numeric paramter for a girder */

double* GetGirderNumericPar( int, const char*, int* ) ;

/* get the klystron number for an element */

int GetKlystron( int ) ;

/* get a klystron numeric parameter */

double* GetKlystronNumericPar( int, const char*, int* ) ;

/* get a klystron's status */

enum KlystronStatus *GetKlystronStatus( int ) ;

/* how many wakefields of each type */

int* GetNumWakes( ) ;

/* tells how many elements are defined */

int nElemInBeamline( ) ;

/* get other global data sizes */

int GetnGirder( ) ;
int GetnKlystron( ) ;
int GetnPS( ) ;


/* return the total number of track flags an element has */

int GetNumTrackFlags( int elemno ) ;

/* return the name and value of one tracking flag */

int GetTrackFlagValue( int ) ;
const char* GetTrackFlagName( int ) ;

/* Add an error message to the pile */

void AddMessage( const char*, int ) ;

/* retrieve the messages and clear the pile */

char** GetAndClearMessages( int* ) ;

/* use Matlab randn function to get a vector of Gaussian-
   distributed random numbers */
#ifdef __CUDACC__
__device__ double RanGaussVecPtr_gpu( curandState_t *rState ) ;
__host__ double* RanGaussVecPtr( int ) ;
#endif
double* RanGaussVecPtr( int ) ;
double* RanGaussVecPtrC( int ) ;

/* use Matlab rand function to get a vector of uniform-
   distributed random numbers if host, else use cudaRand library*/

#ifdef __CUDACC__
__device__ double RanFlatVecPtr_gpu( curandState_t *state) ;
__host__ double* RanFlatVecPtr( int ) ;
#else
double* RanFlatVecPtr( int ) ;
double* RanFlatVecPtrC( int ) ;
#endif

/* Use Matlab sort function to get a sortkey for the rays in  
   a bunch, along a given DOF */

double* GetRaySortkey( double*, int, int ) ;

/* Access the WF global and return pointers to the z positions, kick
   factors; return the bin width as well */

int GetSRWFParameters( int, int, double**, double**, double* ) ;

/* cubic-spline the SRWF and return the splined values at requested
   z locations */

double* SplineSRWF( double*, double*, double*, int, int, int ) ;

/* Use Matlab's pascal function to get a Pascal matrix for use in
   multipole field expansions */

#ifdef __CUDACC__
__host__ double* GetFactorial( ) ;
__host__ double* GetPascalMatrix( ) ;
__host__ double  GetMaxMultipoleIndex( ) ;
__host__ double* GetFactorial_gpu( ) ;
__host__ double* GetPascalMatrix_gpu( ) ;
__host__ double*  GetMaxMultipoleIndex_gpu( ) ;
#else
double* GetFactorial( ) ;
double  GetMaxMultipoleIndex( ) ;
double* GetPascalMatrix( ) ;
#endif

void    ClearMaxMultipoleStuff( ) ;
void    ComputeNewMultipoleStuff( double ) ;

/* Get the class of a transverse-long-range wakefield (ie, time or
   frequency domain */

int GetTLRWakeClass( int ) ;

/* Get the class of a transverse-long-range error wakefield (ie, time or
   frequency domain */

int GetTLRErrWakeClass( int ) ;

/* get a numeric parameter from a TLR */

double* GetTLRNumericPar( int, const char*, int* ) ;

/* get a numeric parameter from an error TLR */

double* GetTLRErrNumericPar( int, const char*, int* ) ;

/* get the shape parameter for a collimator */

int GetCollimatorGeometry( int ) ;

/* get log of gamma function */

#ifdef __CUDACC__
__device__ double GammaLog_gpu( double ) ;
__host__ double GammaLog( double ) ;
#else
double GammaLog( double ) ;
#endif

/* get cube root of R-matrix determinant */

double GetMatrixNormalizer( double* ) ;

/* Calculation and application of CSR wake */

void GetCsrEloss(struct Bunch*, int, int, int, double, double, double ) ;

/* Calculation an application of longitudinal space charge */

double ProcLSC(struct Bunch*, int, double, int ) ;

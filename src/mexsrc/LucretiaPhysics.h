/* Lucretia data, prototypes, functions related to mathematics and physics */

/* AUTH: PT, 02-aug-2004 */
/* MOD:
                         */
#ifndef LUCRETIA_COMMON
  #include "LucretiaCommon.h"
#endif
#define LUCRETIA_PHYSICS
#ifdef __CUDACC__
  #include "curand_kernel.h"
#endif

/* define some constants */

/* value of pi from Matlab 6.1: */
#define PI 3.14159265358979
/* value of electron mass in GeV from 2004 PDG: */
#define ME2C2_GEV 0.000510998918
/* conversion from E[GeV] to Brho [T.m] */
#define GEV2TM 3.335640952
/* speed of light in m/s */
#define CLIGHT 299792458.0 
/* define energy gain which switches between Lcav matrix and drift mat*/
#define MIN_EGAIN 1.e-12
/* min bunch length, replaces sigz if sigz==0 in SRWF convolution */
#define MIN_BUNCH_LENGTH 0.3e-06
/* 300 nm == 1 femtosecond */
#define MIN_TILT 1.e-12
/* if magnet is within MIN_TILT of being a normal magnet, neglect skew terms */

/* define a macro to compute the fall-behind due to relativistic
   effects (actually the fall-behind is this factor times the element
	length) */

#define LORENTZ_DELAY(P) (ME2C2_GEV *ME2C2_GEV / 2 / P / P)

/* return the LucretiaPhysics version string */ 

char* LucretiaPhysicsVersion( ) ;

/* return the R matrix of a drift space */
#ifdef __CUDACC__
__host__ __device__ void GetDriftMap( double, Rmat ) ;
#else
void GetDriftMap( double, Rmat ) ;
#endif

/* return the R matrix and T5xx terms of a quad or plasma lens*/
#ifdef __CUDACC__
__host__ __device__ void GetQuadMap( double, double, double, double, Rmat, double[] ) ;
__host__ __device__ void GetPlensMap( double, double, double, Rmat, double[] ) ;
#else
void GetQuadMap( double, double, double, double, Rmat, double[] ) ;
void GetPlensMap( double, double, double, Rmat, double[] ) ;
#endif

/* return the R matrix and T5xx terms of a solenoid */
#ifdef __CUDACC__
__host__ __device__ void GetSolenoidMap( double, double, double, Rmat, double[] ) ;
#else
void GetSolenoidMap( double, double, double, Rmat, double[] ) ;
#endif

/* return the T matrix terms for a sextupole */
#ifdef __CUDACC__
__host__ __device__ void GetSextMap( double, double, double, double, double[4][10] ) ;
#else
void GetSextMap( double, double, double, double, double[4][10] ) ;
#endif

/* return the R matrix for an RF structure */

void GetLcavMap( double, double, double, double, double,
				 Rmat, int ) ;

/* propagate the transverse coordinates of a ray thru a thin-lens
   multipole */
#ifdef __CUDACC__
__device__ void PropagateRayThruMult_gpu( double, double*, double*, double*, int, double*,
						   double, double, double*, double*, int, int, double,
							double*, int*, double*, double*, int, int, double, double*, double*, double*, curandState_t *rState ) ;
__host__ void PropagateRayThruMult( double, double*, double*, double*, int, double*,
						   double, double, double*, double*, int, int, double,
							double*, int*, double*, double*, int, int, double ) ;
#else
void PropagateRayThruMult( double, double*, double*, double*, int, double*,
						   double, double, double*, double*, int, int, double,
							double*, int*, double*, double*, int, int, double ) ;
#endif

/* emulation of the MAD transport map for a sector bend */

void GetMADSBendMap( double, double, double, double,
						   double, Rmat, double[4][10], 
							double[10], int ) ;

/* transfer map for a sector bend, Lucretia native form: */
#ifdef __CUDACC__
__host__ __device__ void GetLucretiaSBendMap( double , double, double, double,
						  double, Rmat, double[10], double[4][10], double[13] ) ;
#else
void GetLucretiaSBendMap( double , double, double, double,
						  double, Rmat, double[10], double[4][10], double[13] ) ;
#endif

/* return the R matrix for a sector bend fringe field */
#ifdef __CUDACC__
__host__ __device__ void GetBendFringeMap( double, double, double, 
					   double, double, double, 
					   double, double, double, 
					   Rmat, double[10] ) ;
#else
void GetBendFringeMap( double, double, double, 
					   double, double, double, 
					   double, double, double, 
					   Rmat, double[10] ) ;
#endif

/* perform a rotation of an R-matrix through an xy angle */

void RotateRmat( Rmat, double ) ;

/* propagate a set of twiss parameters thru an r matrix */

int TwissThruRmat( Rmat, struct beta0*, struct beta0* ) ;

/* propagate coupled twiss parameters through an R matrix */

int CoupledTwissThruRmat( Rmat, double*, double*, int ) ;

/* convolve a short-range wakefield with the beam */

int ConvolveSRWFWithBeam( struct Bunch*, int, int ) ;

/* bin the rays in a bunch according to the desired bin width */

int BinRays( struct Bunch*, struct SRWF*, double ) ;

/* prepare a bunch to participate in frequency-domain LRWFs */

int PrepareBunchForLRWFFreq(struct Bunch*, int, int,
									 double*, double*, int,
									 double ) ;

/* compute the kick from a frequency-domain LRWF */

int ComputeTLRFreqKicks(struct LRWFFreq*, double, int, 
								struct LRWFFreqKick*, int, int,
								struct LucretiaComplex*, 
								struct LucretiaComplex*, 
								struct LucretiaComplex*, 
								double*, double ) ;

/* complex product operation */

struct LucretiaComplex ComplexProduct( struct LucretiaComplex, 
												   struct LucretiaComplex ) ;

/* synchrotron radiation parameters */
#ifdef __CUDACC__
__host__ __device__ void CalculateSRPars( double, double, double, double*, double*, double*, double* ) ;
#else
void CalculateSRPars( double, double, double, double*, double*, double*, double* ) ;
#endif

/* Poisson-distributed random numbers */
#ifdef __CUDACC__
__host__ int poidev( double ) ;
__device__ int poidev_gpu( double, curandState_t *rState ) ;
#else
int poidev( double ) ;
#endif

/* SR photon distribution via Wolski's method */
#ifdef __CUDACC__
__device__ double SRSpectrumAW_gpu( curandState_t *rState ) ;
__host__ double SRSpectrumAW( ) ;
#else
double SRSpectrumAW( ) ;
#endif

/* SR photon distribution via Burkhardt's method */
#ifdef __CUDACC__
__device__ double SRSpectrumHB_gpu( curandState_t *rState ) ;
__host__ double SRSpectrumHB( ) ;
__host__ __device__ double SynRadC( double ) ;
#else
double SRSpectrumHB( ) ;
double SynRadC( double ) ;
#endif

/* master SR loss function */
#ifdef __CUDACC__
  __device__ double ComputeSRMomentumLoss_gpu( double, double, double, int, curandState_t *rState ) ;
  __host__ double ComputeSRMomentumLoss( double, double, double, int ) ;
#else
  double ComputeSRMomentumLoss( double, double, double, int ) ;
#endif

/* transfer map for a coordinate-change element */

int GetCoordMap( double[6], double[6], Rmat ) ;

/* RMS of beam dimension */
double GetRMSCoord( struct Bunch*, int ) ;

/* Mean of beam dimension */
double GetMeanCoord( struct Bunch*, int ) ;

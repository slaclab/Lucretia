/* LucretiaMatlab.c:
 * Repository for all Lucretia procedures which are specific to the
 * Matlab implementation (ie, excluding all of the Octave ones).
 *
 * Contents:
 *
 * LucretiaMatlabVersions
 * GetElemClass
 * GetElemNumericPar
 * GetPS
 * GetPSNumericPar
 * GetElemGirder
 * GetGirderNumericPar
 * GetKlystron
 * GetKlystronNumericPar
 * GetKlystronStatus
 * GetNumWakes
 * LucretiaMatlabSetup
 * SetDoublesToField
 * nEleminBeamline
 * GetnPS
 * GetnKlys
 * GetnGirder
 * GetNumTrackFlags
 * GetTrackFlagValue
 * GetTrackFlagName
 * CreateStatusCellArray
 * RanGaussVecPtr
 * RanFlatVecPtr
 * GetRaySortkey
 * GetSRWFParameters
 * SplineSRWF
 * GetPascalMatrix
 * GetFactorial
 * GetMaxMultipoleIndex
 * ClearMaxMultipoleStuff
 * ComputeNewMultipoleStuff
 * GetTLRWakeClass
 * GetTLRErrWakeClass
 * GetTLRGeneralWakeClass
 * GetTLRNumericPar
 * GetTLRErrNumericPar
 * GetTLRGeneralNumericPar
 * GetCollimatorGeometry
 * GammaLog
 * GetMatrixNormalizer
 * IsEmpty
 * GetCsrEloss
 * GetExtProcessData
 * TMapGetData
 * TMapParamCheck
 *
 * AUTH: PT, 03-aug-2004
 * MOD:
 * 8-Sept-2016, GW:
 * Add error catching to cuda calls
 * 22-May-2014, GW:
 * Add TMAP data retrieval and verification functions
 * 31-March-2014, GW:
 * Add functionality for GEANT4 interface
 * 08-Oct-2010, GW:
 * LucretiaMatlabSetup tries to read in Lucretia BEAMLINE etc
 * as defined in caller scope in preference to global. This
 * needed for parallel operations with Distributed toolbox
 * 06-Mar-2008, PT:
 * add IsEmpty function to fix problem which appears when
 * switching between 2006 and 2008 versions of Matlab.
 * Update LucretiaMatlabSetup to use IsEmpty.
 * 07-Jan-2008, whitegr:
 * bugfixes related to lattice verification in absence of
 * wakefields.
 * 08-mar-2006, PT:
 * Add GetMatrixNormalizer function.
 * 06-Dec-2005, PT:
 * add RanFlatVecPtr function, and modify ClearPLHSVars in
 * support of same.  Add GammaLog function.
 * 06-Oct-2005, PT:
 * Bugfix:  don't destroy mxArrays used in multipole tracking
 * calculations unless you mxCreate'd them sometime since the
 * last time you mxDestroy'ed them!
 * 05-Oct-2005, PT:
 * Bugfix: don't destroy mxArray which contains the vector
 * of factorials until ready to generate a new one!
 */
#ifdef __CUDACC__
  #include "gpu/mxGPUArray.h"
  #include "curand_kernel.h" /* CUDA device-based random number generator*/
#endif
#include <string.h>
#include <math.h>
//#ifndef mex_h
//  #define mex_h
  #include <mex.h>
//#endif
#include "math.h"
#include "matrix.h"
#include "LucretiaMatlab.h"       /* gets LucretiaCommon.h for free */
#include "LucretiaGlobalAccess.h" /* manipulate Lucretia globals */
#include "LucretiaVersionProto.h" /* prototypes for version functions */
#include "LucretiaCuda.h" /* Definitions and headers required for GPU code */

/* File-scoped variables */

char LucretiaMatlabVersion[] = "LucretiaMatlab Version = 08-Oct-2010" ;

/* Make the mxArray pointers to the global data structures file-scoped;
 * that way when the cmdline-level Lucretia functions are first called
 * the initialization procedure can gain access to the structures for
 * all users in LucretiaMatlab. */

const mxArray *Beamline ; /* global BEAMLINE cell array */
const mxArray *PS       ; /* global PS structure array */
const mxArray *Klystron ; /* global KLYSTRON structure array */
const mxArray *Girder   ; /* global GIRDER structure array */
const mxArray *ZSR      ; /* longitudinal SRWFs */
const mxArray *TSR      ; /* transverse SRWFs */
const mxArray *TLR      ; /* transverse LRWFs */
const mxArray *TLRErr   ; /* transverse error LRWFs */
int nElem ;               /* total # of elements */
int nKlys, nPS, nGirder ; /* other total #'s */
int nWake[4] ;            /* total number of wakefields of each type */
double* key ;             /* sortkey used by GetRaySortkey */
int keylength ;

mxArray* plhs_randn[1]  = {NULL} ; /* return from randn call   */
mxArray* plhs_rand[1]   = {NULL} ; /* return from rand call    */
mxArray* plhs_spline[1] = {NULL} ; /* return from spline call  */
mxArray* plhs_gamln[1]  = {NULL} ; /* return from gammaln call */
mxArray* prhs_det[1]    = {NULL} ; /* send to det function */
mxArray* plhs_det[1]    = {NULL} ; /* return from det function */

/*==================================================================*/

/* Collect the version strings of LucretiaMatlab, LucretiaPhysics, and
 * LucretiaCommon ; return these, plus the version string of the calling
 * procedure, in a cell-vector mxArray. */

/* RET:    a pointer to an mxArray cell array with the 4 version strings
 * therein.
 * ABORT:  Aborts if any of the cells, or the array, can't be allocated.
 * FAIL:   none.                                                   */

mxArray* LucretiaMatlabVersions( const char *CallerVersion )
{
  /* local variables */
  
  mxArray *VersionCells ;      /* pointer to new cell array */
  mxArray *PhysVerString ;     /* Matlab version of PhysVersion */
  mxArray *CommVerString ;     /* Matlab version of CommVersion */
  mxArray *MatlVerString ;     /* Matlab version of LucretiaMatlabVersion */
  mxArray *CallVerString ;     /* Matlab version of CallerVersion */
  
  mwSize nVersion = 4 ;           /* # cells in VersionCells */
  char *PhysVersion ;          /* return from LucretiaPhysicsVersion */
  char *CommVersion ;          /* return from LucretiaCommonVersion */
  
  /* begin by allocating the new mxArray */
  
  VersionCells = mxCreateCellArray( 1, &nVersion ) ;
  if (VersionCells == NULL) /* failure */
    mexErrMsgTxt("Couldn't make new cell array for version data") ;
  
  /* otherwise, start loading it up */
  
  CallVerString = mxCreateString( CallerVersion ) ;
  if (CallVerString == NULL )
    mexErrMsgTxt("Couldn't make new string for version data") ;
  mxSetCell( VersionCells, 0, CallVerString ) ;
  
  PhysVersion = LucretiaPhysicsVersion( ) ;
  PhysVerString = mxCreateString( PhysVersion ) ;
  if (PhysVerString == NULL )
    mexErrMsgTxt("Couldn't make new string for version data") ;
  mxSetCell( VersionCells, 1, PhysVerString ) ;
  
  CommVersion = LucretiaCommonVersion( ) ;
  CommVerString = mxCreateString( CommVersion ) ;
  if (CommVerString == NULL )
    mexErrMsgTxt("Couldn't make new string for version data") ;
  mxSetCell( VersionCells, 2, CommVerString ) ;
  
  MatlVerString = mxCreateString( LucretiaMatlabVersion ) ;
  if (MatlVerString == NULL )
    mexErrMsgTxt("Couldn't make new string for version data") ;
  mxSetCell( VersionCells, 3, MatlVerString ) ;
  
  /* and that's it */
  
  return VersionCells ;
  
}

/*==================================================================*/
/*==================================================================*/
/*==================================================================*/
/*==================================================================*/
/*==================================================================*/

/* Here are the Matlab versions of the procedures prototyped in the
 * LucretiaGlobalAccess file. */

/* Return a C string version of the element class ("DRIF","QUAD",etc) */

/* RET:    A pointer to a string containing the element class name;
 * NULL is returned if the element cell cannot be accessed,
 * if the element cell is not a structure, if "Class" is not
 * a field of the structure, or if "Class" is a non-char field.
 * ABORT:  never.
 * FAIL:   never.                                                  */

char* GetElemClass( int elemno )
{
  mxArray* ElemCell ;   /* pointer to the cell */
  mxArray* ClassField ; /* pointer to "Class" field of cell */
  static char ClassString[17] ; /* for return value */
  
  /* start by getting a pointer to the correct beamline cell */
  
  ElemCell = mxGetCell( Beamline, elemno ) ;
  
  /* if the element is ill-defined throw an abort */
  
  if (ElemCell == NULL)
    return NULL ;
  if (!mxIsStruct(ElemCell))
    return NULL ;
  
  /* now get a pointer to the "Class" string mxArray */
  
  ClassField = mxGetField( ElemCell, 0, "Class" ) ;
  if (ClassField == NULL)
    return NULL ;
  if (!mxIsChar(ClassField))
    return NULL ;
  
  /* return a pointer to the contents of the ClassField */
  
  mxGetString( ClassField, ClassString, 17 ) ;
  return ClassString ;
  
}

/*==================================================================*/

/* Get a numeric parameter of a beamline element. */

/* RET:    A pointer to the double value of the parameter.  NULL is
 * returned if the named parameter is not a field of the
 * element cell, or if it is not a double-precision field.
 * ABORT:  never.
 * FAIL:   if the desired element cannot be accessed.  Since the
 * element's Class must be determined to know which parameters
 * it will have, and the GetElemClass procedure protects
 * against failure to access BEAMLINE{elemno}, it is assumed
 * that we are generally safe against such a failure here. */


double* GetElemNumericPar( int elemno, const char* parname, int* len )
{
  mxArray *ElemCell ; /* pointer to the quad cell array */
  mxArray *Param ;    /* pointer to one element at a time */
  double *parval ;    /* pointer to the double of interest */
  
  if (len != NULL)
    *len = 0 ;
  
  /* get a pointer to the cell */
  ElemCell = mxGetCell( Beamline, elemno ) ;
  /* get field pointer */
  Param = mxGetField( ElemCell, 0, parname ) ;
  if (Param == NULL)
    parval = NULL ;
  else
  {
    parval = mxGetPr( Param ) ;
    if (len != NULL)
      *len = mxGetM(Param)*mxGetN(Param) ;
  }
  
  return parval;
}

/*==================================================================*/

/* get the power supply, if any, which goes with an element */

/* RET:    The index # of the power supply (in Matlab counting, which
 * begins with 1 and not 0) that goes with a given element.
 * Returns zero if NULL return from the GetElemNumericPar
 * call.
 * ABORT:  never.
 * FAIL:   never.                                                   */

int GetPS( int elemno )
{
  double  *PSData ;   /* pointer to the data in the PS Field */
  
  /* use GetElementNumericPar to get the double precision version of the
   * PS index from the Beamline array: */
  
  PSData = GetElemNumericPar( elemno, "PS", NULL ) ;
  if (PSData == NULL)
    return 0;
  else
    return (int)(*PSData) ;
  
}

/*==================================================================*/

/* get a numeric parameter of a given power supply */

/* RET:    Pointer to the value of the numeric parameter of the power
 * supply.  NULL pointer returned if the PS cell array isn't
 * present as a global, if the named field isn't present, if
 * the PS # is out of range, or if the named field isn't of
 * type double.
 * ABORT:  never.
 * FAIL:   never.                                                   */

double* GetPSNumericPar( int PSno, const char* parname, int* len )
{
  mxArray *AmplField ; /* pointer into Ampl field */
  double* parval ;
  
  if (len != NULL)
    *len = 0 ;
  
  if ( PS != NULL )
  {
    AmplField = mxGetField( PS, PSno, parname ) ;
    if (AmplField == NULL)
      return NULL ;
    parval = mxGetPr( AmplField ) ;
    if (len != NULL)
      *len = mxGetM(AmplField)*mxGetN(AmplField) ;
    return parval ;
  }
  else
    return NULL ;
}

/*==================================================================*/

/* get the girder, if any, which goes with an element */

/* RET:    The index # of the girder (in Matlab counting, which
 * begins with 1 and not 0) that goes with a given element.
 * Returns zero if NULL return from the GetElemNumericPar
 * call.
 * ABORT:  never.
 * FAIL:   never.                                                   */

int GetElemGirder( int elemno )
{
  double  *GData ;   /* pointer to the data in the Girder Field */
  
  /* use GetElementNumericPar to get the double precision version of the
   * girder index from the Beamline array: */
  
  GData = GetElemNumericPar( elemno, "Girder", NULL ) ;
  if (GData == NULL)
    return 0;
  else
    return (int)(*GData) ;
  
}

/*==================================================================*/

/* get a numeric parameter of a given girder */

/* RET:    Pointer to the value of the numeric parameter of the girder.
 * NULL pointer returned if the girder structure array isn't
 * present as a global, if the named field isn't present, if
 * the girder # is out of range, or if the named field isn't of
 * type double.
 * ABORT:  never.
 * FAIL:   never.                                                   */

double* GetGirderNumericPar( int Gno, const char* parname, int* len )
{
  mxArray *Field ; /* pointer into field */
  mxArray *Cell  ; /* pointer into field */
  double* parval ;
  
  if (len != NULL)
    *len = 0 ;
  
  if ( Girder != NULL )
  {
    Cell = mxGetCell( Girder, Gno ) ;
    Field = mxGetField( Cell, 0, parname ) ;
    if (Field == NULL)
      return NULL ;
    parval = mxGetPr( Field ) ;
    if (len != NULL)
      *len = mxGetM(Field)*mxGetN(Field) ;
    return parval ;
  }
  else
    return NULL ;
}

/*==================================================================*/

/* get the klystron, if any, which goes with an element */

/* RET:    The index # of the klystron (in Matlab counting, which
 * begins with 1 and not 0) that goes with a given element.
 * Returns zero if NULL return from the GetElemNumericPar
 * call.
 * ABORT:  never.
 * FAIL:   never.                                                   */

int GetKlystron( int elemno )
{
  double  *KData ;   /* pointer to the data in the Klystron Field */
  
  /* use GetElementNumericPar to get the double precision version of the
   * klystron index from the Beamline array: */
  
  KData = GetElemNumericPar( elemno, "Klystron", NULL ) ;
  if (KData == NULL)
    return 0;
  else
    return (int)(*KData) ;
  
}

/*==================================================================*/

/* get a numeric parameter of a given klystron */

/* RET:    Pointer to the value of the numeric parameter of the klystron.
 * ULL pointer returned if the klystron data structure isn't
 * present as a global, if the named field isn't present, if
 * the klys # is out of range, or if the named field isn't of
 * type double.
 * ABORT:  never.
 * FAIL:   never.                                                   */

double* GetKlystronNumericPar( int Kno, const char* parname, int* len )
{
  mxArray *AmplField ; /* pointer into Ampl field */
  double* parval ;
  
  if (len != NULL)
    *len = 0 ;
  
  if ( Klystron != NULL )
  {
    AmplField = mxGetField( Klystron, Kno, parname ) ;
    if (AmplField == NULL)
      return NULL ;
    parval = mxGetPr( AmplField ) ;
    if (len != NULL)
      *len = mxGetM(AmplField)*mxGetN(AmplField) ;
    return parval ;
  }
  else
    return NULL ;
}


/*==================================================================*/

/* get the status of a given klystron */

/* RET:    Pointer to a status enum representing the klystron status.
 * NULL pointer returned if the klystron data structure isn't
 * present as a global, if the status field isn't present, if
 * the klys # is out of range, or if the status field isn't of
 * type string.
 * ABORT:  never.
 * FAIL:   never.                                                   */

enum KlystronStatus* GetKlystronStatus( int Kno )
{
  mxArray* StatField ;
  static enum KlystronStatus retval ;
  char statbuf[16] ;
  
  if ( Klystron != NULL )
  {
    StatField = mxGetField( Klystron, Kno, "Stat" ) ;
    if (StatField == NULL)
      return NULL ;
    if ( !mxIsChar(StatField) )
      return NULL ;
    mxGetString(StatField, statbuf, 16) ;
    
    if (strcmp(statbuf,"ON")==0)
      retval = ON ;
    if (strcmp(statbuf,"STANDBY")==0)
      retval = STANDBY ;
    if (strcmp(statbuf,"TRIPPED")==0)
      retval = TRIPPED ;
    if (strcmp(statbuf,"MAKEUP")==0)
      retval = MAKEUP ;
    if (strcmp(statbuf,"TRIPSTANDBY")==0)
      retval = STANDBYTRIP ;
    return &retval ;
  }
  else
    return NULL ;
}

/*==================================================================*/

/* Return the total number of wakes of all kinds */

int* GetNumWakes( )
{
  return nWake ;
}

/*==================================================================*/

/* Procedure LucretiaMatlabSetup:  gets pointers to the BEAMLINE,
 * PS, GIRDER, KLYSTRON global data objects, which are saved in
 * file-scoped variables.  Does some light status checking (like whether
 * the BEAMLINE and GIRDER are cell arrays, and the PS and KLYSTRON are
 * structure arrays).  Also gets pointers to ZSR and TSR wakefield info
 * from global data structure WF, and nulls key (used for sortkeys).
 *
 * Try to get BEAMLINE/PS/GIRDER/KLYSTRON/WF from caller scope first,
 * then try global - ie if there is a BEAMLINE local variable definition
 * in the MATLAB calling function then this will be taken in preference
 * to the global definition
 *
 * RET:    number of cells in the BEAMLINE array.  Returns zero,
 * indicating that execution must halt, if any of the
 * following conditions are met:
 *
 * -> No global BEAMLINE array
 * -> BEAMLINE is not a cell array
 * -> BEAMLINE has no entries (ie, 0 x 0 matrix)
 * -> BEAMLINE is not one-dimensional
 * -> GIRDER exists but is not:
 * -> a cell array
 * -> one-dimensional
 * -> KLYSTRON exists but is not:
 * -> a structure array
 * -> one-dimensional
 * -> PS exists but is not:
 * -> a structure array
 * -> one-dimensional
 * -> WF exists but is not:
 * -> a 1 x 1 structure
 * -> WF.ZSR exists but is not
 * -> a structure array
 * -> one-dimensional
 * -> WF.TSR exists but is not
 * -> a structure array
 * -> one-dimensional
 * -> WF.TLR exists but is not
 * -> a cell array
 * -> one-dimensional
 * -> WF.TLRErr exists but is not
 * -> a cell array
 * -> one-dimensional
 * FAIL: never.
 *
 * Note that non-existence of GIRDER, KLYSTRON, PS, or WF is permitted -- it is
 * perfectly acceptable to leave these elements out of the accelerator
 * definition (though not very fun).  */
/* Note further that file-scoped variables nElem and nWake,
 * and nTLRModesMax and NTLRErrModesmax are set by this procedure. */

int LucretiaMatlabSetup( )
{
  int M,N ;
  int nelem = 0 ;
  const mxArray* WF ;
  
  nKlys = 0 ;
  nPS = 0 ;
  nGirder = 0 ;
  ZSR = NULL ;
  TSR = NULL ;
  TLR = NULL ;
  TLRErr = NULL ;
  nWake[0] = 0 ;
  nWake[1] = 0 ;
  nWake[2] = 0 ;
  nWake[3] = 0 ;
  
  /* start with the BEAMLINE */
  
  Beamline = GET_GLOBAL_PTR( "BEAMLINE" , "caller" ) ;
  if (Beamline == NULL)
  {
    /* If not found in caller scope, try global scope*/
    Beamline = GET_GLOBAL_PTR( "BEAMLINE" , "global" ) ;
    if (Beamline == NULL)
    {
      AddMessage("No BEAMLINE cell array found",1) ;
      goto egress ;
    }
  }
  
  /* check to make sure that it's okay:
   * is it a cell array? */
  if (!mxIsCell(Beamline))
  {
    AddMessage("Global BEAMLINE is not a cell array",1) ;
    goto egress ;
  }
  /*	is it uni-dimensional and of nonzero length? */
  M = mxGetM(Beamline) ;
  N = mxGetN(Beamline) ;
  if ( (M==0) && (N==0) )
  {
    AddMessage("Global BEAMLINE is zero length",1) ;
    goto egress ;
  }
  if ( (M>1) && (N>1) )
  {
    AddMessage("Global BEAMLINE is multi-dimensional",1) ;
    goto egress ;
  }
  if (M>N)
    nElem = M ;
  else
    nElem = N ;
  
  /* now for the girders */
  
  Girder = GET_GLOBAL_PTR( "GIRDER", "caller" ) ;
  if (Girder == NULL)
  {
    /* If not found in caller scope, try global scope*/
    Girder = GET_GLOBAL_PTR( "GIRDER", "global" ) ;
  }
  if (Girder != NULL)
  {
    M = mxGetM(Girder) ;
    N = mxGetN(Girder) ;
    if (M * N == 0)
      Girder = NULL ;
  }
  if (Girder != NULL)
  {
    
    /* if GIRDER exists, it must be a one-dimensional cell array */
    
    if (!mxIsCell(Girder))
    {
      AddMessage("Global GIRDER is not a cell array",1) ;
      goto egress ;
    }
    if ( (M>1) && (N>1) )
    {
      AddMessage("Global GIRDER is multi-dimensional",1) ;
      goto egress ;
    }
    nGirder = (M>N) ? M : N ;
  }
  
  /* now power supplies (PS) */
  
  PS = GET_GLOBAL_PTR( "PS" , "caller" ) ;
  if (PS == NULL)
  {
    /* If not found in caller scope, try global scope*/
    PS = GET_GLOBAL_PTR( "PS", "global" ) ;
  }
  if (PS != NULL)
  {
    M = mxGetM(PS) ;
    N = mxGetN(PS) ;
    if (M * N == 0)
      PS = NULL ;
  }
  if (PS != NULL)
  {
    
    /* if PS exists, it must be a one-dimensional structure array */
    
    if (!mxIsStruct(PS))
    {
      AddMessage("Global PS is not a structure array",1) ;
      goto egress ;
    }
    if ( (M>1) && (N>1) )
    {
      AddMessage("Global PS is multi-dimensional",1) ;
      goto egress ;
    }
    nPS = (M>N) ? M : N ;
  }
  
  /* now klystrons */
  
  Klystron = GET_GLOBAL_PTR( "KLYSTRON" , "caller" ) ;
  if (Klystron == NULL)
  {
    /* If not found in caller scope, try global scope*/
    Klystron = GET_GLOBAL_PTR( "KLYSTRON", "global" ) ;
  }
  if (Klystron != NULL)
  {
    M = mxGetM(Klystron) ;
    N = mxGetN(Klystron) ;
    if (M * N == 0)
      Klystron = NULL ;
  }
  if (Klystron != NULL)
  {
    
    /* if KLYSTRON exists, it must be a one-dimensional cell array */
    
    if (!mxIsStruct(Klystron))
    {
      AddMessage("Global KLYSTRON is not a structure array",1) ;
      goto egress ;
    }
    if ( (M>1) && (N>1) )
    {
      AddMessage("Global KLYSTRON is multi-dimensional",1) ;
      goto egress ;
    }
    nKlys = (M>N) ? M : N ;
  }
  
  /* now WF */
  
  WF = GET_GLOBAL_PTR( "WF" , "caller" ) ;
  if (WF == NULL)
  {
    /* If not found in caller scope, try global scope*/
    WF = GET_GLOBAL_PTR( "WF", "global" ) ;
  }
  if (WF != NULL)
  {
    M = mxGetM(WF) ;
    N = mxGetN(WF) ;
    if (M * N == 0)
      WF = NULL ;
  }
  /*	if (WF == NULL)
   * {
   * ZSR = NULL ;
   * TSR = NULL ;
   * TLR = NULL ;
   * TLRErr = NULL ;
   * nWake[0] = 0 ;
   * nWake[1] = 0 ;
   * nWake[2] = 0 ;
   * nWake[3] = 0 ;
   * }
   * else */
  if (WF != NULL)
  {
    if (!mxIsStruct(WF))
    {
      AddMessage("Global WF is not a structure",1) ;
      goto egress ;
    }
    if ( (M>1) || (N>1) )
    {
      AddMessage("Global WF dimensions not 1 x 1",1) ;
      goto egress ;
    }
    ZSR = mxGetField(WF,0,"ZSR") ;
    if (IsEmpty(ZSR))
      ZSR = NULL ;
    if (ZSR != NULL)
    {
      if (!mxIsStruct(ZSR))
      {
        AddMessage("Global WF.ZSR is not a structure array",1) ;
        goto egress ;
      }
      M = mxGetM(ZSR) ;
      N = mxGetN(ZSR) ;
      if ( (M>1) && (N>1) )
      {
        AddMessage("Global WF.ZSR is multi-dimensional",1) ;
        goto egress ;
      }
      nWake[0] = (M>N) ? M : N ;
    }
    else
      nWake[0] = 0 ;
    TSR = mxGetField(WF,0,"TSR") ;
    if (IsEmpty(TSR))
      TSR = NULL ;
    if (TSR != NULL)
    {
      if (!mxIsStruct(TSR))
      {
        AddMessage("Global WF.TSR is not a structure array",1) ;
        goto egress ;
      }
      M = mxGetM(TSR) ;
      N = mxGetN(TSR) ;
      if ( (M>1) && (N>1) )
      {
        AddMessage("Global WF.TSR is multi-dimensional",1) ;
        goto egress ;
      }
      nWake[1] = (M>N) ? M : N ;
    }
    else
      nWake[1] = 0 ;
    TLR = mxGetField(WF,0,"TLR") ;
    if (IsEmpty(TLR))
      TLR = NULL ;
    if (TLR != NULL)
    {
      if (!mxIsCell(TLR))
      {
        AddMessage("Global WF.TLR is not a cell array",1) ;
        goto egress ;
      }
      M = mxGetM(TLR) ;
      N = mxGetN(TLR) ;
      if ( (M>1) && (N>1) )
      {
        AddMessage("Global WF.TLR is multi-dimensional",1) ;
        goto egress ;
      }
      nWake[2] = (M>N) ? M : N ;
    }
    else
      nWake[2] = 0 ;
    TLRErr = mxGetField(WF,0,"TLRErr") ;
    if (IsEmpty(TLRErr))
      TLRErr = NULL ;
    if (TLRErr != NULL)
    {
      if (!mxIsCell(TLRErr))
      {
        AddMessage("Global WF.TLRErr is not a cell array",1) ;
        goto egress ;
      }
      M = mxGetM(TLRErr) ;
      N = mxGetN(TLRErr) ;
      if ( (M>1) && (N>1) )
      {
        AddMessage("Global WF.TLRErr is multi-dimensional",1) ;
        goto egress ;
      }
      nWake[3] = (M>N) ? M : N ;
    }
    else
      nWake[3] = 0 ;
  }
  
  /* if we got here, then everything must be okay (or at least not too bad).
   * Set return and exit. */
  
  nelem = nElem ;
  key = NULL ;
  keylength = 0 ;
  
  egress:
        
        return nelem ;
        
}

/*==================================================================*/

/* Procedure to set a vector of doubles into one field of a structure.
 *
 * RET:    int status value: 1 = good,
 * 0 = allocation failure,
 * -1 = ill-defined structure.
 * ABORT:  never.
 * FAIL:   never. */

int SetDoublesToField( mxArray* structure, unsigned int indx, const char* fieldname,
        double* values, int nvalues )
{
  mxArray* fieldptr ;
  double* realptr ;
  int status = 1 ;
  int i ;
  int fieldno ;
  
  /* create a double vector for the values */
  
  fieldptr = mxCreateDoubleMatrix( 1, nvalues, mxREAL ) ;
  if (fieldptr == NULL)
  {
    status = 0 ;
    goto egress;
  }
  realptr = mxGetPr( fieldptr ) ;
  
  for (i=0 ; i<nvalues ; i++)
    realptr[i] = values[i] ;
  
  /* confirm that the structure is a structure, that indx is in range, and
   * that fieldname is a fieldname */
  
  fieldno = mxGetFieldNumber( structure, fieldname ) ;
  if ( (fieldno<0) ||
          (mxGetM(structure)*mxGetN(structure) < indx+1) )
  {
    status = -1 ;
    goto egress ;
  }
  
  /* otherwise, set the field */
  
  mxSetFieldByNumber( structure, indx, fieldno, fieldptr ) ;
  status = 1 ;
  
  egress:
    
    return status ;
    
}

/*==================================================================*/

/* Ask for the total # of elements in BEAMLINE. */

/* RET:    the file-scoped variable nElem ;
 * ABORT:  never.
 * FAIL:   never.  */

int nElemInBeamline( )
{
  return nElem ;
}

/*==================================================================*/

/* Ask for the total # of girders. */

/* RET:    the file-scoped variable nGirder ;
 * ABORT:  never.
 * FAIL:   never.  */

int GetnGirder( )
{
  return nGirder ;
}

/*==================================================================*/

/* Ask for the total # of klystrons. */

/* RET:    the file-scoped variable nKlys ;
 * ABORT:  never.
 * FAIL:   never.  */

int GetnKlystron( )
{
  return nKlys ;
}

/*==================================================================*/

/* Ask for the total # of power supplies. */

/* RET:    the file-scoped variable nPS ;
 * ABORT:  never.
 * FAIL:   never.  */

int GetnPS( )
{
  return nPS ;
}

/*==================================================================*/

/* Since we always need to call GetNumTrackFlags before we can get
 * the flags themselves, we will store a pointer to the most recent
 * TrackFlags structure here in global-land, so that GetSingleTrackFlag
 * can access it */

mxArray* TrkFlgPtr ;

/*==================================================================*/

/* return the number of tracking flags from an element */

/* RET:    number of tracking flags.  If there is a
 * problem with the structure of the tracking flags data,
 * returns -1.
 * ABORT:  never.
 * FAIL:   never. */

int GetNumTrackFlags( int elemno )
{
  mxArray *ElemCell ;
  int iret = -1 ;
  
  /* get a pointer to the TrackFlag field in the requested element */
  
  ElemCell = mxGetCell( Beamline, elemno ) ;
  if (ElemCell == NULL)
    goto egress ;
  TrkFlgPtr = mxGetField( ElemCell, 0, "TrackFlag" ) ;
  if (TrkFlgPtr == NULL)
  {
    
    /* no TrackFlags structure is allowed */
    
    iret = 0 ;
    goto egress ;
    
  }
  
  /* but a TrackFlags field which is not a structure is not allowed */
  
  if (!mxIsStruct(TrkFlgPtr))
    goto egress ;
  
  /* get the total number of tracking flags */
  
  iret = mxGetNumberOfFields( TrkFlgPtr ) ;
  
  /* exit */
  
  egress:
    
    return iret ;
    
}

/*==================================================================*/

/* return the value of a specified tracking flag from an element */

/* RET:    Tracking flag value.  If there is a
 * problem with the structure of the tracking flags data,
 * returns -1.
 * ABORT:  never.
 * FAIL:   never. */

int GetTrackFlagValue( int flagno )
{
  int retval ;
  mxArray* field ;
  double* pr ;
  field = mxGetFieldByNumber( TrkFlgPtr, 0, flagno ) ;
  if (!mxIsDouble(field))
  {
    retval = -1 ;
    goto egress ;
  }
  pr = mxGetPr(field) ;
  retval = (int)pr[0] ;
  
  egress:
    
    return retval ;
    
}

/*==================================================================*/

/* return the name of a specified tracking flag from an element */

/* RET:    Pointer to the character string name of the flag, or
 * NULL in the event of a problem.
 * ABORT:  never.
 * FAIL:   never. */

const char* GetTrackFlagName( int flagno )
{
  const char* flagname ;
  flagname = mxGetFieldNameByNumber( TrkFlgPtr, flagno ) ;
  
  return flagname ;
  
}

/*==================================================================*/

/* Allocate and populate a cell array with the Lucretia status and
 * messages. */

/* RET:    pointer to mxArray with the status cells.
 * ABORT:  Failure of mxCreateCellMatrix, mxCreateString, or
 * CREATE_SCALAR_DOUBLE will
 * cause an exit via mexErrMsgTxt.  If the heap space is
 * so constrained that the error messages cannot be returned,
 * it is probably too much to expect that a soft landing can
 * be engineered.
 * FAIL:   Never. */

mxArray* CreateStatusCellArray( int Status, int Nmsg, char** msgs )
{
  int i ;
  mxArray* cell;
  mxArray* string ;
  mxArray* real ;
  
  /* create the cell array */
  
  cell = mxCreateCellMatrix( Nmsg+1, 1 ) ;
  if (cell == NULL)
    mexErrMsgTxt("Unable to create array for status data!") ;
  
  /* create something to hold the status flag */
  
  real = CREATE_SCALAR_DOUBLE( (double)Status );
  if (real==NULL)
    mexErrMsgTxt("Unable to create array for status data!") ;
  mxSetCell( cell, 0, real ) ;
  
  /* if there are messages, copy them to cell now */
  
  for (i=0 ; i<Nmsg ; i++)
  {
    string = mxCreateString( msgs[i] ) ;
    if (string == NULL)
      mexErrMsgTxt("Unable to create array for status data!") ;
    mxSetCell( cell, i+1, string ) ;
  }
  
  /* set return and exit */
  
  return cell ;
}

/*==================================================================*/

/* use Matlab randn to generate a vector of Gaussian-distributed
 * random numbers. */

/* RET:    pointer to a double vector full of random numbers, or
 * NULL if unsuccessful.
 * ABORT:  never.
 * FAIL:   never.               */
#ifdef __CUDACC__
__device__ double RanGaussVecPtr_gpu( curandState_t *rState )
{
#ifdef LUCRETIA_DPREC
  return curand_normal_double(rState) ;
#else
  return curand_normal(rState) ;
#endif
}
#endif
double* RanGaussVecPtr( int nRandom )
{
#ifdef LUCRETIA_MLRAND  
  mxArray* prhs[2] ;               /* two calling arguments */
  double* vector = NULL ;                 /* pointer to the vector */
  
  if (plhs_randn[0] != NULL)
    mxDestroyArray(plhs_randn[0]) ;
  
  prhs[0] = CREATE_SCALAR_DOUBLE( (double)nRandom ) ;
  if (prhs[0] == NULL)
    goto egress ;
  prhs[1] = CREATE_SCALAR_DOUBLE( 1.0 ) ;
  if (prhs[1] == NULL)
    goto egress ;
  
  /* call matlab's randn */
  
  mexCallMATLAB( 1, plhs_randn, 2, prhs, "randn" ) ;
  if (plhs_randn[0] != NULL)
    vector = mxGetPr( plhs_randn[0] ) ;
  mxDestroyArray( prhs[0] ) ;
  mxDestroyArray( prhs[1] ) ;
  
  
  egress:
    
    return vector ;
#else
  return RanGaussVecPtrC( nRandom ) ;
#endif  
    
}

double* RanGaussVecPtrC( int nRandom )
{
  /* Sending nRandom = 0 clears stored memory and returns NULL */
  static const double a[] =
  {
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00
  };

  static const double b[] =
  {
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01
  };

  static const double c[] =
  {
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00
  };

  static const double d[] =
  {
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00
  };

  static double lowreg = 0.02425 ;
  static double highreg = 0.97575 ;

  double q, r,p ;
  int i ;
  static double *retval = NULL ;
  static int lastNum = 0 ;
  
  /* If requested by sending 0 then clear memory and return NULL */
  if (nRandom==0) {
    if (retval != NULL)
      free(retval) ;
    return NULL ;
  }
  
  /* Assign memory for number of requested random numbers */
  if (retval!=NULL && nRandom!=lastNum) {
    free(retval) ;
    retval = (double*) malloc(sizeof(double)*nRandom) ;
  }
  else if (retval==NULL)
    retval = (double*) malloc(sizeof(double)*nRandom) ;
  lastNum = nRandom ;
  
  
  for (i=0; i<nRandom; i++) {
    p=(double) rand() / (double) RAND_MAX;
    if (p < 0 || p > 1)
    {
      retval[i] = 0.0;
    }
    else if (p == 0)
    {
      retval[i] = -(double) RAND_MAX /* minus "infinity" */;
    }
    else if (p == 1)
    {
      retval[i] = (double) RAND_MAX /* "infinity" */;
    }
    else if (p < lowreg)
    {
      q = sqrt(-2*log(p));
      retval[i] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1) ;
    }
    else if (p > highreg)
    {
      q  = sqrt(-2*log(1-p));
      retval[i] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1) ;
    }
    else
    {
      q = p - 0.5;
      r = q*q;
      retval[i] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
              (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1) ;
    }
  }
  return retval ;
}

void getLucretiaRandSeed( unsigned long long *rseed )
{
  mxArray *rseedMat ;
  double* rPtr ;
  mexEvalString("rseedForMexFunc=ceil(rand*1e9);");
  rseedMat=mexGetVariable("caller","rseedForMexFunc");
  rPtr=mxGetPr(rseedMat);
  *rseed = (unsigned long long) *rPtr ;
}
unsigned int getLucretiaRandSeedC( )
{
  mxArray *rseedMat ;
  double seed ;
  mexEvalString("rseedForMexFunc=ceil(rand*1e9);");
  rseedMat=mexGetVariable("caller","rseedForMexFunc");
  seed=*mxGetPr(rseedMat);
  return (unsigned int) seed ;
}

double* RanFlatVecPtrC( int nRandom )
{
  int i ;
  static double *retval = NULL ;
  static int lastNum = 0 ;
  
  /* If requested by sending 0 then clear memory and return NULL */
  if (nRandom==0) {
    if (retval != NULL)
      free(retval) ;
    return NULL ;
  }
  
  /* Assign memory for number of requested random numbers */
  if (retval!=NULL && nRandom!=lastNum) {
    free(retval) ;
    retval = (double*) malloc(sizeof(double)*nRandom) ;
  }
  else if (retval==NULL)
    retval = (double*) malloc(sizeof(double)*nRandom) ;
  lastNum = nRandom ;
  
  for (i=0; i<nRandom; i++)
    retval[i] = (double) rand() / (double) RAND_MAX ;
  
  return retval ;
  
}

/*==================================================================*/

/* use Matlab rand to generate a vector of uniform-distributed
 * random numbers if running on CPU host, else if running on CUDA device, use cuRand library*/

/* RET:    pointer to a double vector full of random numbers, or
 * NULL if unsuccessful.
 * ABORT:  never.
 * FAIL:   never.               */

#ifdef __CUDACC__
__device__ double RanFlatVecPtr_gpu( curandState_t *rState)
{
#ifdef LUCRETIA_DPREC
  return curand_uniform_double(rState) ;
#else
  return curand_uniform(rState) ;
#endif
}
#endif
double* RanFlatVecPtr( int nRandom )
{
#ifdef LUCRETIA_MLRAND    
  double* vector = NULL ;                 /* pointer to the vector */
  mxArray* prhs[2] ;               /* two calling arguments */
  
  if (plhs_rand[0] != NULL)
    mxDestroyArray(plhs_rand[0]) ;
  
  prhs[0] = CREATE_SCALAR_DOUBLE( (double)nRandom ) ;
  if (prhs[0] == NULL)
    goto egress ;
  prhs[1] = CREATE_SCALAR_DOUBLE( 1.0 ) ;
  if (prhs[1] == NULL)
    goto egress ;
  
  /* call matlab's randn */
  
  mexCallMATLAB( 1, plhs_rand, 2, prhs, "rand" ) ;
  if (plhs_rand[0] != NULL)
    vector = mxGetPr( plhs_rand[0] ) ;
  mxDestroyArray( prhs[0] ) ;
  mxDestroyArray( prhs[1] ) ;
  
  
  egress:
    return vector ;
#else
  double* retvec ;
  retvec = RanFlatVecPtrC( nRandom ) ;
  return retvec ;
#endif  
}

/*==================================================================*/

/* use Matlab rand to compute the log-gamma function. */

/* RET:    double gammaln(x), or zero if unsuccessful.
 * ABORT:  never.
 * FAIL:   never.               */
#ifdef __CUDACC__
__device__ double GammaLog_gpu( double x )
{
#ifdef LUCRETIA_DPREC
  return lgamma(x) ;
#else
  return lgammaf(x) ;
#endif
}
#endif
double GammaLog( double x )
{
  double value ;                   /* one return value */
#ifdef LUCRETIA_MLRAND  
  mxArray* prhs[1] ;               /* one calling arguments */
  
  value = 0 ;
  
  if (plhs_gamln[0] != NULL)
    mxDestroyArray(plhs_gamln[0]) ;
  
  prhs[0] = CREATE_SCALAR_DOUBLE( x ) ;
  if (prhs[0] == NULL)
    goto egress ;
  
  /* call matlab's gammaln */
  
  mexCallMATLAB( 1, plhs_gamln, 1, prhs, "gammaln" ) ;
  if (plhs_gamln[0] != NULL)
    value = *(mxGetPr( plhs_gamln[0] )) ;
  mxDestroyArray( prhs[0] ) ;
  
#else
  value = lgamma( x ) ;
#endif  
#ifdef LUCRETIA_MLRAND    
  egress:
#endif    
    
    return value ;
    
}

/* Use Matlab sort to return a sortkey for the beam along a given DOF
 *
 * RET:    double*, pointer to the sortkey
 * ABORT:  if unable to do Matlab memory allocation.
 * FAIL:   never */

double* GetRaySortkey( double* coords, int nray, int DOF )
{
  
  mxArray* plhs[2] ;    /* 2 returned values */
  mxArray* prhs[2] ;    /* 2 arguments */
  double* matkey ;
  double* spare ;
  int count ;
  
  /* build the arguments */
  
  prhs[1] = CREATE_SCALAR_DOUBLE( 2.0 ) ;
  prhs[0] = mxCreateDoubleMatrix( 6, nray, mxREAL ) ;
  spare = mxGetPr(prhs[0]) ;
  mxSetPr( prhs[0], coords ) ;
  
  /* call the sorter */
  
  mexCallMATLAB( 2, plhs, 2, prhs, "sort" ) ;
  
  /* unpack the result */
  
  matkey = mxGetPr(plhs[1]) ;
  mxSetPr( prhs[0], spare ) ;
  if ( (key != NULL) && (keylength < nray) )
  {
    mxFree( key ) ;
    key = NULL ;
  }
  if (key == NULL)
  {
    key = (double*)mxCalloc( nray, sizeof(double) ) ;
    keylength = nray ;
  }
  for (count=0; count<nray ; count++)
    key[count] = matkey[6*count+DOF-1] - 1. ;
  
  mxDestroyArray( prhs[0] ) ;
  mxDestroyArray( prhs[1] ) ;
  mxDestroyArray( plhs[0] ) ;
  mxDestroyArray( plhs[1] ) ;
  
  /* and that's it */
  
  return key ;
  
}

/*==================================================================*/

/* access global SRWF data and return information on a selected SRWF */

/* RET:   0 if requested WF is missing,
 * -1 if no z vector or z is not a real vector,
 * -2 if z vector has zero length,
 * -3 if z[0] != 0,
 * -4 if no K vector or K is not a real vector,
 * -5 if len(K) != len(z),
 * -6 if no binwidth or binwidth not a real,
 * -7 if binwidth == 0,
 * otherwise the # of values in the wake function table.
 * ABORT: if Matlab dynamic allocation fails.
 * FAIL:  never. */

int GetSRWFParameters( int WFno, int flag, double** zpr, double** kpr,
        double* BinWidth )
{
  const mxArray* WF ;
  mxArray* utility ;
  int retval = 1 ;
  int zlength ;
  double* bw ;
  
  if (flag==0)
    WF = ZSR ;
  else
    WF = TSR ;
  
  if (WF == NULL)
  {
    retval = 0 ;
    goto egress ;
  }
  
  utility = mxGetField( WF, WFno, "z" ) ;
  if (utility == NULL)
  {
    retval = -1 ;
    goto egress ;
  }
  *zpr = mxGetPr( utility ) ;
  if (*zpr == NULL)
  {
    retval = -1 ;
    goto egress ;
  }
  zlength = mxGetNumberOfElements( utility ) ;
  if (zlength == 0)
  {
    retval = -2 ;
    goto egress ;
  }
  if (*zpr[0] != 0.)
  {
    retval = -3 ;
    goto egress ;
  }
  
  utility = mxGetField( WF, WFno, "K" ) ;
  if (utility == NULL)
  {
    retval = -4 ;
    goto egress ;
  }
  *kpr = mxGetPr( utility ) ;
  if (*kpr == NULL)
  {
    retval = -4 ;
    goto egress ;
  }
  if ((int) mxGetNumberOfElements(utility)!=zlength)
  {
    retval = -5 ;
    goto egress ;
  }
  
  utility = mxGetField( WF, WFno, "BinWidth" ) ;
  if (utility == NULL)
  {
    retval = -6 ;
    goto egress ;
  }
  bw = mxGetPr( utility ) ;
  if (bw == NULL)
  {
    retval = -6 ;
    goto egress ;
  }
  else
    *BinWidth = *bw ;
  if (*BinWidth == 0.)
  {
    retval = -7 ;
    goto egress ;
  }
  
  egress:
    
    if (retval == 1)
      return zlength ;
    else
      return retval ;
    
}

/*==================================================================*/

/* Use matlab Spline function to compute the SRWF values at selected
 * z locations, given a table of the SRWF function */

double* SplineSRWF(double* z, double* k, double* zbin,
        int nwake, int i, int nbin)
{
  mxArray* prhs[3] ;                      /* 3 arguments */
  
  double* zz ;
  double zbini ;
  int nbinspline, count ;
  double* spare1 ;
  double* spare2 ;
  double* spare3 ;
  
  if ( plhs_spline[0] != NULL )
    mxDestroyArray( plhs_spline[0] ) ;
  
  /* build the argument data structures */
  
  prhs[0] = mxCreateDoubleMatrix( 1, nwake, mxREAL ) ;
  spare1 = mxGetPr( prhs[0] ) ;
  mxSetPr( prhs[0], z ) ;
  prhs[1] = mxCreateDoubleMatrix( 1, nwake, mxREAL ) ;
  spare2 = mxGetPr( prhs[1] ) ;
  mxSetPr( prhs[1], k ) ;
  nbinspline = nbin-1-i ;
  zbini = zbin[i] ;
  zz = (double*)mxCalloc( nbinspline, sizeof(double) ) ;
  for (count=0 ; count<nbinspline ; count++)
    zz[count] = zbin[count+i+1] - zbini ;
  prhs[2] = mxCreateDoubleMatrix( 1, nbinspline, mxREAL ) ;
  spare3 = mxGetPr( prhs[2] ) ;
  mxSetPr( prhs[2], zz ) ;
  
  /* call spliner */
  
  mexCallMATLAB( 1, plhs_spline, 3, prhs, "spline" ) ;
  
  /* return the splined values */
  
  mxSetPr( prhs[0], spare1 ) ;
  mxSetPr( prhs[1], spare2 ) ;
  mxSetPr( prhs[2], spare3 ) ;
  
  mxDestroyArray( prhs[0] ) ;
  mxDestroyArray( prhs[1] ) ;
  mxDestroyArray( prhs[2] ) ;
  mxFree( zz ) ;
  
  return mxGetPr(plhs_spline[0]) ;
  
}

/*==================================================================*/

/* Since several routines need access to the following variables,
 * they are given a scope higher than the function level */

double*  TheMatrix = NULL ; /* ptr to pre-calc'ed pascal matrix */
double*  TheFact   = NULL ; /* ptr to pre-calc'ed factorial vector */
double   MaxMult = -1 ;     /* size of pascal matrix etc */
mxArray* plhs_pas[1]  ;     /* return from "pascal" call */
mxArray* plhs_abs[1]  ;     /* return from "abs" call */
mxArray* plhs_fact[1] ;     /* return from "factorial" call */
#ifdef __CUDACC__ /* device copies of pre-calc'ed variables above */
double* dTheMatrix = NULL ;
double* dTheFact = NULL ;
double* dMaxMult = NULL ;
double* GetPascalMatrix_gpu( )
{
  return dTheMatrix ;
}
double* GetFactorial_gpu( )
{
  return dTheFact ;
}
double* GetMaxMultipoleIndex_gpu( )
{
  return dMaxMult ;
}
#endif

/* use Matlab functions to get an auxiliary information for computing the
 * kicks of thin-lens multipoles up to a selected pole index: */

double* GetPascalMatrix( )
{
  return TheMatrix ;
}

double* GetFactorial( )
{
  return TheFact ;
}

double GetMaxMultipoleIndex( )
{
  return MaxMult ;
}

void ClearMaxMultipoleStuff( )
{
  TheMatrix = NULL ;
  TheFact   = NULL ;
  
  /* before destroying the arrays of interest for multipole expansions,
   * make sure that they were at some point created! */
  
  if (MaxMult > -1)
  {
    mxDestroyArray( plhs_pas[0] ) ;
    mxDestroyArray( plhs_abs[0] ) ;
    mxDestroyArray( plhs_fact[0] ) ;
    MaxMult = -1 ;
  }
#ifdef __CUDACC__ /* clear device associated variable memory */
  /*cudaFreeHost(dTheMatrix) ;
  cudaFreeHost(dTheFact) ;
  cudaFreeHost(dMaxMult) ; */
  gpuErrchk( cudaFree(dTheMatrix) );
  gpuErrchk( cudaFree(dTheFact) );
  gpuErrchk( cudaFree(dMaxMult) );
#endif  
}

void ComputeNewMultipoleStuff(double MultSize)
{
  /* double* MMptr = NULL ; */
  mxArray* prhs_lin[3]  ;   /* arguments for "linspace" call */
  mxArray* prhs_pas[2]  ;   /* arguments for "pascal" call */
  
  /*	mxArray* plhs_fact[1] ; */  /* return from "factorial" call */
  mxArray* plhs_lin[1]  ;   /* return from "linspace" call */
  
  /* if the pascal matrix and the factorial vector pointers are
   * currently assigned, then we have undestroyed return arrays that
   * should be cleaned up first */
  
  if (TheMatrix != NULL)
    ClearMaxMultipoleStuff( ) ;
  
  /* start by taking absolute value of MultSize */
  
  MaxMult = fabs(MultSize) ;
  
  /* construct the vector from 0 to AbsMultSize */
  
  prhs_lin[0] = CREATE_SCALAR_DOUBLE( 0. ) ;
  prhs_lin[1] = CREATE_SCALAR_DOUBLE( MaxMult ) ;
  prhs_lin[2] = CREATE_SCALAR_DOUBLE( MaxMult+1 ) ;
  mexCallMATLAB( 1, plhs_lin, 3, prhs_lin, "linspace" ) ;
  
  /* use the linspace output to construct a factorial vector */
  
  mexCallMATLAB( 1, plhs_fact, 1, plhs_lin, "factorial" ) ;
  TheFact = mxGetPr( plhs_fact[0] ) ;
  
  /* now construct the pascal matrix */
  
  prhs_pas[0] = CREATE_SCALAR_DOUBLE( MaxMult ) ;
  prhs_pas[1] = CREATE_SCALAR_DOUBLE( 1.0     ) ;
  
  mexCallMATLAB( 1, plhs_pas, 2, prhs_pas, "pascal" ) ;
  mexCallMATLAB( 1, plhs_abs, 1, plhs_pas, "abs" ) ;
  TheMatrix = mxGetPr( plhs_abs[0] ) ;
  
  /* clean up */
  
  mxDestroyArray( prhs_lin[0]  ) ;
  mxDestroyArray( prhs_lin[1]  ) ;
  mxDestroyArray( prhs_lin[2]  ) ;
  mxDestroyArray( prhs_pas[0]  ) ;
  mxDestroyArray( prhs_pas[1]  ) ;
  mxDestroyArray( plhs_lin[0]  ) ;
  /*	mxDestroyArray( plhs_fact[0] ) ; */
  
#ifdef __CUDACC__ /* Create associated variables on the CUDA device */
  /*cudaHostAlloc( (void**) &TheMatrix, MaxMult*MaxMult*sizeof(double), cudaHostAllocMapped );
  cudaHostGetDevicePointer ( &dTheMatrix, TheMatrix, 0 ) ;*/
  gpuErrchk( cudaMalloc((void **)&dTheMatrix, sizeof(double)*MaxMult*MaxMult) );
  gpuErrchk( cudaMemcpy(dTheMatrix, TheMatrix, sizeof(double)*MaxMult*MaxMult, cudaMemcpyHostToDevice) );
  /*cudaHostAlloc( (void**) &TheFact, 3*sizeof(double), cudaHostAllocMapped );
  cudaHostGetDevicePointer ( &dTheFact, TheFact, 0 ) ;*/
  gpuErrchk( cudaMalloc((void **)&dTheFact, sizeof(double)*(MaxMult+1)) );
  gpuErrchk( cudaMemcpy(dTheFact, TheFact, sizeof(double)*(MaxMult+1), cudaMemcpyHostToDevice) );
  /*MMptr = &MaxMult ;
  cudaHostAlloc( (void**) &MMptr, sizeof(double), cudaHostAllocMapped );
  cudaHostGetDevicePointer ( &dMaxMult, &MaxMult, 0 ) ;*/
  gpuErrchk( cudaMalloc((void **)&dMaxMult, sizeof(double)) );
  gpuErrchk( cudaMemcpy(dMaxMult, &MaxMult, sizeof(double), cudaMemcpyHostToDevice) );
#endif
  
  return ;
  
}

/*==================================================================*/

/* Three functions related to figuring out whether a long-range
 * wake is a time- or frequency-domain model */

int GetTLRWakeClass( int Wakenum )
{
  return GetTLRGeneralWakeClass( Wakenum, TLR ) ;
}

int GetTLRErrWakeClass( int Wakenum )
{
  return GetTLRGeneralWakeClass( Wakenum, TLRErr ) ;
}

/* Return an integer indicating whether a particular transverse
 * long-range wakefield is time- or frequency-domain type */

/* RET:   integer TIMEDOMAIN if the wake is time-domain,
 * integer FREQDOMAIN if the wake is frequency-domain,
 * integer UNKNOWNDOMAIN if there is some problem and the
 * wake class cannot be determined.
 * ABORT: Never.
 * FAIL:  Never */

int GetTLRGeneralWakeClass( int Wakenum, const mxArray* TLRGen )
{
  
  mxArray* TheWake ;
  mxArray* ClassField ;
  char class_string[10] ;
  int retval ;
  int classint = UNKNOWNDOMAIN ; /* default = failure */
  
  /* get a pointer to the correct entry in the TLR cell array */
  
  TheWake = mxGetCell( TLRGen, Wakenum ) ;
  if (TheWake == NULL)
    goto egress ;
  
  /* get the class field */
  
  ClassField = mxGetField( TheWake, 0, "Class") ;
  if (ClassField == NULL)
    goto egress ;
  retval = mxGetString(ClassField, class_string, 10) ;
  if (retval == 1)
    goto egress ;
  
  /* if the class field is "Frequency" or "Time", set correct
   * return values now */
  
  if (strcmp(class_string,"Time")==0)
    classint = TIMEDOMAIN ;
  if (strcmp(class_string,"Frequency")==0)
    classint = FREQDOMAIN ;
  
  egress:
    
    return classint ;
    
}

/*==================================================================*/

/* Three functions for interrogating the long range wakefield
 * database for numeric parameters */

double* GetTLRNumericPar( int wakeno, const char* parname, int* parlen )
{
  return GetTLRGeneralNumericPar( wakeno, parname, parlen,
          TLR ) ;
}

double* GetTLRErrNumericPar( int wakeno, const char* parname, int* parlen )
{
  return GetTLRGeneralNumericPar( wakeno, parname, parlen,
          TLRErr ) ;
}

/* get a pointer to a long-range wakefield out of the Matlab
 * database.  While we're at it, get its length.
 *
 * RET:    pointer to the numeric values, or NULL if unsuccessful.
 * As a side effect the length of the numeric value vector
 * will be set.
 * ABORT:  never.
 * FAIL:   never. */

double* GetTLRGeneralNumericPar( int wakeno, const char* parname, int* parlen,
        const mxArray* WakeTable )
{
  
  double* retval = NULL ;
  mxArray* TheWake ;
  mxArray* TheField ;
  
  /* default parlen to zero */
  
  *parlen = 0 ;
  
  /* get a pointer to the correct cell */
  
  TheWake = mxGetCell( WakeTable, wakeno ) ;
  if (TheWake == NULL)
    goto egress ;
  
  /* get a pointer to the desired field */
  
  TheField = mxGetField( TheWake, 0, parname ) ;
  if (TheField == NULL)
    goto egress ;
  
  /* get a pointer to the real values */
  
  retval = mxGetPr( TheField ) ;
  if (retval == NULL)
    goto egress ;
  
  /* if we're still executing, get the size of the vector */
  
  *parlen = mxGetN(TheField) * mxGetM(TheField) ;
  
  egress:
    
    return retval ;
    
}

/*==================================================================*/

/* get the shape parameter of a collimator */

/* RET:    integer values representing elliptical or rectangular
 * collimators if successful, a value representing an unknown
 * geometry otherwise */
/* ABORT:  never.
 * FAIL:   never. */

int GetCollimatorGeometry( int elemno )
{
  mxArray* TheElem ;
  mxArray* shape_ptr ;
  char     geomstr[10] ;
  int dmy ;
  
  /* if the element number is out of range, return the unknown
   * value */
  
  if ( elemno+1 > nElem )
    return COLL_UNKNOWN ;
  
  /* otherwise try to get the shape out of the database */
  
  TheElem = mxGetCell( Beamline, elemno ) ;
  if (TheElem == NULL)
    return COLL_UNKNOWN ;
  
  shape_ptr = mxGetField( TheElem, 0, "Geometry" ) ;
  if (shape_ptr == NULL)
    return COLL_UNKNOWN ;
  
  dmy = mxGetString( shape_ptr, geomstr, 10 ) ;
  if (dmy==1)
    return COLL_UNKNOWN ;
  
  if (strcmp(geomstr,"Ellipse") == 0)
    return COLL_ELLIPSE ;
  else if (strcmp(geomstr,"Rectangle") == 0)
    return COLL_RECTANGLE ;
  else
    return COLL_UNKNOWN ;
  
}

/*==================================================================*/

/* Get the matrix normalizing factor from its determinant */

/* RET:    double precision matrix normalizer
 * ABORT:  never.
 * FAIL:   never. */

double GetMatrixNormalizer( double* R )
{
  
  double* rhsptr ;
  int i ;
  
  /* if we've not yet built the right-hand side mxArray, do it now */
  
  if (prhs_det[0] == NULL)
  {
    prhs_det[0] = mxCreateDoubleMatrix(6,6,mxREAL) ;
  }
  
  /* copy the values over */
  
  rhsptr = mxGetPr(prhs_det[0]) ;
  for (i=0 ; i<36 ; i++)
    rhsptr[i] = R[i] ;
  mexCallMATLAB( 1, plhs_det, 1, prhs_det, "det" ) ;
  
  /* return the cube root of the 6x6 determinant */
  
  return pow(*mxGetPr(plhs_det[0]),1./3.) ;
  
}

/*==================================================================*/

/* clear a couple of matlab return arrays */

/* RET:    none.
 * ABORT:  never.
 * FAIL:   never */

void ClearPLHSVars( )
{
  if (plhs_spline[0] != NULL)
  {
    mxDestroyArray( plhs_spline[0] ) ;
    plhs_spline[0] = NULL ;
  }
  if (plhs_randn[0] != NULL)
  {
    mxDestroyArray( plhs_randn[0] ) ;
    plhs_randn[0] = NULL ;
  }
  if (plhs_rand[0] != NULL)
  {
    mxDestroyArray( plhs_rand[0] ) ;
    plhs_rand[0] = NULL ;
  }
  if (plhs_gamln[0] != NULL)
  {
    mxDestroyArray( plhs_gamln[0] ) ;
    plhs_gamln[0] = NULL ;
  }
  if (prhs_det[0] != NULL)
  {
    mxDestroyArray( prhs_det[0] ) ;
    prhs_det[0] = NULL ;
  }
  if (plhs_det[0] != NULL)
  {
    mxDestroyArray( plhs_det[0] ) ;
    plhs_det[0] = NULL ;
  }
  
  
}

/*==================================================================*/

/* Perform check for an empty field in a way which is compatible with
 * both 2006 and 2008 versions of Matlab.
 *
 * RET:    boolean -- true if the field is empty, false if it has content.
 * ABORT:  never.
 * FAIL:   never */

bool IsEmpty(const mxArray* pa)
{
  if (pa == NULL)
    return true;
  if (mxIsEmpty(pa))
    return true ;
  return false ;
}


/*==================================================================*/

/* Get Energy loss profile due to Coherent Synchrotron Radiation */
void GetCsrEloss(struct Bunch* ThisBunch, int nbin, int smoothVal, int elementNo, double Lpos, double dL, double xmean0 )
{
  mxArray *lhs, *rhs[9] ;
  int iarr ;
  /* If less than 100 particles, don't bother */
  if (ThisBunch->nray<100)
    return;
  /* Allocate RHS entries */
  rhs[0] =  mxCreateDoubleMatrix(6, ThisBunch->nray, mxREAL) ;
  memcpy(mxGetPr(rhs[0]),ThisBunch->x,sizeof(double)*ThisBunch->nray*6) ;
  rhs[1] =  mxCreateDoubleMatrix(1, ThisBunch->nray, mxREAL) ;
  memcpy(mxGetPr(rhs[1]),ThisBunch->Q,sizeof(double)*ThisBunch->nray) ;
  rhs[2] = mxCreateDoubleMatrix(1, ThisBunch->nray, mxREAL) ;
  memcpy(mxGetPr(rhs[2]),ThisBunch->stop,sizeof(double)*ThisBunch->nray) ;
  rhs[3] = mxCreateDoubleScalar((double)nbin);
  rhs[4] = mxCreateDoubleScalar((double)smoothVal);
  rhs[5] = mxCreateDoubleScalar((double)elementNo+1);
  rhs[6] = mxCreateDoubleScalar(Lpos);
  rhs[7] = mxCreateDoubleScalar(dL);
  rhs[8] = mxCreateDoubleScalar(xmean0);
  /* Call applyCSR function to calculate E loss for this element just tracked through */
  /* [beam W dE zOut]=applyCSR(beam,beamQ,nbin,smoothVal,itrack,driftL,driftDL) */
  mexCallMATLAB(1, &lhs, 9, rhs, "applyCSR") ;
  memcpy(ThisBunch->x,mxGetPr(lhs),sizeof(double)*ThisBunch->nray*6);
  for (iarr=0;iarr<9;iarr++)
    mxDestroyArray(rhs[iarr]) ;
  mxDestroyArray(lhs) ;
}

/* Process LSC information
 - Provide element splitting requirements
 - Calculate and apply LSC E modulation to bunch */
double ProcLSC(struct Bunch* ThisBunch, int elemno, double L, int storeDataInd)
{
  mxArray *lhs[2], *rhs[6] ;
  int iarr ;
  double dl_ret ;
  
  /* If less than 100 particles, don't bother */
  if (ThisBunch->nray<100)
    return 0;
  /* Allocate RHS entries */
  rhs[0] =  mxCreateDoubleMatrix(6, ThisBunch->nray, mxREAL) ;
  memcpy(mxGetPr(rhs[0]),ThisBunch->x,sizeof(double)*ThisBunch->nray*6) ;
  rhs[1] =  mxCreateDoubleMatrix(1, ThisBunch->nray, mxREAL) ;
  memcpy(mxGetPr(rhs[1]),ThisBunch->Q,sizeof(double)*ThisBunch->nray) ;
  rhs[2] =  mxCreateDoubleMatrix(1, ThisBunch->nray, mxREAL) ;
  memcpy(mxGetPr(rhs[2]),ThisBunch->stop,sizeof(double)*ThisBunch->nray) ;
  rhs[3] = mxCreateDoubleScalar((double)elemno+1);
  rhs[4] = mxCreateDoubleScalar(L);
  rhs[5] = mxCreateDoubleScalar((double)storeDataInd);
  /* Call applyLSC function to calculate E modulation or split length */
  /* [xOut, dL] = applyLSC(x,Q,stop,elemno,dL,storeBunchInd) */
  mexCallMATLAB(2, lhs, 6, rhs, "applyLSC") ;
  if (L>0)
    memcpy(ThisBunch->x,mxGetPr(lhs[0]),sizeof(double)*ThisBunch->nray*6);
  for (iarr=0;iarr<6;iarr++)
    mxDestroyArray(rhs[iarr]) ;
  dl_ret=*mxGetPr(lhs[1]) ;
  mxDestroyArray(lhs[0]) ;
  mxDestroyArray(lhs[1]) ;
  return dl_ret ; /* dL */
}

/* Get ExtProcess class data from BEAMLINE array */
mxArray* GetExtProcessData(int* elemno, const char *propname)
{
  mxArray* ElemCell ;   /* pointer to the cell */
  mxArray* extProcess ; /* pointer to the extProcess object */
  mxArray* retprop ;
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, *elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return NULL ;
  if (!mxIsStruct(ElemCell))
    return NULL ;
  
  /* now get a pointer to the extProcess object if there */
  extProcess = mxGetField( ElemCell, 0, "ExtProcess" ) ;
  if (extProcess == NULL)
    return NULL ;
  if (!mxIsClass(extProcess, "ExtG4Process"))
    return NULL ;
  
  /* now return the requested property */
  retprop = mxGetProperty(extProcess, 0, propname) ;
  return retprop ;
}

/* RET: void
 * ABORT:  never.
 * FAIL:   never */

mxArray* GetExtProcessPtr(int* elemno )
{
  mxArray* ElemCell ;   /* pointer to the cell */
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, *elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return NULL ;
  if (!mxIsStruct(ElemCell))
    return NULL ;
  
  /* now get a pointer to the extProcess object */
  return mxGetField( ElemCell, 0, "ExtProcess" ) ;
}

mxArray* GetExtProcessPrimariesData(int* elemno)
{
  mxArray* ElemCell ;   /* pointer to the cell */
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, *elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return NULL ;
  if (!mxIsStruct(ElemCell))
    return NULL ;
  
  /* now get a pointer to the extProcess object if there */
  return mxGetField( ElemCell, 0, "ExtProcess_primariesData" ) ;
}

mxArray* GetExtProcessMeshData(int* elemno)
{
  mxArray* ElemCell ;   /* pointer to the cell */
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, *elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return NULL ;
  if (!mxIsStruct(ElemCell))
    return NULL ;
  
  /* now get a pointer to the extProcess object if there */
  return mxGetField( ElemCell, 0, "ExtProcess_meshData" ) ;
}

mxArray* GetExtProcessSecondariesData(int* elemno)
{
  mxArray* ElemCell ;   /* pointer to the cell */
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, *elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return NULL ;
  if (!mxIsStruct(ElemCell))
    return NULL ;
  
  /* now get a pointer to the extProcess object if there */
  return mxGetField( ElemCell, 0, "ExtProcess_secondariesData" ) ;
}

/* Get data from a TMAP BEAMLINE element
   RETURN 1 if OK, 0 if formatting errors */
int TMapGetDataR(int elemno, double R[6][6])
{
  mxArray* ElemCell ;   /* pointer to the cell */
  mxArray* fieldPtr ;
  double* ThisRmat ;
  int i,i2 ;
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return 0 ;
  if (!mxIsStruct(ElemCell))
    return 0 ;
  /* Return pointer to R matrix which must be defined and have dimensions 6 x 6 */
  fieldPtr = mxGetField(ElemCell, 0, "R") ;
  if (fieldPtr==NULL || mxGetM(fieldPtr)!=6 || mxGetN(fieldPtr)!=6)
    return 0;
  ThisRmat = mxGetPr(fieldPtr) ;
  for (i=0; i<6; i++)
    for (i2=0; i2<6; i2++)
      R[i][i2]=ThisRmat[i+6*i2];
  return 1 ;
}
void TMapGetDataLen(int elemno, unsigned short* T_size, unsigned short* U_size, unsigned short* V_size, unsigned short* W_size)
{
  mxArray* ElemCell ;   /* pointer to the cell */
  ElemCell = mxGetCell( Beamline, elemno ) ;
  *T_size = (unsigned short) mxGetNumberOfElements(mxGetField(ElemCell, 0, "T")) ;
  *U_size = (unsigned short) mxGetNumberOfElements(mxGetField(ElemCell, 0, "U")) ;
  *V_size = (unsigned short) mxGetNumberOfElements(mxGetField(ElemCell, 0, "V")) ;
  *W_size = (unsigned short) mxGetNumberOfElements(mxGetField(ElemCell, 0, "W")) ;
}
int TMapGetData(int elemno, double* Offset, double* R, double* T, double* U, double* V, double* W,
        unsigned long* T_inds, unsigned long* U_inds, unsigned long* V_inds, unsigned long* W_inds,
        unsigned short* T_size, unsigned short* U_size, unsigned short* V_size, unsigned short* W_size)
{
  mxArray* ElemCell ;   /* pointer to the cell */
  mxArray* fieldPtr ;
  int dim ;
  double* tmp ;
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return 0 ;
  if (!mxIsStruct(ElemCell))
    return 0 ;
  /* Offset and R fields must be present and intialised with correct dimensions */
  fieldPtr = mxGetField(ElemCell, 0, "Offset") ;
  if (mxGetNumberOfElements(fieldPtr)!=6)
    return 0;
  memcpy(Offset,mxGetPr(fieldPtr),sizeof(double)*6) ;
  fieldPtr = mxGetField(ElemCell, 0, "R") ;
  if (mxGetM(fieldPtr)!=6 || mxGetN(fieldPtr)!=6)
    return 0;
  memcpy(R,mxGetPr(fieldPtr),sizeof(double)*36) ;
  /* Process T elements if provided else return NULL pointers */
  fieldPtr = mxGetField(ElemCell, 0, "T") ;
  if (fieldPtr == NULL) {
    T=NULL;
    T_inds=NULL;
    *T_size=0;
  }
  else {
    if (T_size==NULL || *T_size==0) {
      T=NULL;
      T_inds=NULL;
      *T_size=0;
    }
    else {
      tmp = mxGetPr(fieldPtr) ;
      memcpy(T,tmp,*T_size * sizeof(double)) ;
      fieldPtr = mxGetField(ElemCell, 0, "Tinds") ;
      if (mxGetNumberOfElements(fieldPtr)!=*T_size)
        return 0;
      tmp = mxGetPr(fieldPtr) ;
      for (dim=0;dim<*T_size;dim++)
        T_inds[dim] = (unsigned long) tmp[dim] ;
    }
  }
  /* Process U elements if provided else return NULL pointers */
  fieldPtr = mxGetField(ElemCell, 0, "U") ;
  if (fieldPtr == NULL) {
    U=NULL;
    U_inds=NULL;
    *U_size=0;
  }
  else {
    *U_size = mxGetNumberOfElements(fieldPtr) ;
    if (U_size==NULL || *U_size==0) {
      U=NULL;
      U_inds=NULL;
      *U_size=0;
    }
    else {
      tmp = mxGetPr(fieldPtr) ;
      memcpy(U,tmp,*U_size * sizeof(double)) ;
      fieldPtr = mxGetField(ElemCell, 0, "Uinds") ;
      if (mxGetNumberOfElements(fieldPtr)!=*U_size)
        return 0;
      tmp = mxGetPr(fieldPtr) ;
      for (dim=0;dim<*U_size;dim++)
        U_inds[dim] = (unsigned long) tmp[dim] ;
    }
  }
  /* Process V elements if provided else return NULL pointers */
  fieldPtr = mxGetField(ElemCell, 0, "V") ;
  if (fieldPtr == NULL) {
    V=NULL;
    V_inds=NULL;
    *V_size=0;
  }
  else {
    *V_size = mxGetNumberOfElements(fieldPtr) ;
    if (V_size==NULL || *V_size==0) {
      V=NULL;
      V_inds=NULL;
      *V_size=0;
    }
    else {
      tmp = mxGetPr(fieldPtr) ;
      memcpy(V,tmp,*V_size * sizeof(double)) ;
      fieldPtr = mxGetField(ElemCell, 0, "Vinds") ;
      if (mxGetNumberOfElements(fieldPtr)!=*V_size)
        return 0;
      tmp = mxGetPr(fieldPtr) ;
      for (dim=0;dim<*V_size;dim++)
        V_inds[dim] = (unsigned long) tmp[dim] ;
    }
  }
  /* Process W elements if provided else return NULL pointers */
  fieldPtr = mxGetField(ElemCell, 0, "W") ;
  if (fieldPtr == NULL) {
    W=NULL;
    W_inds=NULL;
    *W_size=0;
  }
  else {
    *W_size = mxGetNumberOfElements(fieldPtr) ;
    if (W_size==NULL || *W_size==0) {
      W=NULL;
      W_inds=NULL;
      *W_size=0;
    }
    else {
      tmp = mxGetPr(fieldPtr) ;
      memcpy(W,tmp,*W_size * sizeof(double)) ;
      fieldPtr = mxGetField(ElemCell, 0, "Winds") ;
      if (mxGetNumberOfElements(fieldPtr)!=*W_size)
        return 0;
      tmp = mxGetPr(fieldPtr) ;
      for (dim=0;dim<*W_size;dim++)
        W_inds[dim] = (unsigned long) tmp[dim] ;
    }
  }
  return 1 ;
}
/* Check formatting of TMAP element class, return 1 if OK, 0 if errors */
int TMapParamCheck(int elemno)
{
  mxArray* ElemCell ;
  mxArray* fieldPtr ;
  mxArray* fieldPtr2 ;
  /* start by getting a pointer to the correct beamline cell */
  ElemCell = mxGetCell( Beamline, elemno ) ;
  /* if the element is ill-defined throw an abort */
  if (ElemCell == NULL)
    return 0 ;
  if (!mxIsStruct(ElemCell))
    return 0 ;
  /* Offset should be 6 x 1 */
  fieldPtr = mxGetField(ElemCell, 0, "Offset") ;
  if (fieldPtr == NULL)
    return 0 ;
  if (mxGetNumberOfElements(fieldPtr) != 6)
    return 0 ;
  /* R should be 6 * 6 */
  fieldPtr = mxGetField(ElemCell, 0, "R") ;
  if (fieldPtr == NULL)
    return 0 ;
  if (mxGetM(fieldPtr) != 6 || mxGetN(fieldPtr) != 6)
    return 0 ;
  /* T,U,V,W can be empty, but if defined Tinds etc fields must have same rows as T etc and be correctly dimensioned*/
  fieldPtr = mxGetField(ElemCell, 0, "T") ;
  if (fieldPtr != NULL) {
    fieldPtr2 = mxGetField(ElemCell, 0, "Tinds") ;
    if (fieldPtr2==NULL || mxGetNumberOfElements(fieldPtr2)!=mxGetNumberOfElements(fieldPtr))
        return 0;
  }
  fieldPtr = mxGetField(ElemCell, 0, "U") ;
  if (fieldPtr != NULL) {
    fieldPtr2 = mxGetField(ElemCell, 0, "Uinds") ;
    if (fieldPtr2==NULL || mxGetNumberOfElements(fieldPtr2)!=mxGetNumberOfElements(fieldPtr))
        return 0;
  }
  fieldPtr = mxGetField(ElemCell, 0, "V") ;
  if (fieldPtr != NULL) {
    fieldPtr2 = mxGetField(ElemCell, 0, "Vinds") ;
    if (fieldPtr2==NULL || mxGetNumberOfElements(fieldPtr2)!=mxGetNumberOfElements(fieldPtr))
        return 0;
  }
  fieldPtr = mxGetField(ElemCell, 0, "W") ;
  if (fieldPtr != NULL) {
    fieldPtr2 = mxGetField(ElemCell, 0, "Winds") ;
    if (fieldPtr2==NULL || mxGetNumberOfElements(fieldPtr2)!=mxGetNumberOfElements(fieldPtr))
        return 0;
  }
  return 1 ;
}

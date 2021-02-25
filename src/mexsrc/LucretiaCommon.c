/* LucretiaCommon.c
 * This file contains routines which are common to the Matlab and Octave
 * implementations of Lucretia, but which are "above" the level of physics
 * operations.  Typically routines in here are called by procedures at the
 * mexfile level, and call procedures at the physics level.
 *
 * Contents:
 *
 * LucretiaCommonVersion
 * RmatCalculate
 * RmatCopy
 * RmatProduct
 * TrackThruMain
 * TrackBunchThruDrift
 * TrackBunchThruQSOS
 * TrackBunchThruMult
 * TrackBunchThruSBend
 * TrackBunchThruRF
 * TrackBunchThruBPM
 * TrackBunchThruInst
 * TrackBunchThruCorrector
 * TrackBunchThruCollimator
 * TrackBunchThruCoord
 * TrackBunchThruTMap
 * GetTrackingFlags
 * CheckAperStopPart
 * CheckP0StopPart
 * CheckPperpStopPart
 * GetTotalOffsetXfrms
 * ApplyTotalXfrm
 * BPMInstIndexingSetup
 * BunchSlotSetup
 * FreeAndNull
 * BadElementMessage
 * BadPSMessage
 * BadKlystronMessage
 * BadTwissMessage
 * BadInverseTwissMessage
 * BadTrackFlagMessage
 * BadApertureMessage
 * BadSROptionsMessage
 * BadParticleMomentumMessage
 * BadParticlePperpMessage
 * BadInitMomentumMessage
 * BadOffsetMessage
 * BadBPMAllocMsg
 * BadSliceAllocMessage
 * BadSRWFAllocMsg
 * BadSRWFMessage
 * NonexistentLRWFmessage
 * BadLRWFAllocMessage
 * BadLRWFBunchOrder
 * BunchStopMessage
 * ClearTrackingVars
 * LcavSliceSetup
 * SBPMSetup
 * SBPMSetS
 * ComputeSBPMReadings
 * ClearConvolvedSRWF
 * ClearBinnedLRWFFreq
 * GetDatabaseParameters
 * GetSpecialSBendPar
 * GetDesignLorentzDelay
 * XYExchange
 * InitialMomentumCheck
 * GetLocalCoordPtrs
 * CheckGirderMoverPars
 * GetDBValue
 * VerifyLattice
 * VerifyParameters
 * ElemIndexLookup
 * UnpackLRWFFreqData
 * ComputeLRWFFreqDamping
 * ComputeLRWFFreqPhase
 * PrepareAllWF
 * GetThisElemTLRFreqKick
 * PutThisElemTLRFreqKick
 * ComputeTSRKicks
 * AccumulateWFBinPositions
 * ClearOldLRWFFreqKicks
 * XGPU2CPU
 * XCPU2GPU
 * ExtProcess
 *
 */ /* AUTH:  PT, 03-aug-2004 */
/* MOD:
 * 8-Sept-2016, GW:
 * Add cuda code for INSTR class and add cuda error catching + cuda debugging
 * 22-May-2014, GW:
 * Add TMAP element class
 * 01-apr-2014, GW:
 * code for implementation of GEANT tracking (ExtProcess)
 * 02-aug-2007, PT:
 * bugfix in CheckAperStopPart, code clean-up in
 * TrackBunchThruSBend, and minor correction to the handling of
 * synchrotron radiation in TrackBunchThruSBend.
 * 24-May-2007, PT:
 * bugfix: check for bad total or transverse momentum on
 * TrackThruMain entry was not performed correctly.  New arglist
 * for CheckP0StopPart which differentiates between performing
 * the check on the upstream vs downstream face of an element.
 * 21-May-2007, PT:
 * support for XYCOR element.
 * 13-feb-2007, PT:
 * bugfix:  correction to loop logic in CheckGirderMoverPars.
 * Correct handling of COORD and DRIF elements in lattice
 * verification.
 * 03-nov-2006, PT:
 * bugfix:  correction to logic for finding the center
 * S position of a long girder in GetTotalOffsetXfrms.
 * 26-jun-2006, PT:
 * bugfix:  multipole magnet power supplies not handled
 * correctly in RmatCalculate.
 * 08-mar-2006, PT:
 * support for coupled Twiss calculations and solenoids.
 * 10-feb-2006, PT:
 * support for COORD element type.
 * 06-dec-2005, PT:
 * support for synchrotron radiation; x/y correctors now
 * produce dispersion in twiss/Rmat operation.
 * 18-oct-2005, PT:
 * support for BPM scale factors.
 * 30-sep-2005, PT:
 * support for transverse cavities (TCAVs).
 * 29-sep-2005, PT:
 * support for multiple power supplies and error factors for
 * a sector bend magnet.
 * 16-sep-2005, PT:
 * change handling of sector bend transport -- use the new
 * GetLucretiaSBendMap function rather than the old
 * GetMADSBendMap.
 * 09-Aug-2005, PT:
 * bugfix to CheckGirderMoverPars and GetTotalOffsetXfrms:
 * tracking would crash if a mover with roll DOF was specified
 * due to an indexing problem in CheckGirderMoverPars coupled
 * with an improperly-handled return status (==2) in the calling
 * routine (GetTotalOffsetXfrms).
 */

#include "LucretiaMatlab.h" // Calls LucretiaCommon.h
//#include "LucretiaCommon.h"       /* data & prototypes for this file + CUDA defs*/

#include "LucretiaDictionary.h"   /* dictionary data */
#include "LucretiaPhysics.h"      /* data & prototypes for physics module */
#include "LucretiaGlobalAccess.h" /* data & prototypes for global var access */
#include "LucretiaVersionProto.h" /* prototypes of version functions */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>  
#include "matrix.h"
/* File-scoped variables: */

#ifdef __CUDACC__
/* rng CUDA variables */
unsigned long long *rSeed = NULL ; /* rng seed sourced from Matlab workspace */
curandState *rngStates = NULL ; /* device generated random number state */
int blocksPerGrid = threadsPerBlock ; /* number of blocks per CUDA grid */
#endif

char LucretiaCommonVers[] = "LucretiaCommon Version = 02-August-2007" ;
Rmat Identity ={
  {1,0,0,0,0,0},
  {0,1,0,0,0,0},
  {0,0,1,0,0,0},
  {0,0,0,1,0,0},
  {0,0,0,0,1,0},
  {0,0,0,0,0,1}
}; /* identity matrix */
double T5xxQuad[10] ;              /* T5xx elements from normal F quadrupole:
 * in order 11,12,13,14,
 * 22,23,24,
 * 33,34,
 * 44 */

/*=====================================================================*/

/* Return the version date of this file to a caller somewhere else
 *
 * RET:   pointer to string LucretiaCommonVers.
 * ABORT: never.
 * FAIL:  never.                                                       */

char* LucretiaCommonVersion( )
{
  return LucretiaCommonVers ;
}

/*=====================================================================*/

/* Perform calculations related to R-matrices, including returning of
 * R-mats and of Twiss parameters */

/* RET:   one of the following:
 * -> a pointer to an Rmat, which contains the R-matrix for the
 * region in question.
 * -> a pointer to an Rstruc, which contains R-matrices for each
 * element in the region of interest.
 * -> a pointer to a Twiss, which contains twiss parameters for
 * each point in the region of interest.
 * -> a pointer to a double matrix, which contains coupled twiss
 * parameters for each point in the region of interest.
 * -> a NULL pointer if a dynamic memory allocation failed.
 * Furthermore, the following conditions will cause an error:
 * -> an ill-constructed element is detected.  This can happen if a
 * critical field is missing or corrupted (ie, improper type).  For
 * all elements the S position, energy, and Class string are critical
 * (energy must be > 0).  For quads, L, B, Tilt are critical (L must
 * be > 0).
 * -> A magnet points at a power supply which has a missing or corrupted
 * Ampl field.
 * -> A twiss propagation fails (there are several possible causes of
 * such a problem).
 * In GetTwiss or RmatAtoB, any of these will cause an immediate exit
 * with Args->Status = 0 (fatal error).  GetRmats will continue but
 * with Args->Status = -1 on exit.
 * FAIL:  never.                                                      */
void* RmatCalculate( struct RmatArgStruc* Args )
{
  /* local variables */
  int count ;                         /* loop variable */
  int start, stop, step ;             /* loop boundaries and direction */
  int count2 ;                        /* for return indexing */
  int nmat ;                          /* number of steps */
  Rmat Relem ;                        /* for capturing R from physics routines */
  static Rmat Rtot ;                  /* A-to-B rmat */
  static struct Rstruc Reach ;        /* structure for each Rij */
  static struct twiss Tall ;          /* structure for all twiss */
  static struct Ctwiss Call ;         /* structure for all coupled twiss */
  
  double *L, *Edes ;                  /* general parameters */
  double *S ;                         /* S position */
  double Bfull, Lfull=0 ;               /* local vars */
  double Egain=0;                       /* momentum gain of element */
  int PS, PS2 ;                       /* x-referencing pointers */
  int ReturnEach ;                    /* A-to-B or each RMAT? */
  int PropagateTwiss ;                /* do we propagate twiss? */
  int stat1 ;                         /* general status indicator */
  char* ElemClass ;                   /* What kind of element is it? */
  double BScale,GScale ;              /* scale factors for sector bends */
  int iplane ;                        /* counter for Wolski dimensions */
  int i ;                             /* general purpose counting */
  
  /* and now for some flags which indicate some sort of problem:*/
  
  int BadAllocate = 0 ;
  int TwissStat ;
  
  /* start by unpacking argument information as needed */
  
  start          = Args->start ;
  stop           = Args->end ;
  ReturnEach     = Args->ReturnEach ;
  PropagateTwiss = Args->PropagateTwiss ;
  
  /* since we no longer permit backwards looping, the number of matrices and the
   * direction are easy to determine */
  
  step = 1 ;
  nmat = stop-start + 1 ;
  
  /* nullify some pointers so that if allocation occurs it will be obvious
   * which ones are allocated and which are not */
  
  Reach.matrix = NULL ;
  Tall.E = NULL ;
  Tall.S = NULL ;
  Tall.TwissPars = NULL ;
  Call.E = NULL ;
  Call.S = NULL ;
  Call.Twiss = NULL ;
  
  /* Are we doing an A-to-B or returning each one?  Do appropriate setup and
   * allocation now.  If any allocation fails, go to the exit without adding
   * a message to the pile (the calling routine will add the message) */
  
  if (ReturnEach == 1) /* allocate RMATs */
  {
    Reach.matrix = (double*)calloc(nmat*36,sizeof(double)) ;
    if (Reach.matrix == NULL)
    {
      BadAllocate = 1 ;
      Args->Status = 0 ;
      goto egress ;
    }
  }
  else                 /* initialize Rtot to the identity */
    RmatCopy(Identity,Rtot) ;
  
  if (PropagateTwiss == 1)
  {
    Tall.nentry = 0 ;
    Tall.S     = (double *)calloc(nmat+1,sizeof(double)) ;
    Tall.E     = (double *)calloc(nmat+1,sizeof(double)) ;
    Tall.TwissPars = (struct beta0*)calloc(nmat+1,sizeof(struct beta0)) ;
    if ( (Tall.S == NULL)    || (Tall.E == NULL)     ||
            (Tall.TwissPars == NULL)                       )
    {
      BadAllocate = 1 ;
      Args->Status = 0 ;
      goto egress ;
    }
    /* now initialize the first entry in each vector in Tall, except for S and E */
    
    Tall.TwissPars[0].betx  = Args->InitialTwiss.betx ;
    Tall.TwissPars[0].alfx  = Args->InitialTwiss.alfx ;
    Tall.TwissPars[0].etax  = Args->InitialTwiss.etax ;
    Tall.TwissPars[0].etapx = Args->InitialTwiss.etapx ;
    Tall.TwissPars[0].nux   = Args->InitialTwiss.nux ;
    
    Tall.TwissPars[0].bety  = Args->InitialTwiss.bety ;
    Tall.TwissPars[0].alfy  = Args->InitialTwiss.alfy ;
    Tall.TwissPars[0].etay  = Args->InitialTwiss.etay ;
    Tall.TwissPars[0].etapy = Args->InitialTwiss.etapy ;
    Tall.TwissPars[0].nuy   = Args->InitialTwiss.nuy ;
    
  }
  
  if (PropagateTwiss == 2) /* coupled Twiss */
  {
    Call.nentry = 0 ;
    Call.S     = (double *)calloc(nmat+1,sizeof(double)) ;
    Call.E     = (double *)calloc(nmat+1,sizeof(double)) ;
    Call.Twiss = (double *)calloc( (nmat+1)*6*6*Args->nWolskiDimensions,
            sizeof(double)) ;
    if ( (Call.S == NULL)    || (Call.E == NULL)     ||
            (Call.Twiss == NULL)                       )
    {
      BadAllocate = 1 ;
      Args->Status = 0 ;
      goto egress ;
    }
    
    /* initialize the first matrix in Call.twiss from the initial values */
    
    for (i=0 ; i<36*Args->nWolskiDimensions ; i++)
      Call.Twiss[i] = Args->InitialWolski[i] ;
    
  }
  
  /* without any further ado, begin the loop over elements,
   * remembering that while the elements are indexed from 1..N in the calling
   * space, they are from 0..N-1 here. */
  
  count2 = 0 ;
  count = start-1-step ;
  
  /* initialize good status */
  
  stat1 = 1 ;
  
  do
  {
    count += step ;
    count2++ ;
    
    /* If a bad element is detected, add a bad element message to the pile.
     * If we are doing a GetRmats, continue with status = -1, otherwise end
     * the loop with status = 0. */
    /* get the element class out of the BEAMLINE array */
    
    ElemClass = GetElemClass( count ) ;
    if (ElemClass == NULL)
    {
      BadElementMessage( count+1 ) ;
      stat1 = 0 ;
      goto HandleStat1 ;
    }
    
    /* get the momentum and make sure it's not zero; get the S position */
    
    Edes = GetElemNumericPar(count,"P", NULL) ;
    Egain = 0. ;
    S    = GetElemNumericPar(count,"S", NULL) ;
    if ( (Edes == NULL) || (S == NULL) )
    {
      BadElementMessage( count+1 ) ;
      stat1 = 0 ;
      goto HandleStat1 ;
    }
    if (*Edes <= 0.)
    {
      BadElementMessage( count+1 ) ;
      stat1 = 0 ;
      goto HandleStat1 ;
    }
    
    /* depending on what the element class is, do different things: */
    
    if (strcmp(ElemClass,"MARK")==0) /* marker */
    {
      RmatCopy(Identity,Relem) ;    /* only zero-length element right now */
      Lfull = 0. ;
    }
    
    else if (strcmp(ElemClass,"QUAD")==0) /* quadrupole */
    {
      stat1 = GetDatabaseParameters( count, nQuadPar,
              QuadPar, RmatPars, ElementTable ) ;
      Lfull = GetDBValue( QuadPar+QuadL ) ;
      if ( (stat1 == 0) || (Lfull <= 0) )
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      Bfull = GetDBValue(QuadPar+QuadB) ;
      Lfull = GetDBValue(QuadPar+QuadL) ;
      if (Lfull <= 0.)
      {
        BadElementMessage( count+1 ) ;
        stat1 = 0 ;
        goto HandleStat1 ;
      }
      
      /* see if it has a power supply.  If so, get its amplitude.  If the
       * amplitude is corrupted, set a bad status and goto egress, otherwise
       * update the magnet B-field setting */
      
      PS = (int)(GetDBValue(QuadPar+QuadPS)) ;
      if (PS != 0)
      {
        stat1 = GetDatabaseParameters( PS-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS, count+1 ) ;
          goto HandleStat1 ;
        }
        Bfull *= GetDBValue(PSPar+PSAmpl) ;
      }
      
      GetQuadMap( *(QuadPar[QuadL].ValuePtr),
              Bfull,
              *(QuadPar[QuadTilt].ValuePtr),
              *(QuadPar[QuadP].ValuePtr),
              Relem, T5xxQuad         ) ;
      
    }
    
    else if (strcmp(ElemClass,"PLENS")==0) /* Plasma lens */
    {
      stat1 = GetDatabaseParameters( count, nQuadPar,
              QuadPar, RmatPars, ElementTable ) ;
      Lfull = GetDBValue( QuadPar+QuadL ) ;
      if ( (stat1 == 0) || (Lfull <= 0) )
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      Bfull = GetDBValue(QuadPar+QuadB) ;
      Lfull = GetDBValue(QuadPar+QuadL) ;
      if (Lfull <= 0.)
      {
        BadElementMessage( count+1 ) ;
        stat1 = 0 ;
        goto HandleStat1 ;
      }
      
      /* see if it has a power supply.  If so, get its amplitude.  If the
       * amplitude is corrupted, set a bad status and goto egress, otherwise
       * update the magnet B-field setting */
      
      PS = (int)(GetDBValue(QuadPar+QuadPS)) ;
      if (PS != 0)
      {
        stat1 = GetDatabaseParameters( PS-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS, count+1 ) ;
          goto HandleStat1 ;
        }
        Bfull *= GetDBValue(PSPar+PSAmpl) ;
      }
      GetPlensMap( *(QuadPar[QuadL].ValuePtr),
              Bfull,
              *(QuadPar[QuadP].ValuePtr),
              Relem, T5xxQuad         ) ;
      
    }
    
    else if (strcmp(ElemClass,"SOLENOID")==0) /* solenoid */
    {
      stat1 = GetDatabaseParameters( count, nSolePar,
              SolePar, RmatPars, ElementTable ) ;
      Lfull = GetDBValue( SolePar+SoleL ) ;
      if ( (stat1 == 0) || (Lfull <= 0) )
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      Bfull = GetDBValue(SolePar+SoleB) ;
      Lfull = GetDBValue(SolePar+SoleL) ;
      if (Lfull <= 0.)
      {
        BadElementMessage( count+1 ) ;
        stat1 = 0 ;
        goto HandleStat1 ;
      }
      
      /* see if it has a power supply.  If so, get its amplitude.  If the
       * amplitude is corrupted, set a bad status and goto egress, otherwise
       * update the magnet B-field setting */
      
      PS = (int)(GetDBValue(SolePar+SolePS)) ;
      if (PS != 0)
      {
        stat1 = GetDatabaseParameters( PS-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS, count+1 ) ;
          goto HandleStat1 ;
        }
        Bfull *= GetDBValue(PSPar+PSAmpl) ;
      }
      
      GetSolenoidMap( *(SolePar[QuadL].ValuePtr),
              Bfull,
              *(SolePar[QuadP].ValuePtr),
              Relem, T5xxQuad         ) ;
      
    }
    
    else if (strcmp(ElemClass,"MULT")==0) /* multipole */
    {
      int pole ;
      RmatCopy(Identity,Relem) ;
      stat1 = GetDatabaseParameters( count, nMultPar,
              MultPar, RmatPars, ElementTable ) ;
      Lfull = GetDBValue( MultPar+MultL ) ;
      Relem[0][1] = Lfull ;
      Relem[2][3] = Lfull ;
      if (stat1 == 0)
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      
      /* see if it has a power supply.  If so, get its amplitude.  If the
       * amplitude is corrupted, set a bad status and goto egress, otherwise
       * update the magnet B-field setting */
      
      PS = (int)(GetDBValue(MultPar+MultPS)) ;
      if (PS != 0)
      {
        stat1 = GetDatabaseParameters( PS-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS, count+1 ) ;
          goto HandleStat1 ;
        }
        Bfull = GetDBValue(PSPar+PSAmpl) ;
      }
      else
        Bfull = 1. ;
      
      /* see if it has either bending or quadrupole fields, if so put them into the
       * matrix */
      
      for (pole=0 ; pole<MultPar[MultPoleIndex].Length ; pole++)
      {
        if (MultPar[MultPoleIndex].ValuePtr[pole] == 0) /* bend */
        {
          double cst = cos(MultPar[MultTilt].ValuePtr[pole]) ;
          double snt = sin(MultPar[MultTilt].ValuePtr[pole]) ;
          Relem[1][5] = MultPar[MultB].ValuePtr[pole] * cst * Bfull
                  / *Edes / GEV2TM ;
          Relem[3][5] = MultPar[MultB].ValuePtr[pole] * snt * Bfull
                  / *Edes / GEV2TM ;
          Relem[0][5] = Relem[1][5] * Lfull / 2 ;
          Relem[2][5] = Relem[3][5] * Lfull / 2 ;
        }
        else if (MultPar[MultPoleIndex].ValuePtr[pole] == 1) /* quad */
        {
          double cst = cos(2*MultPar[MultTilt].ValuePtr[pole]) ;
          double snt = sin(2*MultPar[MultTilt].ValuePtr[pole]) ;
          double Bq = MultPar[MultB].ValuePtr[pole] * Bfull ;
          Relem[1][0] = -Bq * cst / *Edes / GEV2TM ;
          Relem[3][2] =  Bq * cst / *Edes / GEV2TM ;
          Relem[1][2] = -Bq * snt / *Edes / GEV2TM ;
          Relem[3][0] = -Bq * snt / *Edes / GEV2TM ;
        }
      }
    }
    
    else if (strcmp(ElemClass,"SBEN")==0) /* sector bend */
    {
      double intB, intG;
      double E1, E2, H1, H2 ;
      double Hgap, Hgapx, Fint, Fintx ;
      double Theta ;
      Rmat Rface1, Rface2, Rbody, Rtemp ;
      
      stat1 = GetDatabaseParameters( count, nSBendPar,
              SBendPar, RmatPars, ElementTable ) ;
      if ( (stat1 == 0) || (GetDBValue(SBendPar+SBendL)<=0) )
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      intB = GetDBValue(SBendPar+SBendB) ;
      if (SBendPar[SBendB].Length > 1)
        intG = *(SBendPar[SBendB].ValuePtr+1) ;
      else
        intG = 0 ;
      Theta = GetDBValue(SBendPar+SBendAngle) ;
      Lfull = GetDBValue(SBendPar+SBendL) ;
      if (Lfull <= 0)
      {
        BadElementMessage( count+1 ) ;
        stat1 = 0 ;
        goto HandleStat1 ;
      }
      
      /* see if it has a power supply.  If so, get its amplitude.  If the
       * amplitude is corrupted, set a bad status and goto egress, otherwise
       * update the magnet B-field setting */
      
      PS = (int)(GetDBValue(SBendPar+SBendPS)) ;
      if (SBendPar[SBendPS].Length > 1)
        PS2 = (int)(*(SBendPar[SBendPS].ValuePtr+1)) ;
      else
        PS2 = PS ;
      
      BScale = 1.0 ;
      GScale = 1.0 ;
      if (PS > 0)
      {
        stat1 = GetDatabaseParameters( PS-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS, count+1 ) ;
          goto HandleStat1 ;
        }
        /*		     intB *= (GetDBValue(PSPar+PSAmpl)) ;
         * intG *= (GetDBValue(PSPar+PSAmpl)) ;
         */
        BScale = GetDBValue(PSPar+PSAmpl) ;
      }
      
      if (PS2 > 0)
      {
        stat1 = GetDatabaseParameters( PS2-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS2, count+1 ) ;
          goto HandleStat1 ;
        }
        GScale = GetDBValue(PSPar+PSAmpl) ;
      }
      
      intB *= BScale ;
      intG *= GScale ;
      
      /* get parameters related to the upstream / downstream faces */
      
      E1 = GetSpecialSBendPar(&(SBendPar[SBendEdgeAngle]),0) ;
      E2 = GetSpecialSBendPar(&(SBendPar[SBendEdgeAngle]),1) ;
      if ( (cos(E1)==0.) || (cos(E2)==0.) )
      {
        stat1=0 ;
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      
      H1 = GetSpecialSBendPar(&(SBendPar[SBendEdgeCurvature]),0) ;
      H2 = GetSpecialSBendPar(&(SBendPar[SBendEdgeCurvature]),1) ;
      
      Hgap  = GetSpecialSBendPar(&(SBendPar[SBendHGAP]),0) ;
      Hgapx = GetSpecialSBendPar(&(SBendPar[SBendHGAP]),1) ;
      
      Fint  = GetSpecialSBendPar(&(SBendPar[SBendFINT]),0) ;
      Fintx = GetSpecialSBendPar(&(SBendPar[SBendFINT]),1) ;
      
      /* get the map for each face and for the body */
      
      GetBendFringeMap( Lfull, intB, intG, *Edes, E1, H1,
              Hgap, Fint, 1., Rface1, T5xxQuad ) ;
      GetBendFringeMap( Lfull, intB, intG, *Edes, E2, H2,
              Hgapx, Fintx, -1., Rface2, T5xxQuad ) ;
      
      GetLucretiaSBendMap( Lfull, Theta, intB, intG, *Edes, Rbody, NULL, NULL, NULL ) ;
      
      /* construct the total matrix */
      
      RmatProduct( Rbody, Rface1, Rtemp ) ;
      RmatProduct( Rface2, Rtemp, Relem ) ;
      
      /* apply any rotation which is necessary */
      
      RotateRmat( Relem, GetDBValue(SBendPar+SBendTilt) ) ;
      
    }
    
    else if (strcmp(ElemClass,"LCAV") == 0) /* RF structure */
    {
      double voltage, Vsin, phiall, nu ;
      
      stat1 = GetDatabaseParameters( count, nLcavPar,
              LcavPar, RmatPars, ElementTable ) ;
      if (stat1 == 0)
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      voltage = GetDBValue(LcavPar+LcavVolt) ;
      phiall = GetDBValue(LcavPar+LcavPhase) ;
      nu = GetDBValue(LcavPar+LcavFreq) ;
      Lfull = GetDBValue(LcavPar+LcavL) ;
      
      /* get the klystron data for this structure, if any */
      
      PS = (int)(GetDBValue(LcavPar+LcavKlystron)) ;
      if (PS > 0)
      {
        enum KlystronStatus* stat ;
        stat1 = GetDatabaseParameters( PS-1, nKlystronPar,
                KlystronPar, RmatPars, KlystronTable ) ;
        stat  = GetKlystronStatus(PS-1) ;
        if ( (stat1 == 0) || (stat == NULL) )
        {
          BadKlystronMessage( PS, count+1 ) ;
          goto HandleStat1 ;
        }
        if ( (*stat == ON) || (*stat == TRIPPED) )
          voltage *= (GetDBValue(KlystronPar+KlysAmpl))  ;
        else
          voltage = 0. ;
        phiall += GetDBValue(KlystronPar+KlysPhase) ;
      }
      
      /* compute the energy gain and dV/dz of the bunch */
      
      Egain = GetDBValue(LcavPar+LcavEgain)/1000 ;
      Vsin = -voltage*sin(PI*phiall/180) /1000 ;
      
      /* if the element decelerates the beam down to zero, or has zero length,
       * fail out now */
      
      if ( (*Edes + Egain <= 0.) || (Lfull <= 0) )
      {
        BadElementMessage( count+1 ) ;
        stat1 = 0 ;
        goto HandleStat1 ;
      }
      
      /* get the map */
      
      GetLcavMap( Lfull, *Edes, nu, Egain, Vsin, Relem, 3 ) ;
      
    }
    
    /* standard dipole corrector magnet */
    
    else if( (strcmp(ElemClass,"XCOR")==0) ||
            (strcmp(ElemClass,"YCOR")==0)    )
    {
      double ct, st ;
      
      stat1 = GetDatabaseParameters( count, nCorrectorPar,
              CorrectorPar, RmatPars, ElementTable ) ;
      if (stat1 == 0)
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      Lfull = GetDBValue( CorrectorPar+CorL ) ;
      GetDriftMap( Lfull, Relem ) ;
      
      Bfull = GetDBValue( CorrectorPar+CorB ) ;
      ct = cos( GetDBValue( CorrectorPar+CorTilt ) ) ;
      st = sin( GetDBValue( CorrectorPar+CorTilt ) ) ;
      
      /* see if it has a power supply.  If so, get its amplitude.  If the
       * amplitude is corrupted, set a bad status and goto egress, otherwise
       * update the magnet B-field setting */
      
      PS = (int)(GetDBValue(CorrectorPar+CorPS)) ;
      if (PS != 0)
      {
        stat1 = GetDatabaseParameters( PS-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS, count+1 ) ;
          goto HandleStat1 ;
        }
        Bfull *= GetDBValue(PSPar+PSAmpl) ;
      }
      if (strcmp(ElemClass,"XCOR")==0)
      {
        Relem[1][5] = -ct * Bfull / *Edes / GEV2TM ;
        Relem[3][5] = -st * Bfull / *Edes / GEV2TM ;
      }
      else
      {
        Relem[1][5] =  st * Bfull / *Edes / GEV2TM ;
        Relem[3][5] = -ct * Bfull / *Edes / GEV2TM ;
      }
      Relem[0][5] = Relem[1][5] * Lfull / 2 ;
      Relem[2][5] = Relem[3][5] * Lfull / 2 ;
    }
    
    /* combined-function dipole corrector magnet */
    
    else if(strcmp(ElemClass,"XYCOR")==0)
    {
      double ct, st ;
      double B1Full, B2Full ;
      int PS1 ;
      
      stat1 = GetDatabaseParameters( count, nCorrectorPar,
              XYCorrectorPar, RmatPars, ElementTable ) ;
      if (stat1 == 0)
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      Lfull = GetDBValue( XYCorrectorPar+CorL ) ;
      GetDriftMap( Lfull, Relem ) ;
      
      B1Full = GetDBValue(XYCorrectorPar+CorB) ;
      B2Full = *(XYCorrectorPar[CorB].ValuePtr+1) ;
      ct = cos( GetDBValue( XYCorrectorPar+CorTilt ) ) ;
      st = sin( GetDBValue( XYCorrectorPar+CorTilt ) ) ;
      
      /* see if it has power supplies.  If so, get their amplitude.  If the
       * amplitudes are corrupted, set a bad status and goto egress, otherwise
       * update the magnet B-field setting */
      
      PS1 = (int)(GetDBValue(XYCorrectorPar+CorPS)) ;
      PS2 = (int)(*(XYCorrectorPar[CorPS].ValuePtr+1)) ;
      if (PS1 != 0)
      {
        stat1 = GetDatabaseParameters( PS1-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS1, count+1 ) ;
          goto HandleStat1 ;
        }
        B1Full *= GetDBValue(PSPar+PSAmpl) ;
      }
      if (PS2 != 0)
      {
        stat1 = GetDatabaseParameters( PS2-1, nPSPar, PSPar, RmatPars,
                PSTable ) ;
        if (stat1 == 0)
        {
          BadPSMessage( PS2, count+1 ) ;
          goto HandleStat1 ;
        }
        B2Full *= GetDBValue(PSPar+PSAmpl) ;
      }
      Relem[1][5] = (-ct * B1Full + st * B2Full) / *Edes / GEV2TM ;
      Relem[3][5] = (-st * B1Full - ct * B2Full) / *Edes / GEV2TM ;
      Relem[0][5] = Relem[1][5] * Lfull / 2 ;
      Relem[2][5] = Relem[3][5] * Lfull / 2 ;
    }
    
    
    /* change in coordinates */
    
    else if (strcmp(ElemClass,"COORD")==0)
    {
      double dx[6] ;
      
      stat1 = GetDatabaseParameters( count, nCoordPar,
              CoordPar, RmatPars, ElementTable ) ;
      if (stat1 == 0)
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
      stat1 = GetCoordMap( CoordPar[CoordChange].ValuePtr, dx, Relem ) ;
      if (stat1 == 0)
      {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
    }
    
    /* TMap element -> just pull off the R matrix data */
    else if (strcmp(ElemClass,"TMAP")==0)
    {
      stat1 = TMapGetDataR(count, Relem) ;
      if (stat1==0) {
        BadElementMessage( count+1 ) ;
        goto HandleStat1 ;
      }
    }
    
    /* if we made it this far it's something other than a marker or a quad,
     * so we can treat it as a drift element for now... */
    
    
    else
    {
      L = GetElemNumericPar( count, "L", NULL ) ;
      if (L==NULL)
        Lfull = 0. ;
      else
        Lfull = *L ;
      GetDriftMap( Lfull, Relem ) ;
    }
    
    /* Now we can either copy the Relem onto the structure of individual element
     * rmats, or else matrix-multiply it onto the A-to-B rmat */
    
    if (ReturnEach == 1)
    {
      RmatCopy(Relem,(double (*)[6])Reach.matrix+36*(count2-1) )  ;
    }
    else if (PropagateTwiss == 0)
      RmatProduct(Relem,Rtot,Rtot) ;
    
    /* if we are doing twiss propagation, put the energy and S position into the
     * previous element's entries...*/
    
    if (PropagateTwiss == 1)
    {
      Tall.S[count2-1] = *S    ;
      Tall.E[count2-1] = *Edes ;
      
      /* ...and propagate the twiss through the total matrix */
      
      TwissStat = TwissThruRmat( Relem, Tall.TwissPars+count2-1,
              Tall.TwissPars+count2    ) ;
      if (TwissStat == 0)
      {
        BadTwissMessage( count+1 ) ;
        Args->Status = 0 ;
        goto egress ;
      }
      Tall.nentry = count2 ;
    }
    
    /* if we are doing coupled twiss propagation, put the energy and S position into the
     * previous element's entries...*/
    
    if (PropagateTwiss == 2)
    {
      Call.S[count2-1] = *S    ;
      Call.E[count2-1] = *Edes ;
      
      /* ...and propagate the twiss through the total matrix, bearing in mind that
       * Matlab has a different ordering for its indices than C so we need to
       * set the transpose flag; also we need to do this once for each dimension
       * which is included in the propagation */
      
      for (iplane=0 ; iplane<Args->nWolskiDimensions ; iplane++)
      {
        int oldstart = 6*6*(Args->nWolskiDimensions*(count2-1)+iplane) ;
        int newstart = 6*6*(Args->nWolskiDimensions*count2+iplane) ;
        
        TwissStat = CoupledTwissThruRmat( Relem, &(Call.Twiss[oldstart]),
                &(Call.Twiss[newstart]), 1    ) ;
        if (TwissStat == 0)
        {
          BadTwissMessage( count+1 ) ;
          Args->Status = 0 ;
          goto egress ;
        }
      }
      Call.nentry = count2 ;
    }
    
    /* if a bad element was detected on the last iteration, we may need to
     * address it now */
    
    HandleStat1:
      
      if (stat1 != 1) {
        if (!ReturnEach) /* RmatAtoB or GetTwiss: fatal error */
        {
          Args->Status = 0 ;
          goto egress ;
        }
        else /* GetRmats: one missing matrix != fatal error */
        {
          Args->Status = -1 ;
          stat1 = 1 ;
        }
      }
      
      
  } while ( count2 < nmat) ;
  
  /* end of loop over elements. */
  
  /* egress point:  check to see whether some disaster befell us during
   * execution.  If so, handle it. While on the topic, deallocate any
   * locally allocated memory. */
  
  egress:
    
    
    /* if we got here with bad status due to bad dynamic allocation, free
     * anything which was successfully allocated and return a null pointer */
    
    if (BadAllocate != 0)
    {
      if (Reach.matrix != NULL)
        free(Reach.matrix) ;
      if (Tall.S != NULL)
        free(Tall.S) ;
      if (Tall.E != NULL)
        free(Tall.E) ;
      if (Tall.TwissPars != NULL)
        free(Tall.TwissPars) ;
      if (Call.S != NULL)
        free(Call.S) ;
      if (Call.E != NULL)
        free(Call.E) ;
      if (Call.Twiss != NULL)
        free(Call.Twiss) ;
      return NULL ;
    }
    
    
    /*	Now we just have to set the return value and
     * exit.  If we are doing an eachelem, return a pointer to Relem; otherwise,
     * return a pointer to Rtot. */
    
    if (ReturnEach == 1)            /* GetRmats operation */
    {
      Reach.Nmatrix = count2 ;
      return &Reach ;
    }
    else if (PropagateTwiss==1)     /* GetTwiss operation */
    {
      Tall.S[Tall.nentry] = Tall.S[Tall.nentry-1] + Lfull ;
      Tall.E[Tall.nentry] = Tall.E[Tall.nentry-1] + Egain ;
      Tall.nentry++ ;
      
      /* if the original request was for backwards-propagation, then
       * we need to adjust the nu parameters */
      
      if (Args->Backwards == 1)
      {
        double nuxmax = Tall.TwissPars[Tall.nentry-1].nux ;
        double nuymax = Tall.TwissPars[Tall.nentry-1].nuy ;
        int nucount ;
        for (nucount=0 ; nucount<Tall.nentry ; nucount++)
        {
          Tall.TwissPars[nucount].nux +=
                  Args->InitialTwiss.nux - nuxmax ;
          Tall.TwissPars[nucount].nuy +=
                  Args->InitialTwiss.nuy - nuymax ;
        }
      }
      
      return &Tall ;
    }
    else if (PropagateTwiss == 2) /* coupled GetTwiss operation */
    {
      Call.S[Call.nentry] = Call.S[Call.nentry-1] + Lfull ;
      Call.E[Call.nentry] = Call.E[Call.nentry-1] + Egain ;
      Call.nentry++ ;
      return &Call ;
    }
    else                            /* RmatAtoB operation */
      return Rtot ;
}

/*=====================================================================*/

/* Here are some variables which need to be preserved from one call to
 * another of the tracker DLL, and which also need to be accessed by
 * both the main tracker and the dynamic-allocation deleter.  Thus we
 * put them into the file before the tracking function */

int nBunchInBeamOld=0 ;            /* total beam depth on last call */
int FirstBunchOld=0 ;              /* which bunches tracked on last */
int LastBunchOld=-1 ;				  /*   call                        */
int nElemOld=0 ;                   /* size of BEAMLINE on last call */
int dS ;
int maxNmode ;
int maxNmodeOld = 0 ;

/* backbones for BPM, instrument, and SBPM data structures */

struct BPMdat**  bpmdata  = NULL ;
struct INSTdat** instdata = NULL ;
struct SBPMdat** sbpmdata = NULL ;

/* backbone for TLRFreq and TLRErrFreq data */

struct LRWFFreqKick** TLRFreqKickDat    = NULL ;
struct LRWFFreqKick** TLRErrFreqKickDat = NULL ;

/* backbones for the first and last bunches tracked at each RF unit */

int* FirstBunchAtRF = NULL ;
int* LastBunchAtRF  = NULL ;

/* data structure to hold pointers to real data in Matlab cell arrays */

struct LRWFFreqData* TLRFreqData    ;
struct LRWFFreqData* TLRErrFreqData ;

/* number of wakefields of each type */

int* numwakes ;

int FirstBPMElemno ;               /* element # of first BPM in line */
int BPMElemnoLastCall ;            /* utility for BPM tracking */
int FirstInstElemno ;
int InstElemnoLastCall ;
int FirstSBPMElemno ;
int SBPMElemnoLastCall ;

/* here is a variable which needs to be seen by several functions, so
 * it has to have file scope */
int* StoppedParticles = NULL;
#ifdef __CUDACC__
int* StoppedParticles_gpu = NULL;
#endif

/*=====================================================================*/

/* Main procedure for looping through a beamline and tracking a beam
 * through all of the elements, generating and saving data on BPM
 * readings etc along the way. */

/* RET:    none.  However, TrackArgs->Status is set as follows:
 * +1 -> All OK.
 * +2 -> failure before tracking attempted
 * 0 -> failure during tracking
 * -1 -> one or more particles stopped during tracking.
 * ABORT:
 * FAIL:         */

void TrackThruMain( struct TrackArgsStruc* TrackArgs )
{
  
  int OuterLoop, OuterStart, OuterStop, OuterStep ;
  int InnerLoop, InnerStart, InnerStop, InnerStep ;
  int* ElemLoop;
  int* BunchLoop ;
  int LastLoopElem = -1 ;
  int* TFlag=NULL ;
  char* ElemClass=NULL ;
  int TrackStatus ;     /* status returned from functions */
  int GlobalStatus = 1; /* overall status of this function */
  int NewStopped = 0 ;
  int *momStat=NULL ;

#ifndef LUCRETIA_MLRAND
  /* If not using MATLAB calls to get random number then intialize the C random number generator here from Matlab */
  srand( getLucretiaRandSeedC() ) ;
#endif
  
#ifdef __CUDACC__
  int maxpart=0, ib;
  int dosr,iele ;
  /* Get random number seed from Matlab workspace */
  rSeed = (unsigned long long*) calloc(1,sizeof(unsigned long long)) ;
  getLucretiaRandSeed(rSeed) ;
  
  /* Get Max particles in any bunch of the beam */
  for (ib=0; ib<TrackArgs->TheBeam->nBunch; ib++) {
    if (TrackArgs->TheBeam->bunches[ib]->nray>maxpart)
      maxpart=TrackArgs->TheBeam->bunches[ib]->nray;
  }
  
  /* Define blocks per grid for CUDA computations */
  blocksPerGrid = (maxpart + threadsPerBlock - 1) / threadsPerBlock;
  
  /* Make a GPU copy of the TrackFlags integer array */
  int* TFlag_gpu ;
  gpuErrchk( cudaMalloc(&TFlag_gpu, sizeof(int)*(NUM_TRACK_FLAGS+1)) );
  
  /* CUDA rng states*/
  gpuErrchk( cudaMalloc((void **)&rngStates, blocksPerGrid * threadsPerBlock *
          sizeof(curandState)) );
  /* If no sync radiation selected anywhere, then just initialise with zeroes */
  dosr=0;
  for (iele=0; iele<TrackArgs->LastElem; iele++)
  {
    TFlag = GetTrackingFlags( iele ) ;
    if (TFlag[SynRad] > SR_None)
    {
      dosr=1;
      break;
    }
  }
  if (dosr==0) {
    gpuErrchk( cudaMemset(rngStates,0,blocksPerGrid*threadsPerBlock*sizeof(curandState)) );
  }
  else { /* make rng states for each thread*/
    rngSetup_kernel<<<blocksPerGrid, threadsPerBlock>>>(rngStates, *rSeed); 
    gpuErrchk( cudaGetLastError() ) ;
  }
#endif
  
  /* initialize the pointers to freq-domain wakefield data */
  TLRFreqData    = NULL ;
  TLRErrFreqData = NULL ;
  
  /* initialize StoppedParticles to the "none stopped" state */
  StoppedParticles = (int*) calloc(1,sizeof(int)) ;
  *StoppedParticles = 0 ;
  
  /* Map StoppedParticles int onto both host and device memory for CUDA*/
#ifdef __CUDACC__
  //gpuErrchk( cudaSetDeviceFlags(cudaDeviceMapHost) );
  gpuErrchk( cudaHostAlloc( (void**) &StoppedParticles, sizeof(int), cudaHostAllocMapped ) );
  gpuErrchk( cudaHostGetDevicePointer ( &StoppedParticles_gpu, StoppedParticles, 0 ) );
#endif
  
  /* If the total number of elements in the beamline has not changed from
   * the last call, AND the total number of bunches in the beam has not
   * changed, AND the data backbones are allocated, then we do not need to
   * reallocate them.  Otherwise we need to clear them and reallocate */
  
  
  
  if ( (bpmdata == NULL) ||                  /* already allocated */
          (nElemOld != nElemInBeamline( )) )  /* # elts unchanged  */
    
  {
    ClearTrackingVars( ) ;
    nElemOld = nElemInBeamline( ) ;
    nBunchInBeamOld = TrackArgs->TheBeam->nBunch ;
    bpmdata  = (struct BPMdat**)calloc(nElemOld,sizeof(struct BPMdata*)) ;
    instdata = (struct INSTdat**)calloc(nElemOld,sizeof(struct INSTdata*)) ;
    sbpmdata = (struct SBPMdat**)calloc(nElemOld,sizeof(struct SBPMdata*)) ;
    TLRFreqKickDat = (struct LRWFFreqKick **) calloc(nElemOld,sizeof(struct LRWFFreqKick*)) ;
    TLRErrFreqKickDat = (struct LRWFFreqKick **) calloc(nElemOld,sizeof(struct LRWFFreqKick*)) ;
    FirstBunchAtRF = (int *) calloc(nElemOld,sizeof(int)) ;
    LastBunchAtRF = (int *) calloc(nElemOld,sizeof(int)) ;
    if ( (bpmdata==NULL)||(instdata==NULL)||(sbpmdata==NULL)
    ||(TLRFreqKickDat==NULL)||(TLRErrFreqKickDat==NULL)
    ||(FirstBunchAtRF==NULL)||(FirstBunchAtRF==NULL)             )
    {
      GlobalStatus = 2 ;
      AddMessage(
              "Unable to allocate data backbones in TrackThruMain",1) ;
      goto egress ;
    }
    
    /* initialize the First/LastBunchAtRF array to -1, not zero */
    
    for (OuterLoop = 0 ; OuterLoop < nElemOld ; OuterLoop++)
    {
      FirstBunchAtRF[OuterLoop] = -1 ;
      LastBunchAtRF[OuterLoop] = -1 ;
    }
    
  }
  
  /* point the TrackArgs pointers at the newly-allocated backbones */
  
  TrackArgs->bpmdata  = bpmdata ;
  TrackArgs->instdata = instdata ;
  TrackArgs->sbpmdata = sbpmdata ;
  
  /* initialize variables used by the BPM tracker to indicate that no BPM
   * tracking has yet occurred */
  
  FirstBPMElemno = -1 ;
  BPMElemnoLastCall = -1 ;
  FirstInstElemno = -1 ;
  InstElemnoLastCall = -1 ;
  FirstSBPMElemno = -1 ;
  SBPMElemnoLastCall = -1 ;
  TrackArgs->nBPM = 0 ;
  TrackArgs->nINST = 0 ;
  TrackArgs->nSBPM = 0 ;
  maxNmode = 0 ;
  
  dS = 1 ;
  
  
  /* get the total # of wakes of all types */
  
  numwakes = GetNumWakes( ) ;
  
  /* get pointers to the frequency-domain LRWFs and error LRWFs */
  
  if ( numwakes[2] > 0)
  {
    TLRFreqData = UnpackLRWFFreqData(  numwakes[2], TLRTable,
            &maxNmode, &GlobalStatus ) ;
    if (GlobalStatus != 1)
    {
      GlobalStatus = 2 ;
      goto egress ;
    }
  }
  
  if ( numwakes[3] > 0)
  {
    TLRErrFreqData = UnpackLRWFFreqData(  numwakes[3], TLRErrTable,
            &maxNmode, &GlobalStatus ) ;
    if (GlobalStatus != 1)
    {
      GlobalStatus = 2 ;
      goto egress ;
    }
  }
  
  /* if the maximum number of modes in the present implementation is different
   * from the max number from the last track operation, then all of the
   * existing vectors of LRWF frequency-mode kicks are potentially out of date.
   * In that eventuality, clear them now */
  
  if (maxNmode != maxNmodeOld)
  {
    ClearOldLRWFFreqKicks( 0 ) ;
    ClearOldLRWFFreqKicks( 1 ) ;
    if (maxNmodeOld != 0)
      AddMessage("Change in max # of LRWF modes, old kick data discarded",0) ;
  }
  maxNmodeOld = maxNmode ;
  
  /* Depending on how the user wants to do things, we either want to have
   * an outer loop of elements and an inner loop over bunches (more
   * efficient) or an outer loop over bunches and an inner loop over
   * elements (more flexible).  Set that up now. */
  
  if (TrackArgs->BunchwiseTracking == 0) /* default */
  {
    OuterStart = TrackArgs->FirstElem-1 ;
    OuterStop  = TrackArgs->LastElem-1 ;
    OuterStep = 1 ;
    InnerStart = TrackArgs->FirstBunch-1 ;
    InnerStop  = TrackArgs->LastBunch-1 ;
    InnerStep  = 1;
    ElemLoop  = &OuterLoop ;   /* tricky way to make sure that the two */
    BunchLoop = &InnerLoop ;   /* loop vars are correctly configured */
  }
  else
  {
    InnerStart = TrackArgs->FirstElem-1 ;
    InnerStop  = TrackArgs->LastElem-1 ;
    InnerStep = 1 ;
    OuterStart = TrackArgs->FirstBunch-1 ;
    OuterStop  = TrackArgs->LastBunch-1 ;
    OuterStep  = 1;
    ElemLoop  = &InnerLoop ;
    BunchLoop = &OuterLoop ;
  }
  
  /* Check to see whether there are unstopped particles coming in with
   * bad momenta (total or transverse), raise a warning if so. */
  int bloop, RayLoop ;
  struct Bunch* TheBunch ;
  /* loop over bunches */
  for (bloop = TrackArgs->FirstBunch-1 ; bloop < TrackArgs->LastBunch ; bloop++)
  {
    TheBunch = TrackArgs->TheBeam->bunches[bloop] ;
    momStat = (int*) calloc(TheBunch->nray,sizeof(int)) ;
    memset(momStat, 0, sizeof(int)*TheBunch->nray) ;
#ifdef __CUDACC__
    int *momStat_gpu ;
    gpuErrchk( cudaMalloc(&momStat_gpu, sizeof(int)*TheBunch->nray) );
    gpuErrchk( cudaMemset(momStat_gpu, 0, sizeof(int)*TheBunch->nray) );
    InitialMomentumCheck<<<blocksPerGrid,threadsPerBlock>>>(momStat_gpu,TheBunch->stop,TheBunch->ngoodray_gpu,TheBunch->x,TheBunch->y,
            TheBunch->nray, StoppedParticles_gpu ) ;
    gpuErrchk( cudaGetLastError() );
    gpuErrchk( cudaMemcpy(momStat,momStat_gpu,sizeof(int)*TheBunch->nray,cudaMemcpyDeviceToHost) );
#else    
    for (RayLoop = 0 ; RayLoop < TheBunch->nray ; RayLoop++)
      InitialMomentumCheck( momStat, TheBunch->stop, &TheBunch->ngoodray, TheBunch->x, TheBunch->y, RayLoop, StoppedParticles ) ;
#endif    
    for (RayLoop = 0 ; RayLoop < TheBunch->nray ; RayLoop++)
    {
      if (momStat[RayLoop] != 0)
      {
        if (*momStat<0)
          BadParticlePperpMessage( 0, bloop+1, abs(*momStat)+1 ) ;
        else
          BadParticleMomentumMessage( 0, bloop+1, *momStat+1 ) ;
        BadInitMomentumMessage( )  ;
        GlobalStatus = 2 ;
        free(momStat) ;
#ifdef __CUDACC__
        gpuErrchk( cudaFree(momStat_gpu) );
#endif        
        goto egress ;
      }
    }
    free(momStat) ;
    #ifdef __CUDACC__
      gpuErrchk( cudaFree(momStat_gpu) );
    #endif 
  } 
  
  /* without further ado, begin the two loops */
  OuterLoop = OuterStart ;
  do
  {
    
    InnerLoop = InnerStart ;
    
    do
    {
      
      /* if the local variable NewStopped and global variable StoppedParticles are
       * not in agreement, it indicates that some rays were stopped since the last
       * trip through the loop.  In which case, issue a message and update the local
       * variable (so that the message is only issued once).  Also update global
       * status */
      
      if (NewStopped != *StoppedParticles)
      {
        GlobalStatus = -1 ;
        AddMessage("Particles stopped during tracking",0) ;
        NewStopped = *StoppedParticles ;
      }
      
      /* if the bunch is out of good particles, issue a message, set a warning,
       * and set a flag on the bunch indicating that the appropriate authorities
       * have been notified. */
#ifdef __CUDACC__
      gpuErrchk( cudaMemcpy(&TrackArgs->TheBeam->bunches[*BunchLoop]->ngoodray,
              TrackArgs->TheBeam->bunches[*BunchLoop]->ngoodray_gpu, sizeof(int), cudaMemcpyDeviceToHost) );
#endif
      if (TrackArgs->TheBeam->bunches[*BunchLoop]->ngoodray <=0)
      {
        if (TrackArgs->TheBeam->bunches[*BunchLoop]->StillTracking==1)
        {
          BunchStopMessage( (*BunchLoop)+1, (*ElemLoop) ) ;
          GlobalStatus = -1 ;
          TrackArgs->TheBeam->bunches[*BunchLoop]->StillTracking=0 ;
        }
        InnerLoop += InnerStep ;
        continue ;
      }
      
      /* get the tracking flags; if corrupted, send a warning message and
       * attempt to continue */
      
      if (LastLoopElem != *ElemLoop)
      {
        TFlag = GetTrackingFlags( *ElemLoop ) ;
        if (TFlag[NUM_TRACK_FLAGS] == 0)
        {
          GlobalStatus = 0 ;
          BadTrackFlagMessage( (*ElemLoop)+1 ) ;
          goto egress ;
        }
        /* Copy TFlag over to GPU */
#ifdef __CUDACC__
        gpuErrchk( cudaMemcpy(TFlag_gpu, TFlag, sizeof(int)*NUM_TRACK_FLAGS, cudaMemcpyHostToDevice) );
#endif
        
        /* get the element class string */
        
        ElemClass = GetElemClass( *ElemLoop ) ;
        if (ElemClass == NULL)
        {
          GlobalStatus = 0 ;
          BadElementMessage( (*ElemLoop)+1 ) ;
          goto egress ;
        }
      }
      
      LastLoopElem = *ElemLoop ; /* only get track flags and ElemClass if ElemLoop has
       * changed from last loop execution */

      /* track this bunch through this element */
      /* - first deal with all element types that cannot or should not be split or have zero length*/
      if (strcmp(ElemClass,"LCAV")==0)
      {
        /* Split status not supported for LCAV now- too damned complicated to figure out how to split up
         * and correctly deal with wakefield and BPM details */
        TrackStatus = TrackBunchThruRF( *ElemLoop, *BunchLoop,  TrackArgs, TFLAG, 0 ) ;
        if (TrackStatus == 1) 
          postEleTrack( TrackArgs->TheBeam, BunchLoop, ElemLoop, 0, *GetElemNumericPar( *ElemLoop, "S", NULL ), TFlag );
      }
      else if (strcmp(ElemClass,"TCAV")==0)
      {
        /* No split status treatment for TCAV's - see above */
        TrackStatus = TrackBunchThruRF( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 1 ) ;
        if (TrackStatus == 1) 
          postEleTrack( TrackArgs->TheBeam, BunchLoop, ElemLoop, 0, *GetElemNumericPar( *ElemLoop, "S", NULL ), TFlag) ;
      }
      else if ( strcmp(ElemClass,"COORD")==0 )
      {
        TrackStatus = TrackBunchThruCoord( *ElemLoop, *BunchLoop, TrackArgs, TFLAG ) ;
        if (TrackStatus == 1) 
          postEleTrack( TrackArgs->TheBeam, BunchLoop, ElemLoop, 0, *GetElemNumericPar( *ElemLoop, "S", NULL ), TFlag) ;
      }
      else if ( strcmp(ElemClass,"TMAP")==0 )
      {
        TrackStatus = TrackBunchThruTMap( *ElemLoop, *BunchLoop, TrackArgs, TFLAG ) ;
        if (TrackStatus == 1) 
          postEleTrack( TrackArgs->TheBeam, BunchLoop, ElemLoop, 0, *GetElemNumericPar( *ElemLoop, "S", NULL ), TFlag) ;
      }
      else if ( strcmp(ElemClass,"MARK")==0 ) {
        TrackStatus = 1 ;
      }
      else /* all other elements which have a standard split treatment and before/after split tracking actions */
      {
#ifdef __CUDACC__        
        TrackStatus = ElemTracker(ElemClass,ElemLoop,BunchLoop,TFlag_gpu,TFlag,TrackArgs) ;
#else
        TrackStatus = ElemTracker(ElemClass,ElemLoop,BunchLoop,TFlag,TrackArgs) ;
#endif                
        if (TrackStatus == 0) {
          GlobalStatus = TrackStatus ;
          goto egress ;
        }
      }
      
      /* latch non-unity status from the tracking procedures.  If a fatal
       * error occurred, abort. */
      
      if (TrackStatus == -1)
        GlobalStatus = TrackStatus ;
      if (TrackStatus == 0)
      {
        GlobalStatus = TrackStatus ;
        goto egress ;
      }
    
      /* increment the inner loop counter and close the do loop */
      
      InnerLoop += InnerStep ;
    } while (InnerLoop != InnerStop+InnerStep) ;
    
    /* increment the outer loop counter and close the do loop */
    
    OuterLoop += OuterStep ;
    
  } while (OuterLoop != OuterStop+OuterStep) ;
  
  
  egress:
    
    /* put the latched status back into TrackArgs */
    
    TrackArgs->Status = GlobalStatus ;
    
    /* clear all wakefields on all bunches */
    
    for (OuterLoop = TrackArgs->FirstBunch-1 ;
    OuterLoop < TrackArgs->LastBunch ;
    OuterLoop++)
    {
      ClearConvolvedSRWF( TrackArgs->TheBeam->bunches[OuterLoop],
              -1,  0 ) ;
      ClearConvolvedSRWF( TrackArgs->TheBeam->bunches[OuterLoop],
              -1,  1 ) ;
      ClearBinnedLRWFFreq( TrackArgs->TheBeam->bunches[OuterLoop],
              -1,  0 ) ;
      ClearBinnedLRWFFreq( TrackArgs->TheBeam->bunches[OuterLoop],
              -1,  1 ) ;
    }
    
    /* discard the existing families of pointers to LRWF data */
    
    FreeAndNull( (void **) &TLRFreqData ) ;
    FreeAndNull( (void **) &TLRErrFreqData ) ;
    
    /* discard the damping factors */
    
    for (OuterLoop = 0 ; OuterLoop<numwakes[2] ; OuterLoop++)
    {
      FreeAndNull( (void **) &(TrackArgs->TheBeam->TLRFreqDamping[OuterLoop]) ) ;
      FreeAndNull( (void **) &(TrackArgs->TheBeam->TLRFreqxPhase[OuterLoop]) ) ;
      FreeAndNull( (void **) &(TrackArgs->TheBeam->TLRFreqyPhase[OuterLoop]) ) ;
    }
    for (OuterLoop = 0 ; OuterLoop<numwakes[3] ; OuterLoop++)
    {
      FreeAndNull( (void **) &(TrackArgs->TheBeam->TLRErrFreqDamping[OuterLoop]) ) ;
      FreeAndNull( (void **) &(TrackArgs->TheBeam->TLRErrFreqxPhase[OuterLoop]) ) ;
      FreeAndNull( (void **) &(TrackArgs->TheBeam->TLRErrFreqyPhase[OuterLoop]) ) ;
    }
    
    /* clear out the existing pascal matrix and factorial vector */
    
    ClearMaxMultipoleStuff( ) ;
    
    /* Clear GPU allocated memory */
#ifdef __CUDACC__
    free(rSeed);
    gpuErrchk( cudaFreeHost(StoppedParticles) );
    gpuErrchk( cudaFree(TFlag_gpu) );
    gpuErrchk( cudaFree(rngStates) );
#endif
    
    return ;
}

/*=====================================================================*/

/* perform tracking throuhj a Transfer Map element (up to 5th order). */

/* RET:    Status = 1 (success), always.
 * ABORT:  never.
 * FAIL:   Will fail if ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno. Or TMAP element badly formatted */

int TrackBunchThruTMap( int elemno, int bunchno,  struct TrackArgsStruc* ArgStruc, int* TrackFlag )
{
#ifndef __CUDACC__
  int rayloop ;
#endif
  double* L ;
  double dZmod ;               /* lorentz delay for P = Pmod */
  struct Bunch* ThisBunch ;    /* a shortcut */
  double* Pmod ;
  double Offset[6], R[36] ;
  double *T=NULL,*U=NULL,*V=NULL,*W=NULL ;
  unsigned short T_size=0, U_size=0, V_size=0, W_size=0, stat ;
  unsigned long *T_inds=NULL, *U_inds=NULL, *V_inds=NULL, *W_inds=NULL ;
  
  /* get the length of the drift space */
  L = GetElemNumericPar( elemno, "L", NULL ) ;
  
  /* get the design Lorentz delay */
  Pmod = GetElemNumericPar( elemno, "P", NULL ) ;
  dZmod = GetDesignLorentzDelay( Pmod ) ;
  
  /* get the address of the bunch in question */
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  /* Get Transport Map data */
  /* As minimum, contains offsets and R matrix */
  /* Other elements depends on Order of map and number of elements placed */
  TMapGetDataLen(elemno,&T_size,&U_size,&V_size,&W_size) ;
  if (T_size>0) {
    T_inds = (unsigned long*) malloc(T_size * sizeof(unsigned long)) ;
    T = (double*) malloc(T_size * sizeof(double)) ;
  }
  if (U_size>0) {
    U_inds = (unsigned long*) malloc(U_size * sizeof(unsigned long)) ;
    U = (double*) malloc(U_size * sizeof(double)) ;
  }
  if (V_size>0) {
    V_inds = (unsigned long*) malloc(V_size * sizeof(unsigned long)) ;
    V = (double*) malloc(V_size * sizeof(double)) ;
  }
  if (W_size>0) {
    W_inds = (unsigned long*) malloc(W_size * sizeof(unsigned long)) ;
    W = (double*) malloc(W_size * sizeof(double)) ;
  }
  stat = TMapGetData(elemno, Offset, R, T, U, V, W, T_inds, U_inds, V_inds, W_inds, &T_size,&U_size,&V_size,&W_size) ;
  if (stat==0) {
    AddMessage("Badly formatted TMAP element",1) ;
    return 0 ;
  }
  
  /* execute ray tracking kernel (loop over rays) */
#ifdef __CUDACC__
  double* dZmod_gpu ;
  double* L_gpu ;
  double *Offset_gpu, *R_gpu ;
  double *T_gpu,*U_gpu,*V_gpu,*W_gpu ;
  unsigned long *T_inds_gpu, *U_inds_gpu, *V_inds_gpu, *W_inds_gpu ;
  gpuErrchk( cudaMalloc(&dZmod_gpu, sizeof(double)) );
  gpuErrchk( cudaMalloc(&L_gpu, sizeof(double)) );
  gpuErrchk( cudaMemcpy(dZmod_gpu, &dZmod, sizeof(double), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(L_gpu, L, sizeof(double), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMalloc(&Offset_gpu, sizeof(double)*6) );
  gpuErrchk( cudaMalloc(&R_gpu, sizeof(double)*36) );
  gpuErrchk( cudaMemcpy(Offset_gpu, Offset, sizeof(double)*6, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(R_gpu, R, sizeof(double)*36, cudaMemcpyHostToDevice) );
  if (T_size>0) {
    gpuErrchk( cudaMalloc(&T_gpu, sizeof(double)*T_size) );
    gpuErrchk( cudaMemcpy(T_gpu, T, sizeof(double)*T_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&T_inds_gpu, sizeof(unsigned long)*T_size) );
    gpuErrchk( cudaMemcpy(T_inds_gpu, T_inds, sizeof(unsigned long)*T_size, cudaMemcpyHostToDevice) );
  }
  if (U_size>0) {
    gpuErrchk( cudaMalloc(&U_gpu, sizeof(double)*U_size) );
    gpuErrchk( cudaMemcpy(U_gpu, U, sizeof(double)*U_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&U_inds_gpu, sizeof(unsigned long)*U_size) );
    gpuErrchk( cudaMemcpy(U_inds_gpu, U_inds, sizeof(unsigned long)*U_size, cudaMemcpyHostToDevice) );
  }
  if (V_size>0) {
    gpuErrchk( cudaMalloc(&V_gpu, sizeof(double)*V_size) );
    gpuErrchk( cudaMemcpy(V_gpu, V, sizeof(double)*V_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&V_inds_gpu, sizeof(unsigned long)*V_size) );
    gpuErrchk( cudaMemcpy(V_inds_gpu, V_inds, sizeof(unsigned long)*V_size, cudaMemcpyHostToDevice) );
  }
  if (W_size>0) {
    gpuErrchk( cudaMalloc(&W_gpu, sizeof(double)*W_size) );
    gpuErrchk( cudaMemcpy(W_gpu, W, sizeof(double)*W_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&W_inds_gpu, sizeof(unsigned long)*W_size) );
    gpuErrchk( cudaMemcpy(W_inds_gpu, W_inds, sizeof(unsigned long)*W_size, cudaMemcpyHostToDevice) );
  }
  TrackBunchThruTMap_kernel<<<blocksPerGrid, threadsPerBlock>>>(L_gpu, dZmod_gpu, ThisBunch->x, ThisBunch->y,
          ThisBunch->stop, TrackFlag, ThisBunch->nray, Offset_gpu, R_gpu ,T_size,U_size,V_size,W_size,
          T_gpu, U_gpu, V_gpu, W_gpu, T_inds_gpu, U_inds_gpu, V_inds_gpu, W_inds_gpu, *Pmod);
  gpuErrchk( cudaGetLastError() );
  gpuErrchk( cudaFree(dZmod_gpu) ); gpuErrchk( cudaFree(L_gpu) );
  gpuErrchk( cudaFree(R_gpu) ); gpuErrchk( cudaFree(Offset_gpu) );
  if (T_size>0) {
    gpuErrchk( cudaFree(T_gpu) );
    gpuErrchk( cudaFree(T_inds_gpu) );
  }
  if (U_size>0) {
    gpuErrchk( cudaFree(U_gpu) );
    gpuErrchk( cudaFree(U_inds_gpu) );
  }
  if (V_size>0) {
    gpuErrchk( cudaFree(V_gpu) );
    gpuErrchk( cudaFree(V_inds_gpu) );
  }
  if (W_size>0) {
    gpuErrchk( cudaFree(W_gpu) );
    gpuErrchk( cudaFree(W_inds_gpu) );
  }
#else
  for ( rayloop=0 ; rayloop<ThisBunch->nray ; rayloop++ )
    TrackBunchThruTMap_kernel(L, &dZmod, ThisBunch->x, ThisBunch->y, ThisBunch->stop, TrackFlag, rayloop,
            Offset,R,T_size,U_size,V_size,W_size, T, U, V, W, T_inds, U_inds, V_inds, W_inds, *Pmod);
#endif
  if (T_size>0) {
    free(T_inds); free(T);
  }
  if (U_size>0) {
    free(U_inds); free(U);
  }
  if (V_size>0) {
    free(V_inds); free(V);
  }
  if (W_size>0) {
    free(W_inds); free(W);
  }
  return 1;
  
}

/* The TMap tracking Kernel */
#ifdef __CUDA_ARCH__
__global__ void TrackBunchThruTMap_kernel(double *L, double* dZmod, double* xb, double* yb, double* stop, int* TrackFlag, int N,
        double* Offset, double* R, unsigned short T_size, unsigned short U_size, unsigned short V_size,
        unsigned short W_size, double* T, double* U, double* V, double* W, unsigned long* T_inds, unsigned long* U_inds,
        unsigned long* V_inds, unsigned long* W_inds, double Pmod)
#else
void TrackBunchThruTMap_kernel(double *L, double* dZmod, double* xb, double* yb, double* stop, int* TrackFlag, int N,
        double* Offset, double* R, unsigned short T_size, unsigned short U_size, unsigned short V_size,
        unsigned short W_size, double* T, double* U, double* V, double* W, unsigned long* T_inds, unsigned long* U_inds,
        unsigned long* V_inds, unsigned long* W_inds, double Pmod)
#endif        
{
  
  double dZdL=0 ;
  unsigned short mapInd, j, j2 ;
  unsigned long dim[6], i ;
  int powval=10 ;
  
#ifdef __CUDA_ARCH__
  i = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( i >= N ) return;
#else
  i = N;
#endif
  
  /* if the bunch was previously stopped on an aperture, loop */
  if (stop[i] != 0.) return ;
  
  if (TrackFlag[LorentzDelay] == 1)
    dZdL += LORENTZ_DELAY(xb[6*i+5]) - *dZmod ;
  
  /* Apply any offsets */
  for (j=0;j<6;j++)
    yb[6*i+j] = Offset[j];
  
  /* P converted to dP/P */
  xb[6*i+5] = (xb[6*i+5]-Pmod) / Pmod ;
  
  /* Apply R map */
  for (j=0;j<6;j++)
    for (j2=0;j2<6;j2++)
      yb[6*i+j] += R[j+6*j2] * xb[6*i+j2] ;
    
  if (T_size>0) {
    for (mapInd=0;mapInd<T_size;mapInd++) {
      powval=1;
      for (j=0;j<3;j++) {
        dim[2-j] = floor( (double) ( T_inds[mapInd] % (powval*10)) / powval ) - 1 ;
        powval*=10;
      }
      yb[6*i+dim[0]] += T[mapInd]*xb[6*i+dim[1]]*xb[6*i+dim[2]] ;
    }
  }
  if (U_size>0) {
    for (mapInd=0;mapInd<U_size;mapInd++) {
      powval=1;
      for (j=0;j<4;j++) {
        dim[3-j] = floor( (double) (U_inds[mapInd] % (powval*10)) / powval ) - 1;
        powval *= 10 ;
      }
      yb[6*i+dim[0]] += U[mapInd]*xb[6*i+dim[1]]*xb[6*i+dim[2]]*xb[6*i+dim[3]] ;
    }
  }
  if (V_size>0) {
    for (mapInd=0;mapInd<V_size;mapInd++) {
      powval=1;
      for (j=0;j<5;j++) {
        dim[4-j] = floor( (double) (V_inds[mapInd] % (powval*10)) / powval ) - 1;
        powval *= 10 ;
      }
      yb[6*i+dim[0]] += V[mapInd]*xb[6*i+dim[1]]*xb[6*i+dim[2]]*xb[6*i+dim[3]]*xb[6*i+dim[4]] ;
    }
  }
  if (W_size>0) {
    for (mapInd=0;mapInd<W_size;mapInd++) {
      powval=1;
      for (j=0;j<6;j++) {
        dim[5-j] = floor( (double) (W_inds[mapInd] % (powval*10)) / powval ) - 1;
        powval *= 10 ;
      }
      yb[6*i+dim[0]] += W[mapInd]*xb[6*i+dim[1]]*xb[6*i+dim[2]]*xb[6*i+dim[3]]*xb[6*i+dim[4]]*xb[6*i+dim[5]] ;
    }
  }
  
  /* Apply Lorentz delay */
  yb[6*i+4] += *L * dZdL ;
  
  /* Put P back into absolute units */
  yb[6*i+5] = Pmod + yb[6*i+5] * Pmod ;
  
}

/*=====================================================================*/

/* perform tracking of one bunch through one drift matrix.  This
 * procedure also acts as a default tracker, tracking through any
 * elements with unrecognized class names.  If the element has no
 * length field, the element will be treated as zero-length. */

/* RET:    Status = 1 (success), always.
 * ABORT:  never.
 * FAIL:   Will fail if ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno. */

int TrackBunchThruDrift( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TrackFlag, double splitScale )
{
  double* L ;                  /* drift has only 2 pars of any */
#ifndef __CUDACC__
  int rayloop ;
#endif
  double dZmod ;               /* lorentz delay for P = Pmod */
  double Lfull ;
  struct Bunch* ThisBunch ;    /* a shortcut */
  double* Pmod ;
  
  /* get the length of the drift space */
  L = GetElemNumericPar( elemno, "L", NULL ) ;
  
  if ( splitScale == 0 )
    splitScale = 1 ;
  else if ( splitScale < 0 ) /* Case where INSTR or BPM class wants to just apply moment calculations */
    splitScale = 0 ;
  else
    splitScale = splitScale / *L ;
  
  if (L==NULL)
    Lfull = 0. ;
  else
    Lfull = *L * splitScale ;
  
  /* get the design Lorentz delay */
  Pmod = GetElemNumericPar( elemno, "P", NULL ) ;
  dZmod = GetDesignLorentzDelay( Pmod ) ;
  
  /* get the address of the bunch in question */
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  /* since about half of the coordinate data in the "output" bunch
   * is the same as the "input" bunch (px, py, p0), start by simply
   * exchanging the pointers of the input and output bunches */
  XYExchange( ThisBunch ) ;
  
  /* execute ray tracking kernel (loop over rays) */
#ifdef __CUDACC__
  double* Lfull_gpu ;
  double* dZmod_gpu ;
  gpuErrchk( cudaMalloc(&Lfull_gpu, sizeof(double)) ); gpuErrchk( cudaMalloc(&dZmod_gpu, sizeof(double)) );
  gpuErrchk( cudaMemcpy(Lfull_gpu, &Lfull, sizeof(double), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(dZmod_gpu, &dZmod, sizeof(double), cudaMemcpyHostToDevice) );
  TrackBunchThruDrift_kernel<<<blocksPerGrid, threadsPerBlock>>>(Lfull_gpu, dZmod_gpu, ThisBunch->y, ThisBunch->stop, TrackFlag, ThisBunch->nray);
  gpuErrchk( cudaGetLastError() ) ;
  gpuErrchk( cudaFree(Lfull_gpu) ); gpuErrchk( cudaFree(dZmod_gpu) );
#else
  for ( rayloop=0 ; rayloop<ThisBunch->nray ; rayloop++ )
    TrackBunchThruDrift_kernel(&Lfull, &dZmod, ThisBunch->y, ThisBunch->stop, TrackFlag, rayloop);
#endif
  
  return 1;
  
}


/* The DRIFT tracking Kernel */
#ifdef __CUDACC__
__global__ void TrackBunchThruDrift_kernel(double* Lfull, double* dZmod, double* yb, double* stop, int* TrackFlag, int N)
#else
void TrackBunchThruDrift_kernel(double* Lfull, double* dZmod, double* yb, double* stop, int* TrackFlag, int N)
#endif
{
  
  double dZdL=0 ;
  int i;
  
#ifdef __CUDACC__
  i = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( i >= N ) return;
#else
  i = N;
#endif
  /* if the bunch was previously stopped on an aperture, loop */
  if (stop[i] != 0.) return ;
  
  if (TrackFlag[LorentzDelay] == 1)
    dZdL += LORENTZ_DELAY(yb[6*i+5]) - *dZmod ;
  
  if (TrackFlag[ZMotion] == 1)
    dZdL += 0.5*(yb[6*i+1] * yb[6*i+1] +
            yb[6*i+3] * yb[6*i+3]   ) ;
  /* otherwise put in the change in x, y, z positions */
  yb[6*i]   += yb[6*i+1] * *Lfull ;
  yb[6*i+2] += yb[6*i+3] * *Lfull ;
  yb[6*i+4] += *Lfull * dZdL ;
}


/*=====================================================================*/

/* Perform tracking of one bunch through one magnet of types Quad, Sext,
 * Octu, Solenoid, including
 * any transformations necessary to manage magnet, girder, or girder
 * mover position offsets etc.  All strength errors are also included.
 *
 * RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno or if
 * BEAMLINE{elemno} is not some sort of magnet.  */

int TrackBunchThruQSOS( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TrackFlag, int nPoleFlag, double splitScale, double splitS, int TFlagAper )
{
  double L,B,Tilt=0 ;              /* some basic parameters */
  double aper2 ;                 /* square of aperture radius */
  double Xfrms[6][2] ;           /* upstream and downstream coordinate
   * xfrms from magnet + girder + mover
   * offset values */
  double dZmod ;                 /* lorentz delay @ design momentum */
  int PS ;
#ifndef __CUDACC__
  int ray ;              /* shortcut for 6*ray */
#endif
  struct Bunch* ThisBunch ;    /* a shortcut */
  int stat = 1 ;
  int skew ;
  struct LucretiaParameter *ElemPar ;
  int nPar ;
#ifdef __CUDACC__
  double *PascalMatrix, *Bang, *MaxMultInd ;
  double *Xfrms_flat = NULL ;
#endif
  /* get the element parameters from BEAMLINE; exit with bad status if
   * parameters are missing or corrupted. */
  if ( nPoleFlag != 0)
  {
    ElemPar = QuadPar ;
    nPar = nQuadPar ;
  }
  else
  {
    ElemPar = SolePar ;
    nPar = nSolePar ;
  }
  
  stat = GetDatabaseParameters( elemno, nPar, ElemPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  L = GetDBValue(ElemPar+QuadL) ;
  if ( splitScale == 0 )
    splitScale = 1 ;
  else
    splitScale = splitScale / L ;
  L *= splitScale ;
  if (L<=0)
  {
    BadElementMessage( elemno+1 ) ;
    stat = 0 ;
    goto egress ;
  }
  
  B = GetDBValue(ElemPar+QuadB)  ;
  dZmod = GetDesignLorentzDelay( ElemPar[QuadP].ValuePtr ) ;
  if ( nPoleFlag != 0)
    Tilt = GetDBValue(ElemPar+QuadTilt) ;
  aper2 = GetDBValue(ElemPar+Quadaper) ;
  aper2 *= aper2 ;
  
  /* if aperture is zero but aperture track flag is on,
   * it's an error.  Set error status and exit. */
  
  if ( (aper2 == 0.) && (TFlagAper == 1) )
  {
    BadApertureMessage( elemno+1 ) ;
    stat = 0 ;
    goto egress ;
  }
  
  /* now the error parameters */
  
  B *= (1. + GetDBValue(ElemPar+QuaddB) ) ;
  
  /* now get the power supply parameters, if any */
  
  PS = (int)(GetDBValue(ElemPar+QuadPS)) ;
  if (PS > 0)
  {
    
    /* convert from Matlab to C indexing */
    
    PS-- ;
    stat = GetDatabaseParameters( PS, nPSPar, PSPar,
            TrackPars, PSTable ) ;
    if (stat == 0)
    {
      BadPSMessage( elemno+1, PS+1 ) ;
      goto egress ;
    }
    B *= (GetDBValue(PSPar+PSAmpl)) *
            (1. +  GetDBValue(PSPar+PSdAmpl) ) ;
    
  } /* end of PS interlude
   *
   * now we get the complete input- and output- transformations for the
   * element courtesy of the relevant function */
  
  /* Apply split scale */
  
  B *= splitScale ;
  if ( splitS == 0 )
    splitS = *ElemPar[QuadS].ValuePtr ;
  
  stat = GetTotalOffsetXfrms( ElemPar[QuadGirder].ValuePtr,
          &L,
          &splitS,
          ElemPar[QuadOffset].ValuePtr,
          Xfrms ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* since the rotation transformation can be applied to the magnet
   * rather than the beam, do that now; also figure out whether we are
   * dealing with a normal or partially skew magnet */
  
  if ( nPoleFlag != 0 && nPoleFlag <5 )
  {
    Tilt += Xfrms[5][0] ;
    if  (fabs(sin(nPoleFlag*Tilt)) < MIN_TILT)
      skew = 0 ;
    else
      skew = 1 ;
  }
  else if ( nPoleFlag == 0 )  /* solenoids are always "skew" magnets */
    skew = 1 ;
  else
    skew = 0; /* plasma lens cannot be skew */
  
  /* if this is an octupole, we need to make sure that the infrastructure for
   * handling a thin-lens multipole is ready to go */
  
  if (nPoleFlag == 4)
  {
    if ( GetMaxMultipoleIndex( ) < nPoleFlag )
      ComputeNewMultipoleStuff( nPoleFlag ) ;
  }
  
  
  /* make a shortcut to get to the bunch of interest */
  
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  
  /* execute ray tracking kernel (loop over rays) */
#ifdef __CUDACC__
  PascalMatrix = GetPascalMatrix_gpu( ) ;
  Bang         = GetFactorial_gpu( ) ;
  MaxMultInd = GetMaxMultipoleIndex_gpu( ) ;
  double *Xfrms_gpu ;
  gpuErrchk( cudaMalloc(&Xfrms_gpu, sizeof(double)*12) );
  Xfrms_flat = &(Xfrms[0][0]) ;
  gpuErrchk( cudaMemcpy(Xfrms_gpu, Xfrms_flat, sizeof(double)*12, cudaMemcpyHostToDevice) );
  TrackBunchThruQSOS_kernel<<<blocksPerGrid, threadsPerBlock>>>(ThisBunch->nray, ThisBunch->stop, ThisBunch->y, ThisBunch->x, TrackFlag,
          ThisBunch->ngoodray_gpu, elemno, aper2, nPoleFlag, B, L, Tilt, skew, Xfrms_gpu, dZmod, StoppedParticles_gpu,
          PascalMatrix, Bang, MaxMultInd, *rSeed, rngStates, ThisBunch->ptype ) ;
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaFree(Xfrms_gpu) );
#else
  for (ray=0 ;ray<ThisBunch->nray ; ray++)
    TrackBunchThruQSOS_kernel(ray, ThisBunch->stop, ThisBunch->y, ThisBunch->x, TrackFlag, &ThisBunch->ngoodray, elemno, aper2, nPoleFlag,
            B, L, Tilt, skew, Xfrms, dZmod, StoppedParticles, ThisBunch->ptype) ;
#endif
  egress:
    
    return stat;
    
}


/* Quad, Sext, Octu, Solenoid Tracking kernel*/
#ifdef __CUDACC__
__global__ void TrackBunchThruQSOS_kernel(int nray, double* stop, double* yb, double* xb, int* TrackFlag, int* ngoodray,
        int elemno, double aper2, int nPoleFlag, double B, double L, double Tilt,
        int skew, double* pXfrms, double dZmod, int* stp, double* PascalMatrix, double* Bang,
        double* MaxMultInd, unsigned long long rSeed, curandState *rState, unsigned short int* ptype)
#else
        void TrackBunchThruQSOS_kernel(int nray, double* stop, double* yb, double* xb, int* TrackFlag, int* ngoodray,
        int elemno, double aper2, int nPoleFlag, double B, double L, double Tilt,
        int skew, double Xfrms[6][2], double dZmod, int* stp, unsigned short int* ptype)
#endif
{
  int ray,coord,raystart,doStop,icount ;
  double *x, *px, *y, *py, *z, *p0 ;
  double SR_dP = 0. ;
  Rmat Rquad ;                 /* quad linear map */
  double T5xx[10] ;
#ifndef __CUDA_ARCH__
  double LastRayP = 0. ;
#else
  curandState rState_local ;
#endif
#ifdef __CUDACC__
  double Xfrms[6][2] ;
  int irw, icol ;
  for (irw=0;irw<6;irw++)
    for (icol=0;icol<2;icol++)
      Xfrms[irw][icol]=pXfrms[(2*irw)+icol] ;
#endif
  double TijkTransv[4][10] ;
  double nPoleOctu = 3. ;
  double AngVecOctu[2] = {0,0} ;
  double r ;
#ifdef __CUDA_ARCH__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
#else
  ray = nray;
#endif
  raystart = 6*ray ;
  
  /* copy rng state to local memory for efficiency */
#ifdef __CUDA_ARCH__
  rState_local = rState[ray] ;
#endif
  
  /* If positivly charged particle, invert B */
  if (ptype[ray] == 1) {
    B=-B;
  }
  
  /* if the ray was previously stopped just copy it over */
  
  if (stop[ray] > 0.)
  {
    for (coord=0 ; coord<6 ; coord++) {
      yb[raystart+coord] = xb[raystart+coord] ;
      goto egress ; }
  }
  
  /* make ray coordinates into local ones, including offsets etc which
   * are demanded by the transformation structure. */
  GetLocalCoordPtrs(xb, raystart,&x,&px,&y,&py,&z,&p0) ;
  ApplyTotalXfrm( Xfrms, UPSTREAM, TrackFlag, 0 ,x,px,y,py,z,p0) ;
  
  yb[raystart+4] = *z ;
  
  /* entrance-face aperture test, if requested */
  
  if (TrackFlag[Aper] == 1)
  {
    doStop = CheckAperStopPart( xb, yb, stop, ngoodray,elemno,&aper2,ray,UPSTREAM,
            NULL, 0, stp, 0 ) ;
    if (doStop == 1)
      goto egress ;
  }
  
  /* Compute the SR momentum loss, if required, and apply 1/2 of the
   * loss here at the entry face of the element; as long as we're here,
   * check to see whether the particle has lost all of its momentum,
   * and if so stop it. */
  
  if (TrackFlag[SynRad] > SR_None)
  {
    double rpow ;
    r = sqrt(*x * *x + *y * *y) ;
    rpow = r ;
    if ( nPoleFlag >= 3 )
      rpow *= r / 2 ;
    if ( nPoleFlag >= 4 )
      rpow *= r / 3 ;
#ifdef __CUDA_ARCH__
    SR_dP = ComputeSRMomentumLoss_gpu( *p0, rpow*fabs(B), L, TrackFlag[SynRad], &rState_local ) ;
#else
    SR_dP = ComputeSRMomentumLoss( *p0, rpow*fabs(B), L, TrackFlag[SynRad] ) ;
#endif
    yb[raystart+5] = *p0 - SR_dP ;
    doStop = CheckP0StopPart( stop,ngoodray,xb,yb,elemno,ray,*p0-SR_dP, UPSTREAM, stp ) ;
    if (doStop == 1)
      goto egress ;
    *p0 -= SR_dP / 2 ;
  }
  else
    yb[raystart+5] = *p0 ;
  
  /* now we switch on the element class which is being tracked through */
  
  switch ( nPoleFlag )
  {
    case 0 : case 2 : case 5 : /* solenoid or quadrupole or plasma lens */
      
      /* get the quad map */
#ifndef __CUDA_ARCH__
      if (*p0 != LastRayP)
      {
#endif
        if (nPoleFlag == 2)
          GetQuadMap( L, B, Tilt, xb[raystart+5], Rquad, T5xx ) ;
        else if (nPoleFlag == 0)
          GetSolenoidMap( L, B, xb[raystart+5], Rquad, T5xx ) ;
        else
          GetPlensMap( L, B, xb[raystart+5], Rquad, T5xx ) ;
        
#ifndef __CUDA_ARCH__
      }
      LastRayP = *p0 ;
#endif
      
      /* perform the matrix transformation; since we know that the quad has
       * no 5j or 6j terms, just do the 4x4 multiplication */
      
      yb[raystart]   = *x*Rquad[0][0] + *px * Rquad[0][1] ;
      yb[raystart+1] = *x*Rquad[1][0] + *px * Rquad[1][1] ;
      yb[raystart+2] = *y*Rquad[2][2] + *py * Rquad[2][3] ;
      yb[raystart+3] = *y*Rquad[3][2] + *py * Rquad[3][3] ;
      
      if (skew==1 && nPoleFlag<5)
      {
        yb[raystart]   += *y*Rquad[0][2] + *py * Rquad[0][3] ;
        yb[raystart+1] += *y*Rquad[1][2] + *py * Rquad[1][3] ;
        yb[raystart+2] += *x*Rquad[2][0] + *px * Rquad[2][1] ;
        yb[raystart+3] += *x*Rquad[3][0] + *px * Rquad[3][1] ;
      }
      
      
      /* now for the 5xx terms */
      if (TrackFlag[ZMotion] == 1 && nPoleFlag<5)
        yb[raystart+4] +=
                T5xx[0] * xb[raystart+0] * xb[raystart+0]
                +	     T5xx[1] * xb[raystart+0] * xb[raystart+1]
                +	     T5xx[2] * xb[raystart+0] * xb[raystart+2]
                +	     T5xx[3] * xb[raystart+0] * xb[raystart+3]
                +	     T5xx[4] * xb[raystart+1] * xb[raystart+1]
                +	     T5xx[5] * xb[raystart+1] * xb[raystart+2]
                +	     T5xx[6] * xb[raystart+1] * xb[raystart+3]
                +	     T5xx[7] * xb[raystart+2] * xb[raystart+2]
                +	     T5xx[8] * xb[raystart+2] * xb[raystart+3]
                +	     T5xx[9] * xb[raystart+3] * xb[raystart+3]
                ;
      
      
      break ; /* end of quad operations */
      
    case 3:   /* sextupole */
#ifndef __CUDA_ARCH__
      if (*p0 != LastRayP)
#endif
        GetSextMap( L, B, Tilt, xb[raystart+5],
                TijkTransv ) ;
#ifndef __CUDA_ARCH__
      LastRayP = *p0 ;
#endif
      
      /* perform transverse propagation */
      
      yb[raystart  ] = *x + L * *px ;
      yb[raystart+1] = *px ;
      yb[raystart+2] = *y + L * *py ;
      yb[raystart+3] = *py ;
      
      for (icount=0 ; icount<2 ; icount++)
      {
        yb[raystart+icount] +=
                TijkTransv[icount][0] * *x  * *x  +
                TijkTransv[icount][1] * *x  * *px +
                TijkTransv[icount][4] * *px * *px +
                TijkTransv[icount][7] * *y  * *y  +
                TijkTransv[icount][8] * *y  * *py +
                TijkTransv[icount][9] * *py * *py   ;
        
        yb[raystart+icount+2] +=
                TijkTransv[icount+2][2] * *x  * *y  +
                TijkTransv[icount+2][3] * *x  * *py +
                TijkTransv[icount+2][5] * *px * *y  +
                TijkTransv[icount+2][6] * *px * *py   ;
        
        if (skew == 1)
        {
          yb[raystart+icount] +=
                  TijkTransv[icount][2] * *x  * *y  +
                  TijkTransv[icount][3] * *x  * *py +
                  TijkTransv[icount][5] * *px * *y  +
                  TijkTransv[icount][6] * *px * *py   ;
          
          yb[raystart+icount+2] +=
                  TijkTransv[icount+2][0] * *x  * *x  +
                  TijkTransv[icount+2][1] * *x  * *px +
                  TijkTransv[icount+2][4] * *px * *px +
                  TijkTransv[icount+2][7] * *y  * *y  +
                  TijkTransv[icount+2][8] * *y  * *py +
                  TijkTransv[icount+2][9] * *py * *py   ;
        }
      }
      
      
      
      /* longitudinal propagation */
      
      if (TrackFlag[ZMotion] == 1)
        yb[raystart+4] += 0.5*L*( *px * *px + *py * *py ) ;
      
      /*		  yb[raystart+5] = xb[raystart+5] ; */
      
      break ;
      
    case 4: /* octupole -- handled as a special case of the thin-lens
     * multipole, with SR disabled */
#ifdef __CUDA_ARCH__
      PropagateRayThruMult_gpu( L, &B, &Tilt, &nPoleOctu, 1, AngVecOctu,
              1.0, 0.0,
              &(xb[raystart]),
              &(yb[raystart]),
              TrackFlag[ZMotion], SR_None, 0,
              stop,ngoodray,xb,yb, elemno, ray, 1, PascalMatrix, Bang, MaxMultInd, &rState_local ) ;
#else
      PropagateRayThruMult( L, &B, &Tilt, &nPoleOctu, 1, AngVecOctu,
              1.0, 0.0,
              &(xb[raystart]),
              &(yb[raystart]),
              TrackFlag[ZMotion], SR_None, 0,
              stop,ngoodray,xb,yb, elemno, ray, 1 ) ;
#endif
      break ;
  }
  
  /* no matter what kind of element, perform Lorentz delay if requested */
  
  /*		yb[raystart+5] = xb[raystart+5] ; */
  if (TrackFlag[LorentzDelay] == 1)
    yb[raystart+4] += L*(LORENTZ_DELAY(*p0) - dZmod) ;
  
  /* exit-face aperture test, if requested */
  
  if (TrackFlag[Aper] == 1)
  {
    doStop = CheckAperStopPart( xb,yb,stop,ngoodray,elemno,&aper2,ray,DOWNSTREAM,
            NULL, 0, stp, 0 ) ;
    if (doStop == 1)
      goto egress ;
  }
  
  /* undo the coordinate transformations */
  
  GetLocalCoordPtrs(yb, raystart,&x,&px,&y,&py,&z,&p0) ;
  
  ApplyTotalXfrm( Xfrms, DOWNSTREAM, TrackFlag, dZmod,x,px,y,py,z,p0 ) ;
  
  /* check amplitude of outgoing angular momentum */
  
  doStop = CheckPperpStopPart( stop, ngoodray , elemno, ray,
          px, py, stp ) ;
  
  egress:
    /* Copy rng state back to global memory */
#ifdef __CUDA_ARCH__
    rState[ray] = rState_local ;
#endif
    return ;
    
}

/*=====================================================================*/

/* Perform tracking of one bunch through one multipole magnet, including
 * any transformations necessary to manage magnet, girder, or girder
 * mover position offsets etc.  All strength errors are also included.
 *
 * RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno or if
 * BEAMLINE{elemno} is not some sort of magnet. 
 * ===========================================
 * GRW: 1/8/14 : Remove checking of aper>0 and Lrad>0, changed local
 * function to not do aper checking / syn rad if 0 instead*/

int TrackBunchThruMult( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc, int* TrackFlag, double splitScale, double splitS  )
{
  
  double dB, Tilt, L ;           /* some basic parameters */
  double aper2 ;                 /* square of aperture radius */
  double Xfrms[6][2] ;           /* upstream and downstream coordinate
   * xfrms from magnet + girder + mover
   * offset values */
  double dZmod ;                 /* lorentz delay @ design momentum */
  int PS ;
#ifndef __CUDACC__
  int ray ;                    /* shortcut for 6*ray */
#endif
  struct Bunch* ThisBunch ;    /* a shortcut */
  int pole ;
  double maxpole ;
  double dmy ;
  int stat = 1 ;
  double Lrad ;
#ifdef __CUDACC__
  double *PascalMatrix, *Bang, *MaxMultInd ;
  double *Xfrms_flat = NULL ;
#endif
  
  /* get the element parameters from BEAMLINE; exit with bad status if
   * parameters are missing or corrupted. */
  
  stat = GetDatabaseParameters( elemno, nMultPar, MultPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* check to make sure that the B-field, tilt, and PoleIndex fields are all
   * equal in length */
  
  if ( (MultPar[MultB].Length != MultPar[MultTilt].Length )     ||
          (MultPar[MultB].Length != MultPar[MultPoleIndex].Length)    )
  {
    stat = 0 ;
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* check to make sure the pole indices are integer valued.  While we're
   * at it, find the maximum pole index value */
  
  maxpole = -1 ;
  for (pole=0 ; pole<MultPar[MultPoleIndex].Length ; pole++)
  {
    if ( modf(MultPar[MultPoleIndex].ValuePtr[pole],&dmy) != 0 )
    {
      stat = 0 ;
      BadElementMessage( elemno+1 ) ;
      goto egress ;
    }
    if (MultPar[MultPoleIndex].ValuePtr[pole] > maxpole)
      maxpole = MultPar[MultPoleIndex].ValuePtr[pole] ;
  }
  
  /* generate the Pascal matrix and factorial vector, if needed */
  
  if (GetMaxMultipoleIndex( ) < maxpole+1.)
    ComputeNewMultipoleStuff(maxpole+1.) ;
  
  L = GetDBValue(MultPar + MultL) ;
  if ( splitScale ==0 )
    splitScale = 1 ;
  else
    splitScale = splitScale / L ;
  L *= splitScale ;
  
  dZmod = GetDesignLorentzDelay( MultPar[MultP].ValuePtr ) ;
  aper2 = GetDBValue(MultPar+Multaper) ;
  aper2 *= aper2 ;
  Lrad = GetDBValue(MultPar+MultLrad) ;
  if (Lrad<=0)
    Lrad = L ;
  
  Lrad *= splitScale ;
  
  /* now the error parameters */
  
  dB = 1. + GetDBValue(MultPar+MultdB) ;
  
  /* now get the power supply parameters, if any */
  
  PS = (int)(GetDBValue(MultPar+MultPS)) ;
  
  if (PS > 0)
  {
    
    /* convert from Matlab to C indexing */
    
    PS-- ;
    stat = GetDatabaseParameters( PS, nPSPar, PSPar,
            TrackPars, PSTable ) ;
    if (stat == 0)
    {
      BadPSMessage( elemno+1, PS+1 ) ;
      goto egress ;
    }
    dB *= (GetDBValue(PSPar+PSAmpl)) *
            (1. +  GetDBValue(PSPar+PSdAmpl) ) ;
    
  } /* end of PS interlude
   *
   * now we get the complete input- and output- transformations for the
   * element courtesy of the relevant function */
  
  if ( splitS == 0 )
    splitS = *MultPar[MultS].ValuePtr ;
  
  stat = GetTotalOffsetXfrms( MultPar[MultGirder].ValuePtr,
          &L,
          &splitS,
          MultPar[MultOffset].ValuePtr,
          Xfrms ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* since the rotation transformation can be applied to the magnet
   * rather than the beam, do that now */
  
  Tilt = Xfrms[5][0] ;
  
  
  /* make a shortcut to get to the bunch of interest */
  
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  
  /* execute ray tracking kernel (loop over rays) */
#ifdef __CUDACC__
  PascalMatrix = GetPascalMatrix_gpu( ) ;
  Bang = GetFactorial_gpu( ) ;
  MaxMultInd = GetMaxMultipoleIndex_gpu( ) ;
  double *Xfrms_gpu ;
  double *MultAngleValue, *MultBValue, *MultTiltValue, *MultPoleIndexValue ;
  gpuErrchk( cudaMalloc((void **)&Xfrms_gpu, sizeof(double)*12) );
  Xfrms_flat = &(Xfrms[0][0]) ;
  gpuErrchk( cudaMemcpy(Xfrms_gpu, Xfrms_flat, sizeof(double)*12, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMalloc((void **)&MultAngleValue, sizeof(double)*MultPar[MultAngle].Length) );
  gpuErrchk( cudaMalloc((void **)&MultBValue, sizeof(double)*MultPar[MultB].Length) );
  gpuErrchk( cudaMalloc((void **)&MultTiltValue, sizeof(double)*MultPar[MultTilt].Length) );
  gpuErrchk( cudaMalloc((void **)&MultPoleIndexValue, sizeof(double)*MultPar[MultPoleIndex].Length) );
  gpuErrchk( cudaMemcpy(MultAngleValue, MultPar[MultAngle].ValuePtr, sizeof(double)*MultPar[MultAngle].Length, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(MultBValue, MultPar[MultB].ValuePtr, sizeof(double)*MultPar[MultB].Length, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(MultTiltValue, MultPar[MultTilt].ValuePtr, sizeof(double)*MultPar[MultTilt].Length, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(MultPoleIndexValue, MultPar[MultPoleIndex].ValuePtr, sizeof(double)*MultPar[MultPoleIndex].Length, cudaMemcpyHostToDevice) );
  TrackBunchThruMult_kernel<<<blocksPerGrid, threadsPerBlock>>>( MultAngleValue, MultBValue, MultTiltValue, MultPoleIndexValue,
          MultPar[MultPoleIndex].Length, ThisBunch->nray, ThisBunch->stop, ThisBunch->x, ThisBunch->y, TrackFlag,
          ThisBunch->ngoodray_gpu, elemno, aper2, L, dB, Tilt, Lrad, Xfrms_gpu, dZmod, splitScale, StoppedParticles_gpu,
          PascalMatrix, Bang, MaxMultInd, *rSeed, rngStates, ThisBunch->ptype_gpu) ;
  gpuErrchk( cudaGetLastError() ) ;
  gpuErrchk( cudaFree(Xfrms_gpu) );
  gpuErrchk( cudaFree(MultAngleValue) );
  gpuErrchk( cudaFree(MultBValue) );
  gpuErrchk( cudaFree(MultTiltValue) );
  gpuErrchk( cudaFree(MultPoleIndexValue) );
#else
  for (ray=0 ;ray<ThisBunch->nray ; ray++)
    TrackBunchThruMult_kernel( MultPar[MultAngle].ValuePtr, MultPar[MultB].ValuePtr, MultPar[MultTilt].ValuePtr, MultPar[MultPoleIndex].ValuePtr,
          MultPar[MultPoleIndex].Length, ray, ThisBunch->stop, ThisBunch->x, ThisBunch->y, TrackFlag, &ThisBunch->ngoodray, elemno, aper2, L,
          dB, Tilt, Lrad, Xfrms, dZmod, splitScale, StoppedParticles, ThisBunch->ptype) ;
#endif
  
  egress:
    
    return stat;
    
}
#ifdef __CUDACC__
__global__ void TrackBunchThruMult_kernel(double* MultAngleValue, double* MultBValue, double* MultTiltValue, double* MultPoleIndexValue,
        int MultPoleIndexLength, int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray,
        int elemno, double aper2, double L, double dB, double Tilt, double Lrad, double* pXfrms, double dZmod, double splitScale, int* stp,
        double* PascalMatrix, double* Bang, double* MaxMultInd, unsigned long long rSeed, curandState *rState, unsigned short int* ptype)
#else
void TrackBunchThruMult_kernel(double* MultAngleValue, double* MultBValue, double* MultTiltValue, double* MultPoleIndexValue,
        int MultPoleIndexLength, int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray,
        int elemno, double aper2, double L, double dB, double Tilt, double Lrad, double Xfrms[6][2], double dZmod, double splitScale, int* stp,
        unsigned short int* ptype)
#endif
        
{
  int ray, raystart, coord, doStop, count ;
  double *x, *px, *y, *py, *z, *p0 ;
#ifdef __CUDA_ARCH__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
  curandState rState_local = rState[ray];
#else
  ray = nray;
#endif
#ifdef __CUDACC__
  double Xfrms[6][2] ;
  int irw, icol ;
  for (irw=0;irw<6;irw++)
    for (icol=0;icol<2;icol++)
      Xfrms[irw][icol]=pXfrms[(2*irw)+icol] ;
#endif
  
  raystart = 6*ray ;
  
  /* if the ray was previously stopped copy it over */
  
  if (stop[ray] > 0.)
  {
    for (coord=0 ; coord<6 ; coord++)
      yb[raystart+coord] = xb[raystart+coord] ;
    goto egress ;
  }
  
  /* If positivly charged particle, invert B */
  if (ptype[ray] == 1) {
    for (count=0; count<MultPoleIndexLength; count++)
      MultBValue[count]=-MultBValue[count];
  }
  
  /* make ray coordinates into local ones, including offsets etc which
   * are demanded by the transformation structure. */
  
  GetLocalCoordPtrs(xb, raystart,&x,&px,&y,&py,&z,&p0) ;
  
  ApplyTotalXfrm( Xfrms, UPSTREAM, TrackFlag, 0 ,x,px,y,py,z,p0) ;
  
  yb[raystart+4] = *z ;
  yb[raystart+5] = *p0 ;
  
  /* entrance-face aperture test, if requested */
  
  if (TrackFlag[Aper] == 1 && aper2>0)
  {
    doStop = CheckAperStopPart( xb,yb,stop,ngoodray,elemno,&aper2,ray,UPSTREAM,
            NULL, 0, stp, 0 ) ;
    if (doStop == 1)
      goto egress ;
  }
  
  /* Propagate through, and perform SR energy loss within the multipole
   * if required.  This is handled differently from the call to
   * PropagateRayThruMult in the QSO tracker by necessity. */
  
  /*		ThisBunch->y[raystart+4] = ThisBunch->x[raystart+4] ; */
#ifdef __CUDA_ARCH__
//   yb[raystart] = 999 ;
//   yb[raystart+1] = MultPoleIndex ;
//   return ;
  PropagateRayThruMult_gpu( L,
          MultBValue,
          MultTiltValue,
          MultPoleIndexValue,
          MultPoleIndexLength,
          MultAngleValue,
          dB, Tilt,
          &(xb[raystart]),
          &(yb[raystart]),
          TrackFlag[ZMotion],
          TrackFlag[SynRad], Lrad,stop,ngoodray,xb,yb,
          elemno, ray, splitScale, PascalMatrix, Bang, MaxMultInd, &rState_local ) ;
#else
  PropagateRayThruMult( L,
          MultBValue,
          MultTiltValue,
          MultPoleIndexValue,
          MultPoleIndexLength,
          MultAngleValue,
          dB, Tilt,
          &(xb[raystart]),
          &(yb[raystart]),
          TrackFlag[ZMotion],
          TrackFlag[SynRad], Lrad,stop,ngoodray,xb,yb,
          elemno, ray, splitScale ) ;
#endif
  
  if (TrackFlag[LorentzDelay] == 1)
    yb[raystart+4] += L*(LORENTZ_DELAY((*p0)) - dZmod) ;
  /*yb[raystart+5] = xb[raystart+5] ;*/
  
  /* exit-face aperture test, if requested */
  
  if (TrackFlag[Aper] == 1)
  {
    doStop = CheckAperStopPart( xb,yb,stop,ngoodray,elemno,&aper2,ray,DOWNSTREAM,
            NULL, 0, stp, 0 ) ;
    if (doStop == 1)
      goto egress ;
  }
  
  /* undo the coordinate transformations */
  
  GetLocalCoordPtrs(yb, raystart,&x,&px,&y,&py,&z,&p0) ;
  
  ApplyTotalXfrm( Xfrms, DOWNSTREAM, TrackFlag, dZmod,x,px,y,py,z,p0 ) ;
  
  /* check amplitude of outgoing angular momentum */
  
  doStop = CheckPperpStopPart( stop, ngoodray, elemno, ray, px, py, stp ) ;
  
  egress:
    /* copy rng state back to global memory */
#ifdef __CUDA_ARCH__
    rState[ray] = rState_local ;
#endif
    return ;
}


/*=====================================================================*/

/* Perform tracking of one bunch through one bend magnet, including
 * any transformations necessary to manage magnet, girder, or girder
 * mover position offsets etc.  All strength errors are also included.
 *
 * RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if: ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno; if
 * BEAMLINE{elemno} is not some sort of magnet; or if a ray
 * is encountered with momentum <=0 which was not stopped
 * (ie, if some previous procedure degraded a ray's momentum
 * to zero but forgot to stop it). */

int TrackBunchThruSBend( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TrackFlag, int supEdgeEffect1, int supEdgeEffect2, double L, double thisS )
{
  
  double Tilt ;                /* some basic parameters */
  double intB, intG ;
  double BScale, GScale ;
  double E1, E2, H1, H2 ;
  double Theta ;
  double hgap,hgapx,fint,fintx ;
  double hgap2,hgapx2 ;
  double TotalTilt ;
  double cTT, sTT, cT, sT ;
  double Tx, Ty ;
  double splitScale ;
  double Xfrms[6][2] ;           /* upstream and downstream coordinate
   * xfrms from magnet + girder + mover
   * offset values */
  int PS, PS2 ;
#ifndef __CUDACC__
  int ray ;              /* shortcut for 6*ray */
#endif
  struct Bunch* ThisBunch ;    /* a shortcut */
#ifdef __CUDACC__
  double *Xfrms_flat = NULL ;
#endif
  
  int stat = 1 ;
  double OffsetFromTiltError ;
  double AngleFromTiltError ;
  
  /* get the element parameters from BEAMLINE; exit with bad status if
   * parameters are missing or corrupted. */
  
  stat = GetDatabaseParameters( elemno, nSBendPar, SBendPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  splitScale = L / GetDBValue(SBendPar+SBendL) ;
  if (L<=0)
  {
    BadElementMessage( elemno+1 ) ;
    stat = 0 ;
    goto egress ;
  }
  
  intB = GetDBValue(SBendPar+SBendB) * splitScale ;
  if (SBendPar[SBendB].Length > 1)
    intG = *(SBendPar[SBendB].ValuePtr+1) * splitScale ;
  else
    intG = 0. ;
  
  Theta = GetDBValue(SBendPar+SBendAngle) * splitScale ;
  Tilt = GetDBValue(SBendPar+SBendTilt) ;
  
  E1 = GetSpecialSBendPar(&(SBendPar[SBendEdgeAngle]),0) * supEdgeEffect1 ;
  E2 = GetSpecialSBendPar(&(SBendPar[SBendEdgeAngle]),1) * supEdgeEffect2 ;
  if ( (cos(E1)==0.) || (cos(E2)==0.) )
  {
    stat = 0 ;
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  H1 = GetSpecialSBendPar(&(SBendPar[SBendEdgeCurvature]),0) * supEdgeEffect1 ;
  H2 = GetSpecialSBendPar(&(SBendPar[SBendEdgeCurvature]),1) * supEdgeEffect2 ;
  
  hgap  = GetSpecialSBendPar(&(SBendPar[SBendHGAP]),0) ;
  hgapx = GetSpecialSBendPar(&(SBendPar[SBendHGAP]),1) ;
  
  fint  = GetSpecialSBendPar(&(SBendPar[SBendFINT]),0) * supEdgeEffect1 ;
  fintx = GetSpecialSBendPar(&(SBendPar[SBendFINT]),1) * supEdgeEffect2 ;
  hgap2 = hgap * hgap ;
  hgapx2 = hgapx * hgapx ;
  
  /* if aperture is zero but aperture track flag is on,
   * it's an error.  Set error status and exit. */
  
  if ( ( (hgap2 == 0.)  && (TrackFlag[Aper] == 1) )
  ||
          ( (hgapx2 == 0.) && (TrackFlag[Aper] == 1) ) )
  {
    BadApertureMessage( elemno+1 ) ;
    stat = 0 ;
    goto egress ;
  }
  
  /* now the error parameters */
  
  BScale = 1. + GetDBValue(SBendPar+SBenddB) ;
  if (SBendPar[SBenddB].Length > 1)
    GScale = 1. + *(SBendPar[SBenddB].ValuePtr+1) ;
  else
    GScale = BScale ;
  
  intB *= BScale ;
  intG *= GScale ;
  
  /* now get the power supply parameters, if any */
  
  BScale = 1. ;
  GScale = 1. ;
  PS = (int)(GetDBValue(SBendPar+SBendPS)) ;
  if (SBendPar[SBendPS].Length > 1)
    PS2 = (int)(*(SBendPar[SBendPS].ValuePtr + 1)) ;
  else
    PS2 = PS ;
  if (PS > 0)
  {
    
    /* convert from Matlab to C indexing */
    
    PS -= 1 ;
    stat = GetDatabaseParameters( PS, nPSPar, PSPar,
            TrackPars, PSTable ) ;
    if (stat == 0)
    {
      BadPSMessage( elemno+1, PS+1 ) ;
      goto egress ;
    }
    BScale = GetDBValue(PSPar+PSAmpl) * (1.+ GetDBValue(PSPar+PSdAmpl) ) ;
  }
  if (PS2 > 0)
  {
    
    /* convert from Matlab to C indexing */
    
    PS2 -= 1 ;
    stat = GetDatabaseParameters( PS2, nPSPar, PSPar,
            TrackPars, PSTable ) ;
    if (stat == 0)
    {
      BadPSMessage( elemno+1, PS2+1 ) ;
      goto egress ;
    }
    GScale = GetDBValue(PSPar+PSAmpl) * (1.+ GetDBValue(PSPar+PSdAmpl) ) ;
  }
  intB *= BScale ;
  intG *= GScale ;
  
  /* now we get the complete input- and output- transformations for the
   * element courtesy of the relevant function */
  stat = GetTotalOffsetXfrms( SBendPar[SBendGirder].ValuePtr,
          &L,
          &thisS,
          SBendPar[SBendOffset].ValuePtr,
          Xfrms ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* compute the total tilt of the magnet */
  
  TotalTilt = Xfrms[5][0] + Tilt  ;
  
  /* precompute some trig functions */
  
  cTT = cos(TotalTilt) ;
  sTT = sin(TotalTilt) ;
  cT = 1-cos(TotalTilt - Tilt) ;
  sT = sin(TotalTilt - Tilt) ;
  Tx = cT*cos(Tilt) + sT*sin(Tilt) ;
  Ty = -sT*cos(Tilt) + cT*sin(Tilt) ;
  if (Theta != 0.)
    OffsetFromTiltError = L/Theta*(1-cos(Theta)) ;
  else
    OffsetFromTiltError = 0. ;
  AngleFromTiltError = sin(Theta) ;
  
  
  /* make a shortcut to get to the bunch of interest */
  
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  /* execute ray tracking kernel (loop over rays) */
#ifdef __CUDACC__
  double *Xfrms_gpu ;
  gpuErrchk( cudaMalloc((void **)&Xfrms_gpu, sizeof(double)*12) );
  Xfrms_flat = &(Xfrms[0][0]) ;
  gpuErrchk( cudaMemcpy(Xfrms_gpu, Xfrms_flat, sizeof(double)*12, cudaMemcpyHostToDevice) );
  TrackBunchThruSBend_kernel<<<blocksPerGrid, threadsPerBlock>>>( ThisBunch->nray, ThisBunch->x, ThisBunch->y, ThisBunch->stop,
          TrackFlag, Xfrms_gpu, cTT, sTT, Tx, Ty, OffsetFromTiltError, AngleFromTiltError, ThisBunch->ngoodray_gpu, hgap2, intB, intG, L,
          elemno, E1, H1, hgap, fint, Theta, E2, H2, hgapx, fintx, hgapx2, StoppedParticles_gpu, *rSeed, rngStates, ThisBunch->ptype_gpu) ;
  gpuErrchk( cudaGetLastError() );
  gpuErrchk( cudaFree(Xfrms_gpu) );
#else
  for (ray=0 ;ray<ThisBunch->nray ; ray++)
    TrackBunchThruSBend_kernel(ray, ThisBunch->x, ThisBunch->y, ThisBunch->stop, TrackFlag, Xfrms, cTT, sTT, Tx, Ty,
            OffsetFromTiltError, AngleFromTiltError, &ThisBunch->ngoodray, hgap2, intB, intG, L, elemno, E1, H1, hgap, fint, Theta,
            E2, H2, hgapx, fintx, hgapx2, StoppedParticles, ThisBunch->ptype) ;
#endif
  
  egress:
    
    return stat;
    
}
#ifdef __CUDACC__
__global__ void TrackBunchThruSBend_kernel(int nray, double* xb, double* yb, double* stop, int* TrackFlag, double* pXfrms, double cTT,
        double sTT, double Tx, double Ty, double OffsetFromTiltError, double AngleFromTiltError, int* ngoodray, double hgap2,
        double intB, double intG, double L, int elemno, double E1, double H1, double hgap, double fint, double Theta, double E2,
        double H2, double hgapx, double fintx, double hgapx2, int* stp, unsigned long long rSeed, curandState *rState,
        unsigned short int* ptype)
#else
void TrackBunchThruSBend_kernel(int nray, double* xb, double* yb, double* stop, int* TrackFlag, double Xfrms[6][2], double cTT,
        double sTT, double Tx, double Ty, double OffsetFromTiltError, double AngleFromTiltError, int* ngoodray, double hgap2,
        double intB, double intG, double L, int elemno, double E1, double H1, double hgap, double fint, double Theta, double E2,
        double H2, double hgapx, double fintx, double hgapx2, int* stp, unsigned short int* ptype)
#endif
{
  int ray, coord, doStop, raystart ;
  double ctemp ;
  double SR_dP = 0 ;
  double *x, *px, *y, *py, *z, *p0 ;
  double TijkTransv[4][10] ;
  double Tij6[13] ;
  double Pdes, delta, delta2, dp1, dp12 ;
#ifndef __CUDA_ARCH__
  double LastRayP = 0. ;
#endif
  Rmat Rface1, Rface2, Rbody ; /* linear maps */
  double T5xx1[10], T5xx2[10], T5xxbody[10] ; /* 2nd order maps */
  double* temp;
#ifdef __CUDACC__
  double Xfrms[6][2] ;
  int irw, icol ;
  for (irw=0;irw<6;irw++)
    for (icol=0;icol<2;icol++)
      Xfrms[irw][icol]=pXfrms[(2*irw)+icol] ;
#endif
#ifdef __CUDA_ARCH__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
  curandState rState_local = rState[ray] ;
#else
  ray = nray;
#endif
  
  raystart = 6*ray ;
  
  /* If positively charged, flip sign of angles and magnetic fields */
  if (ptype[ray] == 1) {
     intG=-intG;
     intB=-intB;
     Theta=-Theta;
     E1=-E1;
     E2=-E2;
  }
  
  /* if the ray was previously stopped copy it over */
  
  if (stop[ray] > 0.)
  {
    for (coord=0 ; coord<6 ; coord++)
      yb[raystart+coord] = xb[raystart+coord] ;
    goto egress ;
  }
  
  /* make ray coordinates into local ones, including offsets etc which
   * are demanded by the transformation structure. */
  
  GetLocalCoordPtrs(xb, raystart,&x,&px,&y,&py,&z,&p0) ;
  
  ApplyTotalXfrm( Xfrms, UPSTREAM, TrackFlag, 0,x,px,y,py,z,p0 ) ;
    
  /* rotate particles into the coordinate frame of the magnet.  Note that the
   * offsets are performed first, indicating that both rotation and translation
   * of each magnet is in the global coordinate system (ie, the translation is
   * not in the rotated local magnet coordinate system) */
  
  ctemp = (*x) ;
  (*x) = (*x) * cTT + (*y) * sTT ;
  (*y) = (*y) * cTT - ctemp * sTT ;
  ctemp = (*px) ;
  (*px) = (*px) * cTT + (*py) * sTT ;
  (*py) = (*py) * cTT - ctemp * sTT ;
  
  /* entrance-face aperture test, if requested */
  
  if (TrackFlag[Aper] == 1)
  {
    doStop = CheckAperStopPart( xb,yb,stop,ngoodray,elemno,&hgap2,ray,UPSTREAM,
            NULL, 0, stp, 1 ) ;
    if (doStop == 1)
      goto egress ;
  }
  
  yb[raystart+5] = *p0 ;
  
  /* Compute the SR momentum loss, if required, and apply 1/2 of the
   * loss here at the entry face of the element; as long as we're here,
   * check to see whether the particle has lost all of its momentum,
   * and if so stop it. */
  
  if (TrackFlag[SynRad] > SR_None)
  {
    double Beff, Bx, By ;
    By = intB + intG * (*x) ;
    Bx = intG * (*y) ;
    Beff = sqrt(Bx*Bx + By*By) ;
    if (Beff != 0) {
#ifdef __CUDA_ARCH__
      SR_dP = ComputeSRMomentumLoss_gpu( *p0, Beff, L, TrackFlag[SynRad], &rState_local ) ;
#else
      SR_dP = ComputeSRMomentumLoss( *p0, Beff, L, TrackFlag[SynRad] ) ;
#endif
      doStop = CheckP0StopPart( stop,ngoodray,xb,yb,elemno,ray,*p0-SR_dP, UPSTREAM, stp ) ;
      if (doStop == 1)
        goto egress ;
      *p0 -= SR_dP / 2 ;
      
    }
  }
  
  /* entrance face transformation : */
#ifndef __CUDA_ARCH__
  if (*p0 != LastRayP)
#endif
    GetBendFringeMap( L, intB, intG, *p0, E1, H1,
            hgap, fint, 1., Rface1, T5xx1 ) ;
  
  /* make use of the fact that the map is rather sparse */
  
  yb[raystart] = (*x) + T5xx1[0] * (*x) * (*x)
  + T5xx1[1] * (*y) * (*y) ;
  yb[raystart+1] = Rface1[1][0] * (*x) + (*px)
  + T5xx1[2] * (*x) * (*x)
  + T5xx1[3] * (*x) * (*px)
  + T5xx1[4] * (*y) * (*y)
  + T5xx1[5] * (*y) * (*py) ;
  yb[raystart+2] = (*y) + T5xx1[6] * (*x) * (*y) ;
  yb[raystart+3] = Rface1[3][2] * (*y) + (*py)
  + T5xx1[7] * (*x) * (*y)
  + T5xx1[8] * (*x) * (*py)
  + T5xx1[9] * (*px) * (*y) ;
  yb[raystart+4] = *z ;
  yb[raystart+5] = *p0 ;
  
  /* exchange x and y */
  /*XYExchange( &xb, &yb, nray ) ;*/
  temp = xb ;
  xb = yb ;
  yb = temp ;
  
  /* reassign the pointers to the "new" x  */
  GetLocalCoordPtrs(xb, raystart,&x,&px,&y,&py,&z,&p0) ;
  
  /* now for the body:   */
#ifndef __CUDA_ARCH__
  if (*p0 != LastRayP)
#endif
    GetLucretiaSBendMap( L, Theta, intB, intG, *p0, Rbody, T5xxbody, TijkTransv, Tij6 ) ;
  
  /* apply the map; since we rotated the coordinates, we know that the cross-plane
   * terms of the map are zero and can be neglected */
  
  yb[raystart] = *x*Rbody[0][0] + *px * Rbody[0][1]
          + Rbody[0][5] ;
  yb[raystart+1] = *x*Rbody[1][0] + *px * Rbody[1][1]
          + Rbody[1][5] ;
  yb[raystart+2] = *y*Rbody[2][2] + *py * Rbody[2][3] ;
  yb[raystart+3] = *y*Rbody[3][2] + *py * Rbody[3][3] ;
  yb[raystart+4] = *z + *x * Rbody[4][0] + *px * Rbody[4][1]
          + Rbody[4][5]
          + (*x) * (*x) * T5xxbody[0]
          + (*x) * (*px) * T5xxbody[1]
          + (*px) * (*px) * T5xxbody[2]
          + (*y) * (*y) * T5xxbody[3]
          + (*y) * (*py) * T5xxbody[4]
          + (*py) * (*py) * T5xxbody[5] ;
  yb[raystart+5] = *p0 ;
  
  // Second-order transverse terms
  Pdes = intB / Theta / GEV2TM ;
  delta = (*p0-Pdes)/Pdes ; delta2=delta*delta;
  dp1  = delta + 1 ;
  dp12 = dp1 * dp1 ;
  yb[raystart] += TijkTransv[0][0] * *x  * *x +
                  TijkTransv[0][1] * *x  * *px * dp1 +
                  TijkTransv[0][4] * *px * *px * dp12 +
                  TijkTransv[0][9] * *py * *py * dp12 +
                  Tij6[0] * *x * delta + Tij6[1] * *px * delta + Tij6[2] * delta2 ;
  yb[raystart+1] += TijkTransv[1][4] * *px * *px * dp1 +
                    TijkTransv[1][9] * *py * *py * dp12 ;
  yb[raystart+2] += TijkTransv[2][3] * *x * *py * dp1 + 
                    TijkTransv[2][6] * *px * *py * dp12 +
                    Tij6[7] * *py * delta ;
  
  /* now we have to do the coordinate exchange again */
  /*XYExchange( &xb, &yb, nray ) ;*/
  temp = xb ;
  xb = yb ;
  yb = temp ;
  GetLocalCoordPtrs(xb, raystart,&x,&px,&y,&py,&z,&p0) ;
  
  /* exit-face map */
#ifndef __CUDA_ARCH__
  if (*p0 != LastRayP)
#endif
    GetBendFringeMap( L, intB, intG, *p0, E2, H2,
            hgapx, fintx, -1., Rface2, T5xx2 ) ;
  
  
  /* make use of the fact that the map is rather sparse */
  
  yb[raystart] = (*x) + T5xx2[0] * (*x) * (*x)
  + T5xx2[1] * (*y) * (*y) ;
  yb[raystart+1] = Rface2[1][0] * (*x) + (*px)
  + T5xx2[2] * (*x) * (*x)
  + T5xx2[3] * (*x) * (*px)
  + T5xx2[4] * (*y) * (*y)
  + T5xx2[5] * (*y) * (*py) ;
  yb[raystart+2] = (*y) + T5xx2[6] * (*x) * (*y) ;
  yb[raystart+3] = Rface2[3][2] * (*y) + (*py)
  + T5xx2[7] * (*x) * (*y)
  + T5xx2[8] * (*x) * (*py)
  + T5xx2[9] * (*px) * (*y) ;
  yb[raystart+4] = *z ;
  yb[raystart+5] = *p0 ;
  
  
  
  /* preserve the current momentum */
#ifndef __CUDA_ARCH__
  LastRayP = *p0 ;
#endif
  
  /* exit-face aperture test, if requested */
  
  if (TrackFlag[Aper] == 1)
  {
    doStop = CheckAperStopPart( xb,yb,stop,ngoodray,elemno,&hgapx2,ray,DOWNSTREAM,
            NULL, 0, stp, 1 ) ;
    if (doStop == 1)
      goto egress ;
  }
  
  /* apply the 2nd half of SR eloss here */
  
  yb[raystart+5] -= SR_dP / 2 ;
  
  /* undo the coordinate rotations */
  
  ctemp = yb[raystart] ;
  yb[raystart] = yb[raystart] * cTT
          - yb[raystart+2] * sTT ;
  yb[raystart+2] = yb[raystart+2] * cTT
          + ctemp * sTT ;
  ctemp = yb[raystart+1] ;
  yb[raystart+1] = yb[raystart+1] * cTT
          - yb[raystart+3] * sTT ;
  yb[raystart+3] = yb[raystart+3] * cTT
          + ctemp * sTT ;
  
  /* undo the coordinate transformations */
  
  GetLocalCoordPtrs(yb, raystart,&x,&px,&y,&py,&z,&p0) ;
  
  ApplyTotalXfrm( Xfrms, DOWNSTREAM, TrackFlag, 0,x,px,y,py,z,p0 ) ;
  
  /* if the bend magnet has an error rotation, this will cause a deflection of the
   * beam wrt the design coordinate axis.  Apply this deflection now */
  
  yb[raystart] += Tx * OffsetFromTiltError ;
  yb[raystart+1] += Tx * AngleFromTiltError ;
  yb[raystart+2] += Ty * OffsetFromTiltError ;
  yb[raystart+3] += Ty * AngleFromTiltError ;
  
  /* finally, if the transverse momentum has gotten too high, stop the particle */
  
  doStop = CheckPperpStopPart( stop, ngoodray, elemno, ray, px, py, stp ) ;
  
  egress:
#ifdef __CUDA_ARCH__
    /* Copy rng state back to global memory */
    rState[ray] = rState_local ;
#endif
    return ;
}

/*=====================================================================*/

/* Perform tracking of one bunch through one RF structure, including
 * any transformations necessary to manage magnet, girder, or girder
 * mover position offsets etc.  All strength errors are also included.
 *
 * RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if: ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno; if
 * BEAMLINE{elemno} is not an Lcav; if a ray is encountered
 * with momentum <= 0 but which has not been STOPped.  */

int TrackBunchThruRF( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TrackFlag, int Mode )
{
  
  double L,dL,V ;                /* some basic parameters */
  double phi1, freq ;            /* more basic parameters */
  double aper2 ;                 /* square of aperture */
  double Xfrms[6][2] ;           /* upstream and downstream coordinate
   * xfrms from magnet + girder + mover
   * offset values */
  int Klys ;
  int ray,coord ;
  int raystart ;               /* shortcut for 6*ray */
  struct Bunch* ThisBunch ;    /* a shortcut */
  Rmat Rcav ;                  /* structure linear map */
  int Stop ;                   /* did the ray stop? */
  double LastRayP = 0. ;
  double LastRaydP = 0. ;
  double *Q ;
  double *xout, *yout ;
  double dP, Krf ;
  int ZSRno, TSRno, NSBPM ;
  int TLRno, TLRErrno ;
  int TLRClass, TLRErrClass ;
  static int SBPMCounter, SBPMcount2 ;
  double S0 ;
  struct SRWF* ThisZSR ;
  struct SRWF* ThisTSR ;
  int stat = 1 ;
  double phibunch, ddmy ;
  double tbunch  ;
  struct LRWFFreq* ThisTLRFreq ;
  struct LRWFFreq* ThisTLRErrFreq ;
  struct LRWFFreqKick* ThisStrucTLRFreqKick = NULL   ;
  struct LRWFFreqKick* ThisStrucTLRErrFreqKick = NULL ;
  
  
  /* variables related to slicing the structure for SBPMs or wakefields */
  
  static int nslice = 0 ;
  static int nslicealloc = 0 ;
  static int* doSBPM = NULL ;
  static double* Lfrac = NULL ;
  int TWFSliceno ;
  int slicecount ;
  
  double *x, *px, *y, *py, *z, *p0 ;
  
  /* Pointer which allows us to re-use most of this code for either LCAV or
   * TCAV */
  
  struct LucretiaParameter* CavPar=NULL ;
  
  /* rotation parameters, only used for TCAV */
  
  double Tilt, CosTilt=0, SinTilt=0 ;
  double dPKick ;
  
  /* get the element parameters from BEAMLINE; exit with bad status if
   * parameters are missing or corrupted. */
  
  if ( Mode == 0 ) /* LCAV */
  {
    stat = GetDatabaseParameters( elemno, nLcavPar, LcavPar,
            TrackPars, ElementTable ) ;
    CavPar = LcavPar ;
  }
  else if ( Mode == 1 ) /* TCAV */
  {
    stat = GetDatabaseParameters( elemno, nTcavPar, TcavPar,
            TrackPars, ElementTable ) ;
    CavPar = TcavPar ;
  }
  
  if (stat==0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  L = GetDBValue( CavPar+LcavL ) ;
  if (L<=0)
  {
    stat = 0 ;
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  V = GetDBValue( CavPar+LcavVolt ) ;
  phi1 = GetDBValue( CavPar+LcavPhase ) ;
  
  /* now for optional parameters */
  
  aper2 = GetDBValue( CavPar+Lcavaper ) ; aper2 *= aper2 ;
  freq = GetDBValue( CavPar+LcavFreq ) ;
  freq = freq * 1e6 ;                /* convert to Hz */
  Krf = 2* PI * freq / CLIGHT ;      /* wave # in 1/m */
  
  /* compute the time interval since the first bunch,
   * and the resulting RF phase error */
  
  if (FirstBunchAtRF[elemno] == -1)
    FirstBunchAtRF[elemno] = bunchno ;
  if (bunchno < LastBunchAtRF[elemno])
    FirstBunchAtRF[elemno] = bunchno ;
  LastBunchAtRF[elemno] = bunchno ;
  tbunch = (bunchno - FirstBunchAtRF[elemno])
  * ArgStruc->TheBeam->interval ;
  phibunch = modf( tbunch * freq, &ddmy ) ;
  phibunch *= 2 * PI ;
  
  /* if aperture is zero but aperture track flag is on,
   * it's an error.  Set error status and exit. */
  
  if ( (aper2 == 0.) && (TrackFlag[Aper] == 1) )
  {
    BadApertureMessage( elemno+1 ) ;
    stat = 0 ;
    goto egress ;
  }
  
  /* do all wakefield preparation work */
  
  stat = PrepareAllWF(CavPar[LcavWakes].ValuePtr,
          CavPar[LcavWakes].Length,
          TrackFlag, elemno, bunchno,
          ArgStruc->TheBeam,
          &ZSRno, &TSRno, &TLRno, &TLRErrno,
          &TLRClass, &TLRErrClass,
          &ThisZSR, &ThisTSR,
          &ThisTLRFreq, &ThisTLRErrFreq,
          &ThisStrucTLRFreqKick,
          &ThisStrucTLRErrFreqKick ) ;
  if (stat==0)
    goto egress ;
  
  /* now the error parameters */
  
  V *= (1. + GetDBValue( CavPar+LcavdV ) ) ;
  phi1 += GetDBValue( CavPar+LcavdPhase ) ;
  
  /* now get the klystron parameters, if any */
  
  Klys = (int)GetDBValue( CavPar+LcavKlystron ) ;
  if (Klys > 0)
  {
    
    /* convert from Matlab to C indexing */
    
    enum KlystronStatus* kstat ;
    Klys -= 1 ;
    stat = GetDatabaseParameters( Klys, nKlystronPar, KlystronPar,
            TrackPars, KlystronTable ) ;
    if (stat==0)
    {
      BadKlystronMessage( elemno+1, Klys+1 ) ;
      goto egress ;
    }
    V *= GetDBValue( KlystronPar+KlysAmpl )
    * (1. + GetDBValue( KlystronPar+KlysdAmpl ) ) ;
    phi1 += GetDBValue( KlystronPar+KlysPhase )
    + GetDBValue( KlystronPar+KlysdPhase ) ;
    kstat = GetKlystronStatus( Klys ) ;
    if (kstat != NULL)
    {
      if ( (*kstat == TRIPPED) || (*kstat == STANDBY) ||
              (*kstat == STANDBYTRIP)                       )
        V = 0. ;
    }
    else  /* null kstat */
    {
      BadKlystronMessage( elemno+1, Klys+1 ) ;
      stat = 0 ;
      goto egress ;
    }
    
    
  } /* end of klystron interlude */
  
  phi1 *= PI / 180. ; /* convert degrees to radians */
  phi1 += phibunch ;   /* add bunch arrival-time offset */
  V /= 1e3 ;          /* convert to GeV */
  
  /* now we get the complete input- and output- transformations for the
   * element courtesy of the relevant function */
  
  stat = GetTotalOffsetXfrms(             CavPar[LcavGirder].ValuePtr,
          CavPar[LcavL].ValuePtr,
          CavPar[LcavS].ValuePtr,
          CavPar[LcavOffset].ValuePtr,
          Xfrms ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* if this is a transverse cavity, construct the full tilt now */
  
  if ( Mode == 1 )
  {
    Tilt = GetDBValue(CavPar + TcavTilt) + Xfrms[5][0] ;
    CosTilt = cos(Tilt) ;
    SinTilt = sin(Tilt) ;
  }
  
  /* figure out how many slices we need, and what we need to do on each of them */
  
  stat = LcavSliceSetup( ArgStruc, elemno, TrackFlag, &nslice, &nslicealloc,
          &NSBPM, &doSBPM, &Lfrac, &TWFSliceno ) ;
  
  /* if the status is 1 then we can continue; if it's 0 something is very wrong and
   * we need to abort */
  
  if (stat == 0)
    goto egress ;
  
  /* figure out which slot in the SBPM data array we need to use, and do some
   * initialization if we are on bunch 1 */
  
  if (NSBPM > 0)
  {
    stat = SBPMSetup( ArgStruc, elemno, bunchno, NSBPM, &SBPMCounter ) ;
    if (stat == 0)
      goto egress ;
  }
  
  /* on the first bunch, set the S positions of the SBPMs within the structure: */
  
  if ( (bunchno+1 == ArgStruc->FirstBunch) && (NSBPM > 0) )
  {
    S0 = GetDBValue( CavPar+LcavS ) ;
    SBPMSetS( SBPMCounter, S0, L, nslice, doSBPM, Lfrac ) ;
  }
  
  /* make a shortcut to get to the bunch of interest */
  
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  /* loop over structure longitudinal slices */
  
  SBPMcount2 = 0 ;
  for (slicecount=0 ; slicecount<nslice ; slicecount++)
  {
    dL = L * Lfrac[slicecount] ;
    
    /* if this is a transverse cavity, the slice has a drift matrix ; get that now */
    
    if ( Mode == 1 )
      GetDriftMap( dL, Rcav ) ;
    
    /* if this is the slice on which we apply transverse wakes, do the transverse
     * wake calculations now */
    
    if (slicecount == TWFSliceno)
    {
      if (ThisTSR != NULL)
        ComputeTSRKicks( ThisTSR, L ) ;
      if (ThisTLRFreq != NULL)
      {
        stat = ComputeTLRFreqKicks( ThisTLRFreq, L, 0, ThisStrucTLRFreqKick,
                TLRFreqData[TLRno].nModes, bunchno,
                ArgStruc->TheBeam->TLRFreqDamping[TLRno],
                ArgStruc->TheBeam->TLRFreqxPhase[TLRno],
                ArgStruc->TheBeam->TLRFreqyPhase[TLRno],
                TLRFreqData[TLRno].Tilt,
                Xfrms[5][0]
                ) ;
        if (stat != 1)
          goto egress ;
      }
      if (ThisTLRErrFreq != NULL)
      {
        stat = ComputeTLRFreqKicks( ThisTLRErrFreq, L, 1, ThisStrucTLRErrFreqKick,
                TLRErrFreqData[TLRErrno].nModes, bunchno,
                ArgStruc->TheBeam->TLRErrFreqDamping[TLRErrno],
                ArgStruc->TheBeam->TLRErrFreqxPhase[TLRErrno],
                ArgStruc->TheBeam->TLRErrFreqyPhase[TLRErrno],
                TLRErrFreqData[TLRErrno].Tilt,
                Xfrms[5][0]
                ) ;
        if (stat != 1)
          goto egress ;
      }
    }
    
    /* loop over rays in the bunch */
    
    for (ray=0 ;ray<ThisBunch->nray ; ray++)
    {
      
      raystart = 6*ray ;
      
      /* if the ray was previously stopped copy it over */
      
      if (ThisBunch->stop[ray] > 0.)
      {
        for (coord=0 ; coord<6 ; coord++)
          ThisBunch->y[raystart+coord] = ThisBunch->x[raystart+coord] ;
        continue ;
      }
      
      /* make ray coordinates into local ones, including offsets etc which
       * are demanded by the transformation structure. */
      
      GetLocalCoordPtrs(ThisBunch->x, raystart,&x,&px,&y,&py,&z,&p0) ;
      Q = &(ThisBunch->Q[ray]) ;
      
      /* if this is the first slice, transform the ray coords to the reference
       * frame of the element and check the aperture */
      
      if ( slicecount==0 )
      {
        
        ApplyTotalXfrm( Xfrms, UPSTREAM, TrackFlag, 0,x,px,y,py,z,p0 ) ;
        
        /* entrance-face aperture test, if requested */
        
        if (TrackFlag[Aper] == 1)
        {
          Stop = CheckAperStopPart( ThisBunch->x,ThisBunch->y,ThisBunch->stop,&ThisBunch->ngoodray,elemno,&aper2,ray,UPSTREAM,
                  NULL, 0, StoppedParticles, 0 ) ;
          if (Stop == 1)
            continue ;
        }
        
      } /* end of first-slice activities */
      
      /* if this slice starts with a structure-BPM, accumulate the needed data
       * now */
      
      if (doSBPM[slicecount] > 0)
      {
        ArgStruc->sbpmdata[SBPMCounter]->Q[SBPMcount2] += (*Q) ;
        ArgStruc->sbpmdata[SBPMCounter]->x[SBPMcount2] += (*x) * (*Q) ;
        ArgStruc->sbpmdata[SBPMCounter]->y[SBPMcount2] += (*y) * (*Q);
      }
      
      /* if this is the SRWF_T slice, add the deflections to the beam position */
      
      if (slicecount==TWFSliceno)
      {
        if (ThisTSR != NULL)
        {
          (*px) += ThisTSR->binVx[ThisTSR->binno[ray]] / (*p0) ;
          (*py) += ThisTSR->binVy[ThisTSR->binno[ray]] / (*p0) ;
        }
        if (ThisTLRFreq != NULL)
        {
          (*px) += ThisTLRFreq->binVx[ThisTLRFreq->binno[ray]] / (*p0) ;
          (*py) += ThisTLRFreq->binVy[ThisTLRFreq->binno[ray]] / (*p0) ;
        }
        if (ThisTLRErrFreq != NULL)
        {
          (*px) += ThisTLRErrFreq->binVx[ThisTLRErrFreq->binno[ray]] / (*p0) ;
          (*py) += ThisTLRErrFreq->binVy[ThisTLRErrFreq->binno[ray]] / (*p0) ;
        }
      }
      
      /* compute momentum gain for the particle based on its z position, the
       * phase of the RF, the wave #, and the voltage.  Remember that the
       * particle momenta are in GeV, the voltage in GV, the phase in radians,
       * and the wave # in 1/m: 
       * If ptype=1 (e+), then invert Voltage to apply correct lorentz force */
      if (ThisBunch->ptype[ray] == 1) {
        dP = -V * cos( phi1 + Krf * (*z) ) * Lfrac[slicecount] ;
      }
      else {
        dP = V * cos( phi1 + Krf * (*z) ) * Lfrac[slicecount] ;
      }
      if ( Mode == 1)
      {
        dPKick = dP ;
        dP = 0 ;
        if ( TrackFlag[SynRad] != SR_None )
          dP -= ComputeSRMomentumLoss( *p0,
                  fabs(dPKick)*GEV2TM ,
                  dL,
                  TrackFlag[SynRad] ) ;
        dPKick /= *p0 ;
      }
      if (ThisZSR != NULL)
        dP -= dL * ThisZSR->K[ThisZSR->binno[ray]]  ;
      ThisBunch->y[raystart+5] = ThisBunch->x[raystart+5] + dP ;
      
      /* if the exit-momentum is < 0, stop the particle */
      
      Stop = CheckP0StopPart( ThisBunch->stop,&ThisBunch->ngoodray,ThisBunch->x,ThisBunch->y, elemno, ray,
              ThisBunch->y[raystart+5], DOWNSTREAM, StoppedParticles ) ;
      if (Stop != 0)
        continue ;
      
      /* get the structure map for the x-plane only */
      
      if ( Mode == 0 ) /* Track through linac cavity */
      {
        if ( (*p0 != LastRayP) || (dP != LastRaydP) )
          GetLcavMap( dL, *p0, 0., dP, 0., Rcav, 1 ) ;
        LastRayP = *p0 ;
        LastRaydP = dP ;
      }
      
      /* perform the matrix transformation; since we know the RF structure has rotational
       * symmetry use the x-plane 2x2 matrix on both x and y  planes.  Do this for both
       * types of cavity */
      
      ThisBunch->y[raystart]   = *x*Rcav[0][0] + *px * Rcav[0][1] ;
      ThisBunch->y[raystart+1] = *x*Rcav[1][0] + *px * Rcav[1][1] ;
      ThisBunch->y[raystart+2] = *y*Rcav[0][0] + *py * Rcav[0][1] ;
      ThisBunch->y[raystart+3] = *y*Rcav[1][0] + *py * Rcav[1][1] ;
      ThisBunch->y[raystart+4] = *z ;
      
      /* add the deflections and delay for the transverse cavity */
      
      if ( Mode == 1)
      {
        double dPx = dPKick * CosTilt ;
        double dPy = dPKick * SinTilt ;
        double dLov2 = dL / 2 ;
        ThisBunch->y[raystart]   += dPx * dLov2 ;
        ThisBunch->y[raystart+1] += dPx ;
        ThisBunch->y[raystart+2] += dPy * dLov2 ;
        ThisBunch->y[raystart+3] += dPy ;
        if (TrackFlag[ZMotion] == 1)
          ThisBunch->y[raystart+4] +=
                  0.5*L*( *px * *px + *py * *py
                  + *px * dPx + *py * dPy
                  + 0.5*dPx*dPx + 0.5*dPy*dPy ) ;
      }
      
      xout = &(ThisBunch->y[raystart]) ;
      yout = &(ThisBunch->y[raystart+2]) ;
      
      /* if the NEXT slice is the TWF slice, then we need to accumulate the mean
       * position of each bin on THIS slice */
      
      if (TWFSliceno == slicecount+1)
      {
        if (ThisTSR != NULL)
          AccumulateWFBinPositions( ThisTSR->binx,
                  ThisTSR->biny,
                  ThisTSR->binno[ray],
                  *xout, *yout, *Q ) ;
        if (ThisTLRFreq != NULL)
          AccumulateWFBinPositions( ThisTLRFreq->binx,
                  ThisTLRFreq->biny,
                  ThisTLRFreq->binno[ray],
                  *xout, *yout, *Q ) ;
      }
      
      /* Last-slice activities: */
      
      if (slicecount == nslice-1)
      {
        
        /* if the last slice ends with an SBPM, fill it with data now */
        
        if (doSBPM[nslice] > 0)
        {
          ArgStruc->sbpmdata[SBPMCounter]->Q[NSBPM-1] += (*Q) ;
          ArgStruc->sbpmdata[SBPMCounter]->x[NSBPM-1] += (*xout)*(*Q) ;
          ArgStruc->sbpmdata[SBPMCounter]->y[NSBPM-1] += (*yout)*(*Q);
        }
        
        /* exit-face aperture test, if requested */
        
        if (TrackFlag[Aper] == 1)
        {
          Stop = CheckAperStopPart( ThisBunch->x,ThisBunch->y,ThisBunch->stop,&ThisBunch->ngoodray,elemno,&aper2,ray,DOWNSTREAM,
                  NULL, 0, StoppedParticles, 0 ) ;
          if (Stop == 1)
            continue ;
        }
        
        /* undo the coordinate transformations */
        
        GetLocalCoordPtrs(ThisBunch->y, raystart,&x,&px,&y,&py,&z,&p0) ;
        
        ApplyTotalXfrm( Xfrms, DOWNSTREAM, TrackFlag, 0 ,x,px,y,py,z,p0) ;
        
        /* check amplitude of outgoing angular momentum */
        
        Stop = CheckPperpStopPart( ThisBunch->stop, &ThisBunch->ngoodray, elemno, ray,
                px, py, StoppedParticles ) ;
      }
      
    } /* end of coord loop */
    
    /* Apply longitudinal space charge if requested */
    if ( TrackFlag[LSC] > 0 )
      ProcLSC(ThisBunch,elemno,dL,(bunchno+1) * TrackFlag[LSC_storeData]) ;
    
    /* now if this is not the last slice, exchange the x and y coordinates so
     * that the next slice starts tracking the output coords of this slice */
    
    if (slicecount < nslice-1)
    {
      XYExchange( ThisBunch ) ;
    }
    
    /* if this slice was an SBPM slice, update the inner SBPM counter */
    
    if (doSBPM[slicecount] > 0)
      SBPMcount2++ ;
    
  } /* end of slice loop */
  
  /* if this is the last bunch to be tracked, and there are SBPMs,
   * compute the SBPM readings now */
  
  if ( (bunchno+1 == ArgStruc->LastBunch) && (NSBPM > 0) )
  {
    stat = ComputeSBPMReadings( SBPMCounter, elemno, Xfrms[5][0] ) ;
    if (stat==0)
    {
      BadElementMessage( elemno+1 ) ;
    }
    else
      stat = abs(stat) ;
  }
  
  egress:
    
    if (ThisStrucTLRFreqKick != NULL)
      PutThisElemTLRFreqKick( &(ThisStrucTLRFreqKick), elemno, bunchno,
              ArgStruc->LastBunch,
              ArgStruc->BunchwiseTracking,
              0 ) ;
    if (ThisStrucTLRErrFreqKick != NULL)
      PutThisElemTLRFreqKick( &(ThisStrucTLRErrFreqKick), elemno, bunchno,
              ArgStruc->LastBunch,
              ArgStruc->BunchwiseTracking,
              1 ) ;
    
    /* if this is the last bunch, and we are tracking element-wise, forget
     * about which bunch was first/last in this structure */
    
    if ( (bunchno+1 == ArgStruc->LastBunch) &&
            (ArgStruc->BunchwiseTracking == 0)    )
    {
      FirstBunchAtRF[elemno] = -1 ;
      LastBunchAtRF[elemno]  = -1 ;
    }
    
    return stat ;
    
}

/*=====================================================================*/

/* Perform tracking of one bunch through one beam position monitor,
 * including transformations related to the element/girder offset.
 * While doing the tracking, the data structures for the BPM
 * readings etc are accumulated if required.
 *
 * RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno */

int TrackBunchThruBPM( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TFlag, double splitL )
{
  
  static int BPMCounter ;           /* Which BPM is it? */
  int nBunchNeeded ;                /* how many bunches' dataspace? */
  double* retcatch ;
  int BunchSlot ;
  int i,j,k ;
  double Xfrms[6][2] ;
  double sintilt,costilt ;
  struct Bunch* ThisBunch ;
  double dxvec[6] ;
  double* gaussran ;
  double QBS ;
  int stat=1 ;
#ifdef __CUDACC__
  float xread_cnt, yread_cnt, Q_cnt, P_cnt, pxq_cnt, pyq_cnt, z_cnt ; /* bpmdata counters for kernel*/
  float sigma_cnt[36] ;
#else
  double xread_cnt, yread_cnt, Q_cnt, P_cnt, pxq_cnt, pyq_cnt, z_cnt ; /* bpmdata counters for kernel*/
  double sigma_cnt[36] ;
#endif


   /* for CUDA and CPU, TrackFlags is local memory, for CUDA TFlag is device memory*/
#ifdef __CUDACC__
  int* TrackFlags=0;
  TrackFlags=(int*) malloc(sizeof(int)*NUM_TRACK_FLAGS) ;
  memset(TrackFlags,0,sizeof(int)*NUM_TRACK_FLAGS);
  gpuErrchk( cudaMemcpy(TrackFlags, TFlag, sizeof(int)*NUM_TRACK_FLAGS, cudaMemcpyDeviceToHost) );
  float *Xfrms_flat = NULL ;
#else
  int* TrackFlags = TFlag ;
#endif          

  /* Get the BPM parameters */
  
  stat = GetDatabaseParameters( elemno, nBPMPar, BPMPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* start by doing the tracking */
  /* if internally spitting tracking then finish by passing splitL<0 to get BPM data after drift tracking */
  stat = TrackBunchThruDrift( elemno, bunchno, ArgStruc, TFlag, splitL ) ;

  /* if we do not need to save BPM information here, we can return */
  if ( ArgStruc->GetInstData == 0 || splitL > 0 || ( TrackFlags[GetBPMData] == 0 && TrackFlags[GetBPMBeamPars]==0 ) )
    goto egress ;
  
   /* if we're still here, we must want to accumulate some information, so
   * start that process */
  
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
   /* do initialization */
  
  BPMInstIndexingSetup( elemno, &FirstBPMElemno,
          &BPMElemnoLastCall, &BPMCounter ) ;
  
   /* figure out how many bunch-data slots we need for this BPM, based on
   * the number of bunches to be tracked and the multi/single switch.
   * While we're at it, figure out which data slot this bunch's data goes
   * into */
  
  BunchSlotSetup( TrackFlags, ArgStruc, bunchno,
          &nBunchNeeded, &BunchSlot     ) ;
  
   /* If this is the first bunch, then we've never tracked in this BPM
   * before and we should check the allocation of its data in the data
   * backbone */
  
  if (bunchno+1 == ArgStruc->FirstBunch)
  {
    if (bpmdata[BPMCounter] == NULL)
    {
      bpmdata[BPMCounter] = (struct BPMdat*) calloc(1,sizeof(struct BPMdat)) ;
      if (bpmdata[BPMCounter] == NULL)
      {
        BadBPMAllocMsg( elemno+1 ) ;
        stat = 0 ;
        goto egress ;
      }
      bpmdata[BPMCounter]->nbunchalloc = 0 ;
    } /* end allocation block */
    
    /* initialize stuff which has to be initialized on bunch 1 */
    
    bpmdata[BPMCounter]->nBunch      = 0 ;
    bpmdata[BPMCounter]->indx = elemno + 1 ;
    bpmdata[BPMCounter]->GetBeamPars = TrackFlags[GetBPMBeamPars] ;
    bpmdata[BPMCounter]->S = GetDBValue( BPMPar+BPMS ) ;
    bpmdata[BPMCounter]->Pmod = GetDBValue( BPMPar+BPMP )  ;
    
    /* allocate enough space for the BPM data on all bunches required */
    
    if (bpmdata[BPMCounter]->nbunchalloc < nBunchNeeded)
    {
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->xread) ) ;
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->yread) ) ;
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->sigma) ) ;
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->P) ) ;
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->z) ) ;
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->Q) ) ;
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->sumpxq) ) ;
      FreeAndNull( (void **)&(bpmdata[BPMCounter]->sumpyq) ) ;
      
      bpmdata[BPMCounter]->xread  =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      bpmdata[BPMCounter]->yread  =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      bpmdata[BPMCounter]->Q      =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      bpmdata[BPMCounter]->sigma  =
              (double*)calloc(36*nBunchNeeded,sizeof(double)) ;
      bpmdata[BPMCounter]->P      =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      bpmdata[BPMCounter]->z      =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      bpmdata[BPMCounter]->sumpxq =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      bpmdata[BPMCounter]->sumpyq =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      
      
      if ( (bpmdata[BPMCounter]->xread  == NULL) ||
              (bpmdata[BPMCounter]->yread  == NULL) ||
              (bpmdata[BPMCounter]->Q      == NULL) ||
              (bpmdata[BPMCounter]->sigma  == NULL) ||
              (bpmdata[BPMCounter]->P      == NULL) ||
              (bpmdata[BPMCounter]->z      == NULL) ||
              (bpmdata[BPMCounter]->sumpxq == NULL) ||
              (bpmdata[BPMCounter]->sumpyq == NULL)    )
      {
        BadBPMAllocMsg( elemno+1 ) ;
        stat = 0 ;
        goto egress ;
      }
      
      bpmdata[BPMCounter]->nbunchalloc = nBunchNeeded ;
    } /* end allocation of data vectors block */
    
    /* increment the counter which tells how many structures are filled on
     * exit */
    
    ArgStruc->nBPM++ ;
    
  } /* end first-bunch check/initialization/allocation block */
  
  /* we need to clear out the present data slot if we are on the first
   * bunch, or if we are doing multibunch tracking */
  
  if ( (TrackFlags[MultiBunch] == 1) ||
          (bunchno+1 == ArgStruc->FirstBunch) )
  {
    bpmdata[BPMCounter]->xread[BunchSlot]  = 0. ;
    bpmdata[BPMCounter]->yread[BunchSlot]  = 0. ;
    bpmdata[BPMCounter]->P[BunchSlot]      = 0. ;
    bpmdata[BPMCounter]->z[BunchSlot]      = 0. ;
    bpmdata[BPMCounter]->Q[BunchSlot]      = 0. ;
    bpmdata[BPMCounter]->sumpxq[BunchSlot] = 0. ;
    bpmdata[BPMCounter]->sumpyq[BunchSlot] = 0. ;
    for (i=0 ; i<36 ; i++)
      bpmdata[BPMCounter]->sigma[36*BunchSlot+i] = 0. ;
  }
  
  /* get the transformations needed based on BPM mechanical offsets */
  
  stat = GetTotalOffsetXfrms( BPMPar[BPMGirder].ValuePtr,
          BPMPar[BPML].ValuePtr,
          BPMPar[BPMS].ValuePtr,
          BPMPar[BPMOffset].ValuePtr,
          Xfrms ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* pre-compute sine and cosine components of the tilt */
  
  sintilt = sin(Xfrms[5][0]) ;
  costilt = cos(Xfrms[5][0]) ;
  
  /* Accumulate the appropriate summation of particle positions in the
   * data structure.  Bear in mind that the present data vector is the
   * positions after tracking thru the BPM but in the survey coordinate
   * system, so we have to make appropriate transformations to the BPM
   * coordinate system */
  xread_cnt=0; yread_cnt=0; Q_cnt=0; P_cnt=0; pxq_cnt=0; pyq_cnt=0; z_cnt=0;
  memset(sigma_cnt, 0, 36*sizeof(double) );
#ifdef __CUDACC__
  float *Xfrms_gpu ;
  float *xread_gpu, *yread_gpu, *Q_gpu, *P_gpu, *pxq_gpu, *pyq_gpu, *z_gpu, *sigma_gpu ; /* bpmdata counters for kernel*/
  gpuErrchk( cudaMalloc((void **)&Xfrms_gpu, sizeof(float)*12) );
  Xfrms_flat = (float*)&(Xfrms[0][0]) ;
  gpuErrchk( cudaMemcpy(Xfrms_gpu, Xfrms_flat, sizeof(float)*12, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMalloc((void **)&xread_gpu, sizeof(float)) ); 
  gpuErrchk( cudaMalloc((void **)&yread_gpu, sizeof(float)) );
  gpuErrchk( cudaMalloc((void **)&Q_gpu, sizeof(float)) ); gpuErrchk( cudaMalloc((void **)&P_gpu, sizeof(float)) );
  gpuErrchk( cudaMalloc((void **)&pxq_gpu, sizeof(float))); gpuErrchk( cudaMalloc((void **)&pyq_gpu, sizeof(float)) );
  gpuErrchk( cudaMalloc((void **)&z_gpu, sizeof(float)) ); gpuErrchk( cudaMalloc((void **)&sigma_gpu, 36*sizeof(float)) );
  gpuErrchk( cudaMemset(xread_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(yread_gpu, 0, sizeof(float)) );
  gpuErrchk( cudaMemset(Q_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(P_gpu, 0, sizeof(float)) );
  gpuErrchk( cudaMemset(pxq_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(pyq_gpu, 0, sizeof(float)) );
  gpuErrchk( cudaMemset(z_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(sigma_gpu, 0, sizeof(float)*36) );
  TrackBunchThruBPM_kernel<<<blocksPerGrid, threadsPerBlock>>>( ThisBunch->nray, Xfrms_gpu, TFlag, ThisBunch->Q, ThisBunch->y, ThisBunch->stop,
								(float)sintilt, (float)costilt, xread_gpu, yread_gpu, Q_gpu, P_gpu, pxq_gpu, pyq_gpu, z_gpu, sigma_gpu) ;
  gpuErrchk( cudaGetLastError() ) ;
  gpuErrchk( cudaMemcpy(&xread_cnt, xread_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&yread_cnt, yread_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&Q_cnt, Q_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&P_cnt, P_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&pxq_cnt, pxq_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&pyq_cnt, pyq_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&z_cnt, z_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(sigma_cnt, sigma_gpu, sizeof(float)*36, cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaFree(Xfrms_gpu) );
  gpuErrchk( cudaFree(xread_gpu) );
  gpuErrchk( cudaFree(yread_gpu) );
  gpuErrchk( cudaFree(Q_gpu) );
  gpuErrchk( cudaFree(P_gpu) );
  gpuErrchk( cudaFree(pxq_gpu) );
  gpuErrchk( cudaFree(pyq_gpu) );
  gpuErrchk( cudaFree(z_gpu) );
  gpuErrchk( cudaFree(sigma_gpu) );
#else
  for (i=0 ;i<ThisBunch->nray ; i++)
    TrackBunchThruBPM_kernel(i, Xfrms, TFlag, ThisBunch->Q, ThisBunch->y, ThisBunch->stop, sintilt, costilt,
           &xread_cnt, &yread_cnt, &Q_cnt, &P_cnt, &pxq_cnt, &pyq_cnt, &z_cnt, sigma_cnt ) ;
#endif

  /* put accumulated data into necessary data structure slots */
  bpmdata[BPMCounter]->xread[BunchSlot] += xread_cnt ;
  bpmdata[BPMCounter]->yread[BunchSlot] += yread_cnt ;
  bpmdata[BPMCounter]->Q[BunchSlot] += Q_cnt ;

  if (TrackFlags[GetBPMBeamPars]==1) /* P, z, and sigma returned */
  {
    bpmdata[BPMCounter]->P[BunchSlot]      += P_cnt ;
    bpmdata[BPMCounter]->sumpxq[BunchSlot] += pxq_cnt ;
    bpmdata[BPMCounter]->sumpyq[BunchSlot] += pyq_cnt ;
    bpmdata[BPMCounter]->z[BunchSlot]      += z_cnt;

    for (j=0; j<6; j++)
    {
      for (k=j ; k<6 ; k++)
      {
        bpmdata[BPMCounter]->sigma[36*BunchSlot+6*j+k] += sigma_cnt[6*j+k] ;
      }
    } 
  }
  
  /* Do we need to complete the calculation of first and second moments?
   * If we are doing multibunch tracking, the answer is yes; if we are not
   * but we are on the last bunch, the answer is also yes */
  if ( (TrackFlags[MultiBunch] == 1)    ||
          (bunchno+1 == ArgStruc->LastBunch)    )
  {
    
    /* increment the counter which tells TrackThruSetReturn how many bunch
     * data slots are filled */
    
    bpmdata[BPMCounter]->nBunch++ ;
    
    /* normalize the charge out of the first-moment information */
    
    QBS = bpmdata[BPMCounter]->Q[BunchSlot] ;
    if (QBS == 0.)
      QBS = 1 ;
    
    bpmdata[BPMCounter]->xread[BunchSlot] /= QBS ;
    bpmdata[BPMCounter]->yread[BunchSlot] /= QBS ;
    
    if (TrackFlags[GetBPMBeamPars]==1) /* P, z, and sigma returned */
    {
      bpmdata[BPMCounter]->P[BunchSlot]      /= QBS ;
      bpmdata[BPMCounter]->sumpxq[BunchSlot] /= QBS ;
      bpmdata[BPMCounter]->sumpyq[BunchSlot] /= QBS ;
      bpmdata[BPMCounter]->z[BunchSlot]      /= QBS ;
      
      dxvec[0] = bpmdata[BPMCounter]->xread[BunchSlot] ;
      dxvec[1] = bpmdata[BPMCounter]->sumpxq[BunchSlot] ;
      dxvec[2] = bpmdata[BPMCounter]->yread[BunchSlot] ;
      dxvec[3] = bpmdata[BPMCounter]->sumpyq[BunchSlot] ;
      dxvec[4] = bpmdata[BPMCounter]->z[BunchSlot] ;
      dxvec[5] = bpmdata[BPMCounter]->P[BunchSlot] ;
      
      /* normalize the charge out of the 2nd moment information and subtract
       * the appropriate first moment components */
      
      for (j=0; j<6; j++)
      {
        for (k=j ; k<6 ; k++)
        {
          bpmdata[BPMCounter]->sigma[36*BunchSlot+6*j+k] /= QBS ;
          bpmdata[BPMCounter]->sigma[36*BunchSlot+6*j+k] -=
                  dxvec[j] * dxvec[k] ;
          bpmdata[BPMCounter]->sigma[36*BunchSlot+6*k+j] =
                  bpmdata[BPMCounter]->sigma[36*BunchSlot+6*j+k] ;
          
        }
      }
      
    }

    /* Apply the BPM scale factor */
    
    retcatch = BPMPar[BPMdScale].ValuePtr ;
    if ( retcatch != NULL )
    {
      double scale = 1. + *retcatch ;
      bpmdata[BPMCounter]->xread[BunchSlot] *= scale ;
      bpmdata[BPMCounter]->yread[BunchSlot] *= scale ;
    }
    
    /* as a final step, apply the BPM electrical offset and resolution to
     * the BPM reading */
    
    retcatch = BPMPar[BPMElecOffset].ValuePtr ;
    if ( (retcatch != NULL) && (BPMPar[BPMElecOffset].Length > 1) )
    {
      bpmdata[BPMCounter]->xread[BunchSlot] += retcatch[0] ;
      bpmdata[BPMCounter]->yread[BunchSlot] += retcatch[1] ;
    }
    
    retcatch = BPMPar[BPMResolution].ValuePtr ;
    if ( (retcatch != NULL) && (BPMPar[BPMResolution].Length > 0) )
    {
      gaussran = RanGaussVecPtr( 2 ) ;
      bpmdata[BPMCounter]->xread[BunchSlot] +=
              retcatch[0]*gaussran[0] ;
      bpmdata[BPMCounter]->yread[BunchSlot] +=
              retcatch[0]*gaussran[1] ;
    }

  }
  
  /* now return */
  
  egress:
#ifdef __CUDACC__
  free(TrackFlags);
#endif         
    return stat ;
    
}

#ifdef __CUDACC__
__global__ void TrackBunchThruBPM_kernel(int nray, float* pXfrms, int* TrackFlags, double* qb, double* yb, double* stop, float sintilt, float costilt,
        float* xread_ret, float* yread_ret, float* Q_ret, float* P_ret, float* sumpxq_ret,
        float* sumpyq_ret, float* z_ret, float* sigma_ret )
#else
void TrackBunchThruBPM_kernel(int nray, double Xfrms[6][2], int* TrackFlags, double* qb, double* yb, double* stop, double sintilt, double costilt,
        double* xread_cnt, double* yread_cnt, double* Q_cnt, double* P_cnt, double* sumpxq_cnt, double* sumpyq_cnt,
        double* z_cnt, double sigma_cnt[36] )
#endif  
{    
  
  /* Use temp shared variables for variables that need to be accumulated so atomic adds can be efficiently utilized for CUDA code */
int ray, raystart, k, j;
#ifdef __CUDACC__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
  __shared__ float xread_cnt, yread_cnt, Q_cnt ;
  xread_cnt=0; yread_cnt=0; Q_cnt=0;
  __shared__ float P_cnt, sumpxq_cnt, sumpyq_cnt, z_cnt;
  __shared__ float sigma_cnt[36] ;
  P_cnt=0; sumpxq_cnt=0; sumpyq_cnt=0; z_cnt=0;
  memset( sigma_cnt, 0, sizeof(float)*36 ) ;
  __syncthreads();
  float Xfrms[6][2] ;
  int irw, icol ;
  float dx,dpx,dy,dpy,dz,P,Q ;
  float dxvec[6] ;
  for (irw=0;irw<6;irw++)
    for (icol=0;icol<2;icol++)
      Xfrms[irw][icol]=pXfrms[(2*irw)+icol] ;
#else
  double dx,dpx,dy,dpy,dz,P,Q ;
  double dxvec[6] ;
  ray = nray;
#endif
  raystart = 6*ray ;
  if (stop[ray] !=0.)
    return ;

  /* get the contribution from this ray, including its weight and the
   * transformation from the element offset. */
#ifdef __CUDACC__
  dx  = (float)yb[raystart] - Xfrms[0][1]
    + Xfrms[4][1] * (float)yb[raystart+1] ;
  dpx = (float)yb[raystart+1] - Xfrms[1][1] ;
  dy  = (float)yb[raystart+2] - Xfrms[2][1]
    + Xfrms[4][1] * (float)yb[raystart+3] ;
  dpy = (float)yb[raystart+3] - Xfrms[3][1] ;
  dz  = (float)yb[raystart+4] -	Xfrms[4][1]
          + 0.5 * Xfrms[4][1] *
    (  (float)yb[raystart+1] * (float)yb[raystart+1] +
       (float)yb[raystart+3] * (float)yb[raystart+3]   ) ;
  P   = (float)yb[raystart+5] ;
  Q   = (float)qb[ray] ;
#else
  dx  = yb[raystart] - Xfrms[0][1]
    + Xfrms[4][1] * yb[raystart+1] ;
  dpx = yb[raystart+1] - Xfrms[1][1] ;
  dy  = yb[raystart+2] - Xfrms[2][1]
    + Xfrms[4][1] * yb[raystart+3] ;
  dpy = yb[raystart+3] - Xfrms[3][1] ;
  dz  = yb[raystart+4] -	Xfrms[4][1]
          + 0.5 * Xfrms[4][1] *
    (  yb[raystart+1] * yb[raystart+1] +
       yb[raystart+3] * yb[raystart+3]   ) ;
  P   = yb[raystart+5] ;
  Q   = qb[ray] ;
#endif

  /* put into a vector, including rotations (if the BPM is rotated,
   * by convention clockwise, then a +x offset -> -y contribution and a
   * +y offset -> +x contribution) */

  dxvec[0] = dx * costilt + dy * sintilt ;
  dxvec[1] = dpx * costilt + dpy * sintilt ;
  dxvec[2] = -dx * sintilt + dy * costilt ;
  dxvec[3] = -dpx * sintilt + dpy * costilt ;
  dxvec[4] = dz ;
  dxvec[5] = P ;

  /* accumulate this ray, weighted by charge, to go into the necessary slots
   * in the data structure */
#ifdef __CUDACC__
  atomicAdd( &xread_cnt, dxvec[0] * Q ) ;
  atomicAdd( &yread_cnt, dxvec[2] * Q ) ;
  atomicAdd( &Q_cnt, Q ) ;
#else  
  *xread_cnt += dxvec[0] * Q ;
  *yread_cnt += dxvec[2] * Q ;
  *Q_cnt += Q ;
#endif
  if (TrackFlags[GetBPMBeamPars]==1) /* P, z, and sigma returned */
  {
#ifdef __CUDACC__
    atomicAdd( &P_cnt, dxvec[5] * Q ) ;
    atomicAdd( &sumpxq_cnt, dxvec[1] * Q ) ;
    atomicAdd( &sumpyq_cnt, dxvec[3] * Q ) ;
    atomicAdd( &z_cnt, dxvec[4] * Q ) ;
#else    
    *P_cnt      += dxvec[5] * Q ;
    *sumpxq_cnt += dxvec[1] * Q ;
    *sumpyq_cnt += dxvec[3] * Q ;
    *z_cnt      += dxvec[4] * Q ;
#endif
    for (j=0; j<6; j++)
      for (k=j ; k<6 ; k++)
#ifdef __CUDACC__
        atomicAdd( &sigma_cnt[6*j+k], Q * dxvec[k]*dxvec[j] ) ;
#else        
        sigma_cnt[6*j+k] += Q * dxvec[k]*dxvec[j] ;
#endif 
  }
#ifdef __CUDACC__ 
  /* Add the sub-pieces of the accumulated variables that exist in shared memory back into global GPU memory */
  __syncthreads();
  if (threadIdx.x>0)
    return;
  atomicAdd( xread_ret, xread_cnt ) ;
  atomicAdd( yread_ret, yread_cnt ) ;
  atomicAdd( Q_ret, Q_cnt ) ;
  if (TrackFlags[GetBPMBeamPars]==1)
  {
    atomicAdd( P_ret, P_cnt ) ;
    atomicAdd( sumpxq_ret, sumpxq_cnt ) ;
    atomicAdd( sumpyq_ret, sumpyq_cnt ) ;
    atomicAdd( z_ret, z_cnt ) ;
    for (j=0; j<6; j++)
    {
      for (k=0; k<6; k++)
      {
        atomicAdd( &sigma_ret[6*j+k], sigma_cnt[6*j+k] ) ;
      }
    }
  }
#endif    
} 

/*=====================================================================*/

/* Perform tracking of one bunch through one instrument of some sort.
 * Tracking includes transformations related to the element/girder
 * offset.  While doing the tracking, the data structures for beam
 * positions, sigmas, whatever are accumulated if required. */

/* RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno */

int TrackBunchThruInst( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TFlag, double splitL )
{
  
  static int instCounter ;           /* Which inst is it? */
  int nBunchNeeded ;                /* how many bunches' dataspace? */
  int BunchSlot ;
  double Xfrms[6][2] ;
  double sintilt,costilt ;
  struct Bunch* ThisBunch ;
  double QBS ;
  int stat=1 ;
#ifdef __CUDACC__
  float xread_cnt, yread_cnt, Q_cnt, P_cnt, pxq_cnt, pyq_cnt, z_cnt ; /* bpmdata counters for kernel*/
  float sigma_cnt[4] ;
#else
  double xread_cnt, yread_cnt, Q_cnt, P_cnt, pxq_cnt, pyq_cnt, z_cnt ; /* bpmdata counters for kernel*/
  double sigma_cnt[4] ;
#endif
  
     /* for CUDA and CPU, TrackFlags is local memory, for CUDA TFlag is device memory*/
#ifdef __CUDACC__
  int* TrackFlags=0;
  TrackFlags=(int*) malloc(sizeof(int)*NUM_TRACK_FLAGS) ;
  memset(TrackFlags,0,sizeof(int)*NUM_TRACK_FLAGS);
  gpuErrchk( cudaMemcpy(TrackFlags, TFlag, sizeof(int)*NUM_TRACK_FLAGS, cudaMemcpyDeviceToHost) );
  float *Xfrms_flat = NULL ;
#else
  int* TrackFlags = TFlag ;
#endif        
  
  /* Get the element data, exit with bad status if problems */
  stat = GetDatabaseParameters( elemno, nInstPar, InstPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* start by doing the tracking */
  /* if internally spitting tracking then finish by passing splitL<0 to get BPM data after drift tracking */
  stat = TrackBunchThruDrift( elemno, bunchno, ArgStruc, TFlag, splitL ) ;

  /* if we do not need to save BPM information here, we can return */
  if ( ArgStruc->GetInstData == 0 || splitL > 0 || TrackFlags[GetInstData] == 0 )
    goto egress ;
  
  /* if we're still here, we must want to accumulate some information, so
   * start that process */
  
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  /* setup the indexing */
  
  BPMInstIndexingSetup( elemno, &FirstInstElemno,
          &InstElemnoLastCall, &instCounter ) ;
  
  /* figure out how many bunch-data slots we need for this inst, based on
   * the number of bunches to be tracked and the multi/single switch.
   * While we're at it, figure out which data slot this bunch's data goes
   * into */
  
  BunchSlotSetup( TrackFlags, ArgStruc, bunchno,
          &nBunchNeeded, &BunchSlot     ) ;
  
  /* If this is the first bunch, then we've never tracked in this inst
   * before and we should check the allocation of its data in the data
   * backbone */
  
  if (bunchno+1 == ArgStruc->FirstBunch)
  {
    if (instdata[instCounter] == NULL)
    {
      instdata[instCounter] = (struct INSTdat*)calloc(1,sizeof(struct INSTdat)) ;
      if (instdata[instCounter] == NULL)
      {
        BadBPMAllocMsg( elemno+1 ) ;
        stat = 0 ;
        goto egress ;
      }
      instdata[instCounter]->nbunchalloc = 0 ;
    } /* end allocation block */
    
    /* initialize stuff which has to be initialized on bunch 1 */
    
    instdata[instCounter]->nBunch      = 0 ;
    instdata[instCounter]->indx = elemno + 1 ;
    instdata[instCounter]->S = GetDBValue( InstPar+InstS ) ;
    
    /* allocate enough space for the instrument data on all bunches needed */
    
    if (instdata[instCounter]->nbunchalloc < nBunchNeeded )
    {
      FreeAndNull( (void**)&(instdata[instCounter]->x) ) ;
      FreeAndNull( (void**)&(instdata[instCounter]->y) ) ;
      FreeAndNull( (void**)&(instdata[instCounter]->z) ) ;
      FreeAndNull( (void**)&(instdata[instCounter]->sig11) ) ;
      FreeAndNull( (void**)&(instdata[instCounter]->sig13) ) ;
      FreeAndNull( (void**)&(instdata[instCounter]->sig33) ) ;
      FreeAndNull( (void**)&(instdata[instCounter]->sig55) ) ;
      FreeAndNull( (void**)&(instdata[instCounter]->Q) ) ;
      
      instdata[instCounter]->x =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      instdata[instCounter]->y =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      instdata[instCounter]->z =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      instdata[instCounter]->sig11 =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      instdata[instCounter]->sig33 =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      instdata[instCounter]->sig55 =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      instdata[instCounter]->sig13 =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      instdata[instCounter]->Q =
              (double*)calloc(nBunchNeeded,sizeof(double)) ;
      
      if ( (instdata[instCounter]->x     == NULL) ||
              (instdata[instCounter]->y     == NULL) ||
              (instdata[instCounter]->z     == NULL) ||
              (instdata[instCounter]->sig11 == NULL) ||
              (instdata[instCounter]->sig33  == NULL) ||
              (instdata[instCounter]->sig55  == NULL) ||
              (instdata[instCounter]->sig13 == NULL) ||
              (instdata[instCounter]->Q     == NULL)    )
      {
        BadBPMAllocMsg( elemno+1 ) ;
        stat = 0 ;
        goto egress ;
      }
      instdata[instCounter]->nbunchalloc = nBunchNeeded ;
    } /* end of allocation of data vectors block */
    
    /* increment the counter which tells how many structures are filled on
     * exit */
    
    ArgStruc->nINST++ ;
    
  } /* end first-bunch check/initialization/allocation block */
  
  /* we need to clear out the present data slot if we are on the first
   * bunch, or if we are doing multibunch tracking */
  
  if ( (TrackFlags[MultiBunch] == 1) ||
          (bunchno+1 == ArgStruc->FirstBunch) )
  {
    instdata[instCounter]->x[BunchSlot] = 0 ;
    instdata[instCounter]->y[BunchSlot] = 0 ;
    instdata[instCounter]->z[BunchSlot] = 0 ;
    instdata[instCounter]->sig11[BunchSlot] = 0 ;
    instdata[instCounter]->sig33[BunchSlot] = 0 ;
    instdata[instCounter]->sig55[BunchSlot] = 0 ;
    instdata[instCounter]->sig13[BunchSlot] = 0 ;
    instdata[instCounter]->Q[BunchSlot] = 0 ;
  }
  
  /* get the transformations needed based on mechanical offsets */
  
  stat = GetTotalOffsetXfrms( InstPar[InstGirder].ValuePtr,
          InstPar[InstL].ValuePtr,
          InstPar[InstS].ValuePtr,
          InstPar[InstOffset].ValuePtr,
          Xfrms ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* pre-compute sine and cosine components of the tilt */
  
  sintilt = sin(Xfrms[5][0]) ;
  costilt = cos(Xfrms[5][0]) ;
  
  /* Accumulate the appropriate summation of particle positions in the
   * data structure.  Bear in mind that the present data vector is the
   * positions after tracking thru the inst but in the survey coordinate
   * system, so we have to make appropriate transformations to the inst
   * coordinate system */

  xread_cnt=0; yread_cnt=0; Q_cnt=0; P_cnt=0; pxq_cnt=0; pyq_cnt=0; z_cnt=0;
  memset(sigma_cnt, 0, 4*sizeof(double) );
#ifdef __CUDACC__
  float *Xfrms_gpu ;
  float *xread_gpu, *yread_gpu, *Q_gpu, *P_gpu, *pxq_gpu, *pyq_gpu, *z_gpu, *sigma_gpu ; /* bpmdata counters for kernel*/
  gpuErrchk( cudaMalloc((void **)&Xfrms_gpu, sizeof(float)*12) );
  Xfrms_flat = (float*)&(Xfrms[0][0]) ;
  gpuErrchk( cudaMemcpy(Xfrms_gpu, Xfrms_flat, sizeof(float)*12, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMalloc((void **)&xread_gpu, sizeof(float)) ); gpuErrchk( cudaMalloc((void **)&yread_gpu, sizeof(float)) );
  gpuErrchk( cudaMalloc((void **)&Q_gpu, sizeof(float)) ); gpuErrchk( cudaMalloc((void **)&P_gpu, sizeof(float)) );
  gpuErrchk( cudaMalloc((void **)&pxq_gpu, sizeof(float)) ); gpuErrchk( cudaMalloc((void **)&pyq_gpu, sizeof(float)) );
  gpuErrchk( cudaMalloc((void **)&z_gpu, sizeof(float)) ); gpuErrchk( cudaMalloc((void **)&sigma_gpu, 4*sizeof(float)) );
  gpuErrchk( cudaMemset(xread_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(yread_gpu, 0, sizeof(float)) );
  gpuErrchk( cudaMemset(Q_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(P_gpu, 0, sizeof(float)) );
  gpuErrchk( cudaMemset(pxq_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(pyq_gpu, 0, sizeof(float)) );
  gpuErrchk( cudaMemset(z_gpu, 0, sizeof(float)) ); gpuErrchk( cudaMemset(sigma_gpu, 0, sizeof(float)*4) );
  TrackBunchThruInst_kernel<<<blocksPerGrid, threadsPerBlock>>>( ThisBunch->nray, Xfrms_gpu, TFlag, ThisBunch->Q, ThisBunch->y, ThisBunch->stop,
								(float)sintilt, (float)costilt, xread_gpu, yread_gpu, Q_gpu, P_gpu, pxq_gpu, pyq_gpu, z_gpu, sigma_gpu) ;
  gpuErrchk( cudaGetLastError() ) ;
  gpuErrchk( cudaMemcpy(&xread_cnt, xread_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&yread_cnt, yread_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&Q_cnt, Q_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&P_cnt, P_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&pxq_cnt, pxq_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&pyq_cnt, pyq_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(&z_cnt, z_gpu, sizeof(float), cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaMemcpy(sigma_cnt, sigma_gpu, sizeof(float)*4, cudaMemcpyDeviceToHost) );
  gpuErrchk( cudaFree(Xfrms_gpu) );
  gpuErrchk( cudaFree(xread_gpu) );
  gpuErrchk( cudaFree(yread_gpu) );
  gpuErrchk( cudaFree(Q_gpu) );
  gpuErrchk( cudaFree(P_gpu) );
  gpuErrchk( cudaFree(pxq_gpu) );
  gpuErrchk( cudaFree(pyq_gpu) );
  gpuErrchk( cudaFree(z_gpu) );
  gpuErrchk( cudaFree(sigma_gpu) );
#else
  int i ;
  for (i=0 ;i<ThisBunch->nray ; i++)
    TrackBunchThruInst_kernel(i, Xfrms, TFlag, ThisBunch->Q, ThisBunch->y, ThisBunch->stop, sintilt, costilt,
           &xread_cnt, &yread_cnt, &Q_cnt, &P_cnt, &pxq_cnt, &pyq_cnt, &z_cnt, sigma_cnt ) ;
#endif  

  /* put accumulated data into necessary data structure slots */
  instdata[instCounter]->x[BunchSlot] += xread_cnt ;
  instdata[instCounter]->y[BunchSlot] += yread_cnt ;
  instdata[instCounter]->z[BunchSlot] += z_cnt ;
  instdata[instCounter]->Q[BunchSlot] +=  Q_cnt ;
  instdata[instCounter]->sig11[BunchSlot] += sigma_cnt[0] ;
  instdata[instCounter]->sig33[BunchSlot] += sigma_cnt[1] ;
  instdata[instCounter]->sig55[BunchSlot] += sigma_cnt[2] ;
  instdata[instCounter]->sig13[BunchSlot] += sigma_cnt[3] ;
  
  /* Do we need to complete the calculation of first and second moments?
   * If we are doing multibunch tracking, the answer is yes; if we are not
   * but we are on the last bunch, the answer is also yes */
  
  if ( (TrackFlags[MultiBunch] == 1)    ||
          (bunchno+1 == ArgStruc->LastBunch)    )
  {
    
    /* increment the counter which tells TrackThruSetReturn how many bunch
     * data slots are filled */
    
    instdata[instCounter]->nBunch++ ;
    
    /* normalize the charge out of the various data slots */
    
    QBS = instdata[instCounter]->Q[BunchSlot] ;
    if (QBS == 0.)
      QBS = 1. ;
    
    instdata[instCounter]->x[BunchSlot]     /= QBS ;
    instdata[instCounter]->y[BunchSlot]     /= QBS ;
    instdata[instCounter]->z[BunchSlot]     /= QBS ;
    instdata[instCounter]->sig11[BunchSlot] /= QBS ;
    instdata[instCounter]->sig33[BunchSlot] /= QBS ;
    instdata[instCounter]->sig55[BunchSlot] /= QBS ;
    instdata[instCounter]->sig13[BunchSlot] /= QBS ;
    
    /* subtract the <x>*<x> component from the sigmas */
    
    instdata[instCounter]->sig11[BunchSlot] -=
            (instdata[instCounter]->x[BunchSlot] *
            instdata[instCounter]->x[BunchSlot]   ) ;
    instdata[instCounter]->sig33[BunchSlot] -=
            (instdata[instCounter]->y[BunchSlot] *
            instdata[instCounter]->y[BunchSlot]   ) ;
    instdata[instCounter]->sig55[BunchSlot] -=
            (instdata[instCounter]->z[BunchSlot] *
            instdata[instCounter]->z[BunchSlot]   ) ;
    instdata[instCounter]->sig13[BunchSlot] -=
            (instdata[instCounter]->x[BunchSlot] *
            instdata[instCounter]->y[BunchSlot]   ) ;
    
  }
  
  egress:
    
    return stat;
    
}

#ifdef __CUDACC__
__global__ void TrackBunchThruInst_kernel(int nray, float* pXfrms, int* TrackFlags, double* qb, double* yb, double* stop, float sintilt, float costilt,
        float* xread_ret, float* yread_ret, float* Q_ret, float* P_ret, float* sumpxq_ret,
        float* sumpyq_ret, float* z_ret, float* sigma_ret )
#else
void TrackBunchThruInst_kernel(int nray, double Xfrms[6][2], int* TrackFlags, double* qb, double* yb, double* stop, double sintilt, double costilt,
        double* xread_cnt, double* yread_cnt, double* Q_cnt, double* P_cnt, double* sumpxq_cnt, double* sumpyq_cnt,
        double* z_cnt, double sigma_cnt[4] )
#endif  
{    
  
  int ray, raystart, k, j;
#ifdef __CUDACC__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
  __shared__ float xread_cnt, yread_cnt, Q_cnt ;
  xread_cnt=0; yread_cnt=0; Q_cnt=0;
  __shared__ float P_cnt, sumpxq_cnt, sumpyq_cnt, z_cnt;
  __shared__ float sigma_cnt[36] ;
  P_cnt=0; sumpxq_cnt=0; sumpyq_cnt=0; z_cnt=0;
  memset( sigma_cnt, 0, sizeof(float)*36 ) ;
  __syncthreads();
  float Xfrms[6][2] ;
  int irw, icol ;
  float dx,dpx,dy,dpy,dz,P,Q ;
  float dxvec[6] ;
  for (irw=0;irw<6;irw++)
    for (icol=0;icol<2;icol++)
      Xfrms[irw][icol]=pXfrms[(2*irw)+icol] ;
#else
  double dx,dpx,dy,dpy,dz,P,Q ;
  double dxvec[6] ;
  ray = nray;
#endif
  raystart = 6*ray ;
  if (stop[ray] !=0.)
    return ;
  
   /* get the contribution from this ray, including its weight and the
   * transformation from the element offset. */
#ifdef __CUDACC__
  dx  = (float)yb[raystart] - Xfrms[0][1]
    + Xfrms[4][1] * (float)yb[raystart+1] ;
  dpx = (float)yb[raystart+1] - Xfrms[1][1] ;
  dy  = (float)yb[raystart+2] - Xfrms[2][1]
    + Xfrms[4][1] * (float)yb[raystart+3] ;
  dpy = (float)yb[raystart+3] - Xfrms[3][1] ;
  dz  = (float)yb[raystart+4] -	Xfrms[4][1]
          + 0.5 * Xfrms[4][1] *
    (  (float)yb[raystart+1] * (float)yb[raystart+1] +
       (float)yb[raystart+3] * (float)yb[raystart+3]   ) ;
  P   = (float)yb[raystart+5] ;
  Q   = (float)qb[ray] ;
#else
  dx  = yb[raystart] - Xfrms[0][1]
    + Xfrms[4][1] * yb[raystart+1] ;
  dpx = yb[raystart+1] - Xfrms[1][1] ;
  dy  = yb[raystart+2] - Xfrms[2][1]
    + Xfrms[4][1] * yb[raystart+3] ;
  dpy = yb[raystart+3] - Xfrms[3][1] ;
  dz  = yb[raystart+4] -	Xfrms[4][1]
          + 0.5 * Xfrms[4][1] *
    (  yb[raystart+1] * yb[raystart+1] +
       yb[raystart+3] * yb[raystart+3]   ) ;
  P   = yb[raystart+5] ;
  Q   = qb[ray] ;
#endif

  /* put into a vector, including rotations (if the BPM is rotated,
   * by convention clockwise, then a +x offset -> -y contribution and a
   * +y offset -> +x contribution) */

  dxvec[0] = dx * costilt + dy * sintilt ;
  dxvec[1] = dpx * costilt + dpy * sintilt ;
  dxvec[2] = -dx * sintilt + dy * costilt ;
  dxvec[3] = -dpx * sintilt + dpy * costilt ;
  dxvec[4] = dz ;
  dxvec[5] = P ;

  /* accumulate this ray, weighted by charge, to go into the necessary slots
   * in the data structure */
#ifdef __CUDACC__
  atomicAdd( &xread_cnt, dxvec[0] * Q ) ;
  atomicAdd( &yread_cnt, dxvec[2] * Q ) ;
  atomicAdd( &Q_cnt, Q ) ;
#else  
  *xread_cnt += dxvec[0] * Q ;
  *yread_cnt += dxvec[2] * Q ;
  *Q_cnt += Q ;
#endif
  if (TrackFlags[GetBPMBeamPars]==1) /* P, z, and sigma returned */
  {
#ifdef __CUDACC__
    atomicAdd( &P_cnt, dxvec[5] * Q ) ;
    atomicAdd( &sumpxq_cnt, dxvec[1] * Q ) ;
    atomicAdd( &sumpyq_cnt, dxvec[3] * Q ) ;
    atomicAdd( &z_cnt, dxvec[4] * Q ) ;
#else    
    *P_cnt      += dxvec[5] * Q ;
    *sumpxq_cnt += dxvec[1] * Q ;
    *sumpyq_cnt += dxvec[3] * Q ;
    *z_cnt      += dxvec[4] * Q ;
#endif
    for (j=0; j<6; j++)
      for (k=j ; k<6 ; k++)
#ifdef __CUDACC__
        atomicAdd( &sigma_cnt[6*j+k], Q * dxvec[k]*dxvec[j] ) ;
#else        
        sigma_cnt[6*j+k] += Q * dxvec[k]*dxvec[j] ;
#endif 
  }
#ifdef __CUDACC__ 
  /* Add the sub-pieces of the accumulated variables that exist in shared memory back into global GPU memory */
  __syncthreads();
  if (threadIdx.x>0)
    return;
  atomicAdd( xread_ret, xread_cnt ) ;
  atomicAdd( yread_ret, yread_cnt ) ;
  atomicAdd( Q_ret, Q_cnt ) ;
  if (TrackFlags[GetBPMBeamPars]==1)
  {
    atomicAdd( P_ret, P_cnt ) ;
    atomicAdd( sumpxq_ret, sumpxq_cnt ) ;
    atomicAdd( sumpyq_ret, sumpyq_cnt ) ;
    atomicAdd( z_ret, z_cnt ) ;
    for (j=0; j<6; j++)
    {
      for (k=0; k<6; k++)
      {
        atomicAdd( &sigma_ret[6*j+k], sigma_cnt[6*j+k] ) ;
      }
    }
  }
#endif
}

/*=====================================================================*/

/* Perform tracking of one bunch through one corrector magnet, including
 * any transformations necessary to manage magnet, girder, or girder
 * mover rotations.  All strength errors are also included.  Does
 * slightly different things depending on whether the calling routine
 * indicates that the magnet is an XCOR or a YCOR. */

/* RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if: ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno; if
 * BEAMLINE{elemno} is not some sort of magnet;
 * if a ray is encountered which has momentum <= 0 but which
 * has not been STOPped. */

int TrackBunchThruCorrector( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TrackFlag, int xycorflag, double splitL, double splitS )
{
  double Lov2,B,B2=0,dB,Tilt,L=0.0 ;     /* some basic parameters */
  double Xfrms[6][2] ;           /* upstream and downstream coordinate
   * xfrms from magnet + girder + mover
   * offset values */
  int PS ;
#ifndef __CUDACC__
  int ray ;
#endif  
  struct Bunch* ThisBunch ;    /* a shortcut */
  int stat = 1 ;
  double Lrad, fabsB ;
  struct LucretiaParameter* CPar ;
  /* depending on whether this is an XCOR, a YCOR, or an XYCOR, we use
   * different dictionaries */
  
  
  if (xycorflag == XYCOR)
    CPar = XYCorrectorPar ;
  else
    CPar = CorrectorPar ;
  /* get the element parameters from BEAMLINE; exit with bad status if
   * any parameters are missing or corrupted. */
  
  stat = GetDatabaseParameters( elemno, nCorrectorPar, CPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  Lov2 = GetDBValue( CPar+CorL ) / 2. ;
  B = GetDBValue( CPar+CorB ) ;
  Tilt = GetDBValue( CPar+CorTilt ) ;
  Lrad = GetDBValue( CPar+CorLrad ) ;
  if (Lrad <= 0.)
    Lrad = 2*Lov2 ;
  /* Split element treatment */
  if ( splitL == 0 ) {
    splitL = 1 ;
    L = Lov2 * 2;
  }
  else {
    splitL = splitL / (Lov2 * 2) ;
    L *= Lov2 * 2 * splitL ;
    Lrad *= splitL ;
    Lov2 *= splitL ;
  }
  /* now the error parameters */
  
  B *= (1. + GetDBValue( CPar+CordB ) ) ;
  
  /* now get the power supply parameters, if any */
  PS = (int)(GetDBValue( CPar+CorPS ) )  ;
  if (PS > 0)
  {
    
    /* convert from Matlab to C indexing */
    
    PS -= 1 ;
    stat = GetDatabaseParameters( PS, nPSPar, PSPar,
            TrackPars, PSTable ) ;
    if (stat == 0)
    {
      BadPSMessage( elemno+1, PS+1 ) ;
      goto egress ;
    }
    B *= GetDBValue( PSPar+PSAmpl ) * (1.+GetDBValue( PSPar+PSdAmpl ) ) ;
    
  } /* end of PS interlude */
  /* now we handle the case of an XYCOR, which has an additional B field,
   * an additional power supply, and an additional field error.  While we're
   * at it, arrange for B to always be associated with the horizontal kick, and
   * B2 always with the vertical kick.  */
  
  switch (xycorflag) {
    
    case XYCOR :
      
      B2 = *(CPar[CorB].ValuePtr+1) ;
      dB = *(CPar[CordB].ValuePtr+1) ;
      B2 *= (1. + dB) ;
      PS = (int)(*(CPar[CorPS].ValuePtr+1)) ;
      if (PS > 0)
      {
        
        /* convert from Matlab to C indexing */
        
        PS -= 1 ;
        stat = GetDatabaseParameters( PS, nPSPar, PSPar,
                TrackPars, PSTable ) ;
        if (stat == 0)
        {
          BadPSMessage( elemno+1, PS+1 ) ;
          goto egress ;
        }
        B2 *= GetDBValue( PSPar+PSAmpl ) * (1.+GetDBValue( PSPar+PSdAmpl ) ) ;
        
      } /* end of PS interlude */
      break ;
      
    case XCOR :
      
      B2 = 0 ;
      break ;
      
    case YCOR :
      
      B2 = B ;
      B = 0 ;
      break ;
      
  } /* end of switch statement on xycorflag */
  /* Split element treatment */
  B *= splitL ;
  B2 *= splitL ;
  
  
  fabsB = sqrt(B*B + B2*B2) ;
  
  /* now we get the complete input- and output- transformations for the
   * element courtesy of the relevant function */
  if ( splitS == 0 ) {
    if ( CPar[CorS].ValuePtr == NULL )
      splitS = 0 ;
    else
      splitS = *CPar[CorS].ValuePtr ;
  }
  
  stat = GetTotalOffsetXfrms( CPar[CorGirder].ValuePtr,
          &L,
          &splitS,
          CPar[CorOffset].ValuePtr,
          Xfrms ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  /* since the rotation transformation can be applied to the magnet
   * rather than the beam, do that now  (GW 2/21/2015: move to ray loop to allow for +ve particles)*/
  
  Tilt += Xfrms[5][0] ;
  
  /* make a shortcut to get to the bunch of interest */
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  /* use the drift-tracker first */
  stat = TrackBunchThruDrift( elemno, bunchno, ArgStruc, TrackFlag, L ) ;
  /* loop over rays in the bunch */
#ifdef __CUDACC__  
  TrackBunchThruCorrector_kernel<<<blocksPerGrid, threadsPerBlock>>>(ThisBunch->nray, B, B2, ThisBunch->ptype_gpu, ThisBunch->stop,
          Tilt, TrackFlag, fabsB, ThisBunch->x, ThisBunch->y, Lrad, ThisBunch->ngoodray_gpu, elemno, StoppedParticles_gpu, Lov2, rngStates) ;
  gpuErrchk( cudaGetLastError() );
#else  
  for (ray=0 ;ray<ThisBunch->nray ; ray++)
    TrackBunchThruCorrector_kernel(ray, B, B2, ThisBunch->ptype, ThisBunch->stop, Tilt, TrackFlag, fabsB, ThisBunch->x, ThisBunch->y, 
            Lrad, &ThisBunch->ngoodray, elemno, StoppedParticles, Lov2) ;
#endif
  egress:
    
    return stat;
    
}

#ifdef __CUDACC__
__global__ void TrackBunchThruCorrector_kernel(int nray, double B, double B2, unsigned short int* ptype, double* stop, double Tilt, int* TrackFlag,
        double fabsB, double* xb, double* yb, double Lrad, int* ngoodray, int elemno, int* stp, double Lov2, curandState *rState)
#else
void TrackBunchThruCorrector_kernel(int nray, double B, double B2, unsigned short int* ptype, double* stop, double Tilt, int* TrackFlag,
        double fabsB, double* xb, double* yb, double Lrad, int* ngoodray, int elemno, int* stp, double Lov2)
#endif  
{
int ray, raystart, doStop ;
double XField, YField ;      /* x and y deflecting fields */
double dx, dy ;              /* x and y deflection */
double dxp, dyp ;
double SR_dP = 0 ;
#ifdef __CUDACC__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
  curandState rState_local = rState[ray] ;
#else
  ray = nray;
#endif  
  raystart = 6*ray ;
    
  /* if previously stopped, skip this particle;
   * we don't need to copy its coordinates, since TrackBunchThruDrift
   * has already moved the x values to y. */    
  if ( stop[ray] > 0. )
    return ;
  
  /* Flip field sign for positive charged particles */
  if (ptype[ray] == 1) {
    B=-B;
    B2=-B2;
  }
    
  /* Compute correction fields */
  XField = (B * cos(Tilt) - B2*sin(Tilt)) / GEV2TM ;
  YField = (B * sin(Tilt) + B2*cos(Tilt)) / GEV2TM ;
  dx = XField * Lov2 ;
  dy = YField * Lov2 ;
    
  /* apply synchrotron radiation */
  if (TrackFlag[SynRad] != SR_None && Lrad>0 && fabsB>0) {
#ifdef __CUDACC__
    SR_dP = ComputeSRMomentumLoss_gpu( yb[raystart+5], fabsB, Lrad, TrackFlag[SynRad], &rState_local ) ;
#else    
    SR_dP = ComputeSRMomentumLoss( yb[raystart+5],fabsB, Lrad,TrackFlag[SynRad] ) ;
#endif
    SR_dP /= 2 ;
    yb[raystart+5]  -= SR_dP ;
    doStop = CheckP0StopPart(stop,ngoodray,xb,yb,elemno,ray,yb[raystart+5]-SR_dP, UPSTREAM, stp) ;
    if (doStop != 0)
      return ;
  }
    
  /* add the kick.  Note that rays with zero energy
   * are detected and stopped by TrackThruMain, and do not
   * need to be trapped here. */
  dxp = XField / yb[raystart+5] ;
  dyp = YField / yb[raystart+5] ;
  if (TrackFlag[ZMotion] == 1)
    yb[raystart+4] += Lov2 * ( yb[raystart+1]*dxp + yb[raystart+3]*dyp + 0.5 * (dxp*dxp + dyp*dyp) ) ;
  yb[raystart+1] += dxp ;
  yb[raystart]   += dx  / yb[raystart+5] ;
  yb[raystart+3] += dyp ;
  yb[raystart+2] += dy  / yb[raystart+5] ;
  yb[raystart+5] -= SR_dP ;
    
  /* check amplitude of outgoing angular momentum */
  doStop = CheckPperpStopPart( stop, ngoodray , elemno, ray, yb+raystart+1, yb+raystart+3, stp ) ;
}


/*=====================================================================*/

/* Track one bunch thru one collimator
 *
 * RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   Will fail if: ArgStruc does not contain a well-defined
 * and self-consistent structure for bunch # bunchno; if
 * BEAMLINE{elemno} is not some sort of collimator. */

int TrackBunchThruCollimator( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TrackFlag, double splitL, double splitS )
{
#ifdef __CUDACC__  
  double *Xfrms_flat = NULL ;
#else
  int ray ;
#endif  
  int stat = 1 ;
  double Xfrms[6][2] ;           /* upstream and downstream coordinate
   * xfrms from magnet + girder + mover
   * offset values */
  double aper2[2] ;              /* square of apertures */
  int shape ;                    /* collimator shape */
  struct Bunch* ThisBunch ;      /* shortcut */
  double Tilt, *L, *S ;
  
  /* get the element parameters from BEAMLINE; exit with bad status if
   * any parameters are missing or corrupted. */
  stat = GetDatabaseParameters( elemno, nCollPar, CollPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* compute the square of the half-gap in x and y degrees of freedom */
  
  aper2[0] = CollPar[Collaper].ValuePtr[0] *
          CollPar[Collaper].ValuePtr[0]    ;
  aper2[1] = CollPar[Collaper].ValuePtr[1] *
          CollPar[Collaper].ValuePtr[1]    ;
  
  /* get the complete coordinate transforms including any from the girder */
  if (splitL == 0) {
    L = CollPar[CollL].ValuePtr ;
    S = CollPar[CollS].ValuePtr ;
  }
  else {
    L = &splitL ;
    S = &splitS ;
  }
  
  
  stat = GetTotalOffsetXfrms( CollPar[CollGirder].ValuePtr,
          L,
          S,
          CollPar[CollOffset].ValuePtr,
          Xfrms                         ) ;
  
  /* if the status is 1, then everything was found and unpacked OK.
   * If it's zero, then something was seriously wrong so abort. */
  
  if (stat == 0)
  {
    BadOffsetMessage( elemno+1 ) ;
    goto egress ;
  }
  
  Tilt = GetDBValue( CollPar + CollTilt ) + Xfrms[5][0] ;
  
  /* get the geometry of the collimator from the database */
  
  shape = GetCollimatorGeometry( elemno ) ;
  if (shape == COLL_UNKNOWN)
  {
    BadElementMessage( elemno+1 ) ;
    stat = 0 ;
    goto egress ;
  }
  
  /* make a shortcut to get to the bunch of interest */
  
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  /* if apertures are turned on, do the upstream transformation and
   * aperture check */
#ifdef __CUDACC__
  double *Xfrms_gpu ;
  gpuErrchk( cudaMalloc(&Xfrms_gpu, sizeof(double)*12) );
  Xfrms_flat = &(Xfrms[0][0]) ;
  gpuErrchk( cudaMemcpy(Xfrms_gpu, Xfrms_flat, sizeof(double)*12, cudaMemcpyHostToDevice) );
  TrackBunchThruCollimator_kernel<<<blocksPerGrid, threadsPerBlock>>>(ThisBunch->nray, 1, ThisBunch->x, ThisBunch->y, ThisBunch->stop, TrackFlag,
          ThisBunch->ngoodray_gpu, elemno, aper2[0], aper2[1], shape, Tilt, StoppedParticles_gpu, Xfrms_gpu) ;
  gpuErrchk( cudaPeekAtLastError() );
#else  
  for (ray=0 ;ray<ThisBunch->nray ; ray++)
    TrackBunchThruCollimator_kernel(ray, 1, ThisBunch->x, ThisBunch->y, ThisBunch->stop, TrackFlag, &ThisBunch->ngoodray, elemno,
      aper2[0], aper2[1], shape, Tilt, StoppedParticles, Xfrms) ;
#endif      
  
  /* now track thru the intervening drift */
  stat = TrackBunchThruDrift( elemno, bunchno, ArgStruc, TrackFlag, *L ) ;
  if (stat==0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* if apertures are turned on, do the downstream aperture check and
   * transform */
#ifdef __CUDACC__
  TrackBunchThruCollimator_kernel<<<blocksPerGrid, threadsPerBlock>>>(ThisBunch->nray, 0, ThisBunch->x, ThisBunch->y, ThisBunch->stop, TrackFlag,
          ThisBunch->ngoodray_gpu, elemno, aper2[0], aper2[1], shape, Tilt, StoppedParticles_gpu, Xfrms_gpu) ;
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaFree(Xfrms_gpu) );
#else  
  for (ray=0 ;ray<ThisBunch->nray ; ray++)
   TrackBunchThruCollimator_kernel(ray, 0, ThisBunch->x, ThisBunch->y, ThisBunch->stop, TrackFlag, &ThisBunch->ngoodray, elemno,
      aper2[0], aper2[1], shape, Tilt, StoppedParticles, Xfrms) ;
#endif  
  
  egress :
    
    return stat ;
    
}

#ifdef __CUDACC__
__global__ void TrackBunchThruCollimator_kernel(int nray, int isupstream, double* xb, double* yb, double* stop, int* TrackFlag, int* ngoodray, int elemno,
        double aper2_0, double aper2_1, int shape, double Tilt, int* stp, double* pXfrms)
#else
void TrackBunchThruCollimator_kernel(int nray, int isupstream, double* xb, double* yb, double* stop, int* TrackFlag, int* ngoodray, int elemno,
        double aper2_0, double aper2_1, int shape, double Tilt, int* stp, double Xfrms[6][2])
#endif
{
  int ray, raystart, Stop ;
  double *x, *px, *y, *py, *z, *p0 ;
  double aper2[2] ;
  aper2[0]=aper2_0; aper2[1]=aper2_1;
#ifdef __CUDACC__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
  double Xfrms[6][2] ;
  int irw, icol ;
  for (irw=0;irw<6;irw++)
    for (icol=0;icol<2;icol++)
      Xfrms[irw][icol]=pXfrms[(2*irw)+icol] ;
#else
  ray = nray;
#endif  
  raystart = 6*ray ;
  
  /* only check aperture if flag set */
  if (TrackFlag[Aper] <= 0)
    return;
  
   /* if the ray was previously stopped ignore it  */
  if (stop[ray] > 0.)
    return ;

  /* make ray coordinates into local ones, including offsets etc which
   * are demanded by the transformation structure. */
  if (isupstream==1) {
    GetLocalCoordPtrs(xb, raystart,&x,&px,&y,&py,&z,&p0) ;
    ApplyTotalXfrm( Xfrms, UPSTREAM, TrackFlag, 0 ,x,px,y,py,z,p0) ;
    Stop = CheckAperStopPart( xb,yb,stop,ngoodray, elemno, aper2, ray,
            UPSTREAM, &shape, Tilt, stp, 0 ) ;
  }
  else {
    GetLocalCoordPtrs(yb, raystart,&x,&px,&y,&py,&z,&p0) ;
    Stop = CheckAperStopPart( xb,yb,stop,ngoodray, elemno, aper2, ray,
            DOWNSTREAM, &shape, Tilt, stp, 0 ) ;
  }
  if (Stop == 1)
    return ;       
  if (isupstream==0)
    ApplyTotalXfrm( Xfrms, DOWNSTREAM, TrackFlag, 0,x,px,y,py,z,p0 ) ;
}
        
        
/*=====================================================================*/

/* Track one bunch thru one coordinate change element
 *
 * RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:   never. */

int TrackBunchThruCoord( int elemno, int bunchno,
        struct TrackArgsStruc* ArgStruc,
        int* TrackFlag                   )
{
  int stat = 1 ;
  struct Bunch* ThisBunch ;      /* shortcut */
  double dx[6] ;
  Rmat R ;
#ifdef __CUDACC__
  int hStop=0 ;
  double *R_flat = NULL ;
#else
  int ray ;
#endif  
  /* get the element parameters from BEAMLINE; exit with bad status if
   * any parameters are missing or corrupted. */
  stat = GetDatabaseParameters( elemno, nCoordPar, CoordPar,
          TrackPars, ElementTable ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* compute the transfer map, exit with bad status if not successful */
  stat = GetCoordMap( CoordPar[CoordChange].ValuePtr, dx, R ) ;
  if (stat == 0)
  {
    BadElementMessage( elemno+1 ) ;
    goto egress ;
  }
  
  /* make a shortcut to the bunch of interest */
  ThisBunch = ArgStruc->TheBeam->bunches[bunchno] ;
  
  /* loop over rays */
#ifdef __CUDACC__
  int *Stop ;
  double *R_gpu ;
  double *dx_gpu ;
  gpuErrchk( cudaMalloc(&R_gpu, sizeof(double)*36) );
  gpuErrchk( cudaMalloc(&Stop, sizeof(int)) ) ;
  gpuErrchk( cudaMalloc(&dx_gpu, sizeof(double)*6) );
  R_flat = &(R[0][0]) ;
  gpuErrchk( cudaMemcpy(R_gpu, R_flat, sizeof(double)*36, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(Stop, &hStop, sizeof(int), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(dx_gpu, dx, sizeof(double)*6, cudaMemcpyHostToDevice) );
  TrackBunchThruCoord_kernel<<<blocksPerGrid, threadsPerBlock>>>(ThisBunch->nray, ThisBunch->stop, ThisBunch->x, ThisBunch->y,
          TrackFlag, &ThisBunch->ngoodray, StoppedParticles_gpu, Stop, R_gpu, dx_gpu, elemno) ;
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaMemcpy(&hStop, Stop, sizeof(int), cudaMemcpyDeviceToHost ) );
  gpuErrchk( cudaFree(R_gpu) );
  gpuErrchk( cudaFree(Stop) );
  gpuErrchk( cudaFree(dx_gpu) );
  if (hStop>0)
    BadParticlePperpMessage( elemno+1, bunchno+1, hStop+1 ) ;
#else
  for (ray=0 ;ray<ThisBunch->nray ; ray++)
    TrackBunchThruCoord_kernel(ray, ThisBunch->stop, ThisBunch->x, ThisBunch->y, TrackFlag, &ThisBunch->ngoodray, StoppedParticles, R, dx, elemno, bunchno) ;
#endif    
    
  egress:
    
    return stat ;
    
}

#ifdef __CUDACC__
__global__ void TrackBunchThruCoord_kernel(int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray, int* stp, int* Stop, double* R_flat,
        double* dx, int elemno)
#else
void TrackBunchThruCoord_kernel(int nray, double* stop, double* xb, double* yb, int* TrackFlag, int* ngoodray, int* stp, Rmat R, double dx[6], int elemno, int bunchno)
#endif
{
  int ray, raystart, thisStop ;
  double x1, y1 ;
  double *x,*px,*y,*py,*z,*p0;
#ifdef __CUDACC__
  ray = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( ray >= nray ) return;
  double R[6][6] ;
  int irw, icol ;
  for (irw=0;irw<6;irw++)
    for (icol=0;icol<6;icol++)
      R[irw][icol]=R_flat[(6*irw)+icol] ;
#else
  ray = nray;
#endif  
  raystart = 6*ray ;
    
  /* if the ray was previously stopped ignore it  */
  if (stop[ray] > 0.)
    return ;

  /* make ray coordinates into local ones */
  GetLocalCoordPtrs(xb, raystart,&x,&px,&y,&py,&z,&p0) ;

  /* make transformed coordinates from original coordinates */
  x1 = *x + dx[0] ;
  y1 = *y + dx[2] ;
  yb[raystart] = R[0][0]*x1 + R[0][1]* *px + R[0][2]*y1 + R[0][3]* *py ;
  yb[raystart+2] = R[2][0]*x1 + R[2][1]* *px + R[2][2]*y1 + R[2][3]* *py ;
  yb[raystart+4] = *z + dx[4] ;
  if (TrackFlag[ZMotion] == 1)
    yb[raystart+4] += R[4][0]*x1 + R[4][1]* *px + R[4][2]*y1 + R[4][3]* *py ;
  yb[raystart+1] = R[1][1] * *px + R[1][3]* *py + dx[1] ;
  yb[raystart+3] = R[3][1] * *px + R[3][3]* *py + dx[3] ;
  yb[raystart+5] = *p0 ;

  /* check to see if, as a result of the coordinate transformation, any
   * of the particles are now going perpendicular to the new coordinate
   * axes, if so stop them now. */
  thisStop = CheckPperpStopPart( stop, ngoodray, elemno, ray, yb+raystart+1 , yb+raystart+3, stp ) ;
  if (thisStop != 0)
#ifdef __CUDACC__    
    atomicCAS(Stop, 0, ray) ;
#else
    BadParticlePperpMessage( elemno+1, bunchno+1, ray+1 ) ;
#endif    
}

/*=====================================================================*/
/*=====================================================================*/
/*=====================================================================*/
/*=====================================================================*/
/*=====================================================================*/

/* Some utility procedures follow */

/* Copy R matrix Rsource into R matrix Rtarget
 *
 * RET:   none
 * ABORT: never.
 * FAIL:  never.                                               */

void RmatCopy( Rmat Rsource, Rmat Rtarget )
{
  int i,j ;
  for (i=0 ; i<6 ; i++){
    for (j=0 ; j<6 ; j++){
      Rtarget[i][j] = Rsource[i][j] ;
    }
  }
  return ;
}

/*=====================================================================*/

/* Multiply R matrix Rearly by R matrix Rlate, put results into Rprod;
 * the contents of Rprod are (obviously) destroyed.  In matrix terms,
 *
 * Rprod = Rlate * Rearly ;
 *
 * RET:   none.
 * ABORT: never.
 * FAIL:  never.                                               */



void RmatProduct( Rmat Rlate, Rmat Rearly, Rmat Rprod )
{
  Rmat Rtemp = {
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0}
  } ;
  int i,j,k ;
  
  /* Since it is possible that Rprod and Rearly are the same (ie the calling
   * routine would like to ultimately replace Rearly with Rprod), we must
   * carefully do the multiplication in a way that does not corrupt Rearly.
   * Specifically, perform the multiplication into Rtemp, then do an RmatCopy. */
  
  for (i = 0 ; i < 6 ; i++){
    for (j = 0 ; j < 6 ; j++){
      for (k = 0 ; k < 6 ; k++){
        Rtemp[i][j] += Rlate[i][k]*Rearly[k][j] ;
      }
    }
  }
  RmatCopy(Rtemp,Rprod) ;
  return ;
}

/*=====================================================================*/

/* Return the values of the tracking flags to the calling procedure.*/

/* RET:    a vector of ints with the values of the flags.  Note that
 * the last of the flags returned is a status, which indicates
 * whether subsidiary procedure GetTrackFlagGlobals executed
 * successfully.  Also indicates which flags are set, and how
 * many times, via the global TrackFlagSet vector.
 * ABORT:  never.
 * FAIL:   never. */

int* GetTrackingFlags( int elemno )
{
  
  const char* FlagName ; /* flag names from BEAMLINE cell array */
  int  FlagValue ; /* values from BEAMLINE cell array */
  int  nFlagNames   ; /* how many there were */
  static int DecodedFlagValue[NUM_TRACK_FLAGS+1] ;
  int i,j ;
  int OutOfBounds=0 ;
  
  /* initialize all tracking flags to zero (defaults) */
  
  for (i=0 ; i<NUM_TRACK_FLAGS ; i++)
  {
    DecodedFlagValue[i] = 0 ;
    TrackFlagSet[i] = 0 ;
  }
  DecodedFlagValue[NUM_TRACK_FLAGS] = 0 ;
  
  /* use global access to get the names of the tracking flags and their
   * values.  If there is not a TrackFlag structure in the BEAMLINE elt
   * of interest, return with good status (no law says you must have a
   * tracking flags structure on an element).  If however there is such
   * a structure but it is corrupted, return with bad status. */
  
  nFlagNames = GetNumTrackFlags( elemno )  ;
  
  if (nFlagNames == 0)
  {
    DecodedFlagValue[NUM_TRACK_FLAGS] = 1 ;
    goto egress ;
  }
  if (nFlagNames < 0)
  {
    DecodedFlagValue[NUM_TRACK_FLAGS] = 0 ;
    goto egress ;
  }
  
  
  /* otherwise loop over the flags and compare them to the prod names: */
  
  for (i=0 ; i<nFlagNames ; i++)
  {
    FlagName = GetTrackFlagName( i ) ;
    FlagValue = GetTrackFlagValue( i ) ;
    for (j=0 ; j<NUM_TRACK_FLAGS ; j++)
    {
      if (strcmp(FlagName,TrackFlagNames[j]) == 0)
      {
        DecodedFlagValue[j] = FlagValue ;
        if ( (FlagValue > TrackFlagMaxValue[j]) ||
                (FlagValue < TrackFlagMinValue[j])    )
          OutOfBounds++ ;
        TrackFlagSet[j]++ ;
        break ;
      }
    }
  }
  if (OutOfBounds==0)
    DecodedFlagValue[NUM_TRACK_FLAGS] = 1 ;
  
  egress:
    
    return DecodedFlagValue ;
    
}

#ifdef __CUDACC__
__host__ __device__ int my_strcmp(const char *s1, const char *s2)
{
    for ( ; *s1 == *s2; s1++, s2++)
	if (*s1 == '\0')
	    return 0;
    return ((*(unsigned char *)s1 < *(unsigned char *)s2) ? -1 : +1);
}
#endif

/*=====================================================================*/

/* Check to see whether a particle has gone outside of the aperture;
 * if so, stop it from further tracking. */

/* RET:    status int, 0 == did not stop particle, 1 == particle was
 * stopped.
 * ABORT:  never.
 * FAIL:   never. */
int CheckAperStopPart( double *x, double *y, double *stop, int *ngoodray,
        int elemno,
        double* aper2,
        int ray, int UpstreamDownstream,
        int* shape, double tilt, int* stp, int isbend )
{
  double* posvec ;
  int retval = 0 ;
  double x_rotated, y_rotated ;
  int i ;
  
  /* if UpstreamDownstream is zero, point posvec at x (pre-tracking
   * particle positions), otherwise at y (post-tracking positions) */
  
  if (UpstreamDownstream == UPSTREAM)
    posvec = x ;
  else
    posvec = y ;
  
  /* check the position */
  if (shape == NULL) /* indicates circular, element aperture */
  {
    /* If it is a bend, just look at the vertical aperture */
    if ( isbend==1 && posvec[6*ray+2]*posvec[6*ray+2] >= aper2[0] )
      retval = 1 ;
    else if ( (posvec[6*ray+0]*posvec[6*ray+0] + posvec[6*ray+2]*posvec[6*ray+2]) >= aper2[0] ) {
      retval = 1 ;
    }
  }
  else               /* rectangular or elliptical */
  {
    x_rotated = posvec[6*ray+0] * cos(tilt) - posvec[6*ray+2] * sin(tilt) ;
    y_rotated = posvec[6*ray+0] * sin(tilt) + posvec[6*ray+2] * cos(tilt) ;
    
    if ( (*shape == COLL_ELLIPSE && ( x_rotated*x_rotated / aper2[0]  + y_rotated*y_rotated / aper2[1]  >= 1 ))
         ||
         (*shape == COLL_RECTANGLE &&  ( (x_rotated*x_rotated >= aper2[0]) || (y_rotated*y_rotated >= aper2[1]) ) ) ) {
        retval = 1 ;
    }
  }
  if (retval == 1)
  {
    
    /* set the stopping point in the ThisBunch->stop vector */
    
    stop[ray] = (double)(elemno+1)  ; /* in Matlab indexing */
    *ngoodray=*ngoodray-1 ;
    
    /* set the global stopped-particle variable */
    *stp = 1 ;
    
    /* if we are doing the upstream face, copy the ray from input to output
     * vector */
    
    if (UpstreamDownstream == UPSTREAM)
    {
      for (i=6*ray ; i<6*ray+5 ; i++)
        y[i] = x[i] ;
    }
  }
  
  return retval ;
  
}

/*==================================================================*/

/* Check whether a particle needs to stop on total momentum <= 0.
 *
 * RET:    status int, 0 == did not stop particle, 1 == particle was
 * stopped.
 * ABORT:  never.
 * FAIL:   never. */

int CheckP0StopPart( double *stop, int *ngoodray, double *x, double *y, int elemno,
        int rayno, double P0, int upstreamdownstream, int* stp )
{
  int stat = 0, i ;
  if (P0 <= 0.)
  {
    stat = 1 ;
    stop[rayno] = (double)(elemno+1) ;
    ngoodray-- ;
    *stp = 1 ;
    
    /* If the check is done on the upstream face, copy the
     * x, px, y, py, z coordinates from x to y (ie, we want to preserve
     * the incoming particle coordinates, and the negative momentum, in
     * the y so that they are handed to the user on exit); also copy the
     * bad momentum into the returned-data vector */
    if (upstreamdownstream == UPSTREAM)
    {
      for (i=6*rayno ; i<6*rayno+4 ; i++)
        y[i] = x[i] ;
      y[6*rayno+5] = P0 ;
    }
    
  }
  return stat ;
}

/*==================================================================*/

/* Check whether a particle needs to stop on |Pperp| >= 1.
 *
 * RET:    status int, 0 == did not stop particle, 1 == particle was
 * stopped.
 * ABORT:  never.
 * FAIL:   never. */

int CheckPperpStopPart( double* stop, int* ngoodray, int elemno,
        int rayno, double* Px, double* Py, int* stp )
{
  int stat = 0 ;
  double Pperp ;
  
  Pperp = (*Px) * (*Px) + (*Py) * (*Py) ;
  if (Pperp >= 1.)
  {
    stat = 1 ;
    stop[rayno] = (double)(elemno+1) ;
    ngoodray-- ;
    *stp = 1 ;
  }
  return stat ;
}

/*==================================================================*/

/* Add an error message about a bad element */

void BadElementMessage( int elemno )
{
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Error in element definition:  Element %d",elemno) ;
  AddMessage( outmsg, 1 ) ;
  
}

/*==================================================================*/

/* Add an error message about a bad power supply */

void BadPSMessage( int elemno, int PSno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Error in definition of PS %d, Element %d",PSno,elemno) ;
  AddMessage( outmsg, 1 ) ;
}

/*==================================================================*/

/* Add an error message about a klystron supply */

void BadKlystronMessage( int elemno, int Kno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Error in definition of Klystron %d, Element %d",Kno,elemno) ;
  AddMessage( outmsg, 1 ) ;
}

/*==================================================================*/

/* Add an error message about a bad Twiss propagation */

void BadTwissMessage( int elemno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Error in Twiss propagation, Element %d",elemno) ;
  AddMessage( outmsg, 1 ) ;
}

/*==================================================================*/

/* Add an error message about bad inverse-twiss propagation */

void BadInverseTwissMessage( int e1, int e2 )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,
          "Error in back-propagating Twiss, elements %d to %d",
          e1, e2) ;
  AddMessage( outmsg, 1 ) ;
}

/*==================================================================*/

/* Add an error message about bad tracking flags */

void BadTrackFlagMessage( int elemno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Corrupted tracking flags, Element %d",elemno) ;
  AddMessage( outmsg, 0 ) ;
}

/*==================================================================*/

/* Add an error message about aperture problems */

void BadApertureMessage( int elemno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Missing or zero aperture, Element %d",elemno) ;
  AddMessage( outmsg, 0 ) ;
}

/*==================================================================*/

/* Add an error message about SR options */

void BadSROptionsMessage( int elemno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"SR selected for element with zero L/Lrad, Element %d",elemno) ;
  AddMessage( outmsg, 0 ) ;
}

/*==================================================================*/

/* Add an error message about a bad momentum */

void BadParticleMomentumMessage( int elemno, int bunchno, int rayloop )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Negative or zero particle momentum: Element %d, Bunch %d, Particle %d",
          elemno, bunchno, rayloop ) ;
  AddMessage( outmsg, 0 ) ;
}

/*==================================================================*/

/* Add an error message about a bad transverse momentum */

void BadParticlePperpMessage( int elemno, int bunchno, int rayloop )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Transverse momentum > Total momentum: Element %d, Bunch %d, Particle %d",
          elemno, bunchno, rayloop ) ;
  AddMessage( outmsg, 0 ) ;
}

/*==================================================================*/

/* Add an error message about bad initial momenta */

void BadInitMomentumMessage( )
{
  
  char outmsg[90] ;
  sprintf(outmsg,"Invalid total or perpendicular momenta in initial rays") ;
  AddMessage( outmsg, 1 ) ;
}

/*==================================================================*/

/* Add an error message about a bad offset */

void BadOffsetMessage( int elemno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Corrupted offset/girder/mover: Element %d",
          elemno ) ;
  AddMessage( outmsg, 0 ) ;
}

/*==================================================================*/

/* Add an error message about a bad allocation of BPM stuff */

void BadBPMAllocMsg( int elemno )
{
  
  char outmsg[90]; /* output message */
  
  sprintf(outmsg,"Unable to allocate BPM/INST dataspace: Element %d",
          elemno ) ;
  AddMessage( outmsg, 0 ) ;
}

/*==================================================================*/

/* Add a message about being unable to alloce LCAV slice attributes */

void BadSliceAllocMessage( int elemno )
{
  
  char outmsg[90] ;
  
  sprintf(outmsg,"Unable to allocate LCAV attribute dataspace: Element %d",
          elemno ) ;
  AddMessage( outmsg, 0 ) ;
  
}

/*==================================================================*/

/* Add a message about being unable to allocate space for the convolution
 * of a SRWF */

void BadSRWFAllocMsg( int WF, int flag )
{
  
  char outmsg[90] ;
  
  if (flag==0)
    sprintf(outmsg,
            "Unable to allocate convolved-ZSR dataspace, ZSRno = %d",WF) ;
  else
    sprintf(outmsg,
            "Unable to allocate convolved-TSR dataspace, TSRno = %d",WF) ;
  AddMessage( outmsg, 0 ) ;
  
}

/*==================================================================*/

/* Add a message about a corrutpted SRWF */

void BadSRWFMessage( int WF, int flag )
{
  
  char outmsg[90] ;
  
  if (flag==0)
    sprintf(outmsg,
            "Corrupted ZSR data, ZSRno = %d",WF) ;
  else
    sprintf(outmsg,
            "Corrupted TSR data, TSRno = %d",WF) ;
  AddMessage( outmsg, 0 ) ;
  
}

/*==================================================================*/

/* Add a message about a nonexistent LRWF */

void NonExistentLRWFmessage( int elemno, int wakeno, int tableno )
{
  
  char outmsg[90] ;
  
  if ( tableno == TLRTable )
    sprintf(outmsg,
            "Element %d points to nonexistent LRWF %d",
            elemno, wakeno) ;
  else
    if ( tableno == TLRTable )
      sprintf(outmsg,
              "Element %d points to nonexistent error LRWF %d",
              elemno, wakeno) ;
  AddMessage( outmsg, 0 ) ;
  
}

/*==================================================================*/

/* Add a message about being unable to allocate space for the convolution
 * of a LRWF */

void BadLRWFAllocMsg( int WF, int flag )
{
  
  char outmsg[90] ;
  
  if (flag==0)
    sprintf(outmsg,
            "Unable to allocate bins for bunch-TLR interaction, TLRno = %d",WF) ;
  else
    sprintf(outmsg,
            "Unable to allocate bins for bunch-error TLR interaction, TLRno = %d",WF) ;
  AddMessage( outmsg, 0 ) ;
  
}

/*==================================================================*/

/* add a message about bad bunch tracking order on LRWFs */

void BadLRWFBunchOrder( int LastBunch, int ThisBunch )
{
  
  char outmsg[90] ;
  sprintf(outmsg,
          "Bad bunch order for LRWF:  last bunch = %d, this bunch = %d",
          LastBunch, ThisBunch ) ;
  AddMessage(outmsg,0) ;
}


/*==================================================================*/

/* Add a message about a stopped bunch */

void BunchStopMessage( int bunch, int elem )
{
  
  char outmsg[90] ;
  
  sprintf( outmsg,
          "BUNCHSTOP:  All rays stopped, bunch %d, element %d",
          bunch, elem ) ;
  
  
  AddMessage( outmsg, 0 ) ;
  
}


/*==================================================================*/

/* Compute the coordinate transformation required for an element based
 * on its offset, the offset of its girder, and the offset of its
 * mover.  The upstream and downstream transforms are returned in
 * Xfrms.  The returned integer is a status value. */

/* RET:    1 if all required fields are present, and all required and
 * optional fields are well-formed.
 * 0 if required fields are missing or any fields are
 * incorrectly formed.
 * ABORT:  never.
 * FAIL:   never.                           */

int GetTotalOffsetXfrms( double* ElemGirdNo, double* ElemLen,
        double* ElemS, double* ElemOffset,
        double Xfrms[6][2] )
{
  
  double s0, s ;                   /* S positions */
  double d1, d2 ;                  /* face distances from girder center */
  double L ;                       /* element length */
  double Lov2 ;                    /* element half-length */
  double GO2[6] =
  {0,0,0,0,0,0}
  ;
  double* utility ;
  int stat = 1 ;
  int i,j ;
  int girdno ;
  int MovrAxes[6] ;
  int NumAxes ;
  
  /* start with the element parameters themselves; they are guaranteed
   * not to be NULL by the dictionary lookup process */
  
  L = *ElemLen ;
  Lov2 = 0.5 * L ;
  s = *ElemS ;
  
  /* no offset field is legal */
  
  if (ElemOffset == NULL)
  {
    for (i=0 ; i<6 ; i++)
    {
      for (j=0 ; j<2 ; j++)
      {
        Xfrms[i][j] = 0. ;
      }
    }
  }
  else
  {
    
    /* put in the front-face transformation */
    
    Xfrms[0][0] = ElemOffset[0] - Lov2 * ElemOffset[1] ;
    Xfrms[0][1] = ElemOffset[0] + Lov2 * ElemOffset[1] ;
    
    Xfrms[1][0] = ElemOffset[1] ;
    Xfrms[1][1] = ElemOffset[1] ;
    
    Xfrms[2][0] = ElemOffset[2] - Lov2 * ElemOffset[3] ;
    Xfrms[2][1] = ElemOffset[2] + Lov2 * ElemOffset[3] ;
    
    Xfrms[3][0] = ElemOffset[3] ;
    Xfrms[3][1] = ElemOffset[3] ;
    
    Xfrms[4][0] = ElemOffset[4] ;
    Xfrms[4][1] = ElemOffset[4] ;
    
    Xfrms[5][0] = ElemOffset[5] ;
    Xfrms[5][1] = ElemOffset[5] ;
    
  }
  
  /* Exit if the element has no girder */
  
  if (ElemGirdNo == NULL)
    goto egress ;
  else
  {
    girdno = (int)(*ElemGirdNo) ;
    if (girdno == 0)
      goto egress ;
  }
  
  
  /* if we're still here, get girder parameters */
  
  stat = GetDatabaseParameters( girdno-1, nGirderPar,
          GirderPar, TrackPars, GirderTable ) ;
  if (stat==0)
    goto egress ;
  if (stat==-1)
    stat = 1 ;
  
  /* make sure that the NDOF of the mover, if any, and the position of the
   * mover have the same length */
  
  stat = CheckGirderMoverPars( GirderPar, TrackPars ) ;
  if (stat!=1)
  {
    stat = 0 ;
    goto egress ;
  }
  
  s0 = *(GirderPar[GirderS].ValuePtr) ;
  if (GirderPar[GirderS].Length > 1)
    /*		s0 = 0.5 * (*(GirderPar[GirderS].ValuePtr+1)-s0) ; */
    s0 = 0.5 * (*(GirderPar[GirderS].ValuePtr+1)+s0) ;
  
  /* compute the distance from the girder center to the faces of the
   * element, with the convention that if the face is upstream of the
   * girder center then the offset is negative */
  
  d1 = s-Lov2 - s0 ; /* dist gird center to upstream face */
  d2 = d1 + L ;      /* dist gird center to downstream face */
  
  /* Now:  if d1>0 and the girder angle > 0, then the element has a
   * positive offset and angle from the survey line.  This is emulated
   * by moving the beam negative in position and angle.  However, since
   * the Xfrm values on the upstream face are **SUBTRACTED**, we find
   * that the change in Xfrm has to be positive if d1 and the angle are
   * positive, or if the offset is positive: */
  
  utility = GirderPar[GirderOffset].ValuePtr ;
  
  Xfrms[0][0] += utility[0] + d1 * utility[1] ;
  Xfrms[0][1] += utility[0] + d2 * utility[1] ;
  
  Xfrms[1][0] += utility[1] ;
  Xfrms[1][1] += utility[1] ;
  
  Xfrms[2][0] += utility[2] + d1 * utility[3] ;
  Xfrms[2][1] += utility[2] + d2 * utility[3] ;
  
  Xfrms[3][0] += utility[3] ;
  Xfrms[3][1] += utility[3] ;
  
  Xfrms[4][0] += utility[4] ;
  Xfrms[4][1] += utility[4] ;
  
  Xfrms[5][0] += utility[5] ;
  Xfrms[5][1] += utility[5] ;
  
  /* Get the girder mover parameters.  There are 2 vectors here, the
   * first is a vector of coordinate indices (ie, a 3 DOF mover with
   * x,y, xyrotate would have [1 2 6] for this vector */
  
  /* If there is no mover, go to the exit */
  
  if (GirderPar[GirderMover].ValuePtr == NULL)
    goto egress ;
  
  /* otherwise combine values in Xfrms with the mover values */
  
  NumAxes = GirderPar[GirderMover].Length ;
  utility = GirderPar[GirderMover].ValuePtr ;
  for (i=0 ; i<NumAxes ;i++)
    MovrAxes[i] = (int)utility[i]-1 ;
  
  /* if there's a mover but no MoverPos, or MoverPos has the wrong length,
   * it's an error */
  
  utility = GirderPar[GirderMoverPos].ValuePtr ;
  
  /* extract mover position values */
  
  for (i=0 ; i<NumAxes ; i++)
    GO2[MovrAxes[i]] = utility[i] ;
  
  /* set them into the transform databank just the way that the girder
   * offsets were set in */
  
  Xfrms[0][0] += GO2[0] + d1 * GO2[1] ;
  Xfrms[0][1] += GO2[0] + d2 * GO2[1] ;
  
  Xfrms[1][0] += GO2[1] ;
  Xfrms[1][1] += GO2[1] ;
  
  Xfrms[2][0] += GO2[2] + d1 * GO2[3] ;
  Xfrms[2][1] += GO2[2] + d2 * GO2[3] ;
  
  Xfrms[3][0] += GO2[3] ;
  Xfrms[3][1] += GO2[3] ;
  
  Xfrms[4][0] += GO2[4] ;
  Xfrms[4][1] += GO2[4] ;
  
  Xfrms[5][0] += GO2[5] ;
  Xfrms[5][1] += GO2[5] ;
  
  /* that's it, set return and exit */
  
  egress :
    
    return stat ;
    
}

/*==================================================================*/

/* Apply an element's xyz translations and xz/yz rotations to the
 * coordinates of a particle (at the upstream face), or undo the
 * transformations (at the downstream face). */

/* RET:    none.
 * ABORT:  never.
 * FAIL:   never. */

void ApplyTotalXfrm( double Xfrms[6][2], int face, int* TrackFlag,
        double dzmod, double *x, double *px, double *y, double *py, double *z, double *p0 )
{
  
  /* The pointers to the particle coordinates have global scope, thus
   * are not passed as arguments.  Are we on the upstream or downstream
   * face? */
  if (face == UPSTREAM)
  {
    (*x)  += Xfrms[4][0] * (*px) - Xfrms[0][0] ;
    (*px) -= Xfrms[1][0] ;
    (*y)  += Xfrms[4][0] * (*py) - Xfrms[2][0] ;
    (*py) -= Xfrms[3][0] ;
    
    if (TrackFlag[ZMotion] == 1)
      (*z) += 0.5 * Xfrms[4][0] *
              ( (*px) * (*px) + (*py) * (*py) )
              - Xfrms[4][0] ;
  }
  else if (face == DOWNSTREAM)
  {
    (*x)  += Xfrms[0][1] - Xfrms[4][1] * (*px) ;
    (*px) += Xfrms[1][1] ;
    (*y)  += Xfrms[2][1] - Xfrms[4][1] * (*py) ;
    (*py) += Xfrms[3][1] ;
    if (TrackFlag[ZMotion] == 1)
      (*z) += Xfrms[4][1] -
              0.5 * Xfrms[4][0] *
              ( (*px) * (*px) + (*py) * (*py) ) ;
    if (TrackFlag[LorentzDelay] == 1)
      (*z) += LORENTZ_DELAY( (*p0) ) - dzmod ;
  }
  
  return ;
  
}

/*==================================================================*/

/* Perform some common logic for BPMs and INSTs related to setting
 * the indexing between the BEAMLINE list (element order) and the
 * BPM or INST list (only BPMs or INSTs that return data are indexed). */

/* RET:    none.
 * Abort:  never.
 * FAIL:   never. */

void BPMInstIndexingSetup( int elemno, int* FirstElemno,
        int *ElemnoLastCall, int* Counter )
{
  
  /* if FirstElemno is -1, means it ain't been set yet.  Set it now. */
  
  if ( *FirstElemno == -1 )
    *FirstElemno = elemno ;
  
  /* if we're on the first inst/bpm element, zero the counter of such
   * devices; if we are not on the first element, AND we are not on the
   * same element as last time, increment the counter */
  
  if ( *FirstElemno == elemno )
    *Counter = 0 ;
  else if (elemno != *ElemnoLastCall)
  {
    (*Counter)++ ;
    *ElemnoLastCall = elemno ;
  }
  
  return ;
  
}

/*==================================================================*/

/* figure out how many bunch slots are needed, and which one we are
 * on now */

/* RET:    none.
 * ABORT:  never.
 * FAIL:   never. */

void BunchSlotSetup( int *TrackFlags,
        struct TrackArgsStruc *ArgStruc, int bunchno,
        int* nBunchNeeded, int* BunchSlot)
{
  if (TrackFlags[MultiBunch] == 1)
  {
    *nBunchNeeded = ArgStruc->LastBunch - ArgStruc->FirstBunch + 1 ;
    *BunchSlot    = bunchno - (ArgStruc->FirstBunch-1) ;
  }
  else
  {
    *nBunchNeeded = 1 ;
    *BunchSlot = 0 ;
  }
  
  return ;
  
}

/*==================================================================*/

/* free dynamic memory and null the pointer to it */

/* RET:   none.
 * ABORT: never.
 * FAIL:  will fail if told to free memory that was not dynamically
 * allocated in the first place */

void FreeAndNull( void** GeneralPointer )
{
  if (*GeneralPointer != NULL)
  {
    free(*GeneralPointer) ;
    *GeneralPointer = NULL ;
  }
  
  return ;
  
}

/*==================================================================*/

/* clear tracking variables which otherwise stick around from one call
 * to TrackThru to another */

/* RET:    none.
 * ABORT:  never.
 * FAIL:   */

void ClearTrackingVars( )
{
  
  int i ;
  
  /* start with the bpmdata global:  is it allocated? */
  
  if (bpmdata != NULL)
  {
    
    /* loop over entries in bpmdata until we find one that is not assigned */
    
    i=0 ;
    while (bpmdata[i] != NULL)
    {
      
      /* free the dynamically-allocated data vectors */
      
      FreeAndNull( (void**)&(bpmdata[i]->xread) ) ;
      FreeAndNull( (void**)&(bpmdata[i]->yread) ) ;
      FreeAndNull( (void**)&(bpmdata[i]->sigma) ) ;
      FreeAndNull( (void**)&(bpmdata[i]->P) ) ;
      FreeAndNull( (void**)&(bpmdata[i]->z) ) ;
      FreeAndNull( (void**)&(bpmdata[i]->Q) ) ;
      FreeAndNull( (void**)&(bpmdata[i]->sumpxq) ) ;
      FreeAndNull( (void**)&(bpmdata[i]->sumpyq) ) ;
      
      /* free the bpmdata entry */
      
      FreeAndNull( (void**)&(bpmdata[i]) ) ;
      i++ ;
      
    }
    
    /* free the backbone itself */
    
    free( bpmdata ) ;
    bpmdata = NULL ;
    
  }
  
  /* now apply the same logic to insts */
  
  if ( instdata != NULL )
  {
    i=0 ;
    while (instdata[i] != NULL)
    {
      FreeAndNull( (void**)&(instdata[i]->x) ) ;
      FreeAndNull( (void**)&(instdata[i]->y) ) ;
      FreeAndNull( (void**)&(instdata[i]->z) ) ;
      FreeAndNull( (void**)&(instdata[i]->Q) ) ;
      FreeAndNull( (void**)&(instdata[i]->sig11) ) ;
      FreeAndNull( (void**)&(instdata[i]->sig33) ) ;
      FreeAndNull( (void**)&(instdata[i]->sig55) ) ;
      FreeAndNull( (void**)&(instdata[i]->sig13) ) ;
      
      FreeAndNull( (void**)&(instdata[i]) ) ;
      i++ ;
    }
    free( instdata ) ;
    instdata = NULL ;
  }
  
  /*  SBPMs */
  
  if (sbpmdata != NULL)
  {
    i=0 ;
    while (sbpmdata[i] != NULL)
    {
      FreeAndNull( (void**)&(sbpmdata[i]->x) ) ;
      FreeAndNull( (void**)&(sbpmdata[i]->y) ) ;
      FreeAndNull( (void**)&(sbpmdata[i]->Q) ) ;
      FreeAndNull( (void**)&(sbpmdata[i]->S) ) ;
      
      FreeAndNull( (void**)&(sbpmdata[i]) ) ;
      i++ ;
    }
    free( sbpmdata ) ;
    sbpmdata = NULL ;
  }
  
  /* frequency-mode LRWF_T's  */
  
  if (TLRFreqKickDat != NULL)
  {
    for (i=0 ; i<nElemOld ; i++)
    {
      if (TLRFreqKickDat[i] != NULL)
      {
        FreeAndNull( (void**)&(TLRFreqKickDat[i]->xKick) ) ;
        FreeAndNull( (void**)&(TLRFreqKickDat[i]->yKick) ) ;
        FreeAndNull( (void**)&(TLRFreqKickDat[i]) ) ;
      }
    }
    free( TLRFreqKickDat ) ;
    TLRFreqKickDat = NULL ;
  }
  
  /* frequency-mode LRWF_error's  */
  
  if (TLRErrFreqKickDat != NULL)
  {
    for (i=0 ; i<nElemOld ; i++)
    {
      if (TLRErrFreqKickDat[i] != NULL)
      {
        FreeAndNull( (void**)&(TLRErrFreqKickDat[i]->xKick) ) ;
        FreeAndNull( (void**)&(TLRErrFreqKickDat[i]->yKick) ) ;
        FreeAndNull( (void**)&(TLRErrFreqKickDat[i]) ) ;
      }
    }
    free( TLRErrFreqKickDat ) ;
    TLRErrFreqKickDat = NULL ;
  }
  
  /* first/last-bunch-at-RF backbone */
  
  if (FirstBunchAtRF != NULL)
    FreeAndNull( (void**)&FirstBunchAtRF ) ;
  if (LastBunchAtRF != NULL)
    FreeAndNull( (void**)&LastBunchAtRF ) ;
  
  
  return ;
  
}

/*=====================================================================*/

/* Perform slice setup -- figure out how many slices to use when tracking
 * the LCAV (for TSRs and SBPMs), make sure that there's a valid SBPM data
 * structure ready for the data, make sure that the correct BPM number is
 * being pointed to.
 *
 * RET:    Status, +1 for success, 0 for failure.
 * ABORT:  never.
 * FAIL:   never.    */

int LcavSliceSetup( struct TrackArgsStruc *ArgStruc, int elemno, int* TrackFlag,
        int* nslice, int* nslicealloc, int* NSBPM, int** doSBPM,
        double** Lfrac, int* TWFSliceno )
{
  
  double* dmy ;
  double dL ;
  div_t nslov2 ;
  int count ;
  int stat = 1 ;
  
  /* Are we doing SBPMs, and does this particular structure have SBPMs?  If
   * so, we need to figure out how many SBPMs are in this structure */
  
  *NSBPM = 0 ;
  if ( (ArgStruc->GetInstData==1) && (TrackFlag[GetSBPMData] == 1))
  {
    dmy = GetElemNumericPar( elemno, "NBPM", NULL ) ;
    if (dmy==NULL)
      *NSBPM = 0 ;
    else
      *NSBPM = (int)(*dmy) ;
  }
  
  /* figure out the number of slices needed based on the number of SBPMs on
   * this structure */
  
  switch (*NSBPM)
  {
    case 0  : *nslice = 1 ;
    break ;
    case 1  : *nslice = 2 ;
    break ;
    default : *nslice = (*NSBPM) - 1 ;
    break ;
  }
  
  /* if we have enough slices allocated, great.  Otherwise, allocate now.
   * Remember to allocate at least 2 extra slices over the total needed from
   * the calculation above */
  
  if (*nslicealloc < (*nslice)+2)
  {
    FreeAndNull( (void**)doSBPM ) ;
    FreeAndNull( (void**)Lfrac ) ;
    *doSBPM = (int*)calloc( (*nslice)+2,sizeof(int) ) ;
    *Lfrac  = (double*)calloc( (*nslice)+2,sizeof(double) ) ;
    if ( (*doSBPM == NULL) || (*Lfrac == NULL) )
    {
      BadSliceAllocMessage( elemno+1 ) ;
      stat = 0 ;
      goto egress ;
    }
    *nslicealloc = (*nslice)+2 ;
  }
  
  /* set the Lfrac and doSBPM values */
  
  dL = 1. / (*nslice) ;
  for (count=0 ; count<(*nslice) ; count++)
  {
    *(*Lfrac+count) = dL ;
    *(*doSBPM+count) = *NSBPM ;
  }
  *(*doSBPM+*nslice) = *NSBPM ;
  *(*Lfrac+*nslice) = 0. ;
  if (*NSBPM == 1)
  {
    **doSBPM = 0 ;
    *(*doSBPM+2) = 0 ;
  }
  
  /* now:  if transverse wakefieldss are requested, and the number of
   * slices is odd, that means that we need to "split" a slice to
   * ensure that we track to exactly the 50% point in the structure.
   */
  
  if ( (TrackFlag[SRWF_T]) || (TrackFlag[LRWF_T]) || (TrackFlag[LRWF_ERR]) )
  {
    nslov2 = div(*nslice,2) ;
    if (nslov2.rem == 1)
    {
      (*nslice)++ ;
      for (count = (*nslice) ; count >nslov2.quot+1 ; count--)
      {
        *(*doSBPM+count) = *(*doSBPM+count-1) ;
        *(*Lfrac+count) = *(*Lfrac+count-1) ;
      }
      *(*Lfrac+nslov2.quot+1) = dL / 2. ;
      *(*Lfrac+nslov2.quot)   = dL / 2. ;
      *(*doSBPM+nslov2.quot+1) = 0 ;
      nslov2 = div(*nslice,2) ;
    }
    *TWFSliceno = nslov2.quot ;
  }
  else
    *TWFSliceno = -1 ;
  
  egress:
    
    return stat;
    
}

/*=====================================================================*/

/* Perform setup and initialization of the SBPM data structures during
 * LCAV tracking */

/* RET:    Status, 1 = success, 0 = failure.
 * ABORT:  never.
 * FAIL:    */

int SBPMSetup( struct TrackArgsStruc* ArgStruc, int elemno,
        int bunchno, int NSBPM, int* SBPMCounter     )
{
  int count ;
  int stat = 1 ;
  
  /* if NSBPM is less than one, do nothing and return */
  
  if (NSBPM < 1)
    goto egress ;
  
  /* find the correct index into the SBPM backbone */
  
  BPMInstIndexingSetup( elemno, &FirstSBPMElemno,
          &SBPMElemnoLastCall, SBPMCounter ) ;
  
  /* if this is not the first bunch to be tracked, we can exit */
  
  if (bunchno+1 != ArgStruc->FirstBunch)
    goto egress ;
  
  /* if on the other hand this is the first bunch, then we need to: */
  
  /* Check to make sure that the correct backbone is allocated: */
  
  if (sbpmdata[*SBPMCounter] == NULL)
  {
    sbpmdata[*SBPMCounter] = (struct SBPMdat*)calloc(1,sizeof(struct SBPMdat)) ;
    if (sbpmdata[*SBPMCounter] == NULL)
    {
      BadBPMAllocMsg( elemno+1) ;
      stat = 0 ;
      goto egress ;
    }
    sbpmdata[*SBPMCounter]->nbpmalloc = 0 ;
  }
  
  /* set the SBPM index value */
  
  sbpmdata[*SBPMCounter]->indx = elemno+1 ;
  
  /* check to make sure enough BPMs are allocated */
  
  if (sbpmdata[*SBPMCounter]->nbpmalloc < NSBPM)
  {
    FreeAndNull( (void**)&(sbpmdata[*SBPMCounter]->x) ) ;
    FreeAndNull( (void**)&(sbpmdata[*SBPMCounter]->y) ) ;
    FreeAndNull( (void**)&(sbpmdata[*SBPMCounter]->S) ) ;
    FreeAndNull( (void**)&(sbpmdata[*SBPMCounter]->Q) ) ;
    
    sbpmdata[*SBPMCounter]->x =
            (double*)calloc(NSBPM,sizeof(double)) ;
    sbpmdata[*SBPMCounter]->y =
            (double*)calloc(NSBPM,sizeof(double)) ;
    sbpmdata[*SBPMCounter]->S =
            (double*)calloc(NSBPM,sizeof(double)) ;
    sbpmdata[*SBPMCounter]->Q =
            (double*)calloc(NSBPM,sizeof(double)) ;
    if ( (sbpmdata[*SBPMCounter]->x == NULL) ||
            (sbpmdata[*SBPMCounter]->y == NULL) ||
            (sbpmdata[*SBPMCounter]->S == NULL) ||
            (sbpmdata[*SBPMCounter]->Q == NULL)    )
    {
      BadBPMAllocMsg( elemno+1 ) ;
      stat = 0 ;
      goto egress ;
    }
    sbpmdata[*SBPMCounter]->nbpmalloc = NSBPM ;
  }
  
  /* increment ArgStruc's SBPM counter */
  
  ArgStruc->nSBPM++ ;
  
  /* set the nbpm value in the SBPM structure */
  
  sbpmdata[*SBPMCounter]->nbpm = NSBPM ;
  
  /* initialize all data accumulators on all BPMs to zero */
  
  for (count=0 ; count < NSBPM ; count++)
  {
    sbpmdata[*SBPMCounter]->x[count] = 0. ;
    sbpmdata[*SBPMCounter]->y[count] = 0. ;
    sbpmdata[*SBPMCounter]->S[count] = 0. ;
    sbpmdata[*SBPMCounter]->Q[count] = 0. ;
  }
  
  
  egress:
    
    return stat;
    
}

/*=====================================================================*/

/* Set S positions of SBPMs within a given RF structure. */

/* RET:    none.
 * ABORT:  never.
 * FAIL:   never. */

void SBPMSetS( int SBPMCounter,
        double S0, double L,
        int nslice, int* doSBPM,
        double* Lfrac )
{
  int count ;
  int sbpmno=-1 ;
  double Lcum = 0. ;
  
  for (count=0 ; count<=nslice ; count++)
  {
    if (doSBPM[count] > 0)
    {
      sbpmno++ ;
      sbpmdata[SBPMCounter]->S[sbpmno] = S0 + dS * Lcum ;
    }
    Lcum = Lcum + Lfrac[count] * L ;
  }
  
  return ;
}

/*=====================================================================*/

/* charge-normalize the data in the SBPMs, add offsets and electrical
 * noise */

/* RET:    status int == 1 for all okay, == -1 if no offset or resolution
 * info, == 0 if offsets corrupted.
   ABORT:  never.
   FAIL:   */


int ComputeSBPMReadings( int SBPMCounter, int elemno, double dTilt )
{
  int count ;
  double* offset;
  int noffset ;
  double* resol ;
  double* noise=NULL ;
  int stat = 1 ;
  double QC ;
  double x,y ;
  double costilt, sintilt ;
  
  /* get the BPM offsets from the BEAMLINE data structure */
  
  offset = GetElemNumericPar( elemno, "BPMOffset", &noffset ) ;
  if (offset == NULL)
    stat = -1 ;
  else if (noffset != 2*sbpmdata[SBPMCounter]->nbpm)
  {
    stat = 0 ;
    goto egress ;
  }
  costilt = cos(dTilt) ;
  sintilt = sin(dTilt) ;
  
  resol = GetElemNumericPar( elemno, "BPMResolution", NULL ) ;
  if (resol == NULL)
    stat = -1 ;
  else if (*resol != 0.)
    noise = RanGaussVecPtr(2*sbpmdata[SBPMCounter]->nbpm) ;
  
  /* loop over SBPM slots */
  
  for (count=0 ; count<sbpmdata[SBPMCounter]->nbpm ; count++)
  {
    QC = sbpmdata[SBPMCounter]->Q[count] ;
    if (QC==0.)
      QC = 1. ;
    
    x = sbpmdata[SBPMCounter]->x[count] / QC ;
    y = sbpmdata[SBPMCounter]->x[count] / QC ;
    
    sbpmdata[SBPMCounter]->x[count] =  x*costilt + y*sintilt ;
    sbpmdata[SBPMCounter]->y[count] = -x*sintilt + y*costilt ;
    if (offset != NULL)
    {
      sbpmdata[SBPMCounter]->x[count] += offset[2*count]  ;
      sbpmdata[SBPMCounter]->y[count] += offset[2*count+1]  ;
    }
    if (noise != NULL)
    {
      sbpmdata[SBPMCounter]->x[count] += noise[2*count]   * (*resol) ;
      sbpmdata[SBPMCounter]->y[count] += noise[2*count+1] * (*resol) ;
    }
  }
  
  egress:
    
    return stat ;
    
}

/*=====================================================================*/

/* Clear out one or more convolved wakefields from a bunch.
 *
 * RET:    none.
 * ABORT:  none.
 * FAIL:   none. */

void ClearConvolvedSRWF( struct Bunch* ThisBunch, int wake, int flag )
{
  int first, last, count ;
  struct SRWF* TheWF ;
  struct SRWF** backbone ;
  int* numwake ;
  
  /* are we interested in ZSRs or TSRs? */
  
  if (flag==0)
    backbone = ThisBunch->ZSR ;
  else
    backbone = ThisBunch->TSR ;
  
  /* how many? */
  
  numwake = GetNumWakes( ) ;
  
  if (wake==-1)
  {
    first = 0 ;
    last = numwake[flag] ;
  }
  else
  {
    first = wake ;
    last = wake+1 ;
  }
  
  /* loop over wakes and begin clearing */
  
  for (count=first ; count<last ; count++)
  {
    TheWF = backbone[count] ;
    if (TheWF==NULL)
      continue ;
    FreeAndNull( (void**)&(TheWF->binno) ) ;
    FreeAndNull( (void**)&(TheWF->binQ) ) ;
    FreeAndNull( (void**)&(TheWF->ptype) ) ;
    FreeAndNull( (void**)&(TheWF->binx) ) ;
    FreeAndNull( (void**)&(TheWF->biny) ) ;
    FreeAndNull( (void**)&(TheWF->binVx) ) ;
    FreeAndNull( (void**)&(TheWF->binVy) ) ;
    FreeAndNull( (void**)&(TheWF->K) ) ;
    FreeAndNull( (void**)&(TheWF) ) ;
    backbone[count] = TheWF ;
  }
  
  return ;
  
}

/*=====================================================================*/

/* Clear out one or more binning-vectors related to a LRWF in Freq domain */

/* RET:    none.
 * ABORT:  none.
 * FAIL:   none. */

void ClearBinnedLRWFFreq( struct Bunch* ThisBunch, int wake, int flag )
{
  int first, last, count ;
  struct LRWFFreq* TheWF ;
  struct LRWFFreq** backbone ;
  int* numwake ;
  
  /* are we interested in ZSRs or TSRs? */
  
  if (flag==0)
    backbone = ThisBunch->TLRFreq ;
  else
    backbone = ThisBunch->TLRErrFreq ;
  
  /* how many? */
  
  numwake = GetNumWakes( ) ;
  
  if (wake==-1)
  {
    first = 0 ;
    last = numwake[flag+2] ;
  }
  else
  {
    first = wake ;
    last = wake+1 ;
  }
  
  /* loop over wakes and begin clearing */
  
  for (count=first ; count<last ; count++)
  {
    TheWF = backbone[count] ;
    if (TheWF==NULL)
      continue ;
    FreeAndNull( (void**)&(TheWF->binno) ) ;
    FreeAndNull( (void**)&(TheWF->binQ) ) ;
    FreeAndNull( (void**)&(TheWF->binVx) ) ;
    FreeAndNull( (void**)&(TheWF->binVy) ) ;
    FreeAndNull( (void**)&(TheWF->binx) )  ;
    FreeAndNull( (void**)&(TheWF->biny) ) ;
    FreeAndNull( (void**)&(TheWF->Wx) ) ;
    FreeAndNull( (void**)&(TheWF->Wy) ) ;
    FreeAndNull( (void**)&(TheWF->xphase) ) ;
    FreeAndNull( (void**)&(TheWF->yphase) ) ;
    FreeAndNull( (void**)&(TheWF) ) ;
    backbone[count] = TheWF ;
  }
  
  return ;
  
}

/*=====================================================================*/

/* Get the required parameters for an operation out of the database.
 * If requested, check the length of the returned scalar/vector of
 * parameters.  Make sure that all parameters needed for the operation
 * are present, and that all parameters for which the parameter length
 * is crucial meet that tolerance.  Signal failure, success, or warn
 * that optional parameters are missing or have the wrong length. */

/* RET:    +1 if all required and optional parameters are present, and
 * all parameters with a required or optional length tolerance
 * are within that tolerance
 * -1 if optional parameters are missing, or parameters for which
 * the length tolerance is optional are not within tolerance
 * 0 if required parameters are missing, or parameters with a
 * required length tolerance are not within tolerance.
 * ABORT:  Never.
 * FAIL:   Never. */

int GetDatabaseParameters( int elemno, int nParam,
        struct LucretiaParameter Dictionary[],
        int WhichPars, int WhichTable )
{
  
  int count ;
  int stat = 1 ;
  int LengthOK ;
  
  /* loop over parameters */
  
  for (count=0 ; count < nParam ; count++)
  {
    
    /* is the parameter needed, based on the calling operation? */
    
    if (Dictionary[count].Requirement[WhichPars] != Ignored)
    {
      
      /* figure out which table to interrogate, and go to it! */
      
      switch( WhichTable )
      {
        case ElementTable:
          Dictionary[count].ValuePtr =
                  GetElemNumericPar( elemno,
                  Dictionary[count].name,
                  &(Dictionary[count].Length) ) ;
          break ;
        case PSTable:
          Dictionary[count].ValuePtr =
                  GetPSNumericPar( elemno,
                  Dictionary[count].name,
                  &(Dictionary[count].Length) ) ;
          break ;
        case GirderTable:
          Dictionary[count].ValuePtr =
                  GetGirderNumericPar( elemno,
                  Dictionary[count].name,
                  &(Dictionary[count].Length) ) ;
          break ;
        case KlystronTable:
          Dictionary[count].ValuePtr =
                  GetKlystronNumericPar( elemno,
                  Dictionary[count].name,
                  &(Dictionary[count].Length) ) ;
          break ;
        case TLRTable:
          Dictionary[count].ValuePtr =
                  GetTLRNumericPar( elemno,
                  Dictionary[count].name,
                  &(Dictionary[count].Length) ) ;
          break ;
        case TLRErrTable:
          Dictionary[count].ValuePtr =
                  GetTLRErrNumericPar( elemno,
                  Dictionary[count].name,
                  &(Dictionary[count].Length) ) ;
          break ;
        default:
          stat = 0 ;
          goto egress ;
      }
      
      /* handle a missing parameter */
      
      if (Dictionary[count].ValuePtr == NULL)
      {
        if (Dictionary[count].Requirement[WhichPars] == Required)
        {
          stat = 0 ;       /* missing required par => return bad status */
          goto egress ;
        }
        stat = -1 ;         /* missing optional par => return warning */
        
      }
      
      /* if the parameter is not missing, handle any requirements on its length */
      
      else if (Dictionary[count].LengthRequirement[WhichPars] != Ignored)
      {
        LengthOK = ( (Dictionary[count].Length >= Dictionary[count].MinLength) &&
                (Dictionary[count].Length <= Dictionary[count].MaxLength)    ) ;
        if ( !LengthOK )
        {
          if (Dictionary[count].LengthRequirement[WhichPars] == Required)
          {
            stat = 0 ;
            goto egress ;
          }
          stat = -1 ;
        }
      }
      
    }
    
  }
  
  egress:
    
    return stat ;
    
}

/*=====================================================================*/

/* Return the value of a bend magnet parameter for which the user can
 * specify either 1 value or 2, and for which the calling routine wants
 * either the first, or the second (which is equal to the first if no
 * second is specified). */

/* RET:    double precision value of desired parameter.
 * ABORT:  never.
 * FAIL:   never. */

double GetSpecialSBendPar(struct LucretiaParameter* ThePar, int index)
{
  double* dptr ;
  
  dptr = ThePar->ValuePtr ;
  
  if ( (index == 1) && (ThePar->Length == 2) )
    dptr++ ;
  
  return *dptr ;
  
}

/*=====================================================================*/

/* Get the design Lorentz delay of an element based on its design
 * momentum.  Return zero if the design momentum is zero.
 *
 * RET:    double precision value of Lorentz delay.
 * ABORT:  never.
 * FAIL:   never. */

double GetDesignLorentzDelay( double* pmod )
{
  double dzmod ;
  
  if (pmod == NULL)
    dzmod = 0. ;
  else if (*pmod == 0.)
    dzmod = 0. ;
  else
    dzmod = LORENTZ_DELAY(*pmod) ;
  
  return dzmod ;
  
}

/*=====================================================================*/

/* Exchange the x and y coordinate vectors of a bunch */
/* Mod :
02-apr-2014 GRW
- Simplify calling with just Bunch structure
call */
/* RET:    None.
 * ABORT:  never.
 * FAIL:   never. */
void XYExchange( struct Bunch* ThisBunch )
{
  double* temp ;
  double** xb = &ThisBunch->x ;
  double** yb = &ThisBunch->y ;

#ifdef __CUDACC__
  int nray = ThisBunch->nray ;
  gpuErrchk( cudaMalloc((void**)&temp, 6*nray*sizeof(double)) );
  gpuErrchk( cudaMemcpy(temp, *xb, 6*nray*sizeof(double), cudaMemcpyDeviceToDevice) );
  gpuErrchk( cudaMemcpy(*xb, *yb, 6*nray*sizeof(double), cudaMemcpyDeviceToDevice) );
  gpuErrchk( cudaMemcpy(*yb, temp, 6*nray*sizeof(double), cudaMemcpyDeviceToDevice) );
  gpuErrchk( cudaFree(temp) );
#else
  temp = *xb ;
  *xb = *yb ;
  *yb = temp ;
#endif
  
  return ;
  
}

/*=====================================================================*/

/* Perform initial check of all momenta in a tracked bunch.  If some are
 * bad (P0 <= 0 or Pperp >= 1) raise a warning and issue a message. */

/* RET:    Status (stat), 0 = all incoming particles OK (good p0 and Pperp, or
 * bad p0 and/or Pperp but stopped), >0 = bad p <0 = bad pt
 * ABORT:  never.
 * FAIL:   never. */
#ifdef __CUDACC__
__global__ void InitialMomentumCheck( int* stat, double* bstop, int *ngoodray, double* x, double* y, int RayLoop, int* stp )
#else
void InitialMomentumCheck( int* stat, double* bstop, int *ngoodray, double* x, double* y, int RayLoop, int* stp )
#endif
{
  int stop, i ;
  double *px, *py ;
  
#ifdef __CUDA_ARCH__
  i = blockDim.x * blockIdx.x + threadIdx.x ;
  if ( i >= RayLoop ) return;
#else
  i = RayLoop;
#endif
  
  stat[i] = 0 ;
  if (bstop[i] == 0)
  {
    stop = CheckP0StopPart( bstop, ngoodray, x, y, 0, i, x[6*i+5], UPSTREAM, stp ) ;
    if (stop != 0)
    {
      stat[i] = 1 ;
      return ;
    }
    else
    {
      px = &(x[6*i+1]) ;
      py = &(x[6*i+3]) ;
      stop = CheckPperpStopPart( bstop, ngoodray, 0, i, px, py, stp ) ;
      if (stop != 0)
      {
        stat[i] = -1 ;
        return ;
      }
    }
  }
}

/*=====================================================================*/

/* Set local coordinate pointers (*x, etc) to point at a given ray. */


/* RET:    None.
 * ABORT:  never.
 * FAIL:   If selected rayno > # of rays in the data vector */

void GetLocalCoordPtrs( double coordvec[], int raystart, double** x, double** px,double** y,double** py,double** z,double** p0 )
{
  *x  = &(coordvec[raystart]  ) ;
  *px = &(coordvec[raystart+1]) ;
  *y  = &(coordvec[raystart+2]) ;
  *py = &(coordvec[raystart+3]) ;
  *z  = &(coordvec[raystart+4]) ;
  *p0 = &(coordvec[raystart+5]) ;
  
}

/*=====================================================================*/

/* consistency check for girder mover parameter lengths */

/* RET:    +1 if mover parameters are consistent (ie either all
 * present or all absent, and all the same length)
 * +2 if Mover(...) has invalid DOFs (ie values which
 * are not between 1 and 6 inclusive, or duplicates)
 * 0 otherwise.
 * ABORT:  never.
 * FAIL:   never. */

int CheckGirderMoverPars( struct LucretiaParameter gpar[],
        int WhichPars )
{
  int FirstMovrPar = GirderMover;
  int LastMovrPar ;
  int MovrPar ;
  int stat ;
  int i ;
  int dof;
  int nTimesFound[6] = {0,0,0,0,0,0} ;
  
  stat = 1 ;
  if (WhichPars == TrackPars)
    LastMovrPar = GirderMoverPos ;
  else
    LastMovrPar = GirderMoverStep ;
  
  if (gpar[FirstMovrPar].ValuePtr == NULL)
  {
    for (MovrPar = FirstMovrPar+1 ; MovrPar <= LastMovrPar ; MovrPar++)
    {
      if (gpar[MovrPar].ValuePtr != NULL)
        stat = 0 ;
    }
  }
  else
  {
    for (MovrPar = FirstMovrPar+1 ; MovrPar <= LastMovrPar ; MovrPar++)
    {
      if (gpar[MovrPar].ValuePtr == NULL)
        stat = 0 ;
      if (gpar[MovrPar].Length != gpar[FirstMovrPar].Length)
        stat = 0 ;
    }
    for (i = 0 ; i < gpar[FirstMovrPar].Length ; i++)
    {
      dof = (int)(*(gpar[FirstMovrPar].ValuePtr+i)) ;
      if ( (dof<1) || (dof>6) )
      {
        stat = 2 ;
        goto egress ;
      }
      nTimesFound[dof-1]++ ;
      if (nTimesFound[dof-1]>1)
      {
        stat = 2 ;
        goto egress ;
      }
    }
    
  }
  
  egress:
    
    return stat ;
    
}

/*=====================================================================*/

/* Derefernce a double pointer; if pointer is null, return default value. */

/* RET:   The value of the double which is pointed to by the argument,
 * or a default value if null pointer.
 * ABORT: never.
 * FAIL:  never. */

double GetDBValue( struct LucretiaParameter* Pardat )
{
  double* valptr ;
  
  valptr = (*Pardat).ValuePtr ;
  if (valptr != NULL)
    return *valptr ;
  else
    return (*Pardat).DefaultValue ;
}

/*=====================================================================*/

/* Perform lattice verification, putting messages related to errors
 * and warnings into Lucretia's text message queue.
 *
 * RET:    none.
 * ABORT:  never.
 * FAIL:   never. */

void VerifyLattice( )
{
  
#include "LucretiaVerifyMsg.h"
  
  int nElem, nPS, nGirder, nKlys ;
  int* nWakes ;
  int nDat, ElemP ;
  int count1, count2, count0, count3 ;
  int tlrtypecount, tlrKcountFact ;
  int wakecondition ;
  const char* wf ;
  char message[100] ;
  int paramstat[15], paramlenstat[15] ;
  int stat ;
  double* retval=NULL ;
  int retlen ;
  int elemno ;
  char* ElemClass ;
  int GirderIndex=0, KlysIndex=0, PSIndex=0 ;
  int PSVec[MaxPSPerDevice],
          KlysVec[MaxKlysPerDevice],
          GirderVec[MaxGirderPerDevice] ;
  int LenPSVec, LenKlysVec, LenGirderVec ;
  int* HardwareVec ;
  int LenHardwareVec ;
  int PIndex=0, LIndex=0, aperindex=0 ;
  int WFIndex=0, WFlen ;
  double* WFPtr ;
  struct LucretiaParameter* Dictionary=NULL ;
  int nPar=0 ;
  enum KlystronStatus* kstat ;
  int* AllowedTrackFlag=NULL ;
  int* ActualTrackFlag ;
  int pole ;
  double dmy ;
  int LradIndex=0 ;
  int ZeroLenOK=0 ;
  
  
  /* get sizes of globals */
  
  nElem = nElemInBeamline( )  ;
  nPS = GetnPS( ) ;
  nKlys = GetnKlystron( ) ;
  nGirder = GetnGirder( ) ;
  nWakes = GetNumWakes( ) ;
  
  /* if no elements, it's an error */
  
  if (nElem==0)
    AddMessage("Error: no BEAMLINE or zero elements in BEAMLINE",0) ;
  
  /* if the others are missing, it's just warnings */
  
  if (nPS==0)
    AddMessage("Warning: no PS or zero power supplies in PS",0) ;
  if (nKlys==0)
    AddMessage("Warning: no KLYSTRON or zero klystrons in KLYSTRON",0) ;
  if (nGirder==0)
    AddMessage("Warning: no GIRDER or zero girders in GIRDER",0) ;
  if (nWakes[0]==0)
    AddMessage("Warning: no short-range longitudinal wakefields",0) ;
  if (nWakes[1]==0)
    AddMessage("Warning: no short-range transverse wakefields",0) ;
  if (nWakes[2]==0)
    AddMessage("Warning: no long-range transverse wakefields",0) ;
  
  /* Wakefield verification:  each SRWF must have a vector of z values
   * and a vector of K values which are equal in length, and must have
   * a bin width / spline accuracy factor: */
  
  for (count1 = 0 ; count1 < 2 ; count1++ )
  {
    if (count1==0)
      wf = zwf ;
    else
      wf = twf ;
    for (count2 = 0 ; count2 < nWakes[count1] ; count2++)
    {
      double *z, *k, bw ;
      wakecondition = GetSRWFParameters( count2, count1, &z, &k, &bw) ;
      
      /* if wakecondition is zero or less it signifies an error condition */
      
      switch ( wakecondition )
      {
        case  0 : /* no such WF */
          sprintf( message, "Error: %s wake %d missing",wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
        case -1 : /* z vector problems */
          sprintf( message, "Error: %s wake %d z vector missing",
                  wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
        case -2 : /* z vector has zero length */
          sprintf( message, "Error: %s wake %d z vector has zero length",
                  wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
        case -3 : /* z[0] != 0 */
          sprintf( message, "Error: %s wake %d z(0) ~= 0",
                  wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
        case -4 : /* K vector problems */
          sprintf( message, "Error: %s wake %d K vector missing",
                  wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
        case -5 : /* K vector has wrong length */
          sprintf( message, "Error: %s wake %d z vector length ~= k vector length",
                  wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
        case -6 : /* no binwidth */
          sprintf( message, "Error: %s wake %d BinWidth missing",
                  wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
        case -7 : /* zero binwidth */
          sprintf( message, "Error: %s wake %d BinWidth == 0",
                  wf,count2+1) ;
          AddMessage( message, 0 ) ;
          break ;
      }
    }
  } /* end of srwf verification */
  
  /* lrwf verification:  each transverse long-range wakefield must be (for now)
   * recognizably in the frequency domain, and each must have a full set of
   * equal-length vectors for Freq, dFreq, K, Q, Tilt, and a scalar BinWidth */
  
  for (tlrtypecount = 2 ; tlrtypecount<4 ; tlrtypecount++)
  {
    if (tlrtypecount==2) /* regular LRWF */
    {
      tlrKcountFact = 2 ;
      wf = TLRStr ;
    }
    else                 /* error LRWF */
    {
      tlrKcountFact = 1 ;
      wf = TLRErrStr ;
    }
    for (count2=0 ; count2 < nWakes[tlrtypecount] ; count2++)
    {
      int wakeclass ;
      if (tlrtypecount==2)
        wakeclass = GetTLRWakeClass( count2 ) ;
      else
        wakeclass = GetTLRErrWakeClass( count2 ) ;
      switch ( wakeclass )
      {
        case UNKNOWNDOMAIN :
          sprintf(message, "Error:  %s # %d Class unknown (not Time or Frequency)",
                  wf, count2+1) ;
          AddMessage(message,0) ;
          continue ;
        case TIMEDOMAIN :
          sprintf(message, "Error: %s # %d in Time domain -- not yet supported",
                  wf, count2+1) ;
          continue ;
        case FREQDOMAIN :
          
          /* get the parameters using the dictionary and the VerifyParameters function */
          
          stat = VerifyParameters( count2, nLRWFFreqPar, LRWFFreqPar,
                  TLRTable, paramstat, paramlenstat ) ;
          
          /* since there are no optional parameters, and there are no dictionary constraints
           * on parameters lengths, the only thing that can be reported by the verifier is
           * missing, corrupted, or non-numeric fields where well-formed numeric fields are
           * expected */
          
          if (stat != 1)
            for (count1=0 ; count1 < nLRWFFreqPar ; count1++)
              if (paramstat[count1] == 0)
              {
                sprintf(message, MissPar, ErrStr, wf, count1+1, reqdStr,
                        LRWFFreqPar[count1].name);
                AddMessage(message,0) ;
              }
          
          /* the other thing that can be wrong with a frequency-domain wakefield is that
           * the vector fields lengths can fail to match */
          
          if (
                  (LRWFFreqPar[LRWFFreqQ].Length     != LRWFFreqPar[LRWFFreqFreq].Length)
                  ||
                  (tlrKcountFact*LRWFFreqPar[LRWFFreqK].Length != LRWFFreqPar[LRWFFreqFreq].Length)
                  ||
                  (2*LRWFFreqPar[LRWFFreqTilt].Length  != LRWFFreqPar[LRWFFreqFreq].Length) )
          {
            sprintf(message,"Error: WF.TLR %d vector parameters have unequal length",
                    count2+1) ;
            AddMessage(message,0) ;
          }
          break ;
      }
    }
  }
  
  /* Now we verify each of the tables, which is a semi-repetitive operation with
   * a few differences for each table */
  
  for (count0 = ElementTable ; count0<=KlystronTable ; count0++)
  {
    if (count0 == ElementTable)
    {
      nDat = nElem ;
      Keywd = ElemStr ;
      ElemP = 0 ;
    }
    else if (count0 == PSTable)
    {
      nPar = nPSPar ;
      Dictionary = PSPar ;
      nDat = nPS ;
      Keywd = PSStr ;
      ElemP = PSElem ;
    }
    else if (count0 == GirderTable)
    {
      nPar = nGirderPar ;
      Dictionary = GirderPar ;
      nDat = nGirder ;
      Keywd = GirderStr ;
      ElemP = GirderElem ;
    }
    else if (count0 == KlystronTable)
    {
      nPar = nKlystronPar ;
      Dictionary = KlystronPar ;
      nDat = nKlys ;
      Keywd = KlysStr ;
      ElemP = KlysElem ;
    }
    
    for (count1 = 0 ; count1 < nDat ; count1++)
    {
      /* for elements, assign dictionary etc now */
      if (count0 == ElementTable)
      {
        ZeroLenOK = 0 ;
        ElemClass = GetElemClass( count1 ) ;
        if (ElemClass == NULL)
        {
          sprintf( message, "Error: Element %d has no Class",count1+1) ;
          AddMessage( message, 0 ) ;
          continue ;
        }
        if (strcmp(ElemClass,"MARK")==0) /* marker */
        {
          double *P, *S ;
          P = GetElemNumericPar( count1, "P", NULL ) ;
          S = GetElemNumericPar( count1, "S", NULL ) ;
          if (P==NULL)
          {
            sprintf( message, "Error: Element %d required parameter P missing",
                    count1+1 ) ;
            AddMessage( message, 0 ) ;
          }
          else if (*P <= 0.)
          {
            sprintf( message, "Error: Element %d design momentum <= 0",
                    count1+1 ) ;
            AddMessage( message, 0 ) ;
          }
          if (S == NULL)
          {
            sprintf( message, "Error: Element %d required parameter S missing",
                    count1+1 ) ;
            AddMessage( message, 0 ) ;
          }
          continue ;
        }
        else if ( (strcmp(ElemClass,"QUAD") == 0) ||
                (strcmp(ElemClass,"SEXT") == 0) ||
                (strcmp(ElemClass,"OCTU") == 0) ||  (strcmp(ElemClass,"PLENS") == 0)  )/* quad, sext, octupole magnet */
        {
          Dictionary = QuadPar ;
          nPar = nQuadPar ;
          PIndex = QuadP ;
          LIndex = QuadL ;
          aperindex = Quadaper ;
          AllowedTrackFlag = QuadTrackFlag ;
          GirderIndex = QuadGirder ;
          PSIndex = QuadPS ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if (strcmp(ElemClass,"SOLENOID") == 0)/* solenoid magnet */
        {
          Dictionary = SolePar ;
          nPar = nSolePar ;
          PIndex = SoleP ;
          LIndex = SoleL ;
          aperindex = Soleaper ;
          AllowedTrackFlag = SoleTrackFlag ;
          GirderIndex = SoleGirder ;
          PSIndex = SolePS ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if (strcmp(ElemClass,"MULT") == 0) /* thin lens multipole */
        {
          Dictionary = MultPar ;
          nPar = nMultPar ;
          PIndex = MultP ;
          LIndex = MultL ;
          aperindex = Multaper ;
          AllowedTrackFlag = MultTrackFlag ;
          GirderIndex = MultGirder ;
          PSIndex = MultPS ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
          LradIndex = MultLrad ;
          ZeroLenOK = 1 ;
        }
        else if (strcmp(ElemClass,"SBEN") == 0) /* sector bend */
        {
          Dictionary = SBendPar ;
          nPar = nSBendPar ;
          PIndex = SBendP ;
          LIndex = SBendL ;
          aperindex = SBendHGAP ;
          AllowedTrackFlag = SBendTrackFlag ;
          GirderIndex = SBendGirder ;
          PSIndex = SBendPS ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if (strcmp(ElemClass,"LCAV") == 0) /* linac structure */
        {
          Dictionary = LcavPar ;
          nPar = nLcavPar ;
          PIndex = LcavP ;
          LIndex = LcavL ;
          aperindex = Lcavaper ;
          AllowedTrackFlag = LcavTrackFlag ;
          GirderIndex = LcavGirder ;
          PSIndex = 0 ;
          KlysIndex = LcavKlystron ;
          WFIndex = LcavWakes ;
        }
        else if (strcmp(ElemClass,"TCAV") == 0) /* linac structure */
        {
          Dictionary = TcavPar ;
          nPar = nTcavPar ;
          PIndex = TcavP ;
          LIndex = TcavL ;
          aperindex = Tcavaper ;
          AllowedTrackFlag = TcavTrackFlag ;
          GirderIndex = TcavGirder ;
          PSIndex = 0 ;
          KlysIndex = TcavKlystron ;
          WFIndex = TcavWakes ;
        }
        else if ( (strcmp(ElemClass,"HMON") == 0) ||
                (strcmp(ElemClass,"VMON") == 0) ||
                (strcmp(ElemClass,"MONI") == 0)    ) /* BPM */
        {
          Dictionary = BPMPar ;
          nPar = nBPMPar ;
          PIndex = BPMP ;
          LIndex = 0 ;
          aperindex = 0 ;
          AllowedTrackFlag = BPMTrackFlag ;
          GirderIndex = BPMGirder ;
          PSIndex = 0 ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if ( (strcmp(ElemClass,"PROF") == 0) ||
                (strcmp(ElemClass,"WIRE") == 0) ||
                (strcmp(ElemClass,"BLMO") == 0) ||
                (strcmp(ElemClass,"SLMO") == 0) ||
                (strcmp(ElemClass,"IMON") == 0) ||
                (strcmp(ElemClass,"INST") == 0)    ) /* instrument */
        {
          Dictionary = InstPar ;
          nPar = nInstPar ;
          PIndex = InstP ;
          LIndex = 0 ;
          aperindex = 0 ;
          AllowedTrackFlag = InstTrackFlag ;
          GirderIndex = InstGirder ;
          PSIndex = 0 ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if ( (strcmp(ElemClass,"XCOR") == 0) ||
                (strcmp(ElemClass,"YCOR") == 0)    ) /* corrector */
        {
          Dictionary = CorrectorPar ;
          nPar = nCorrectorPar ;
          PIndex = CorP ;
          LIndex = CorL ;
          aperindex = 0 ;
          AllowedTrackFlag = CorrectorTrackFlag ;
          GirderIndex = CorGirder ;
          PSIndex = CorPS ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
          LradIndex = CorLrad ;
          ZeroLenOK = 1 ;
        }
        else if (strcmp(ElemClass,"XYCOR") == 0)  /* combined x-y corrector */
        {
          Dictionary = XYCorrectorPar ;
          nPar = nCorrectorPar ;
          PIndex = CorP ;
          LIndex = CorL ;
          aperindex = 0 ;
          AllowedTrackFlag = CorrectorTrackFlag ;
          GirderIndex = CorGirder ;
          PSIndex = CorPS ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
          LradIndex = CorLrad ;
          ZeroLenOK = 1 ;
        }
        else if (strcmp(ElemClass,"COLL") == 0) /* collimator */
        {
          Dictionary = CollPar ;
          nPar = nCollPar ;
          PIndex = CollP ;
          LIndex = 0 ;
          aperindex = Collaper ;
          AllowedTrackFlag = CollTrackFlag ;
          GirderIndex = CollGirder ;
          PSIndex = 0 ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if (strcmp(ElemClass,"COORD") == 0) /* coordinate change */
        {
          Dictionary = CoordPar ;
          nPar = nCoordPar ;
          PIndex = CoordP ;
          LIndex = 0 ;
          aperindex = 0 ;
          AllowedTrackFlag = CoordTrackFlag ;
          GirderIndex = 0 ;
          PSIndex = 0 ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if ( (strcmp(ElemClass,"TMAP") == 0) ) /* TMap */
        {
          Dictionary = TMapPar ;
          nPar = nTMapPar ;
          PIndex = TMapP ;
          LIndex = 0 ;
          aperindex = 0 ;
          AllowedTrackFlag = TMapTrackFlag ;
          GirderIndex = 0 ;
          PSIndex = 0 ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else if ( (strcmp(ElemClass,"DRIF") == 0) ) /* Drift */
        {
          Dictionary = DrifPar ;
          nPar = nDrifPar ;
          PIndex = DriftP ;
          LIndex = 0 ;
          aperindex = 0 ;
          AllowedTrackFlag = DrifTrackFlag ;
          GirderIndex = 0 ;
          PSIndex = 0 ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
        else /* unrecognized element class */
        {
          sprintf( message, "Warning: Element %d Class %s unknown, Drift assumed",
                  count1+1, ElemClass ) ;
          AddMessage( message, 0 ) ;
          Dictionary = DrifPar ;
          nPar = nDrifPar ;
          PIndex = DriftP ;
          LIndex = 0 ;
          aperindex = 0 ;
          AllowedTrackFlag = DrifTrackFlag ;
          GirderIndex = 0 ;
          PSIndex = 0 ;
          KlysIndex = 0 ;
          WFIndex = 0 ;
        }
      }
      
      stat = VerifyParameters( count1, nPar, Dictionary, count0,
              paramstat, paramlenstat ) ;
      if (stat != 1)
      {
        for (count2 = 0 ; count2 < nPar ; count2++)
        {
          
          parname = Dictionary[count2].name ;
          
          /* If this is a girder with no mover, we only actually need 1 info message about missing
           * mover parameters (there are 4 such parameters).  Suppress the unwanted ones. */
          
          if ( (count0 == GirderTable) &&
                  ( (count2==GirderMoverPos)   ||
                  (count2==GirderMoverSetpt) ||
                  (count2==GirderMoverStep)     ) )
            continue ;
          if ( (count0 == GirderTable) &&
                  (count2 == GirderMover) &&
                  (paramstat[count2]==-1)    )
          {
            sprintf(message, MissPar, InfoStr, Keywd, count1+1, optStr, parname) ;
            AddMessage(message,0) ;
            continue ;
          }
          
          /* If the element has no klystron, PS, or girder, but that table doesn't exist at all,
           * don't issue a message (the message about no klystron, PS, or girder table should
           * be sufficient */
          
          if ( (count0 == ElementTable) &&
                  ( ( (count2==GirderIndex) && (GirderIndex != 0) && (nGirder==0) ) ||
                  ( (count2==KlysIndex)   && (KlysIndex != 0)   && (nKlys==0)   ) ||
                  ( (count2==PSIndex)     && (PSIndex != 0)     && (nPS==0)     )    ) )
            continue ;
          
          switch ( paramstat[count2] )
          {
            case  1 : /* all OK */
              break ;
            case  0 : /* error */
              sprintf(message, MissPar, ErrStr, Keywd, count1+1, reqdStr, parname);
              AddMessage( message, 0 ) ;
              break ;
            case -1 : /* warning */
              sprintf(message, MissPar, WarnStr, Keywd, count1+1, optStr, parname);
              AddMessage( message, 0 ) ;
              break ;
          }
          switch ( paramlenstat[count2] )
          {
            case  1 : /* all OK */
              break ;
            case  0 : /* error */
              sprintf(message, BadLen, ErrStr, Keywd, count1+1, parname) ;
              AddMessage( message, 0 ) ;
              break ;
            case -1 : /* warning */
              sprintf(message, BadLen, WarnStr, Keywd, count1+1, parname) ;
              AddMessage( message, 0 ) ;
              break ;
          }
        }
      }
      
      /* now check its list of elements */
      
      if (ElemP != 0)
      {
        if (Dictionary[ElemP].Length == 0)
        {
          sprintf( message, ZeroElt, WarnStr, Keywd, count1+1 ) ;
          AddMessage( message, 0 ) ;
        }
        else
        {
          for (count2=0 ; count2 < Dictionary[ElemP].Length ; count2++)
          {
            elemno = (int)(*(Dictionary[ElemP].ValuePtr+count2))-1 ;
            retval = GetElemNumericPar( elemno, Keywd, &retlen ) ;
            if (retval!=NULL)
            {
              stat = ElemIndexLookup(count1+1,retval,retlen) ;
              if (stat == -1)
                
              {
                sprintf(message, Inconsis, WarnStr, Keywd,
                        count1+1, ElemStr, elemno+1) ;
                AddMessage( message, 0 ) ;
              }
            }
          }
        }
      }
      
      /* for klystrons, we additionally check that the klystron status is present
       * and valid */
      
      if (count0 == KlystronTable)
      {
        kstat = GetKlystronStatus( count1 ) ;
        if (kstat==NULL)
        {
          sprintf( message, "Error: Klystron %d required parameter Status missing",
                  count1+1 ) ;
          AddMessage( message, 0 ) ;
        }
        else if ( (*kstat != TRIPPED) &&
                (*kstat != STANDBY) &&
                (*kstat != STANDBYTRIP) &&
                (*kstat != ON) &&
                (*kstat != MAKEUP) )
        {
          sprintf( message, "Error: Klystron %d Status value invalid",
                  count1+1 ) ;
          AddMessage( message, 0 ) ;
        }
      }
      
      /* for girders, there may be magnet movers which need checking */
      
      if (count0 == GirderTable)
      {
        if (Dictionary[GirderMover].ValuePtr != NULL)
        {
          stat = CheckGirderMoverPars( Dictionary, VerifyPars ) ;
          if (stat == 0) /* inconsistent lengths */
          {
            sprintf( message, "Error:  Girder %d mover data vectors have different lengths",
                    count1+1 ) ;
            AddMessage( message, 0 ) ;
          }
          else if (stat == 2) /* bad DOFs */
          {
            sprintf( message, "Error: Girder %d mover has invalid DOFs",count1+1 ) ;
            AddMessage( message, 0 ) ;
          }
        }
      }
      
      /* for elements, there are a number of additional checks */
      
      if (count0 == ElementTable)
      {
        /* start with elements with illegal lengths (not all classes are involved */
        
        if ( (LIndex > 0) && (*(Dictionary[LIndex].ValuePtr)<=0) &&
                (ZeroLenOK == 0)                                       )
        {
          sprintf( message, "Error: Element %d length <= 0",count1+1 ) ;
          AddMessage( message, 0 ) ;
        }
        
        /* now momentum <= 0 */
        
        if (*(Dictionary[PIndex].ValuePtr)<=0)
        {
          sprintf( message, "Error: Element %d design momentum <= 0",count1+1 ) ;
          AddMessage( message, 0 ) ;
        }
        
        /* Check for pointers to girder, klystron, PS, and wakefield tables */
        
        if (PSIndex != 0)
        {	int CountPSVec ;
          LenPSVec = Dictionary[PSIndex].Length ;
          LenPSVec = (LenPSVec<=MaxPSPerDevice)
          ? LenPSVec : MaxPSPerDevice ;
          for (CountPSVec=0 ;
          CountPSVec<LenPSVec ;
          CountPSVec++          )
          {
            PSVec[CountPSVec] =
                    (int)*(Dictionary[PSIndex].ValuePtr+CountPSVec) ;
          }
        }
        else
        {
          PSVec[0] = 0 ;
          LenPSVec = 1 ;
        }
        /*				PSIndex = (int)(GetDBValue(Dictionary+PSIndex)) ; */
        if (GirderIndex != 0)
        {	int CountGirderVec ;
          LenGirderVec = Dictionary[GirderIndex].Length ;
          LenGirderVec = (LenGirderVec<=MaxGirderPerDevice)
          ? LenGirderVec : MaxGirderPerDevice ;
          for (CountGirderVec=0 ;
          CountGirderVec<LenGirderVec ;
          CountGirderVec++              )
          {
            GirderVec[CountGirderVec] =
                    (int)*(Dictionary[GirderIndex].ValuePtr+CountGirderVec) ;
          }
        }
        else
        {
          GirderVec[0] = 0 ;
          LenGirderVec = 1 ;
        }
        /*				GirderIndex = (int)(GetDBValue(Dictionary+GirderIndex)) ; */
        if (KlysIndex != 0)
        {	int CountKlysVec ;
          LenKlysVec = Dictionary[KlysIndex].Length ;
          LenKlysVec = (LenKlysVec<=MaxKlysPerDevice)
          ? LenKlysVec : MaxKlysPerDevice ;
          for (CountKlysVec=0 ;
          CountKlysVec<LenKlysVec ;
          CountKlysVec++            )
          {
            KlysVec[CountKlysVec] =
                    (int)*(Dictionary[KlysIndex].ValuePtr+CountKlysVec) ;
          }
        }
        else
        {
          KlysVec[0] = 0 ;
          LenKlysVec = 1 ;
        }
        /*				KlysIndex = (int)(GetDBValue(Dictionary+KlysIndex)) ; */
        
        if (WFIndex != 0)
        {
          WFPtr = Dictionary[WFIndex].ValuePtr ;
          WFlen = Dictionary[WFIndex].Length ;
        }
        else
        {
          WFPtr = NULL ;
          WFlen = 0 ;
        }
        
        /* if the wakefields are not valid, issue an error message now */
        
        if (WFPtr != NULL)
        {
          if ( (WFPtr[0]<0) || (WFPtr[0]>nWakes[0]) )
          {
            sprintf( message, "Error: Element %d points to invalid ZSR %d",
                    count1+1, (int)(WFPtr[0]) ) ;
            AddMessage( message, 0 ) ;
          }
          if (WFlen > 1)
          {
            if ( (WFPtr[1]<0) || (WFPtr[1]>nWakes[1]) )
            {
              sprintf( message, "Error: Element %d points to invalid TSR %d",
                      count1+1, (int)(WFPtr[1]) ) ;
              AddMessage( message, 0 ) ;
            }
          }
          if (WFlen > 2)
          {
            if ( (WFPtr[2]<0) || (WFPtr[2]>nWakes[2]) )
            {
              sprintf( message, "Error: Element %d points to invalid TLR %d",
                      count1+1, (int)(WFPtr[2]) ) ;
              AddMessage( message, 0 ) ;
            }
          }
          if (WFlen > 3)
          {
            if ( (WFPtr[3]<0) || (WFPtr[3]>nWakes[3]) )
            {
              sprintf( message, "Error: Element %d points to invalid Error TLR %d",
                      count1+1, (int)(WFPtr[3]) ) ;
              AddMessage( message, 0 ) ;
            }
          }
        }
        
        /* for a collimator, make sure its geometry parameter is sensible */
        
        if (strcmp(ElemClass,"COLL")==0)
          if (GetCollimatorGeometry(count1) == COLL_UNKNOWN)
          {
            sprintf(message, "Error:  Element %d has unknown aperture geometry",
                    count1+1) ;
            AddMessage(message, 0) ;
          }
        
        /* additional checks are needed in the case of a multipole magnet */
        
        if (strcmp(ElemClass,"MULT")==0)
        {
          if ( (MultPar[MultB].Length != MultPar[MultTilt].Length)      ||
                  (MultPar[MultB].Length != MultPar[MultPoleIndex].Length)    )
          {
            sprintf(message,"Error: Element %d multipole parameters not equal in length",
                    count1+1) ;
            AddMessage(message,0) ;
          }
          for (pole=0 ; pole<MultPar[MultPoleIndex].Length ; pole++)
          {
            if (modf(MultPar[MultPoleIndex].ValuePtr[pole],&dmy)!=0.)
            {
              sprintf(message,"Error: Element %d PoleIndex entry %d not an integer",
                      count1+1, pole+1) ;
              AddMessage(message,0) ;
            }
          }
        }
        
        /* formatting checks for TMap element fields */
        if (strcmp(ElemClass,"TMAP")==0)
        {
          if ( TMapParamCheck(count1) == 0 ) {
            sprintf(message,"Error: Element %d Incorrect TMap field formatting",count1+1) ;
            AddMessage(message,0) ;
          }
        }
        
        /* now we make sure that its girder, klystron, or PS are pointed at an existing
         * table entry, and that that entry points back at this element */
        
        for (count3=PSTable; count3<=KlystronTable ; count3++)
        {
          const char* Keywd2 ;
          int TablIndx ;
          int TablElem ;
          int nTabl ;
          int HardwareCount ;
          switch (count3)
          {
            case PSTable:
              Keywd2 = PSStr ;
              /*					TablIndx = PSIndex ; */
              HardwareVec = PSVec ;
              LenHardwareVec = LenPSVec ;
              nTabl = nPS ;
              TablElem = PSElem ;
              Dictionary = PSPar ;
              break ;
            case GirderTable :
              Keywd2 = GirderStr ;
              /*					TablIndx = GirderIndex ; */
              HardwareVec = GirderVec ;
              LenHardwareVec = LenGirderVec ;
              nTabl = nGirder ;
              TablElem = GirderElem ;
              Dictionary = GirderPar ;
              break ;
            case KlystronTable:
              Keywd2 = KlysStr ;
              /*					TablIndx = KlysIndex ; */
              HardwareVec = KlysVec ;
              LenHardwareVec = LenKlysVec ;
              nTabl = nKlys ;
              TablElem = KlysElem ;
              Dictionary = KlystronPar ;
              break ;
          }
          for (HardwareCount=0 ; HardwareCount<LenHardwareVec ; HardwareCount++)
          {
            TablIndx = *(HardwareVec+HardwareCount) ;
            if (TablIndx != 0)
            {
              if (nTabl < TablIndx)
              {
                sprintf( message, NoSuch, ErrStr, Keywd, count1+1,
                        Keywd2, TablIndx ) ;
                AddMessage( message, 0 ) ;
              }
              else
              {
                switch (count3)
                {
                  case PSTable:
                    retval = GetPSNumericPar( TablIndx-1,
                            Dictionary[TablElem].name, &retlen ) ;
                    break ;
                  case GirderTable:
                    retval = GetGirderNumericPar( TablIndx-1,
                            Dictionary[TablElem].name, &retlen ) ;
                    break ;
                  case KlystronTable:
                    retval = GetKlystronNumericPar( TablIndx-1,
                            Dictionary[TablElem].name, &retlen ) ;
                    break ;
                }
                stat = ElemIndexLookup( count1+1, retval, retlen ) ;
                if (stat == -1)
                {
                  sprintf( message, Inconsis, WarnStr, Keywd, count1+1,
                          Keywd2, TablIndx ) ;
                  AddMessage( message, 0 ) ;
                }
              }
            }
          }
        }
        
        /* finally examine the tracking flags and make sure they are kosher */
        
        if (AllowedTrackFlag != NULL)
        {
          ActualTrackFlag = GetTrackingFlags( count1 ) ;
          if (ActualTrackFlag[NUM_TRACK_FLAGS] != 1)
          {
            sprintf( message, "Warning:  Element %d TrackFlag %s corrupted",
                    count1+1, TrackFlagNames[count2] ) ;
            AddMessage( message, 0 ) ;
            continue ;
          }
          for (count2=0 ; count2 < NUM_TRACK_FLAGS ; count2++)
          {
            if ( (AllowedTrackFlag[count2]==0) &&
                    (TrackFlagSet[count2]>0)         )
            {
              sprintf( message, "Warning:  Element %d TrackFlag %s is not valid",
                      count1+1, TrackFlagNames[count2] ) ;
              AddMessage( message, 0 ) ;
            }
            else if ( (AllowedTrackFlag[count2]==1) &&
                    (TrackFlagSet[count2]==0)         )
            {
              sprintf( message, "Warning:  Element %d TrackFlag %s is not present",
                      count1+1, TrackFlagNames[count2] ) ;
              AddMessage( message, 0 ) ;
            }
            else if (TrackFlagSet[count2] > 1)
            {
              sprintf( message, "Warning:  Element %d TrackFlag %s set more than once",
                      count1+1, TrackFlagNames[count2] ) ;
              AddMessage( message, 0 ) ;
            }
            else if ( (ActualTrackFlag[count2] > TrackFlagMaxValue[count2])
            ||
                    (ActualTrackFlag[count2] < TrackFlagMinValue[count2]) )
            {
              sprintf( message, "Warning:  Element %d TrackFlag %s value out of bounds",
                      count1+1, TrackFlagNames[count2] ) ;
              AddMessage( message, 0 ) ;
            }
          }
          if ( (AllowedTrackFlag[Aper]==1) &&
                  (ActualTrackFlag[Aper]>0)   &&
                  (aperindex > 0)                )
          {
            if (GetDBValue(Dictionary+aperindex) == 0.)
            {
              sprintf( message, "Error:  Element %d TrackFlag %s set but aperture == 0",
                      count1+1, TrackFlagNames[Aper] ) ;
              AddMessage( message, 0 ) ;
            }
            if ( (Dictionary[aperindex].Length==2)        &&
                    (Dictionary[aperindex].ValuePtr[1] == 0)    )
            {
              sprintf( message, "Error:  Element %d TrackFlag %s set but aperture == 0",
                      count1+1, TrackFlagNames[Aper] ) ;
              AddMessage( message, 0 ) ;
            }
          }
          if (    (ActualTrackFlag[SynRad] > SR_None)    &&
                  ( (strcmp(ElemClass,"MULT")) ||
                  (strcmp(ElemClass,"XCOR")) ||
                  (strcmp(ElemClass,"YCOR"))           )     )
          {
            double Lrad = GetDBValue(Dictionary+LradIndex) ;
            if (Lrad<=0)
              Lrad = GetDBValue(Dictionary+LIndex) ;
            if (Lrad<=0)
            {
              sprintf( message,
                      "Error: Element %d TrackFlag %s enabled but L/Lrad <= 0",
                      count1+1, TrackFlagNames[SynRad] ) ;
              AddMessage( message, 0 ) ;
            }
          }
          
        }
        
      } /* end of Element special verification */
      
      
    } /* end of loop over tables */
  }
  
  return ;
  
}

/*=====================================================================*/

/* Get the parameters for a given element, klystron, PS or girder out
 * of the data structures, for purposes of the verification algorithm.
 * This function is similar to GetDatabaseParameters, but sufficiently
 * different that a separate function was mandated.
 *
 * RET:    +1 if all required parameters are present, and
 * all parameters with a required length tolerance
 * are within that tolerance
 * -1 if optional parameters are missing, or parameters for which
 * the length tolerance is optional are not within tolerance
 * 0 if required parameters are missing, or parameters with a
 * required length tolerance are not within tolerance.
 * In addition, the slots in paramstat and paramlenstat are
 * filled to indicate to the calling routine exactly which parameters
 * are troubled and which are not.
 * ABORT:  Never.
 * FAIL:   Never. */

int VerifyParameters( int elemno, int nParam,
        struct LucretiaParameter Dictionary[],
        int WhichTable,
        int paramstat[], int paramlenstat[] )
{
  
  int count ;
  int errors = 0 ;
  int warnings = 0 ;
  int LengthOK ;
  
  /* loop over parameters */
  
  for (count=0 ; count < nParam ; count++)
  {
    
    /* figure out which table to interrogate, and go to it! */
    
    paramlenstat[count] = 1 ;
    switch( WhichTable )
    {
      case ElementTable:
        Dictionary[count].ValuePtr =
                GetElemNumericPar( elemno,
                Dictionary[count].name,
                &(Dictionary[count].Length) ) ;
        break ;
      case PSTable:
        Dictionary[count].ValuePtr =
                GetPSNumericPar( elemno,
                Dictionary[count].name,
                &(Dictionary[count].Length) ) ;
        break ;
      case GirderTable:			Dictionary[count].ValuePtr =
              GetGirderNumericPar( elemno,
              Dictionary[count].name,
              &(Dictionary[count].Length) ) ;
      break ;
      case KlystronTable:
        Dictionary[count].ValuePtr =
                GetKlystronNumericPar( elemno,
                Dictionary[count].name,
                &(Dictionary[count].Length) ) ;
        break ;
      case TLRTable:
        Dictionary[count].ValuePtr =
                GetTLRNumericPar( elemno,
                Dictionary[count].name,
                &(Dictionary[count].Length) ) ;
        break ;
    }
    
    /* handle a missing parameter */
    
    if (Dictionary[count].ValuePtr == NULL)
    {
      if ( (Dictionary[count].Requirement[RmatPars]==Required) ||
              (Dictionary[count].Requirement[TrackPars]==Required)   )
      {
        errors++ ;
        paramstat[count] = 0 ;
      }
      else
      {
        warnings++ ;
        paramstat[count] = -1 ;
      }
    }
    
    /* if the parameter is not missing, handle any requirements on its length */
    
    else
    {
      paramstat[count] = 1 ;
      if ( (Dictionary[count].LengthRequirement[RmatPars] != Ignored) ||
              (Dictionary[count].LengthRequirement[TrackPars] != Ignored)   )
      {
        LengthOK = ( (Dictionary[count].Length >= Dictionary[count].MinLength) &&
                (Dictionary[count].Length <= Dictionary[count].MaxLength)    ) ;
        if ( !LengthOK )
        {
          if ( (Dictionary[count].LengthRequirement[RmatPars]==Required) ||
                  (Dictionary[count].LengthRequirement[TrackPars]==Required)   )
          {
            errors++ ;
            paramlenstat[count] = 0 ;
          }
          else
          {
            warnings++ ;
            paramlenstat[count] = -1 ;
          }
        }
      }
    }
  }
  
  if (errors != 0)
    return 0 ;
  else if (warnings != 0)
    return -1 ;
  else
    return 1 ;
  
}

/*=====================================================================*/

/* look up an element's index number in a vector of element numbers */

/* RET:    The position of the element index in the vector, or -1 if
 * not found.
 * ABORT:  never.
 * FAIL:   if veclen > the actual length of vector */

int ElemIndexLookup( int ElemIndex, double* vector, int veclen )
{
  int i ;
  int position = -1 ;
  
  if (vector==NULL)
    goto egress ;
  if (veclen==0)
    goto egress ;
  
  for (i=0 ; i<veclen ; i++)
  {
    if (ElemIndex==(int)vector[i])
    {
      position = i ;
      break ;
    }
  }
  
  egress:
    
    return position ;
    
}

/*=====================================================================*/

/* Get pointers to real data for frequency-mode LRWFs and put into a
 * convenient array for later use. */

/* RET:    A pointer to the array which is allocated and filled.  As
 * a side effect, a status integer is set to 1 if successful,
 * 0 if unsuccessful (dynamic allocation problem and/or
 * corrupted data in the Matlab tables).
 * ABORT:  never.
 * FAIL:   never. */

struct LRWFFreqData* UnpackLRWFFreqData( int nWakes, int WhichTable,
        int* maxmodes, int* Status )
{
  struct LRWFFreqData* backbone ;
  int WakeLoop ;
  int WakeClass ;
  char Message[90] ;
  
  /* assume failure for now */
  
  *Status = 0 ;
  
  /* allocate the return data structure (or try to) */
  backbone = (struct LRWFFreqData*)calloc(nWakes,sizeof(struct LRWFFreqData)) ;
  if (backbone == NULL)
  {
    AddMessage("Unable to allocate memory for LRWF data",0) ;
    *maxmodes = 0 ;
    goto egress ;
  }
  
  /* loop over the desired wakes and find out whether they are frequency-
   * or time-domain */
  
  for (WakeLoop = 0 ; WakeLoop < nWakes ; WakeLoop++)
  {
    if ( WhichTable == TLRTable )
      WakeClass = GetTLRWakeClass( WakeLoop ) ;
    else
      WakeClass = GetTLRErrWakeClass( WakeLoop ) ;
    if (WakeClass == TIMEDOMAIN)
      continue ;
    if (WakeClass == UNKNOWNDOMAIN)
    {
      if (WhichTable == TLRTable)
        sprintf(Message,
                "Frequency domain TLR # %d corrupted",WakeLoop+1) ;
      else
        sprintf(Message,
                "Frequency domain error TLR # %d corrupted",WakeLoop+1) ;
      AddMessage(Message,0) ;
      goto egress ;
    }
    
    /* if frequency domain, get its parameters */
    
    *Status = GetDatabaseParameters( WakeLoop, nLRWFFreqPar,
            LRWFFreqPar, TrackPars, WhichTable ) ;
    if (*Status != 1)
    {
      if (WhichTable == TLRTable)
        sprintf(Message,
                "Frequency domain TLR # %d corrupted",WakeLoop+1) ;
      else
        sprintf(Message,
                "Frequency domain error TLR # %d corrupted",WakeLoop+1) ;
      AddMessage(Message,0) ;
      goto egress ;
    }
    
    /* Make sure that the vector parameters are of correct length:
     * For regular wakes the Freq and Q vectors should be 2x as long as
     * the K and Tilt vectors;
     * For error wakes the Freq, Q, and K vectors should be 2x as long as
     * the Tilt vector;
     * in all cases the Tilt vector length gives the number of modes */
    
    if ( WhichTable == TLRTable )
    {
      if (
              (LRWFFreqPar[LRWFFreqFreq].Length !=
              LRWFFreqPar[LRWFFreqQ].Length     ) ||
              (LRWFFreqPar[LRWFFreqFreq].Length !=
              2*LRWFFreqPar[LRWFFreqK].Length     ) ||
              (LRWFFreqPar[LRWFFreqFreq].Length !=
              2*LRWFFreqPar[LRWFFreqTilt].Length  )    )
      {
        sprintf(Message,
                "Frequency domain TLR # %d corrupted",WakeLoop+1) ;
        AddMessage(Message,0) ;
        goto egress ;
      }
    }
    else
    {
      if (
              (LRWFFreqPar[LRWFFreqFreq].Length !=
              LRWFFreqPar[LRWFFreqQ].Length       ) ||
              (LRWFFreqPar[LRWFFreqFreq].Length !=
              LRWFFreqPar[LRWFFreqK].Length       ) ||
              (LRWFFreqPar[LRWFFreqFreq].Length !=
              2*LRWFFreqPar[LRWFFreqTilt].Length  )    )
      {
        sprintf(Message,
                "Frequency domain error TLR # %d corrupted",WakeLoop+1) ;
        AddMessage(Message,0) ;
        goto egress ;
      }
    }
    
    
    /* fill up the backbone data structure */
    
    backbone[WakeLoop].Freq  = LRWFFreqPar[LRWFFreqFreq].ValuePtr ;
    backbone[WakeLoop].Q     = LRWFFreqPar[LRWFFreqQ].ValuePtr ;
    backbone[WakeLoop].K     = LRWFFreqPar[LRWFFreqK].ValuePtr ;
    backbone[WakeLoop].Tilt  = LRWFFreqPar[LRWFFreqTilt].ValuePtr ;
    backbone[WakeLoop].BinWidth = *(LRWFFreqPar[LRWFFreqBinWidth].ValuePtr) ;
    backbone[WakeLoop].nModes = LRWFFreqPar[LRWFFreqTilt].Length ;
    if (backbone[WakeLoop].nModes > *maxmodes)
      *maxmodes = backbone[WakeLoop].nModes ;
    
  }
  
  /* set status and return */
  
  *Status = 1 ;
  
  egress:
    
    if (*Status == 1)
      return backbone ;
    else
      return NULL ;
    
}

/*=====================================================================*/

/* function to compute the frequency damping factor, exp(-wt/2Q), for
 * all the modes in a selected wakefield, based on the intra-bunch
 * spacing.  Since there can be different frequencies for the x and the
 * y modes, the calculation is made to get a damping factor for each
 * mode. */

/* RET:    a pointer to a vector of complex values, where the x damping
 * factor is in the Real component and the y damping factor is
 * in the Imag component.  Returns NULL if calculation was not
 * successful.
 * ABORT:  never.
 * FAIL:   never. */

struct LucretiaComplex* ComputeLRWFFreqDamping( double dt,
        double* Freq, double* Q, int nModes )
{
  struct LucretiaComplex* damping ;
  double wx,wy ;
  int count ;
  
  /* start by allocating a vector of LucretiaComplex to hold the results */
  damping = (struct LucretiaComplex*)calloc( nModes, sizeof(struct LucretiaComplex) ) ;
  if (damping == NULL)
  {
    AddMessage("Allocation failure in ComputeLRWFFreqDamping",0) ;
    goto egress ;
  }
  
  /* loop over modes and perform the calculations */
  
  for (count=0 ; count < nModes ; count++)
  {
    wx = 2*PI*( Freq[2*count]  ) * 1e6 ;
    wy = 2*PI*( Freq[2*count+1] ) * 1e6 ;
    damping[count].Real = exp(-wx*dt/2/Q[2*count]) ;
    damping[count].Imag = exp(-wy*dt/2/Q[2*count+1]) ;
  }
  egress:
    
    return damping ;
    
}

/*=====================================================================*/

/* function to return the bunch-to-bunch complex phase advance vector
 * for all the modes in a frequency-domain long-range wakefield */

struct LucretiaComplex* ComputeLRWFFreqPhase( double dt,
        double* Freq, int flag, int nModes )
{
  struct LucretiaComplex* phase ;
  double w ;
  int count ;
  
  /* start by allocating a vector of LucretiaComplex to hold the results */
  phase = (struct LucretiaComplex*)calloc( nModes, sizeof(struct LucretiaComplex) ) ;
  if (phase == NULL)
  {
    AddMessage("Allocation failure in ComputeLRWFFreqPhase",0) ;
    goto egress ;
  }
  
  /* loop over modes and perform the calculations */
  
  for (count=0 ; count < nModes ; count++)
  {
    if (flag==0) /* x frequency */
      w = 2*PI*( Freq[2*count]  ) * 1e6 ;
    else         /* y frequency */
      w = 2*PI*( Freq[2*count+1]  ) * 1e6 ;
    phase[count].Real =  cos(w*dt) ;
    phase[count].Imag = -sin(w*dt) ;
  }
  
  egress:
    
    return phase ;
}

/*=====================================================================*/

/* Master function which performs all necessary preparations for a bunch
 * to interact with short- and long-range wakefields */

/* RET:   +1 if successful, 0 if unsuccessful.
 * Side effects: */
/* ABORT: never.
 * FAIL:  never */

int PrepareAllWF(
        /* pure inputs */
        double* WakeIndices, int NumWakeIndices, int* TrackFlag,
        int elemno, int bunchno,
        /* inputs which are modified during execution */
        struct Beam* TheBeam,
        /* pure outputs */
        int* ZSRno, int* TSRno, int* TLRno,    int* TLRErrno,
        int* TLRClass, int* TLRErrClass,
        struct SRWF** ThisZSR, struct SRWF** ThisTSR,
        struct LRWFFreq** ThisTLRFreq,
        struct LRWFFreq** ThisTLRErrFreq,
        struct LRWFFreqKick** ThisElemTLRFreqKick,
        struct LRWFFreqKick** ThisElemTLRErrFreqKick )
        /* 18 arguments as of 03-mar-2005, a new record for me! -PT */
{
  
  struct Bunch* ThisBunch ;
  int stat = 1 ;
  
  /* initialize the variables for the current wakefields to their "do nothing"
   * state */
  
  *ZSRno    = -1 ;
  *TSRno    = -1 ;
  *TLRno    = -1 ;
  *TLRErrno = -1 ;
  *TLRClass    = UNKNOWNDOMAIN ;
  *TLRErrClass = UNKNOWNDOMAIN ;
  *ThisZSR = NULL ;
  *ThisTSR = NULL ;
  *ThisTLRFreq    = NULL ;
  *ThisTLRErrFreq = NULL ;
  *ThisElemTLRFreqKick    = NULL ;
  *ThisElemTLRErrFreqKick = NULL ;
  
  /* make a shortcut to the current bunch */
  
  ThisBunch = TheBeam->bunches[bunchno] ;
  
  /* start with short-range, longitudinal wakefields */
  
  if ( (NumWakeIndices > 0) && (TrackFlag[SRWF_L] > 0) )
  {
    *ZSRno = (int)(WakeIndices[0]) - 1 ;
    
    /* protect against a ZSRno that exceeds the total number of ZSRs in WF */
    
    if ( (*ZSRno)+1 > numwakes[0] )
    {
      stat = 0 ;
      BadSRWFMessage( *(ZSRno)+1, 0 ) ;
      goto egress ;
    }
    
    /* another special case:  No ZSRs, element points to ZSR # 0, but SRWF_Z flag is ON */
    
    if (*ZSRno > -1)
    {
      stat = ConvolveSRWFWithBeam( ThisBunch, *ZSRno, 0 ) ;
      if (stat == 0)
      {
        goto egress ;
      }
      else if (stat == -1)
      {
        stat = 1 ;
        *ZSRno = -1 ;
      }
      else
        *ThisZSR = ThisBunch->ZSR[*ZSRno] ;
    }
  }
  
  /* now short-range transverse wakes, pretty similar */
  
  if ( (NumWakeIndices > 1) && (TrackFlag[SRWF_T] > 0) )
  {
    *TSRno = (int)(WakeIndices[1]) - 1 ;
    
    /* protect against a ZSRno that exceeds the total number of ZSRs in WF */
    
    if ( (*TSRno)+1 > numwakes[1] )
    {
      stat = 0 ;
      BadSRWFMessage( *(TSRno)+1, 1 ) ;
      goto egress ;
    }
    /* another special case:  No TSRs, element points to TSR # 0, but SRWF_T flag is ON */
    
    if (*TSRno > -1)
    {
      stat = ConvolveSRWFWithBeam( ThisBunch, *TSRno, 1 ) ;
      if (stat == 0)
      {
        goto egress ;
      }
      else if (stat == -1)
      {
        stat = 1 ;
        *TSRno = -1 ;
      }
      else
        *ThisTSR = ThisBunch->TSR[*TSRno] ;
    }
  }
  
  /* for long-range transverse wakes, we also need to unpack the wakefield data
   * from the Matlab structures, figure out whether this is a frequency or time
   * domain wakefield, compute damping for the frequency domain type, and get a
   * pointer to the earlier wakefield kicks at this structure */
  
  if ( (NumWakeIndices > 2)    &&
          (TrackFlag[LRWF_T] > 0) &&
          (WakeIndices[2] > 0)        )
  {
    *TLRno = (int)(WakeIndices[2]) - 1 ;
    if ( (*TLRno > numwakes[2]-1) || (*TLRno < -1) )
    {
      NonExistentLRWFmessage( elemno+1, *TLRno+1, TLRTable ) ;
      stat = 0 ;
      goto egress ;
    }
    if ( TLRFreqData[*TLRno].Freq != NULL )
      *TLRClass = FREQDOMAIN ;
    else
      *TLRClass = TIMEDOMAIN ;
    if (*TLRClass == FREQDOMAIN)
    {
      if (TheBeam->TLRFreqDamping[*TLRno] == NULL)
      {
        TheBeam->TLRFreqDamping[*TLRno] =
                ComputeLRWFFreqDamping(
                TheBeam->interval,
                TLRFreqData[*TLRno].Freq,
                TLRFreqData[*TLRno].Q,
                TLRFreqData[*TLRno].nModes ) ;
        if (TheBeam->TLRFreqDamping[*TLRno] == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
        TheBeam->TLRFreqxPhase[*TLRno] =
                ComputeLRWFFreqPhase(
                TheBeam->interval,
                TLRFreqData[*TLRno].Freq,
                0,
                TLRFreqData[*TLRno].nModes ) ;
        if (TheBeam->TLRFreqxPhase[*TLRno] == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
        TheBeam->TLRFreqyPhase[*TLRno] =
                ComputeLRWFFreqPhase(
                TheBeam->interval,
                TLRFreqData[*TLRno].Freq,
                1,
                TLRFreqData[*TLRno].nModes ) ;
        if (TheBeam->TLRFreqyPhase[*TLRno] == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
      }
      stat = PrepareBunchForLRWFFreq( ThisBunch,
              *TLRno, 0,
              TLRFreqData[*TLRno].Freq,
              TLRFreqData[*TLRno].K,
              TLRFreqData[*TLRno].nModes,
              TLRFreqData[*TLRno].BinWidth ) ;
      if (stat==0)
        goto egress ;
      else if (stat == -1)
      {
        *TLRno = -1 ;
        stat = 1 ;
      }
      else if (stat == 1)
      {
        *ThisTLRFreq = ThisBunch->TLRFreq[*TLRno] ;
        *ThisElemTLRFreqKick =
                GetThisElemTLRFreqKick( elemno, 0 ) ;
        if (*ThisElemTLRFreqKick == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
      }
    }
  }
  
  /* the long-range error wake is very much parallel to the long range dipole
   * wake, so much so that a truly awesome programmer (ie, a better one than I
   * am) would write a separate function that does either TLR or TLRErr depending
   * on its arguments.  I am obviously not that programmer. */
  
  if ( (NumWakeIndices > 3)     &&
          (TrackFlag[LRWF_ERR] > 0) &&
          (WakeIndices[3] > 0)          )
  {
    *TLRErrno = (int)(WakeIndices[3]) - 1 ;
    if ( (*TLRErrno > numwakes[3]-1) || (*TLRErrno < -1) )
    {
      NonExistentLRWFmessage( elemno+1, *TLRErrno+1, TLRErrTable ) ;
      stat = 0 ;
      goto egress ;
    }
    if ( TLRErrFreqData[*TLRErrno].Freq != NULL )
      *TLRErrClass = FREQDOMAIN ;
    else
      *TLRErrClass = TIMEDOMAIN ;
    if (*TLRErrClass == FREQDOMAIN)
    {
      if (TheBeam->TLRErrFreqDamping[*TLRErrno] == NULL)
      {
        TheBeam->TLRErrFreqDamping[*TLRErrno] =
                ComputeLRWFFreqDamping(
                TheBeam->interval,
                TLRErrFreqData[*TLRErrno].Freq,
                TLRErrFreqData[*TLRErrno].Q,
                TLRErrFreqData[*TLRErrno].nModes ) ;
        if (TheBeam->TLRErrFreqDamping[*TLRErrno] == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
        TheBeam->TLRErrFreqxPhase[*TLRErrno] =
                ComputeLRWFFreqPhase(
                TheBeam->interval,
                TLRErrFreqData[*TLRErrno].Freq,
                0,
                TLRErrFreqData[*TLRErrno].nModes ) ;
        if (TheBeam->TLRErrFreqxPhase[*TLRErrno] == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
        TheBeam->TLRErrFreqyPhase[*TLRErrno] =
                ComputeLRWFFreqPhase(
                TheBeam->interval,
                TLRErrFreqData[*TLRErrno].Freq,
                1,
                TLRErrFreqData[*TLRErrno].nModes ) ;
        if (TheBeam->TLRErrFreqyPhase[*TLRErrno] == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
      }
      stat = PrepareBunchForLRWFFreq( ThisBunch,
              *TLRErrno, 1,
              TLRErrFreqData[*TLRErrno].Freq,
              TLRErrFreqData[*TLRErrno].K,
              TLRErrFreqData[*TLRErrno].nModes,
              TLRErrFreqData[*TLRErrno].BinWidth ) ;
      if (stat==0)
        goto egress ;
      else if (stat == -1)
      {
        *TLRErrno = -1 ;
        stat = 1 ;
      }
      else if (stat == 1)
      {
        *ThisTLRErrFreq = ThisBunch->TLRErrFreq[*TLRErrno] ;
        *ThisElemTLRErrFreqKick =
                GetThisElemTLRFreqKick( elemno, 1 ) ;
        if (*ThisElemTLRErrFreqKick == NULL)
        {
          stat = 0 ;
          goto egress ;
        }
      }
    }
  }
  
  egress:
    
    return stat ;
    
}

/*=====================================================================*/

/* Now the Get and Put functions for LRWF frequency domain kick vectors
 * need to see the LRWF kick buffer, so the buffer gets scope above the
 * function level */

struct LRWFFreqKick* TLRFreqBuffer    = NULL ;
struct LRWFFreqKick* TLRErrFreqBuffer = NULL ;

/* now the functions that handle them */

/* function to return a LRWF frequency-domain kick vector for the current
 * element */

struct LRWFFreqKick* GetThisElemTLRFreqKick( int elemno, int flag     )
{
  struct LRWFFreqKick* retval ;
  
  /* do different things for TLR versus TLR error */
  
  if (flag==0) /* TLR */
  {
    retval = TLRFreqKickDat[elemno] ;
    if (retval == NULL)
    {
      retval = TLRFreqBuffer ;
      TLRFreqBuffer = NULL ;
    }
  }
  else         /* TLRErr */
  {
    retval = TLRErrFreqKickDat[elemno] ;
    if (retval == NULL)
    {
      retval = TLRErrFreqBuffer ;
      TLRErrFreqBuffer = NULL ;
    }
  }
  
  /* if the buffer still is null, need to allocate something */
  
  if (retval == NULL)
  {
    retval = (struct LRWFFreqKick *)calloc(1,sizeof(struct LRWFFreqKick)) ;
    retval->LastBunch = -1 ;
    if (retval != NULL)
    {
      retval->xKick = (struct LucretiaComplex*)calloc(maxNmode,sizeof(struct LucretiaComplex)) ;
      retval->yKick = (struct LucretiaComplex*)calloc(maxNmode,sizeof(struct LucretiaComplex)) ;
      if ( (retval->xKick == NULL) || (retval->yKick == NULL) )
      {
        FreeAndNull((void**)&(retval->xKick)) ;
        FreeAndNull((void**)&(retval->yKick)) ;
        FreeAndNull((void**)&retval) ;
      }
    }
  }
  
  return retval ;
  
}

void PutThisElemTLRFreqKick( struct LRWFFreqKick** ThisKick, int elemno,
        int bunchno, int lastbunch, int bunchwise,
        int flag )
{
  struct LRWFFreqKick* TheKick ;
  
  TheKick = *ThisKick ;
  
  /* if there are more bunches coming in this track operation, then we want to
   * save this kick vector on the element backbone.  If there are no more bunches
   * we still want to save it if we are doing bunchwise tracking, since in that
   * case it's likely the next track will also do bunchwise on the same set of
   * elements */
  
  if ( (bunchno+1 < lastbunch) || (bunchwise == 1) )
  {
    if (flag==0)
      TLRFreqKickDat[elemno]    = *ThisKick ;
    else
      TLRErrFreqKickDat[elemno] = *ThisKick ;
  }
  else /* done with elementwise tracking, check this buffer back in.  Note
   * that if we previously did bunchwise tracking, then lots of RF units
   * have a kick vector allocated to them.  The first one we check back
   * in can go "on the hook" of TLRFreqBuffer, but the rest need to be
   * deallocated */
  {
    TheKick->LastBunch = -1 ;
    if (flag==0)
    {
      if (TLRFreqBuffer == NULL)
        TLRFreqBuffer    = *ThisKick ;
      else
      {
        FreeAndNull((void**)&(TheKick->xKick)) ;
        FreeAndNull((void**)&(TheKick->yKick)) ;
        FreeAndNull((void**)&TheKick) ;
      }
      TLRFreqKickDat[elemno] = NULL ;
    }
    else
    {
      if (TLRErrFreqBuffer == NULL)
        TLRErrFreqBuffer = *ThisKick ;
      else
      {
        FreeAndNull((void**)&(TheKick->xKick)) ;
        FreeAndNull((void**)&(TheKick->yKick)) ;
        FreeAndNull((void**)&TheKick) ;
      }
      TLRErrFreqKickDat[elemno] = NULL ;
    }
  }
  *ThisKick = NULL ;
  
  return ;
  
}

/* The function to clear the old LRWF kicks also needs to see the
 * buffers */

void ClearOldLRWFFreqKicks( int flag )
{
  int count ;
  
  for (count=0 ; count<nElemOld ; count++)
  {
    if (flag==0)
    {
      FreeAndNull( (void**)&(TLRFreqKickDat[count]) ) ;
      FreeAndNull( (void**)&(TLRFreqBuffer) ) ;
    }
    else
    {
      FreeAndNull( (void**)&(TLRErrFreqKickDat[count]) ) ;
      FreeAndNull( (void**)&(TLRFreqBuffer) ) ;
    }
  }
  
}

/*=====================================================================*/

/* compute short-range transverse wakefield kicks */

void ComputeTSRKicks( struct SRWF* ThisTSR, double L )
{
  int ibin, jbin;
  double *Kbin ;
  
  
  
  for (ibin=0 ; ibin<ThisTSR->nbin ; ibin++)
  {
    ThisTSR->binVx[ibin] =0. ;
    ThisTSR->binVy[ibin] =0. ;
  }
  for (ibin=0 ; ibin<ThisTSR->nbin ; ibin++)
    for (jbin=ibin+1 ; jbin<ThisTSR->nbin ; jbin++)
    {
      Kbin = &(ThisTSR->K[ThisTSR->nbin*ibin+jbin]) ;
      ThisTSR->binVx[jbin] += (*Kbin) *
              ThisTSR->binx[ibin] ;
      ThisTSR->binVy[jbin] += (*Kbin) *
              ThisTSR->biny[ibin] ;
    }
  for (ibin=0 ; ibin<ThisTSR->nbin ; ibin++)
  {
    ThisTSR->binVx[ibin] *= L ;
    ThisTSR->binVy[ibin] *= L ;
    ThisTSR->binx[ibin] = 0 ;
    ThisTSR->biny[ibin] = 0 ;
  }
  
  return ;
  
}

/*=====================================================================*/

/* accumulate ray position into appropriate WF bins */

void AccumulateWFBinPositions( double* binx, double* biny, int binno,
        double xpos, double ypos, double Q )
{
  
  binx[binno] += Q * xpos ;
  biny[binno] += Q * ypos ;
  
  return ;
  
}

/*=====================================================================*/


/* deal with CSR tracking flags */
double GetCsrTrackFlags( int elemNo, int* TFlag, int* csrSmoothFactor, char* ElemClass, struct Bunch* ThisBunch, double* thisS, double* dL, double rstd )
{
  static double dsL[10000] ;
  static double usL=0 ;
  static int lPointer=0 ;
  static int smoothFactor=3 ;
  static int csrActive=0 ;
  double lDecay,rmsDim,R ;
  double *L, *S, *ANG, *P, *B ;
  int iL ;
  S = GetElemNumericPar(elemNo, "S", NULL) ;
  L = GetElemNumericPar(elemNo, "L", NULL) ;
  
  /* Drift or other downstream of CSR element */
  if (TFlag[CSR]==0 && csrActive>0) {
    if (elemNo == 0) return 0; /* Return if first element in deck*/
    *csrSmoothFactor = smoothFactor ;
    if ( *L == 0 )
      return 0 ;
    while (dsL[lPointer] < *S && dsL[lPointer]!=0 && lPointer<10000 )
      lPointer++ ;
    if ( lPointer >= 10000 || dsL[lPointer] == 0 )
      return 0 ;
    lPointer++;
    *csrSmoothFactor = TFlag[CSR_SmoothFactor] ;
    
    /* if next pointer after element or after CSR consideration point, return 0 else return dl and l*/
    if ( dsL[lPointer]>(*S+*L) ) {
      lPointer--;
      csrActive=0;
      return 0 ;
    }
    else if ( dsL[lPointer] == 0 )
      return 0 ;
    else {
      *thisS = dsL[lPointer-1] ;
      if ( (lPointer-1) == 0 )
        *dL = *thisS - usL ;
      else
        *dL = *thisS - dsL[lPointer-2] ;
      return *thisS - usL ; /* distance to u/s bend face */
    }
  }
  else if (TFlag[CSR]==0)
    return 0 ;
  
  csrActive=0;
  
  *csrSmoothFactor = TFlag[CSR_SmoothFactor] ;
  
  if ( TFlag[Split] == 1 ) /* Things break if ask for split of 1 */
    TFlag[Split] = 2 ;
  
  if ( TFlag[CSR] == 0 ) {
    dsL[0] = 0 ;
    lPointer = 0 ;
    return 0 ;
  }
  
  /* Record Bend downstrem face location and Angle if bend, else use effective bend from integrated field at rms beam radius */
  usL = *S + *L ;
  if (strcmp(ElemClass,"SBEN")==0) {
    ANG = GetElemNumericPar(elemNo, "Angle", NULL) ;
    R=*L / sin(fabs(*ANG)) ;
  }
  else if (strcmp(ElemClass,"PLENS")==0) {
    P = GetElemNumericPar(elemNo, "P", NULL) ;
    B = GetElemNumericPar(elemNo, "B", NULL) ;
    R= *L / ( ( *B * rstd ) / (*L * *P * GEV2TM) );
  }
  
  /* Calculate downstream splits */
  if ( TFlag[CSR_DriftSplit] > 0 ) {
    rmsDim = GetRMSCoord( ThisBunch, 4 ) ;
    lDecay=3.0*pow(24*rmsDim*pow(fabs(R),2),0.333333333);
    for ( iL=0; iL<10000; iL++ ) {
      if ( iL < TFlag[CSR_DriftSplit]-1 )
        dsL[iL] = usL + pow(10,-3.0+iL*3.0/(TFlag[CSR_DriftSplit]-1)) * lDecay ;
      else if ( iL == TFlag[CSR_DriftSplit]-1 )
        dsL[iL] = usL + lDecay ;
      else
        dsL[iL] = 0 ;
    }
  }
  else
    dsL[0] = 0 ;
  lPointer = 0 ;
  
  thisS = NULL ;
  
  csrActive=1;
  return 0 ;
}

/* Kernel to setup CUDA random number generator */
#ifdef __CUDACC__
__global__ void rngSetup_kernel(curandState *state, unsigned long long rSeed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence
   * number, no offset */
  curand_init(rSeed, id, 0, &state[id]);
}
#endif

/* Actions to perform post tracking through an element */
/* RET:    None.
 * ABORT:  never.
 * FAIL:   never. */
void postEleTrack( struct Beam* TheBeam, int* bunchno, int* elemno, double L, double S, int* TrackFlag )
{
 struct Bunch* ThisBunch = TheBeam->bunches[*bunchno] ;
 /* Exchange post-track rays (y) with pre-tracked rays (x) for next track operation */
 mxArray* pForceProcess = GetExtProcessData(elemno,"ForceProcess") ;
 if (pForceProcess==NULL || !*(bool*)mxGetLogicals( pForceProcess )) /* don't exchange if forcing ExtProcess (Lucretia tracking not happened) */
  XYExchange( ThisBunch ) ;
 #ifndef __CUDACC__
 /* Perform any external processes (e.g. GEANT4 tracking) */
 #ifdef LUCRETIA_G4TRACK
 ExtProcess( elemno, TheBeam, bunchno, L, &S, TrackFlag) ;
 #endif
 #endif
 return ;
}

/* Interface function to external (GEANT4) process routines */
#ifdef LUCRETIA_G4TRACK
void ExtProcess(int* elemno, struct Beam* TheBeam, int* bunchno, double L, double* S, int* TrackFlag)
{
  /* If there are stopped particles in this bunch that correspond to this beamline element
     and there is an associated extProcess object attached, then pass bunch over to
     GEANT4 tracking to go through the requested material */
  int stoppedParticles=0 ;
  int ray, stat ;
  int len, numRaysResumed ;
  double Xfrms[6][2] ;
  double* ElemOffset ;
  double GirdNo ;
  mxArray* pMaterial;
  double *x, *px, *y, *py, *z, *p0 ;
  struct Bunch* ThisBunch = TheBeam->bunches[*bunchno] ;
  /* First check to see if there is ExtProcess stuff defined for this element */
  pMaterial = GetExtProcessData(elemno, "Material") ;
  if (pMaterial == NULL)
    return ;
  /* Look to see if any particles flagged as stopped for this element -> signal to send to ExtProcess */
  for (ray=0; ray<ThisBunch->nray; ray++) {
    if (ThisBunch->stop[ray] == *elemno+1) {
      stoppedParticles=1;
      break ;
    }
  }
  if (stoppedParticles==0)
    return ;
  if (L == 0) /* If L passed as 0, need to look up from BEAMLINE in case of split tracking */
    L = *GetElemNumericPar( *elemno, "L", NULL ) ;
  
  if (L == 0) /* If L is still 0 then no point passing to GEANT */
      return ;
  /* make ray coordinates into local ones */
  GirdNo = (double) GetElemGirder( *elemno )  ;
  ElemOffset = GetElemNumericPar( *elemno, "Offset", &len ) ;
  if (len != 6)
    ElemOffset = NULL ;
  stat = GetTotalOffsetXfrms( &GirdNo , &L, S, ElemOffset, Xfrms ) ;
  if (stat == 1) {
    for (ray=0 ;ray<ThisBunch->nray ; ray++)
    {
      /* if the ray was previously stopped, ignore it  */
      if (ThisBunch->stop[ray] != *elemno+1)
        continue ;
      GetLocalCoordPtrs(ThisBunch->x, ray*6, &x,&px,&y,&py,&z,&p0) ;
      ApplyTotalXfrm( Xfrms, UPSTREAM, TrackFlag, 0 ,x,px,y,py,z,p0) ;
    }
  }
  /* Perform required GEANT4 tracking */
  numRaysResumed = g4track(elemno, bunchno, TheBeam, &L, ElemOffset) ;
  if (numRaysResumed<0) { // error condition
    printf("Error running GEANT4 interface... skipping for ele: %d\n",*elemno+1) ;
    return ;
  }
  /* adjust stopped particles counter to reflected those re-entered */
  ThisBunch->NGOODRAY += numRaysResumed ;
  
  /* Transform back to accelerator coordinates */
  if (stat == 1) {
    for (ray=0 ;ray<ThisBunch->nray ; ray++)
    {
      /* if ray in stopped state, ignore it - if it came out of G4 tracking stopped then last co-ordinates are where it stopped in volume */
      if (ThisBunch->stop[ray] != 0)
        continue ;
      GetLocalCoordPtrs(ThisBunch->x, ray*6, &x,&px,&y,&py,&z,&p0) ;
      ApplyTotalXfrm( Xfrms, DOWNSTREAM, TrackFlag, 0,x,px,y,py,z,p0 ) ;
    }
  }
}
#endif

/* Element tracker - with standard actions associated with application of coherent processes
   and/or user-defined splitting of element */
#ifdef __CUDACC__        
int ElemTracker(char* ElemClass,int* ElemLoop,int* BunchLoop,int* TFlag_gpu, int* TFlag,struct TrackArgsStruc* TrackArgs)
#else
int ElemTracker(char* ElemClass,int* ElemLoop,int* BunchLoop, int* TFlag, struct TrackArgsStruc* TrackArgs)
#endif                
{
  int nsplit=0, TrackStatus=0, csrSmoothFactor=0, firstSplit=1, skipCSRLSC=0, ray ;
  double splitDL=0, L, lastS, thisS=0, docsrDrift, S_csr=0, S_other=0, LSC_drift=0, S_lsc=0, S_last=0, csrDL, lastS_lsc ;
  double xmean0=0 ; /* Initial mean x position at last bend entrance */
  double xstd[2] = {0,0} ; /* Initial rms x/y size at entrance to last non-bend magnet with CSR flag active */
  double rstd = 0 ; /* radius calculated from above */
  mxArray *pForceProcess;
  /* Get initial S location and initialize lastS reference, and get total unsplit element length */
  lastS = *GetElemNumericPar( *ElemLoop, "S", NULL ) ;
  lastS_lsc = lastS ;
  L = *GetElemNumericPar(*ElemLoop, "L", NULL) ;
  S_last = lastS+L ;

  /* Get plens related csr data */
  if ( strcmp(ElemClass,"PLENS")==0 && abs(TFlag[CSR]) > 0 ) { /* calulate mean r at entrance to quad with CSR flag set to later get ave radiation profile */
    xstd[0] = GetRMSCoord( TrackArgs->TheBeam->bunches[*BunchLoop], 0 ) ;
    xstd[1] = GetRMSCoord( TrackArgs->TheBeam->bunches[*BunchLoop], 2 ) ;
    rstd = sqrt( xstd[0]*xstd[0] + xstd[1]*xstd[1] ) ;
  }
  
  /* Has the CSR process requested this element be split? */
  /* Returns distance to upstream SBEN (docsrDrift) [0 if nothing to do here], S_csr is S location of CSR application, csrDL is segment lenth */
  docsrDrift = GetCsrTrackFlags( *ElemLoop, TFlag, &csrSmoothFactor, ElemClass, TrackArgs->TheBeam->bunches[*BunchLoop], &S_csr, &csrDL, rstd ) ;
  /* Get list of user and/or LSC requested split trackpoints, take the more demanding */
  if ( L > 0 ) {
    if ( TFlag[LSC] > 0 ) { /* Get required LSC drift length if this Track Flag active */
      LSC_drift = ProcLSC(TrackArgs->TheBeam->bunches[*BunchLoop],*ElemLoop,0,*BunchLoop * TFlag[LSC_storeData]) ;
      S_lsc = lastS + LSC_drift ;
    }
    if ( TFlag[Split] > 0 ) { /* User requested element splitting */
      splitDL = L / TFlag[Split] ;
      nsplit = TFlag[Split];
      S_other=lastS+splitDL ;
    }
    if (nsplit>1 && TFlag[LSC]>0 ) { /* Setting Split TrackFlag overrides LSC auto splitting */
      S_lsc=0;
      LSC_drift=0;
    }
    if (nsplit==1) { /* Just pass to regular unsplit tracking */
      S_other=0;
      nsplit=0;
    }
    if (S_other!=0 && S_lsc!=0)
      printf("Something wrong in TrackThru:ElemTracker (Split and LSC both getting set!)\n");
    S_other=S_other+S_lsc ;
    splitDL=splitDL+LSC_drift;
  }
  /* If ExtProcess present and wanting to force tracking to go to ExtProcess, label particles stopped for this element*/
  pForceProcess = GetExtProcessData(ElemLoop,"ForceProcess") ;
  if (pForceProcess!=NULL && *(bool*)mxGetLogicals( pForceProcess )) {
    for (ray=0; ray< TrackArgs->TheBeam->bunches[*BunchLoop]->nray; ray++) {
      if (TrackArgs->TheBeam->bunches[*BunchLoop]->stop[ray] == 0) {
        TrackArgs->TheBeam->bunches[*BunchLoop]->stop[ray] = *ElemLoop+1;
      }
    }
  }
  /* Track bunch through this element with the requested split points, if LSC/User and CSR requested, take
     the larger number of splits from LSC/User and merge with CSR requests */
  if ( (docsrDrift > 0 && S_csr<=S_last) || (S_other>0 && S_other<=S_last) ) {
    while (lastS<S_last) {
      /* Set next tracking point */
      if ( docsrDrift>0 ) {
        if ( S_other == 0 || S_csr < S_other )
          thisS=S_csr ;
        else
          thisS=S_other ;
      }
      else
        thisS=S_other;
      if (thisS>=S_last || thisS==0) {
        skipCSRLSC=1;
        thisS=S_last;
      }
      if (thisS==lastS || fabs(thisS-lastS)<1e-6)
        break;
      /* Execute intergrater through this element with desired length */
      if (pForceProcess!=NULL && *(bool*)mxGetLogicals( pForceProcess ))
	      TrackStatus=1;
      else if (strcmp(ElemClass,"QUAD")==0)
        TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 2, thisS-lastS, lastS, TFlag[Aper] ) ;
      else if (strcmp(ElemClass,"SEXT")==0)
        TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 3, thisS-lastS, lastS, TFlag[Aper] ) ;
      else if (strcmp(ElemClass,"OCTU")==0)
        TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 4, thisS-lastS, lastS, TFlag[Aper] ) ;
      else if (strcmp(ElemClass,"PLENS")==0) 
        TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 5, thisS-lastS, lastS, TFlag[Aper] ) ;
      else if (strcmp(ElemClass,"SOLENOID")==0)
        TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0, thisS-lastS, lastS, TFlag[Aper] ) ;
      else if (strcmp(ElemClass,"MULT")==0)
        TrackStatus = TrackBunchThruMult( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, thisS-lastS, lastS ) ;
      else if (strcmp(ElemClass,"SBEN")==0) {
        if ( firstSplit == 1 ) { /* Track with just upstream edge effects */
          /* Get mean x position at entrance of bend - used by 2D CSR calculation */
          xmean0 = GetMeanCoord( TrackArgs->TheBeam->bunches[*BunchLoop], 0 ) ;
          TrackStatus = TrackBunchThruSBend( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 1, 0, thisS-lastS, lastS ) ;
        }
        else if ( nsplit > 1 ) /* Track with no edge effects */
          TrackStatus = TrackBunchThruSBend( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0, 0, thisS-lastS, lastS  ) ;
        else /* Track with just dowstream edge effects */
          TrackStatus = TrackBunchThruSBend( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0, 1, thisS-lastS, lastS  ) ;
      }
      else if ( strcmp(ElemClass,"HMON")==0 || strcmp(ElemClass,"VMON")==0 || strcmp(ElemClass,"MONI")==0 )
        TrackStatus = TrackBunchThruBPM( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, thisS-lastS ) ;
      else if ( strcmp(ElemClass,"PROF")==0 || strcmp(ElemClass,"WIRE")==0 || strcmp(ElemClass,"BLMO")==0 ||
                strcmp(ElemClass,"SLMO")==0 || strcmp(ElemClass,"IMON")==0 || strcmp(ElemClass,"INST")==0 )
        TrackStatus = TrackBunchThruInst( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, thisS-lastS ) ;
      else if ( strcmp(ElemClass,"XCOR")==0 )
        TrackStatus = TrackBunchThruCorrector( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, XCOR, thisS-lastS, lastS ) ;
      else if ( strcmp(ElemClass,"YCOR")==0 )
        TrackStatus = TrackBunchThruCorrector( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, YCOR, thisS-lastS, lastS ) ;
      else if ( strcmp(ElemClass,"XYCOR")==0 )
        TrackStatus = TrackBunchThruCorrector( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, XYCOR, thisS-lastS, lastS ) ;
      else if ( strcmp(ElemClass,"COLL")==0 )
        TrackStatus = TrackBunchThruCollimator( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, thisS-lastS, lastS ) ;
      else /* default = drift */
        TrackStatus = TrackBunchThruDrift( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, thisS-lastS ) ;
      /* Signal first pass through splitting iterator */
      firstSplit = 0 ;
      
      /* Return if tracker flags an error status */
      if (TrackStatus == 0)
        return TrackStatus ;

      /* Perform post-tracking actions */
      postEleTrack( TrackArgs->TheBeam, BunchLoop, ElemLoop, thisS-lastS, lastS, TFlag) ;
      
      /* Apply CSR if requested (apply if this is a "CSR split" or it shares an S location with an LSC one) 
         - then get pointer to next CSR S location to track to if there is one */
      if ( ( docsrDrift != 0 && ( S_other==0 || S_csr<=S_other ) ) || ( strcmp(ElemClass,"SBEN")==0 && abs(TFlag[CSR]) > 0 ) ) {
        GetCsrEloss(TrackArgs->TheBeam->bunches[*BunchLoop], TFlag[CSR], csrSmoothFactor, *ElemLoop, docsrDrift, csrDL, xmean0 ) ;
        if ( docsrDrift > 0 )
          docsrDrift = GetCsrTrackFlags( *ElemLoop, TFlag, &csrSmoothFactor, ElemClass, TrackArgs->TheBeam->bunches[*BunchLoop], &S_csr, &csrDL, rstd ) ;
      }

      /* If no other physics to apply until next element then break out of tracking this element now*/
      if (skipCSRLSC==1)
        break;
      
      /* Increment S pointer */
      lastS = thisS ;
      
      /* Apply LSC if requested (apply if this is not a "CSR split" or it shares an S location with one) 
         - if just splitting the element manually then just increment S pointer */
      if ( S_other > 0 && ( docsrDrift <= 0 || S_other<=S_csr) ) {
        if ( TFlag[LSC] ) {
          ProcLSC(TrackArgs->TheBeam->bunches[*BunchLoop],*ElemLoop,thisS-lastS_lsc, (*BunchLoop+1) * TFlag[LSC_storeData]) ;
          lastS_lsc=thisS;
        }
        nsplit-- ;
        if (S_other==S_last || ( TFlag[Split] > 0 && nsplit<=0 ) )
          S_other=0;
        else {
          S_other = lastS + splitDL ;
          if ( S_other > S_last )
            S_other = S_last ;
        }
      }
    }
    /* Post split iterator actions for elements that require them */
    if ( (strcmp(ElemClass,"HMON")==0) || (strcmp(ElemClass,"VMON")==0) || (strcmp(ElemClass,"MONI")==0) )
      TrackStatus = TrackBunchThruBPM( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, -1 ) ;
    else if ( strcmp(ElemClass,"PROF")==0 || strcmp(ElemClass,"WIRE")==0 || strcmp(ElemClass,"BLMO")==0 ||
                strcmp(ElemClass,"SLMO")==0 || strcmp(ElemClass,"IMON")==0 || strcmp(ElemClass,"INST")==0 )
      TrackStatus = TrackBunchThruInst( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, -1 ) ;
  }
  else { /* Just normal, unsplit tracking */
    /* Execute integrator through this element through BEAMLINE element length */
    if (pForceProcess!=NULL && *(bool*)mxGetLogicals( pForceProcess ))
      TrackStatus=1;
    else if (strcmp(ElemClass,"QUAD")==0) {
      TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 2, 0, 0, TFlag[Aper] ) ;
    }
    else if (strcmp(ElemClass,"SEXT")==0)
      TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 3, 0, 0, TFlag[Aper] ) ;
    else if (strcmp(ElemClass,"OCTU")==0)
      TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 4, 0, 0, TFlag[Aper] ) ;
    else if (strcmp(ElemClass,"PLENS")==0)
      TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 5, 0, 0, TFlag[Aper] ) ;
    else if (strcmp(ElemClass,"SOLENOID")==0)
      TrackStatus = TrackBunchThruQSOS( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0, 0, 0, TFlag[Aper] ) ;
    else if (strcmp(ElemClass,"MULT")==0)
      TrackStatus = TrackBunchThruMult( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0, 0 ) ;
    else if (strcmp(ElemClass,"SBEN")==0)
      TrackStatus = TrackBunchThruSBend( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 1, 1, L, lastS  ) ;
    else if ( (strcmp(ElemClass,"HMON")==0) || (strcmp(ElemClass,"VMON")==0) || (strcmp(ElemClass,"MONI")==0) )
      TrackStatus = TrackBunchThruBPM( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0 ) ;
    else if ( strcmp(ElemClass,"PROF")==0 || strcmp(ElemClass,"WIRE")==0 || strcmp(ElemClass,"BLMO")==0 ||
                strcmp(ElemClass,"SLMO")==0 || strcmp(ElemClass,"IMON")==0 || strcmp(ElemClass,"INST")==0 )
      TrackStatus = TrackBunchThruInst( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0 ) ;
    else if ( strcmp(ElemClass,"XCOR")==0 )
      TrackStatus = TrackBunchThruCorrector( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, XCOR, 0, 0 ) ;
    else if ( strcmp(ElemClass,"YCOR")==0 )
      TrackStatus = TrackBunchThruCorrector( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, YCOR, 0, 0 ) ;
    else if ( strcmp(ElemClass,"XYCOR")==0 )
      TrackStatus = TrackBunchThruCorrector( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, XYCOR, 0, 0 ) ;
    else if ( strcmp(ElemClass,"COLL")==0 )
      TrackStatus = TrackBunchThruCollimator( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0, 0 ) ;
    else /* default = drift */
      TrackStatus = TrackBunchThruDrift( *ElemLoop, *BunchLoop, TrackArgs, TFLAG, 0 ) ;
    /* Process any errors */
    if (TrackStatus == 0)
      return TrackStatus ;
    /* Perform post-tracking actions */
    postEleTrack( TrackArgs->TheBeam, BunchLoop, ElemLoop, 0, lastS, TFlag) ;
  }
  /* Wakefield clearance where required */
  if (strcmp(ElemClass,"SBEN")==0) {
    /* since the SBend has momentum compaction, clear the wakefields
         * which are available to the just-tracked bunch */
    ClearConvolvedSRWF( TrackArgs->TheBeam->bunches[*BunchLoop],
            -1,  0 ) ;
    ClearConvolvedSRWF( TrackArgs->TheBeam->bunches[*BunchLoop],
            -1,  1 ) ;
    ClearBinnedLRWFFreq( TrackArgs->TheBeam->bunches[*BunchLoop],
            -1,  0 ) ;
    ClearBinnedLRWFFreq( TrackArgs->TheBeam->bunches[*BunchLoop],
            -1,  1 ) ;
  }
  return 1 ;
}

#pragma GCC diagnostic ignored "-Wwrite-strings"
/* This file contains the top-level functions for the Matlab version of
 * Lucretia command TrackThru.  Usage options:
 *
 * [stat,beamout] = TrackThru(ielem,felem,beamin,ibunch,fbunch,ordflag)
 *
 * tracks bunches ibunch thru fbunch of beam structure beamin through
 * elements ielem thru felem, returning the resulting beam to beamout.
 * Only the bunches which are actually tracked will be returned to
 * beamout, any bunch in beamin which is not between ibunch and fbunch
 * inclusive will be ignored.  Flag ordflag tells whether
 * the tracking is element-by-element (more efficient) or bunch by bunch;
 * if ordflag is zero or missing, tracking is element by element, if it
 * is present and nonzero then tracking is bunch by bunch.
 *
 * [stat,beamout,data] = TrackThru( ... )
 *
 * returns a 1 x 3 cell array in addition to the extracted beam.  The
 * returned array contains BPM data in cell 1, data from other instrument
 * classes (INST,IMON,BLMO,SLMO, PROF,WIRE) in cell 2, and S-BPM data in
 * cell 3.
 *
 * V = TrackThru( 'version' )
 *
 * returns a cell array containing the version dates of Lucretia components.
 *
 * TrackThru( 'clear' )
 *
 * commands TrackThru to clear out its local, dynamic-allocated data structures.
 *
 * In addition to gateway routine mexFunction, this file contains two Matlab
 * auxilliary functions:
 *
 * TrackThruGetCheckArgs checks that the calling arguments are all right,
 * and returns a compressed argument structure.
 * TrackThruSetReturn sets the return values.
 *
 * AUTH:  PT, 16-aug-2004
 * MOD:
 * 21-aug-2013, GW:
 * Implement GPU code
 * 03-oct-2005, PT:
 * bugfix:  attempt to deallocate TrackArgs components only if
 * they were allocated in the first place.
 *
 * /*========================================================================*/

/* include files */


#include "mex.h"                /* Matlab API prototypes */
#include "matrix.h"
#include "LucretiaMatlab.h"     /* Lucretia-specific matlab fns */
#ifndef LUCRETIA_COMMON
#include "LucretiaCommon.h"   /* Matlab/Octave common fns */
#endif
#include "LucretiaGlobalAccess.h"
#include <string.h>
/* --- CUDA STUFF --- */
#ifdef __CUDACC__
  #include "gpu/mxGPUArray.h"
#endif

/* file-scoped variables */

char TrackThruVersion[] = "TrackThru Matlab version = 13-September-2013" ;

/*========================================================================*/

/* Gateway procedure for initiating bunch tracking thru the BEAMLINE */

/* RET:    none
 * /* ABORT:  never.
 * /* FAIL:   never.                                                         */

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
{
  
  struct TrackArgsStruc* TrackArgs ;           /* arguments for trackin' */
  int DidTracking = 0 ;
  
  /* Initialize Matlab GPU library */
#ifdef __CUDACC__  
  mxInitGPU() ;
#endif  
  
  /* process the calling arguments and make sure that they are well-conditioned.
   * Return a pointer to the TrackArgs summary */
  
  TrackArgs = TrackThruGetCheckArgs( nlhs, plhs, nrhs, prhs ) ;
  
  /* If the arguments do not check out for some reason, an error-exit
   * is mandated.  Do that now. */
  
  if ( TrackArgs->Status == 0 ) /* bad arguments */
  {
    ClearTrackingVars( ) ;         /* wipe out local dynalloc vars */
    AddMessage( "Improper argument format for TrackThru",1 ) ;
    goto egress ;
  }
  
  /* if TrackThruGetCheckArgs had a failure of dynamic allocation, an
   * error-exit is mandated.  Do that now... */
  
  if ( TrackArgs->Status == -1)
  {
    ClearTrackingVars( ) ;
    AddMessage( "Unable to allocate return variables for tracking",1 ) ;
    TrackArgs->Status = 0 ;
    goto egress ;
  }
  
  /* if LucretiaMatlabSetup bombed, handle that case now */
  
  if (TrackArgs->Status == -2)
  {
    ClearTrackingVars( ) ;
    TrackArgs->Status = 0 ;
    goto egress ;
  }
  
  /* if the user just wanted to clear the local arrays, do that */
  
  if ( TrackArgs->ClearLocal == 1 )
  {
    ClearTrackingVars( ) ;
    return ;
  }
  
  /* if on the other hand they wanted to see the version number, do that */
  
  else if ( TrackArgs->GetVersion == 1 )
  {
    plhs[0] = LucretiaMatlabVersions( TrackThruVersion ) ;
    return ;
  }
  
  /* if on the third hand the user really wants to track, do that */
  
  else
  {
    DidTracking = 1 ;
    TrackThruMain( TrackArgs ) ;
    ClearPLHSVars( ) ;
    if ( TrackArgs->Status == 2 )
    {
      DidTracking = 0 ;
      TrackArgs->Status = 0 ;
    }
  }
  
  egress:
    
    TrackThruSetReturn( TrackArgs, nlhs, plhs, DidTracking ) ;
    /*     cudaDeviceReset() ; // For profing */
    
}

/*========================================================================*/

/* the following pair of character string arrays are needed by both the
 * GetCheckArgs and the SetReturn functions, so put them outside of the
 * function block definitions */

const char* BeamFieldName[] = {           /* 2 field names */
  "BunchInterval","Bunch"
} ;
const char* BunchFieldName[] = {          /* 4 field names */
  "x", "Q", "stop", "ptype"
};

/* for efficiency, when we get the number of wakes we can keep it around
 * for both TrackThruGetCheckArgs and TrackThruSetReturn */

int* nWakes ;


/*========================================================================*/

/* Unpack the calling arguments for the TrackThru command, and verify that
 * they are well-formed.  Do some preparation on the output args as well,
 * and some preparation of local vars related to the global data
 * structures.
 * Also here, allocate GPU-resident array memory and point them to arrays
 * in Matlab space */

/* RET:    a pointer to a TrackArgsStruc with the necessary information
 * for the tracking engine.
 * /* ABORT:  never.
 * /* FAIL:   never. */

struct TrackArgsStruc* TrackThruGetCheckArgs( int nlhs, mxArray* plhs[],
        int nrhs, const mxArray* prhs[] )
{
  static struct TrackArgsStruc ArgStruc ;    /* the object of interest */
  char prhsString[8] ;                       /* the string arg, if any */
  static struct Beam TheBeam ;               /* beam data */
  int nElemTot ;                             /* # elts in BEAMLINE cell array */
  int nBunchTot ;                            /* # bunches in beam */
  mxArray* IntervalField ;                   /* pointer beam interval field */
  mxArray* BunchField ;                      /* pointer to beam bunch field */
  mxArray* newBunchField ;                   /* beam bunch field on exit beam */
#ifdef __CUDACC__
  mwSize xArraySize ;
#endif
  mxArray *x, *q, *stop ;                    /* Bunch data for CPU code */
  double *stop_local ;
  int i,j ;                                  /* counters */
  int npart ;                                /* # particles / bunch */
  int goodargs ;
  int wCount, sCount ;
  
  /* begin by setting the initial default values of the ArgStruc */
  
  ArgStruc.Status = 0 ;             /* default condition is failure */
  ArgStruc.GetVersion = 0 ;
  ArgStruc.ClearLocal = 0 ;
  ArgStruc.GetInstData = 0 ;        /* default = ignore BPMs, INSTs */
  ArgStruc.FirstElem = 0 ;
  ArgStruc.LastElem = 0 ;
  ArgStruc.FirstBunch = 0 ;
  ArgStruc.LastBunch = 0 ;
  ArgStruc.BunchwiseTracking = 0 ;  /* default = element-wise tracking */
  
  /* in terms of the number of ways that TrackThru can be called, and the
   * number of arguments, they are:
   *
   * call for clearing nlhs = 0, nrhs = 1
   * call for version  nlhs = 0 or 1, nrhs = 1
   * call for tracking nlhs = 2 or 3, nrhs = 5 or 6
   *
   * Start by just checking the argument counts */
  
  goodargs = 0 ;
  if ( (nrhs==1) &&
          (  (nlhs==0) || (nlhs==1)  )
          )
    goodargs = 1 ;
  if ( (  (nrhs==5) || (nrhs==6)  ) &&
          (  (nlhs==2) || (nlhs==3)  )    )
    goodargs = 1 ;
  if (goodargs == 0)
    goto egress ;
  
  /*   start with the "clear" and "version" conditions */
  
  if ( (nrhs == 1) && ((nlhs == 0)||(nlhs==1)) )
  {
    
    if ( (!mxIsChar(prhs[0])) ||
            (mxGetM(prhs[0])!=1)    )
      goto egress ;
    
    if (mxGetN(prhs[0])==5)
    {
      mxGetString( prhs[0] , prhsString, 6 ) ;
      if ( strcmp(prhsString,"clear") ==0 )
      {
        ArgStruc.Status = 1 ;
        ArgStruc.ClearLocal = 1 ;
      }
      goto egress ;
    }
    else if (mxGetN(prhs[0])==7)
    {
      mxGetString( prhs[0] , prhsString, 8 ) ;
      if ( strcmp(prhsString,"version") ==0 )
      {
        ArgStruc.Status = 1 ;
        ArgStruc.GetVersion = 1 ;
      }
      goto egress ;
    }
    else
      goto egress ;
    
  }
  
  
  /* now process the case of 2 or 3 return args and 5 or 6 calling args */
  
  if (   ( (nrhs == 5) || (nrhs == 6) )
  &&
          ( (nlhs == 2) || (nlhs == 3) )    )
  {
    if (nrhs == 2)
      ArgStruc.GetInstData = 0 ;
    if (nlhs == 5)
      ArgStruc.BunchwiseTracking = 0 ;
    
    /* first two arguments should be scalar integers between 1 and length(BEAMLINE);
     * check that now */
    
    nElemTot = LucretiaMatlabSetup( ) ;
    if (nElemTot < 1)
    {
      ArgStruc.Status = -2 ;
      goto egress ;
    }
    nWakes = GetNumWakes( ) ;
    
    if ( (!mxIsDouble(prhs[0])) || (!mxIsDouble(prhs[1])) ||
            (mxGetM(prhs[0])!=1)   || (mxGetN(prhs[0])!=1)   ||
            (mxGetM(prhs[1])!=1)   || (mxGetN(prhs[1])!=1)      )
      goto egress ;
    ArgStruc.FirstElem = (int)(*mxGetPr(prhs[0])) ;
    ArgStruc.LastElem  = (int)(*mxGetPr(prhs[1])) ;
    if ( (ArgStruc.FirstElem>nElemTot) || (ArgStruc.LastElem > nElemTot) )
      goto egress ;
    if ( (ArgStruc.FirstElem<1) || (ArgStruc.LastElem<1) )
      goto egress ;
    if (ArgStruc.FirstElem > ArgStruc.LastElem)
      goto egress ;
    
    /* the third argument should be a Matlab data structure with 2 fields, one
     * of which is "BunchInterval", a scalar, and the other of which is "Bunch", a
     * structure vector.  Check that now. */
    
    if (  (!mxIsStruct(prhs[2])) ||
            (mxGetM(prhs[2])!=1)   ||
            (mxGetN(prhs[2])!=1)     )
      goto egress ;
    
    IntervalField = mxGetField(prhs[2],0,BeamFieldName[0]) ;
    if ( IntervalField == NULL )
      goto egress ;
    if ( (mxGetM(IntervalField)!=1)    ||
            (mxGetN(IntervalField)!=1)    ||
            (!mxIsNumeric(IntervalField))    )
      goto egress ;
    
    BunchField = mxGetField(prhs[2],0,BeamFieldName[1]) ;
    if ( BunchField == NULL )
      goto egress ;
    if ( (mxGetM(BunchField)!=1) &&
            (mxGetN(BunchField)!=1)    )
      goto egress ;
    if ( !mxIsStruct(BunchField) )
      goto egress ;
    
    /* as long as we are here, verify that the fields for "Bunch" are okay */
    x = mxGetField(BunchField,0,BunchFieldName[0]) ;
    q = mxGetField(BunchField,0,BunchFieldName[1]) ;
    stop = mxGetField(BunchField,0,BunchFieldName[2]) ;
    mxArray* ptypePtr = mxGetField(BunchField,i,BunchFieldName[3]) ;
    if ( (x==NULL) || (q==NULL) || (stop==NULL) )// OK to not have ptype, will assume it is e- in this case
      goto egress ;
    
    /* get the total number of bunches in the beam */
    
    if (mxGetM(BunchField) > mxGetN(BunchField))
      nBunchTot = mxGetM(BunchField) ;
    else
      nBunchTot = mxGetN(BunchField) ;
    
    /* the fourth and fifth arguments should be scalar numerics which tell the
     * bunch numbers for tracking.  Unpack them and check them now. */
    
    if ( (!mxIsDouble(prhs[3])) || (!mxIsDouble(prhs[4])) ||
            (mxGetM(prhs[3])!=1)   || (mxGetN(prhs[3])!=1)   ||
            (mxGetM(prhs[4])!=1)   || (mxGetN(prhs[4])!=1)      )
      goto egress ;
    ArgStruc.FirstBunch = (int)(*mxGetPr(prhs[3])) ;
    ArgStruc.LastBunch  = (int)(*mxGetPr(prhs[4])) ;
    if ( (ArgStruc.FirstBunch>nBunchTot) || (ArgStruc.LastBunch > nBunchTot) )
      goto egress ;
    if ( (ArgStruc.FirstBunch<1) || (ArgStruc.LastBunch<1) )
      goto egress ;
    if ( ArgStruc.LastBunch < ArgStruc.FirstBunch )
      goto egress ;
    
    /* compute the number of bunches which will be tracked */
    
    ArgStruc.nBunch = ArgStruc.LastBunch - ArgStruc.FirstBunch + 1 ;
    
    /* now take care of the case in which there is an additional calling and/or
     * return argument */
    
    if (nlhs == 3)
      ArgStruc.GetInstData = 1 ;
    if (nrhs == 6)
    {
      if ( (!mxIsNumeric(prhs[5])) ||
              (mxGetM(prhs[5])!=1)    ||
              (mxGetN(prhs[5])!=1)       )
        goto egress ;
      ArgStruc.BunchwiseTracking = (int)(*mxGetPr(prhs[5])) ;
    }
    
    /* At this point we begin to set up more complicated data structures which
     * are needed for tracking and/or returning data.  First duplicate the
     * inter-bunch spacing into TheBeam: */
    
    TheBeam.interval = *mxGetPr(IntervalField) ;
    
    /* we need to allocate enough space to hold a vector of damping exponentials
     * for each freqency-domain long-range wakefield */
    
    if (nWakes[2] > 0)
    {
      TheBeam.TLRFreqDamping = (struct LucretiaComplex**)mxCalloc( nWakes[2] , sizeof(struct LucretiaComplex*) ) ;
      TheBeam.TLRFreqxPhase = (struct LucretiaComplex**)mxCalloc( nWakes[2] , sizeof(struct LucretiaComplex*) ) ;
      TheBeam.TLRFreqyPhase = (struct LucretiaComplex**)mxCalloc( nWakes[2] , sizeof(struct LucretiaComplex*) ) ;
    }
    if (nWakes[3] > 0)
    {
      TheBeam.TLRErrFreqDamping = (struct LucretiaComplex**)mxCalloc( nWakes[3] , sizeof(struct LucretiaComplex*) ) ;
      TheBeam.TLRErrFreqxPhase = (struct LucretiaComplex**)mxCalloc( nWakes[3] , sizeof(struct LucretiaComplex*) ) ;
      TheBeam.TLRErrFreqyPhase = (struct LucretiaComplex**)mxCalloc( nWakes[3] , sizeof(struct LucretiaComplex*) ) ;
    }
    
    
    /* the number of bunches field in TheBeam is the total number of bunches,
     * and the array of bunch pointers is allocated to this size: */
    
    TheBeam.nBunch = nBunchTot ;
    TheBeam.bunches = (struct Bunch**)mxMalloc( TheBeam.nBunch * sizeof(struct Bunch*) ) ;
    
    /* However, we only allocate space for the bunches which will actually be
     * tracked on this call to TrackThru: */
    for (i=ArgStruc.FirstBunch-1 ; i < ArgStruc.LastBunch ; i++) TheBeam.bunches[i] = (struct Bunch*)mxMalloc(sizeof(struct Bunch)) ;
    if (TheBeam.bunches == NULL)
    {
      ArgStruc.Status = -1 ;
      goto egress ;
    }
    
    /* allocate a new mxArray for the extracted beam */
    
    plhs[1] = mxCreateStructMatrix(1,1,2,BeamFieldName) ;
    if (plhs[1] == NULL)
    {
      ArgStruc.Status = -1 ;
      goto egress ;
    }
    
    /* copy the BunchInterval over */
    
    IntervalField = CREATE_SCALAR_DOUBLE( TheBeam.interval ) ;
    if (IntervalField == NULL)
    {
      ArgStruc.Status = -1 ;
      goto egress ;
    }
    mxSetField(plhs[1],0,BeamFieldName[0],IntervalField) ;
    
    /* generate a structure mxArray with the correct fields for a beam */
    if (ptypePtr==NULL) // Don't create ptype if not present in the passed Bunch structure
      newBunchField = mxCreateStructMatrix(1,ArgStruc.nBunch,3,BunchFieldName);
    else
      newBunchField = mxCreateStructMatrix(1,ArgStruc.nBunch,4,BunchFieldName);
    
    if (newBunchField == NULL)
    {
      ArgStruc.Status = -1 ;
      goto egress ;
    }
    
    
    /* loop over the bunches to be tracked.  On each bunch, we will do several
     * things:
     *
     * -> check that the x    field is 6 x npart
     * -> check that the Q    field is 1 x npart
     * -> check that the stop field is 1 x npart
     * -> check that the ptype field is 1 x npart if there
     * -> copy the bunch from the input beam to the output beam
     * -> allocate a bunch structure for TheBeam
     * -> get pointers to the output beam x, Q, stop fields
     * -> allocate another field equal in size to x (called y)
     * -> attach pointers to x,y,Q,stop double fields to slots in the
     * most recent bunch allocated
     *
     * Note that there is an index offset between the bunch number in the input
     * beam structure and that in the output, but that there is no offset between
     * the input beam structure and the NewBeam data structure.  We handle this
     * offset by using i as the index into BunchField and TheBeam.bunches (which
     * have counting offset from zero) and j as the index into NewBunchField
     * (which counts from zero).
     *
     *
     */
    
    for (i = ArgStruc.FirstBunch-1 ; i < ArgStruc.LastBunch ; i++)
    {
      j = i + 1 - ArgStruc.FirstBunch ;
      x=mxGetField(BunchField,i,BunchFieldName[0]) ;
#ifdef __CUDACC__ /* check for existence of Matlab GPU arrays if running GPU version */
      TheBeam.bunches[i]->x_gpu = mxGPUCopyFromMxArray( x ) ;
      xArraySize = mxGPUGetNumberOfElements( TheBeam.bunches[i]->x_gpu ) / 6;
      npart = (int)xArraySize;
#else
      if ( (mxGetM(x)!=6) || (!mxIsDouble(x)) )
        goto egress ;
      npart = mxGetN(x) ;
#endif      
      TheBeam.bunches[i]->nray = npart ;
      TheBeam.bunches[i]->ngoodray = npart ;
#ifdef __CUDACC__
      TheBeam.bunches[i]->x=(double *)mxGPUGetData(TheBeam.bunches[i]->x_gpu) ;
      TheBeam.bunches[i]->y_gpu = mxGPUCopyGPUArray(TheBeam.bunches[i]->x_gpu) ;
      TheBeam.bunches[i]->y=(double *)mxGPUGetData(TheBeam.bunches[i]->y_gpu) ;
      cudaMalloc((void**)&TheBeam.bunches[i]->ngoodray_gpu, sizeof(int)) ;
#else
      mxSetField(newBunchField,j,BunchFieldName[0], mxDuplicateArray(x) ) ;
      TheBeam.bunches[i]->x = mxGetPr(mxGetField(newBunchField,j,
              BunchFieldName[0])) ;
      TheBeam.bunches[i]->y = (double*)mxMalloc(6*npart*sizeof(double)) ;
#endif
      if (nWakes[0] > 0)
      {
        TheBeam.bunches[i]->ZSR = (struct SRWF**)mxMalloc(nWakes[0]*sizeof(struct SRWF*)) ;
        for (wCount = 0 ; wCount < nWakes[0] ; wCount++)
          TheBeam.bunches[i]->ZSR[wCount] = NULL ;
      }
      else
        TheBeam.bunches[i]->ZSR = NULL ;
      if (nWakes[1] > 0)
      {
        TheBeam.bunches[i]->TSR = (struct SRWF**)mxMalloc(nWakes[1]*sizeof(struct SRWF*)) ;
        for (wCount = 0 ; wCount < nWakes[1] ; wCount++)
          TheBeam.bunches[i]->TSR[wCount] = NULL ;
      }
      else
        TheBeam.bunches[i]->TSR = NULL ;
      if (nWakes[2] > 0)
      {
        TheBeam.bunches[i]->TLRFreq = (struct LRWFFreq**)mxMalloc(nWakes[2]*sizeof(struct LRWFFreq*)) ;
        for (wCount = 0 ; wCount < nWakes[2] ; wCount++)
          TheBeam.bunches[i]->TLRFreq[wCount] = NULL ;
      }
      else
        TheBeam.bunches[i]->TLRFreq = NULL ;
      if (nWakes[3] > 0)
      {
        TheBeam.bunches[i]->TLRErrFreq = (struct LRWFFreq**)mxMalloc(nWakes[3]*sizeof(struct LRWFFreq*)) ;
        for (wCount = 0 ; wCount < nWakes[3] ; wCount++)
          TheBeam.bunches[i]->TLRErrFreq[wCount] = NULL ;
      }
      else
        TheBeam.bunches[i]->TLRErrFreq = NULL ;
      
      q=mxGetField(BunchField,i,BunchFieldName[1]) ;
#ifdef __CUDACC__
      TheBeam.bunches[i]->Q_gpu = mxGPUCopyFromMxArray( q ) ;
      TheBeam.bunches[i]->Q = (double *)mxGPUGetData(TheBeam.bunches[i]->Q_gpu) ;
#else      
      mxSetField(newBunchField,j,BunchFieldName[1], mxDuplicateArray(q) ) ;
      if ( (mxGetM(q)!=1)     ||
              (mxGetN(q)!=npart) ||
              (!mxIsDouble(q)  )    )
        goto egress ;
      TheBeam.bunches[i]->Q = mxGetPr(mxGetField(newBunchField,j,
              BunchFieldName[1])) ;
#endif 
      stop=mxGetField(BunchField,i,BunchFieldName[2]) ;
#ifdef __CUDACC__
      TheBeam.bunches[i]->stop_gpu = mxGPUCopyFromMxArray( stop ) ;
      TheBeam.bunches[i]->stop = (double *)mxGPUGetData(TheBeam.bunches[i]->stop_gpu) ;
#else      
      mxSetField(newBunchField,j,BunchFieldName[2], mxDuplicateArray(stop) ) ;
      if ( (mxGetM(stop)!=1)     ||
              (mxGetN(stop)!=npart) ||
              (!mxIsDouble(stop)  )    )
        goto egress ;
      TheBeam.bunches[i]->stop = mxGetPr(mxGetField(newBunchField,j,
              BunchFieldName[2])) ;
#endif      
#ifdef __CUDACC__
      stop_local = (double *) calloc(TheBeam.bunches[i]->nray,sizeof(double)) ;
      cudaMemcpy(stop_local, TheBeam.bunches[i]->stop, sizeof(double)*TheBeam.bunches[i]->nray, cudaMemcpyDeviceToHost) ;
#else
      stop_local = TheBeam.bunches[i]->stop ;
#endif
      for (sCount=0 ; sCount<TheBeam.bunches[i]->nray ; sCount++)
        if (stop_local[sCount] != 0)
          TheBeam.bunches[i]->ngoodray-- ;
      TheBeam.bunches[i]->StillTracking = 1 ;
#ifdef __CUDACC__      
      cudaMemcpy(TheBeam.bunches[i]->ngoodray_gpu, &TheBeam.bunches[i]->ngoodray, sizeof(int), cudaMemcpyHostToDevice) ;
      free(stop_local) ;
#endif   
      unsigned short int* ptype = (unsigned short int*) malloc( sizeof(unsigned short int)*TheBeam.bunches[i]->nray ) ;
      if (ptypePtr==NULL) {
        for (sCount=0 ; sCount<TheBeam.bunches[i]->nray ; sCount++)
          ptype[sCount] = 0 ;
      }
      else {
        double* ptypeVal = (double*) mxGetPr( ptypePtr ) ;
        for (sCount=0 ; sCount<TheBeam.bunches[i]->nray ; sCount++)
          ptype[sCount]=(unsigned short int) ptypeVal[sCount];
        mxSetField(newBunchField,j,BunchFieldName[3], mxDuplicateArray(ptypePtr) ) ; // copy ptype over to output bunch
      }
#ifdef __CUDACC__
      cudaMemcpy(TheBeam.bunches[i]->ptype, &ptype, sizeof(unsigned short int)*TheBeam.bunches[i]->nrayTheBeam.bunches[i]->nray, cudaMemcpyHostToDevice) ;
#else    
      TheBeam.bunches[i]->ptype=ptype;
#endif            
    }
    
    /* hook the newBunchField to the output beam data structure */
    
    mxSetField(plhs[1],0,BeamFieldName[1],newBunchField) ;
    
    /* If there is a third return argument, then set it up as a 1 x 3 cell
     * array.  If we knew how many BPMs, SBPMs, or INSTs we were going to
     * see, we could create the relevant structure arrays now; since we
     * don't, we can only create the cell matrix at this time. */
    
    if (nlhs == 3)
    {
      plhs[2] = mxCreateCellMatrix( 1, 3 ) ;
      if (plhs[2] == NULL)
      {
        ArgStruc.Status = -1 ;
        goto egress ;
      }
    }
    
    
    /* hook TheBeam to ArgStruc */
    
    ArgStruc.TheBeam = &TheBeam ;
    
    
    /* if we got this far, then we must be successful! */
    
    ArgStruc.Status = 1 ;
    
  }
  
  /* now all that remains is to set return value */
  
  egress:
    return &ArgStruc ;
    
}

/*========================================================================*/

/* Package the data from the tracking operation into the return arguments.
 *
 * /* RET:    None.
 * /* ABORT:  Failure to allocate memory for returning data.
 * /* FAIL:   Never.                                                         */

void TrackThruSetReturn( struct TrackArgsStruc* TrackArgs,
        int nlhs, mxArray* plhs[], int DidTracking )
{
  
  /* some Matlab structure field name definitions */
  
  static char* BPMDatFieldName[] = {
    "Index","S", "Pmod","x","y","z","P","sigma"
  };
  static char* INSTDatFieldName[] = {
    "Index","S","x","sig11","y","sig33","sig13","z","sig55"
  } ;
  static char* SBPMDatFieldName[] = {
    "Index","S","x","y"
  } ;
  int i,j,nBunch ;
  int is,js ;
  int status = 1 ;            /* default to good status */
  mxArray* ReturnBunches ;
  mxArray* bpmdat ;
  mxArray* instdat ;
  mxArray* sbpmdat ;
  mxArray* Sdat ;
  double*  realpr ;
  size_t dims[3] ;              /* 3-d array for sigma matrices */
  char** messages ;
  int nmsg ;
  
  /* Start by getting messages and status into the first return arg: */
  
  messages = GetAndClearMessages( &nmsg ) ;
  plhs[0] = CreateStatusCellArray( TrackArgs->Status, nmsg, messages ) ;
  
  /* if we never even attempted tracking, go to the exit now */
  
  if (DidTracking == 0)
  {
    for (i=1 ; i<nlhs ; i++)
      if (plhs[i] == NULL)
        plhs[i] = mxCreateCellMatrix(0,0) ;
    goto egress ;
  }
  
  /* get a pointer to the "Bunch" field in the returned beam */
  
  ReturnBunches = mxGetField( plhs[1], 0, BeamFieldName[1] ) ;
  
  /* loop over bunches */
  
  nBunch = TrackArgs->nBunch ;
  
  for (i=0 ; i < nBunch ; i++)
  {
    
    /* get a pointer to the ray position field in the i'th bunch,
     * remembering that the bunches in TheBeam are not set up from 0 to
     * nBunch-1 but from FirstBunch-1 to LastBunch-1*/
    j = i + TrackArgs->FirstBunch - 1 ;
#ifdef __CUDACC__ /* If running on GPU, just return mxGPUArray links back to tracked beam data*/
    mxSetField( ReturnBunches, i, BunchFieldName[0], mxGPUCreateMxArrayOnGPU(TrackArgs->TheBeam->bunches[j]->x_gpu) ) ;
    mxSetField( ReturnBunches, i, BunchFieldName[1], mxGPUCreateMxArrayOnGPU(TrackArgs->TheBeam->bunches[j]->Q_gpu) ) ;
    mxSetField( ReturnBunches, i, BunchFieldName[2], mxGPUCreateMxArrayOnGPU(TrackArgs->TheBeam->bunches[j]->stop_gpu) ) ;
#else    
    mxSetPr( mxGetField( ReturnBunches, i, BunchFieldName[0] ), TrackArgs->TheBeam->bunches[j]->x ) ;
#endif    
  }
  
  /* if no instrument data was to be returned, jump to the exit */
  
  if ( TrackArgs->GetInstData == 0 )
    goto egress ;
  
  /* Now handle the case where data is to be returned.
   * start by creating a structure array for BPM data */
  
  bpmdat = mxCreateStructMatrix( 1, TrackArgs->nBPM, 8, (const char **)BPMDatFieldName ) ;
  if (bpmdat == NULL)
  {
    status = 0 ;
    goto egress ;
  }
  
  /* loop over BPMs */
  
  for (i=0 ; i<TrackArgs->nBPM ; i++)
  {
    
    /* copy the index, S position and Pmod from the TrackArgs structure to
     * the returnstructure */
    
    mxSetField( bpmdat, i, BPMDatFieldName[0],
            CREATE_SCALAR_DOUBLE((double)TrackArgs->bpmdata[i]->indx) ) ;
    mxSetField( bpmdat, i, BPMDatFieldName[1],
            CREATE_SCALAR_DOUBLE(TrackArgs->bpmdata[i]->S) ) ;
    mxSetField( bpmdat, i, BPMDatFieldName[2],
            CREATE_SCALAR_DOUBLE(TrackArgs->bpmdata[i]->Pmod) ) ;
    
    /* copy the x data to the output structure */
    
    status = SetDoublesToField( bpmdat, i, BPMDatFieldName[3],
            TrackArgs->bpmdata[i]->xread,
            TrackArgs->bpmdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    /* now do the same for y data */
    
    status = SetDoublesToField( bpmdat, i, BPMDatFieldName[4],
            TrackArgs->bpmdata[i]->yread,
            TrackArgs->bpmdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    /* The P, z, and sigma fields are different in that they may or may not
     * be filled with useful data -- it depends on whether or not the BPM in
     * question had its GetBeamPars flag set.  This flag is copied to the
     * bpmdata structure.  Interrogate it now, and if it's zero go on to the
     * next BPM */
    
    if (TrackArgs->bpmdata[i]->GetBeamPars == 0)
      continue ;
    
    /* otherwise, collect data and move to the output structure */
    
    status = SetDoublesToField( bpmdat, i, BPMDatFieldName[5],
            TrackArgs->bpmdata[i]->z,
            TrackArgs->bpmdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( bpmdat, i, BPMDatFieldName[6],
            TrackArgs->bpmdata[i]->P,
            TrackArgs->bpmdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    /* the sigma matrix is a bit special, since we are here copying the moral
     * equivalent of a 3-D matrix into an actual 3-d matrix. */
    
    dims[0] = 6 ; dims[1] = 6 ; dims[2] = TrackArgs->bpmdata[i]->nbunchalloc ;
    
    Sdat = mxCreateNumericArray( 3, dims, mxDOUBLE_CLASS, mxREAL ) ;
    if (Sdat == NULL)
    {
      status = 0 ;
      goto egress ;
    }
    realpr = mxGetPr( Sdat ) ;
    for (j=0 ; j<TrackArgs->bpmdata[i]->nBunch ; j++)
    {
      for (is=0 ; is<6 ; is++)
      {
        for (js=0 ; js<6 ; js++)
        {
          realpr[36*j+6*js+is] =
                  TrackArgs->bpmdata[i]->sigma[36*j+6*is+js] ;
        }
      }
    }
    
    mxSetField( bpmdat, i, BPMDatFieldName[7], Sdat ) ;
    
  } /* end of BPM loop */
  
  /* now we can set the first cell in the 3rd return argument to carry the
   * BPM return information */
  
  mxSetCell(plhs[2],0,bpmdat) ;
  
  /* now we do the instrument data, pretty similar to the BPMs but without
   * any optional fields and/or multi-dimensional arrays */
  
  instdat = mxCreateStructMatrix( 1, TrackArgs->nINST, 9, (const char **)INSTDatFieldName ) ;
  if (instdat == NULL)
  {
    status = 0 ;
    goto egress ;
  }
  
  /* loop over INSTs */
  
  for (i=0 ; i<TrackArgs->nINST ; i++)
  {
    
    /* capture the index and S position */
    
    mxSetField( instdat, i, INSTDatFieldName[0],
            CREATE_SCALAR_DOUBLE((double)TrackArgs->instdata[i]->indx) ) ;
    mxSetField( instdat, i, INSTDatFieldName[1],
            CREATE_SCALAR_DOUBLE(TrackArgs->instdata[i]->S) ) ;
    
    /* in turn capture x, sigx, y, sigy, z, sigz */
    
    status = SetDoublesToField( instdat, i, INSTDatFieldName[2],
            TrackArgs->instdata[i]->x,
            TrackArgs->instdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( instdat, i, INSTDatFieldName[3],
            TrackArgs->instdata[i]->sig11,
            TrackArgs->instdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( instdat, i, INSTDatFieldName[4],
            TrackArgs->instdata[i]->y,
            TrackArgs->instdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( instdat, i, INSTDatFieldName[5],
            TrackArgs->instdata[i]->sig33,
            TrackArgs->instdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( instdat, i, INSTDatFieldName[6],
            TrackArgs->instdata[i]->sig13,
            TrackArgs->instdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( instdat, i, INSTDatFieldName[7],
            TrackArgs->instdata[i]->z,
            TrackArgs->instdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( instdat, i, INSTDatFieldName[8],
            TrackArgs->instdata[i]->sig55,
            TrackArgs->instdata[i]->nBunch ) ;
    if (status != 1)
      goto egress ;
    
  } /* end of loop over instruments */
  
  /* now set the 2nd cell in the 3rd return argument to the instrument data */
  
  mxSetCell( plhs[2], 1, instdat ) ;
  
  /* now for the RF structure BPMs.  These are somewhat different in that
   * each structure can have multiple BPMs, and the indexing of the data is by
   * structure, not BPM. */
  
  sbpmdat = mxCreateStructMatrix( 1, TrackArgs->nSBPM, 4, (const char **)SBPMDatFieldName ) ;
  if (instdat == NULL)
  {
    status = 0 ;
    goto egress ;
  }
  
  /* loop over RF structures */
  
  for (i=0 ; i<TrackArgs->nSBPM ; i++)
  {
    
    /* capture the index */
    
    mxSetField( sbpmdat, i, SBPMDatFieldName[0],
            CREATE_SCALAR_DOUBLE((double)TrackArgs->sbpmdata[i]->indx) ) ;
    
    /* unlike the other structures, here S is a vector like x and y. Capture all
     * now. */
    
    status = SetDoublesToField( sbpmdat, i, SBPMDatFieldName[1],
            TrackArgs->sbpmdata[i]->S,
            TrackArgs->sbpmdata[i]->nbpm ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( sbpmdat, i, SBPMDatFieldName[2],
            TrackArgs->sbpmdata[i]->x,
            TrackArgs->sbpmdata[i]->nbpm ) ;
    if (status != 1)
      goto egress ;
    
    status = SetDoublesToField( sbpmdat, i, SBPMDatFieldName[3],
            TrackArgs->sbpmdata[i]->y,
            TrackArgs->sbpmdata[i]->nbpm ) ;
    if (status != 1)
      goto egress ;
    
  } /* end of loop over INST data */
  
  /* now attach the SBPM data to cell 3 of return argument 3 */
  
  mxSetCell( plhs[2], 2, sbpmdat ) ;
  
  /* now we can get outta here.  If an allocation failed, send an error
   * message, otherwise just exit. */
  
  egress:
    
    /* cleanup variables which were mxCalloc'ed, if they are here (ie, if
     * TrackArgs->TheBeam points at TheBeam).  The necessary referencing
     * occurs right before TrackThruGetCheckArgs set good status, and if
     * the status is not good in that routine then the pointing never happens
     * at all.  In that event it is possible that these variables are mxMalloc'ed
     * and do not get cleared by this routine, in which case I am relying on
     * Matlab's memory manager to take care of the problem.  It's not a good
     * practice, but I'm doing it anyway.  Cope. */
    
    if (TrackArgs->Status == 1)
    {
      if (nWakes[2] > 0)
      {
        mxFree( TrackArgs->TheBeam->TLRFreqDamping ) ;
        mxFree( TrackArgs->TheBeam->TLRFreqxPhase  ) ;
        mxFree( TrackArgs->TheBeam->TLRFreqyPhase  ) ;
      }
      if (nWakes[3] > 0)
      {
        mxFree( TrackArgs->TheBeam->TLRErrFreqDamping ) ;
        mxFree( TrackArgs->TheBeam->TLRErrFreqxPhase  ) ;
        mxFree( TrackArgs->TheBeam->TLRErrFreqyPhase  ) ;
      }
      
      for (i = TrackArgs->FirstBunch-1 ; i < TrackArgs->LastBunch ; i++)
      {
#ifdef __CUDACC__
        /* This doesn't actually clear the x, Q, stop data whilst there is still a
         * GPUArray object linking to these in Matlab in the beamout structure. these
         * Can only be cleared with a reset command to the GPU device from Matlab */
        mxGPUDestroyGPUArray( TrackArgs->TheBeam->bunches[i]->x_gpu ) ;
        mxGPUDestroyGPUArray( TrackArgs->TheBeam->bunches[i]->Q_gpu ) ;
        mxGPUDestroyGPUArray( TrackArgs->TheBeam->bunches[i]->stop_gpu ) ;
        cudaFree( TrackArgs->TheBeam->bunches[i]->ngoodray_gpu ) ;
        mxGPUDestroyGPUArray( TrackArgs->TheBeam->bunches[i]->y_gpu ) ;
#else
        mxFree( TrackArgs->TheBeam->bunches[i]->y ) ;
        free( TrackArgs->TheBeam->bunches[i]->ptype ) ;
#endif
        if (nWakes[0] > 0)
          mxFree( TrackArgs->TheBeam->bunches[i]->ZSR ) ;
        if (nWakes[1] > 0)
          mxFree( TrackArgs->TheBeam->bunches[i]->TSR ) ;
        if (nWakes[2] > 0)
          mxFree( TrackArgs->TheBeam->bunches[i]->TLRFreq ) ;
        if (nWakes[3] > 0)
          mxFree( TrackArgs->TheBeam->bunches[i]->TLRErrFreq ) ;
        
        mxFree( TrackArgs->TheBeam->bunches[i] ) ;
      }
      mxFree( TrackArgs->TheBeam->bunches ) ;
    }
    
    if (status == 0)
    {
      ClearTrackingVars( ) ;
      mexErrMsgTxt("Unable to allocate memory for returned data") ;
    }
    if (status == -1)
    {
      ClearTrackingVars( ) ;
      mexErrMsgTxt("Internal error -- field assign to mxArray failed") ;
    }
    
    return ;
    
}

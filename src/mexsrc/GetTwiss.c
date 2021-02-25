/* This file contains the top-level source for the Matlab version of 
   mexfile GetTwiss (the octave version will be in LucretiaOctave.c).

   Matlab usage:

		[stat,twiss] = GetTwiss( start, finish, itwissx, itwissy ) ;

   where 

		stat = cell array of status information.
	   start, finish = the start and finish of the region of interest
		   for computing the R-matrix.  If start > finish, Twiss will
			be back-propagated from the start element to the finish
			element.
		twiss = data structure with 12 fields, each of which is a 
		   1 x |finish-start| + 2 double array.  Fields:

				S
				P
				betax
				alphax
				etax
				etapx
				nux
				betay
				alphay
				etay
				etapy 
				nuy

         Note that the array includes the parameters at the entry of the
		   first element and the exit of the last element, so there is one
		   extra entry over and above the # of elements.

      itwissx, itwissy : data structures.  Each structure has 5 fields,
		each of which is a double scalar (beta,alpha,eta,etap,nu).  If
		start > finish, these are the final Twiss rather than the initial
		Twiss.

	Alternately, GetTwiss can be called:

		[stat,T] = GetTwiss(start,finish,W)

	where
	
	  W == 6 x 6 x n Matlab real array with the coupled Twiss parameters
	  at the beginning of the line, in Wolski notation.

   In this case, T is a 6 x 6 x n x (|finish-start|+2) double array with
	the coupled Twiss parameters at all elements including the end of the
	line.

   GetTwiss can also be called with one argument, the string "version".
	In this case GetTwiss returns its own version date, as well as the
	version dates of LucretiaCommon, LucretiaPhysics, and LucretiaMatlab.

   This file also includes GetTwissGetCheckArgs and GetTwissSetReturn, which
	process the calling args and return args, respectively; and 
	GetTwissUnpackTwiss, which unpacks twiss parameters from a matlab data
	structure.

   AUTH:  PT, 09-aug-2004
	MOD:

			08-Mar-2006, PT:
				support for coupled Twiss propagation.

========================================================================*/

/* include files */

//#ifndef mex_h
//  #define mex_h
  #include <mex.h>
//#endif
#include "matrix.h"
#include "LucretiaMatlab.h"     /* Lucretia-specific matlab fns */
#ifndef LUCRETIA_COMMON
  #include "LucretiaCommon.h"   /* Matlab/Octave common fns */
#endif
#include "LucretiaPhysics.h"
#include "LucretiaGlobalAccess.h" 
#include <string.h>
#include <stdlib.h>

/* file-scoped variables */

char GetTwissVersion[] = "GetTwiss Matlab version = 08-Mar-2006" ;

mxArray* WolskiInit = NULL ; /* holding tank for initial parameters, if
									     needed by reverse Twiss propagation
										  process */


/*========================================================================*/

/* Gateway function for calculation of Twiss parameters */

/* RET:    none. 
ABORT:  never.
FAIL:   never.                                                   */

void mexFunction( int nlhs, mxArray *plhs[],
				  int nrhs, const mxArray *prhs[] ) 
{

   struct RmatArgStruc* RmatArgs ; /* arguments used by Rmat operation */
   char BadArgs[] = "Improper argument format for GetTwiss" ;

	void *T = NULL ; /* pointer to Twiss or to 4-d double array */

/* Check the input arguments and make sure that they are OK.  Figure
   out whether this is a "version" call or a real request for a matrix.
   If the latter make sure that the BEAMLINE array is present and is
   large enough. Return a pointer to an appropriate argument structure. */

   RmatArgs = GetTwissGetCheckArgs( nlhs, nrhs, prhs ) ;

/* If the arguments do not check out for some reason, an error-return
   is mandated.  Do that now. */
   
   if ( RmatArgs->Status == 0 ) /* bad arguments */
	{
		AddMessage( BadArgs, 1 ) ;
		goto egress ;
	}

/* if the LucretiaMatlabSetup failed, this will be indicated to this routine
   by a -1 in RmatArgs->Status.  Change that to zero (immediate stop),
	and go to the exit (no need for a message, as LucretiaMatlabSetup
	issues messages as it executes). */

	if ( RmatArgs->Status == -1 )
	{
		RmatArgs->Status = 0 ;
		goto egress ;
	}

/* If the call was for a list of versions, let LucretiaMatlabVersions 
   build a cell array of version strings */

   if ( RmatArgs->Version == 1 )
	{
	 plhs[0] = LucretiaMatlabVersions( GetTwissVersion ) ;
	 return ;
	}

   else /* do some real math! */
   {

/* if the propagation is backwards, adjust the initial parameters
   now */

	  if (RmatArgs->Backwards == 1)
	  {
		  ReverseTwissDirection( RmatArgs ) ;
		  if (RmatArgs->Status == 0)
			  goto egress ;
	  }
     T = (struct twiss *)RmatCalculate( RmatArgs ) ;

	  if (T==NULL)
		  AddMessage("Dynamic allocation failure in RmatCalculate",1) ;
   }

/* and that's it! */

egress:  

	if ( RmatArgs->PropagateTwiss == 1)
	  GetTwissSetReturn( RmatArgs->Status, plhs, (struct twiss*)T ) ;
	else
	  GetTwissCoupledSetReturn( RmatArgs->Status, 
	  RmatArgs->nWolskiDimensions, 
				    plhs, (struct Ctwiss*)T ) ;
	return ;

}

/*========================================================================*/

/* Procedure GetTwissGetCheckArgs:  make sure that GetTwiss was called with
   good arguments.  Configure a data structure to convey proper arguments
	to RmatCalculate.  While we're at it, perform some general Lucretia
	initialization. */

/* RET:    pointer to an RmatArgStruc.
   ABORT:  never.
   FAIL:   never.                                  */

struct RmatArgStruc* GetTwissGetCheckArgs( int nlhs, int nrhs, 
														 const mxArray* prhs[] ) 
{

	static struct RmatArgStruc ArgStruc ;    /* the object of interest */
	char prhsString[8] ;              /* the string arg, if any */
	int nElemTot ;                    /* # elts in BEAMLINE cell array */
	double *twissvec ;                /* capture initial conditions */

/* initialize the ArgStruc fields: */

	ArgStruc.start = 0 ;
	ArgStruc.end = 0 ;
	ArgStruc.ReturnEach = 0 ;     /* we want just the matrix for the region */
	ArgStruc.PropagateTwiss = 1 ;
	ArgStruc.Version = 0 ;
	ArgStruc.Status = 0 ;         /* assume bad status for now */

/* There are three valid ways to execute GetTwiss from the Lucretia 
   command line:  
	
	  VersionCell = GetTwiss('version') 

	will return a cell array with version information; 

		[stat,Tstruc] = GetTwiss(first,last,itwissx,itwissy)

   will return a structure which has fields of uncoupled Twiss vectors;
	
	   [stat,Tstruc] = GetTwiss(first,last,W)

   will return a structure filled with coupled Twiss parameters.

	If the number
	of calling args is not 1, 3 or 4 then something is wrong. */

	if ( (nrhs != 1) && (nrhs != 3) && (nrhs != 4) )
		goto Egress ;

/* now handle the case of 1 argument: */

	if ( nrhs == 1 )
	{

/* if the argument is anything but the word "version" it's wrong, so do the
	following checks:  prhs[0] should be a 1x1 character mxArray, and its
	contents should be the word "version" */

		if ( (!mxIsChar(prhs[0])) || 
			  (mxGetM(prhs[0])!=1) ||
			  (mxGetN(prhs[0])!=7)    )
		   goto Egress ;
		mxGetString( prhs[0] , prhsString, 8 ) ;
		if ( strcmp(prhsString,"version") !=0 )
			goto Egress ;

/* if we got here then there must be 1 string argument containing the word
	"version."  So, set the Version flag in ArgStruc and the good status
	flag */

		ArgStruc.Version = 1 ;
		ArgStruc.Status = 1 ;
	}

/* now for the case of 3 or 4 arguments */

	if ( (nrhs == 3) || (nrhs == 4) )
	{

/* for real math, we need 2 return arguments.  If we don't have them,
   error exit! */

		if ( nlhs != 2 )
			goto Egress ;


/* each of the first 2 arguments should be a scalar integer -- check that now */

		if ( (!mxIsDouble(prhs[0])) || (!mxIsDouble(prhs[1])) ||
			  (mxGetM(prhs[0])!=1)   || (mxGetN(prhs[0])!=1)   ||
			  (mxGetM(prhs[1])!=1)   || (mxGetN(prhs[1])!=1)      )
			  goto Egress ;
		ArgStruc.start = (int)(*mxGetPr(prhs[0])) ;
		ArgStruc.end   = (int)(*mxGetPr(prhs[1])) ;

/* if we're going to do real accelerator physics calculations, we need to 
   set up access to the BEAMLINE,PS,KLYS,GIRDER data structures.  Do that
	next. */

		nElemTot = LucretiaMatlabSetup( ) ;
		if (nElemTot < 1)
		{
			ArgStruc.Status = -1 ;
			goto Egress ;
		}

/* make sure that there are enough elements in the beamline */

		if ( (ArgStruc.start>nElemTot) || (ArgStruc.end > nElemTot) )
			goto Egress ;
		if ( (ArgStruc.start<1) || (ArgStruc.end<1) )
			goto Egress ;
		if (ArgStruc.start > ArgStruc.end)
		{
			ArgStruc.Backwards = ArgStruc.start ;
			ArgStruc.start = ArgStruc.end ;
			ArgStruc.end = ArgStruc.Backwards ;
			ArgStruc.Backwards = 1 ;
		}
		else
			ArgStruc.Backwards = 0 ;
	}

/* now we do somewhat different initializations for coupled and
   uncoupled Twiss */

	if (nrhs == 4)
	{

/* Arguments 3 and 4 must be initial Twiss parameters.  Unpack them now */

		twissvec = GetTwissUnpackTwiss(prhs[2]) ;
		if (twissvec == NULL) 
			goto Egress;
		ArgStruc.InitialTwiss.betx  = twissvec[0] ;
		ArgStruc.InitialTwiss.alfx  = twissvec[1] ;
		ArgStruc.InitialTwiss.etax  = twissvec[2] ;
		ArgStruc.InitialTwiss.etapx = twissvec[3] ;
		ArgStruc.InitialTwiss.nux   = twissvec[4] ;
		twissvec = GetTwissUnpackTwiss(prhs[3]) ;
		if (twissvec == NULL)
			goto Egress ;
		ArgStruc.InitialTwiss.bety  = twissvec[0] ;
		ArgStruc.InitialTwiss.alfy  = twissvec[1] ;
		ArgStruc.InitialTwiss.etay  = twissvec[2] ;
		ArgStruc.InitialTwiss.etapy = twissvec[3] ;
		ArgStruc.InitialTwiss.nuy   = twissvec[4] ;

		ArgStruc.InitialWolski = NULL ;
		ArgStruc.nWolskiDimensions = 0 ;

	}

	if (nrhs == 3) 
	{

/* make sure that we have a 6 x 6 x n double-precision matrix */

		int nDim = mxGetNumberOfDimensions( prhs[2] ) ;
		const mwSize* dims = mxGetDimensions( prhs[2] ) ;

		if (mxIsDouble( prhs[2] ) != 1)
			goto Egress ;

		switch (nDim) 
		{

		case 2:
		{
			if ( (dims[0] != 6) || (dims[1] != 6) )
				goto Egress ;
			ArgStruc.nWolskiDimensions = 1 ;
			break ;
		}
		
		case 3:
		{			
			if ( (dims[0] != 6) || (dims[1] != 6) )
				goto Egress ;
			if ( (dims[2] < 1) || (dims[2] > 3) )
				goto Egress ;
			ArgStruc.nWolskiDimensions = dims[2] ;
			break ;
		}

		default:
			goto Egress ;

		}

/* if that's okay, then the initial values are equal to the numeric
   values in the argument, so get a pointer to them now */

		ArgStruc.InitialWolski = mxGetPr( prhs[2] ) ;

/* set the PropagateTwiss to indicate coupled propagation */

		ArgStruc.PropagateTwiss = 2 ;

	}

/* if we made it this far, we must be successful.  Set success indicator */

	ArgStruc.Status = 1 ;



/* once we've gotten here, all we need to do is exit! */

Egress:

	return &ArgStruc ;

}

/*========================================================================*/

/* Procedure GetTwissUnpackTwiss:  unpack the twiss initial conditions from
   a Matlab data structure */

/* RET:    pointer to a vector of Twiss parameters; NULL pointer if the
           argument is not a structure mxArray containing fields which are
			  Twiss parameters, and numeric.
   ABORT:  never.
   FAIL:   never.                                                         */

double *GetTwissUnpackTwiss( const mxArray* TwissStructure )
{
   static double twissvals[5] ;
	mxArray *FieldPtr ;
	double *DataPtr ;

/* if the object ain't a matlab structure, fail */

	if (!mxIsStruct(TwissStructure))
		return NULL ;

/* get the betatron function if possible */

	FieldPtr = mxGetField(TwissStructure,0,"beta") ;
	if (FieldPtr == NULL)
		return NULL ;
	if (!mxIsDouble(FieldPtr))
		return NULL ;
	DataPtr = mxGetPr( FieldPtr ) ;
	if (DataPtr == NULL)
		return NULL ;
	twissvals[0] = DataPtr[0] ;
/* one extra check -- beta must be positive-definite */
	if (twissvals[0] <= 0.)
		return NULL ;

/* get alpha if possible */

	FieldPtr = mxGetField(TwissStructure,0,"alpha") ;
	if (FieldPtr == NULL)
		return NULL ;
	if (!mxIsDouble(FieldPtr))
		return NULL ;
	DataPtr = mxGetPr( FieldPtr ) ;
	if (DataPtr == NULL)
		return NULL ;
	twissvals[1] = DataPtr[0] ;

/* get eta if possible */

	FieldPtr = mxGetField(TwissStructure,0,"eta") ;
	if (FieldPtr == NULL)
		return NULL ;
	if (!mxIsDouble(FieldPtr))
		return NULL ;
	DataPtr = mxGetPr( FieldPtr ) ;
	if (DataPtr == NULL)
		return NULL ;
	twissvals[2] = DataPtr[0] ;

/* get eta' if possible */

	FieldPtr = mxGetField(TwissStructure,0,"etap") ;
	if (FieldPtr == NULL)
		return NULL ;
	if (!mxIsDouble(FieldPtr))
		return NULL ;
	DataPtr = mxGetPr( FieldPtr ) ;
	if (DataPtr == NULL)
		return NULL ;
	twissvals[3] = DataPtr[0] ;

/* get nu if possible */

	FieldPtr = mxGetField(TwissStructure,0,"nu") ;
	if (FieldPtr == NULL)
		return NULL ;
	if (!mxIsDouble(FieldPtr))
		return NULL ;
	DataPtr = mxGetPr( FieldPtr ) ;
	if (DataPtr == NULL)
		return NULL ;
	twissvals[4] = DataPtr[0] ;

/* now return */

	return twissvals ;
}

/*========================================================================*/

/* Procedure ReverseTwissDirection -- take the final twiss parameters and
   the region of interest and determine the correct starting parameters. */

/* RET:    None.
   ABORT:  Never.
   FAIL:           */

void ReverseTwissDirection( struct RmatArgStruc* ArgStruc )
{
	mxArray* plhs_rmat[2] ;              /* for RmatAtoB call argument */
	mxArray* prhs_rmat[2] ;              /* return from RmatAtoB call  */
	mxArray* statarray    ;
	int statvalue         ;
	mxArray* plhs_inv[1]  ;              /* return from inv call       */
	double*  rmat_inv     ;
	struct beta0 betai    ;
	struct beta0 betaf    ;
	Rmat R_inv     ;
	int i,j               ;

/* get the matrix from the start to the end by doing a call to RmatAtoB: */

	prhs_rmat[0] = CREATE_SCALAR_DOUBLE( (double)(ArgStruc->start) ) ;
	prhs_rmat[1] = CREATE_SCALAR_DOUBLE( (double)(ArgStruc->end)   ) ;
	statvalue = mexCallMATLAB( 2, plhs_rmat, 2, prhs_rmat, "RmatAtoB" ) ;

/* get return status from RmatAtoB from the first return argument */

	statarray = mxGetCell(plhs_rmat[0],0) ;
	statvalue = (int)(*(mxGetPr(statarray))) ;
	if (statvalue != 1)
	{
		ArgStruc->Status = 0 ;
		goto egress ;
	}

/* if the return was okay, get the matrix and invert it */

	statvalue = mexCallMATLAB( 1, plhs_inv, 1, &(plhs_rmat[1]), "inv" ) ;
	rmat_inv = mxGetPr( plhs_inv[0] ) ;

/* since Matlab stores arrays in FORTRAN-order, not C-order,
   correct the order of the R-matrix terms */

	for (i=0 ; i<6 ; i++)
		for (j=0 ; j<6 ; j++)
			R_inv[i][j] = *(rmat_inv+6*j+i) ;

/* Uncoupled Twiss case: */

	if (ArgStruc->PropagateTwiss == 1)
	{

/* construct the "initial" Twiss, reversing the signs of the alpha and
   etap terms */

		betai.betx =   ArgStruc->InitialTwiss.betx  ;
		betai.alfx =   ArgStruc->InitialTwiss.alfx  ;
		betai.etax =   ArgStruc->InitialTwiss.etax  ;
		betai.etapx =  ArgStruc->InitialTwiss.etapx ;
		betai.nux   = 0 ;
		betai.bety =   ArgStruc->InitialTwiss.bety  ;
		betai.alfy =   ArgStruc->InitialTwiss.alfy  ;
		betai.etay =   ArgStruc->InitialTwiss.etay  ;
		betai.etapy =  ArgStruc->InitialTwiss.etapy ;
		betai.nuy   = 0 ;

/* do the propagation */

		statvalue = TwissThruRmat( R_inv, &betai, &betaf ) ;

/* did we make it through OK?  */

		if (statvalue != 1)
		{
			BadInverseTwissMessage( ArgStruc->end , ArgStruc->start ) ;
			ArgStruc->Status = 0 ;
			goto egress ;
		}

/* if we got here, set the "final" twiss as initial, again reversing the
   relevant signs */

		ArgStruc->InitialTwiss.betx =   betaf.betx  ;
		ArgStruc->InitialTwiss.alfx =   betaf.alfx  ;
		ArgStruc->InitialTwiss.etax =   betaf.etax  ;
		ArgStruc->InitialTwiss.etapx =  betaf.etapx ;

		ArgStruc->InitialTwiss.bety =   betaf.bety  ;
		ArgStruc->InitialTwiss.alfy =   betaf.alfy  ;
		ArgStruc->InitialTwiss.etay =   betaf.etay  ;
		ArgStruc->InitialTwiss.etapy =  betaf.etapy ;

	}

	if (ArgStruc->PropagateTwiss == 2)
	{

/* allocate a Matlab array for the new initial parameters */
    const size_t nwd=ArgStruc->nWolskiDimensions ;
		const size_t dims[3]={6, 6, nwd } ;
		int dloop ; 
		double *WolskiInitPtr ;

		WolskiInit = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL) ;
		WolskiInitPtr = mxGetPr(WolskiInit) ;

/* loop over dimensions and perform the coupled Twiss propagation */

		for (dloop = 0 ; dloop<ArgStruc->nWolskiDimensions ; dloop++)
		{
			statvalue = CoupledTwissThruRmat( R_inv, 
				&(ArgStruc->InitialWolski[36*dloop]),
				&WolskiInitPtr[36*dloop], 1 ) ;

/* did we make it through OK?  */

			if (statvalue != 1)
			{
				BadInverseTwissMessage( ArgStruc->end , ArgStruc->start ) ;
				ArgStruc->Status = 0 ;
				goto egress ;
			}

		}

/* set the intial values in the ArgStruc array to the values just
   returned by the CoupledTwissThruRmat propagator */

		ArgStruc->InitialWolski = WolskiInitPtr ;

	}


egress:

/* cleanup */

	mxDestroyArray( plhs_rmat[0] ) ;
	mxDestroyArray( plhs_rmat[1] ) ;
	mxDestroyArray( prhs_rmat[0] ) ;
	mxDestroyArray( prhs_rmat[1] ) ;
	mxDestroyArray( plhs_inv[0]  ) ;


	return ;
}

/*========================================================================*/

/* Procedure GetTwissSetReturn -- packages data from a call of GetTwiss into
   the return variable.  */

/* RET:    None.
   ABORT:  if creation of the mxArray fails.
   FAIL:   if dim(plhs) < 2; this should have been addressed in 
           GetTwissGetCheckArgs.  */

void GetTwissSetReturn( int Status, mxArray* plhs[], struct twiss* T )
{

	mxArray *ReturnStruc ;         /* the thing we want to return */
	mxArray *mxrealvec ;           /* vector of reals */
	double *mxrealp ;              /* pointer into real vectors */
	static const char *fieldname[] = {   /* 12 field names */
	"S","P","betax","alphax","etax","etapx","nux",
           "betay","alphay","etay","etapy","nuy"
	} ;
	int i ;
	

	char** messages ; 
	int nmsg ;
	

/* Set the return status and any messages */

   messages = GetAndClearMessages( &nmsg ) ;
	plhs[0] = CreateStatusCellArray( Status, nmsg, messages ) ;

/* if the T pointer is NULL, then there's no point even trying to
   unpack the data structure because it doesn't exist.  In that case,
	head for the exit now. */

	if (T == NULL)
	{
		Status = 1 ;
		plhs[1] = mxCreateCellMatrix( 0 , 0 ) ;
		goto egress ;
	}

/* we can now use Status for the local status of this function */

	Status = 0 ;

/* get a Matlab structure mxArray for the return */

	ReturnStruc = mxCreateStructMatrix( 1, 1, 12, fieldname ) ;
	if (ReturnStruc == NULL)
		goto egress ;

/* now we can get somewhat repetitive:  for each of the 12 fields, 
   allocate a double vector up to T->nentry entries and fill it. 
	As we do so, deallocate the allocated vectors. */

/* S */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->S[i] ;
	mxSetField( ReturnStruc, 0, fieldname[0], mxrealvec ) ;
/* P */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->E[i] ;
	mxSetField( ReturnStruc, 0, fieldname[1], mxrealvec ) ;
/* betx */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].betx ;
	mxSetField( ReturnStruc, 0, fieldname[2], mxrealvec ) ;
/* alfx */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].alfx ;
	mxSetField( ReturnStruc, 0, fieldname[3], mxrealvec ) ;
/* etax */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].etax ;
	mxSetField( ReturnStruc, 0, fieldname[4], mxrealvec ) ;
/* etapx */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].etapx ;
	mxSetField( ReturnStruc, 0, fieldname[5], mxrealvec ) ;
/* nux */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].nux ;
	mxSetField( ReturnStruc, 0, fieldname[6], mxrealvec ) ;
/* bety */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].bety ;
	mxSetField( ReturnStruc, 0, fieldname[7], mxrealvec ) ;
/* alfy */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].alfy ;
	mxSetField( ReturnStruc, 0, fieldname[8], mxrealvec ) ;
/* etay */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].etay ;
	mxSetField( ReturnStruc, 0, fieldname[9], mxrealvec ) ;
/* etapy */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].etapy ;
	mxSetField( ReturnStruc, 0, fieldname[10], mxrealvec ) ;
/* nuy */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->TwissPars[i].nuy ;
	mxSetField( ReturnStruc, 0, fieldname[11], mxrealvec ) ;

	plhs[1] = ReturnStruc ;
	Status = 1 ; /* if we got here then we're golden */

/* return and exit */

egress:

/* if T is present, free its bits and pieces */

	if (T != NULL)
	{
		free(T->S) ;
		free(T->E) ;
		free(T->TwissPars) ;
	}

	if (Status == 0)
		mexErrMsgTxt("Unable to allocate return structure for GetTwiss") ;

	return  ;

}


/*========================================================================*/

/* Procedure GetTwissCoupledSetReturn -- packages data from a call 
   of GetTwiss into the return variable if the call propagated the
	coupled twiss parameters.  */

/* RET:    None.
   ABORT:  if creation of the mxArray fails.
   FAIL:   if dim(plhs) < 2; this should have been addressed in 
           GetTwissGetCheckArgs.  */

void GetTwissCoupledSetReturn( int Status, 
										 int nDims, 
										  mxArray* plhs[], 
										  struct Ctwiss* T )
{

	mxArray *ReturnStruc ;         /* the thing we want to return */
	mxArray *mxrealvec ;           /* subsidiary mxArray for ReturnStruc */
	double *mxrealp ;              /* pointer into real vectors */
	int i ;
	size_t dims[4]={0,0,0,0} ;
	static const char *fieldname[] = {   /* 3 field names */
		"S","P","beta"} ;	

	char** messages ; 
	int nmsg ;
	

/* Set the return status and any messages */

   messages = GetAndClearMessages( &nmsg ) ;
	plhs[0] = CreateStatusCellArray( Status, nmsg, messages ) ;

/* if the T pointer is NULL, then there's no point even trying to
   unpack the data structure because it doesn't exist.  In that case,
	head for the exit now. */

	if (T == NULL)
	{
		Status = 1 ;
		plhs[1] = mxCreateCellMatrix( 0 , 0 ) ;
		goto egress ;
	}

/* we can now use Status for the local status of this function */

	Status = 0 ;

/* get a Matlab structure mxArray for the return */

	dims[0] = 6 ; 
	dims[1] = 6 ;
	dims[2] = nDims ;
	dims[3] = T->nentry ;
	ReturnStruc = mxCreateStructMatrix( 1, 1, 3, fieldname ) ;

/* set the values of the real vectors now: */

/* S */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->S[i] ;
	mxSetField( ReturnStruc, 0, fieldname[0], mxrealvec ) ;
	
/* P */
	mxrealvec = mxCreateDoubleMatrix(1,T->nentry,mxREAL) ;
	if (mxrealvec == NULL)
		goto egress ;

	mxrealp = mxGetPr( mxrealvec ) ;
	if (mxrealp == NULL)
		goto egress ;

	for (i=0 ; i<T->nentry ; i++ )
		mxrealp[i] = T->E[i] ;
	mxSetField( ReturnStruc, 0, fieldname[1], mxrealvec ) ;

/* now for the multi-dimensional Twiss array */	

	dims[0] = 6 ; 
	dims[1] = 6 ;
	dims[2] = nDims ;
	dims[3] = T->nentry ;
	mxrealvec = mxCreateNumericArray( 4, dims, mxDOUBLE_CLASS, mxREAL) ;
	mxrealp = mxGetPr( mxrealvec ) ;

/* fill the array with data */

	for (i=0 ; i<6*6*nDims*T->nentry ; i++)
		mxrealp[i] = T->Twiss[i] ;
	mxSetField( ReturnStruc, 0, fieldname[2], mxrealvec ) ;

/* set return variable */

	plhs[1] = ReturnStruc ;
	Status = 1 ;

/* cleanup and exit */

egress:

	if (T != NULL)
	{
	  FreeAndNull((void**)&(T->S)) ;
	  FreeAndNull((void**)&(T->E)) ;
	  FreeAndNull((void**)&(T->Twiss)) ;
	}

	if (WolskiInit != NULL)
	{
		mxDestroyArray(WolskiInit) ;
		WolskiInit = NULL ;
	}

	return ;

}

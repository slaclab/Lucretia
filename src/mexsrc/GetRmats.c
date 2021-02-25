/*    This file contains the top-level source for the Matlab version of
      mexfile GetRmats (the octave version will be in LucretiaOctave.c).
	  
      Matlab usage:

	     [stat,R] = GetRmats( start, finish ) ;

      where

         stat = cell matrix of status integer + messages

         R = 1 x (|finish-start|+1) structure of 6x6 arrays, containing
		     the R-matrices of the various elements.
         start, finish = the start and finish of the beamline region
		 which are to be computed.  If finish < start GetRmats will loop
		 backwards.
		 
		 GetRmats can also be called with one argument, the string
		 "version".  In this case, GetRmats returns its own version
		 date, as well as the version date of LucretiaCommon,
		 LucretiaMatlab, and LucretiaPhysics files.  

		 Also included in this file:  GetRmatsCheckArgs and 
		 GetRmatsSetReturn, which process the calling args and the
		 return args, respectively.

      AUTH:  PT, 02-aug-2004
	   MOD:

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
#include "LucretiaGlobalAccess.h" 
#include <string.h>
#include <stdlib.h>

/* File-scoped variables */

char GetRmatsVersion[] = "GetRmats Matlab version = 18-Apr-2005" ;

/*========================================================================*/

/* Gateway procedure for computing each R-matrix over a range */

/* RET:    none. 
   ABORT:  never.
   FAIL:   never.                                                         */

void mexFunction( int nlhs, mxArray *plhs[],
				  int nrhs, const mxArray *prhs[] ) 
{

   struct RmatArgStruc* RmatArgs ; /* arguments used by Rmat operation */
   char BadArgs[] = "Improper argument format for GetRmats" ;
	struct Rstruc* Rmats = NULL ; /* capture pointer to data structure */ 

/* Check the input arguments and make sure that they are OK.  Figure
   out whether this is a "version" call or a real request for rmats.
   If the latter make sure that the BEAMLINE array is present and is
   large enough. Return a pointer to an appropriate argument structure. */

   RmatArgs = GetRmatsGetCheckArgs( nlhs, nrhs, prhs ) ;

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
		plhs[0] = LucretiaMatlabVersions( GetRmatsVersion ) ;
		return ;
	}

   else /* do some real math! */
   {
     Rmats = (struct Rstruc *)RmatCalculate( RmatArgs ) ;

	  if (Rmats == NULL) 
		  AddMessage("Dynamic allocation failure in RmatCalculate",1) ;
   }

/* and that's it! */

egress:

	GetRmatsSetReturn( RmatArgs->Status, plhs, Rmats ) ;
	return ;

}

/*========================================================================*/

/* Procedure GetRmatsGetCheckArgs:  make sure that GetRmats was called with
   good arguments.  Configure a data structure to convey proper arguments
	to RmatCalculate.  While we're at it, perform some general Lucretia
	initialization. */

/* RET:    pointer to an RmatArgStruc.
   ABORT:  never.
   FAIL:   never.                         */

struct RmatArgStruc* GetRmatsGetCheckArgs( int nlhs, int nrhs, 
														 const mxArray* prhs[] ) 
{

	static struct RmatArgStruc ArgStruc ;    /* the object of interest */
	char prhsString[8] ;              /* the string arg, if any */
	int nElemTot ;                    /* # elts in BEAMLINE cell array */

/* initialize the ArgStruc fields: */

	ArgStruc.start = 0 ;
	ArgStruc.end = 0 ;
	ArgStruc.ReturnEach = 1 ;     /* we want all the rmats, not just 1 */
	ArgStruc.PropagateTwiss = 0 ;
	ArgStruc.Version = 0 ;
	ArgStruc.Status = 0 ;         /* assume very bad status for now */

/* There are two valid ways to execute GetRmats from the Lucretia command
   line:  
	
	  VersionCell = GetRmats('version') 

	will return a cell array with version information, while

		[Stat,Rstruc] = GetRmats(first,last)

   will return a status cell matrix plus a
	structure array full of 6x6 matrices.  Obviously if there
	are fewer than 1 or more than 2 RHS arguments, something is wrong. */

	if ( (nrhs < 1) || (nrhs > 2) )
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

/* now for the case of 2 arguments */

	if ( nrhs == 2 )
	{

/* for real math, we need 2 return arguments.  If we don't have them,
   error exit! */

		if ( nlhs != 2 )
			goto Egress ;

/* each of the 2 RHS arguments should be a scalar integer -- check that now */

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

/* make sure that the start <= the finish */

		if (ArgStruc.start > ArgStruc.end)
			goto Egress ;

/* if we made it this far, we must be successful.  Set success indicator */

		ArgStruc.Status = 1 ;

	}

/* once we've gotten here, all we need to do is exit! */

Egress:

	return &ArgStruc ;

}

/*========================================================================*/

/* Procedure GetRmatsSetReturn -- packages data from a GetRmats call into
   the return variable.  */

/* RET:    None.
   ABORT:  if creation of the mxArray fails.
   FAIL:   if dim(plhs) < 2; this should have been addressed in 
           GetRmatsGetCheckArgs.  */

void GetRmatsSetReturn( int Status, 
							   mxArray* plhs[], 
								const struct Rstruc* Rmats )
{


	mxArray *ReturnStruc ;         /* the thing we want to return */
	const char *fieldname ;             /* name of field = "RMAT" */
	char rmatname[5] = "RMAT" ;   
	int count ;
	mxArray *matrix6x6 ;           /* pointer to 6x6 numeric matrix */
	double *mxrealp ;              /* pointer into matrix6x6 */
	int i,j ;
	char** messages ; 
	int nmsg ;

/* Set the return status and any messages */

   messages = GetAndClearMessages( &nmsg ) ;
	plhs[0] = CreateStatusCellArray( Status, nmsg, messages ) ;

/* if the Rmats pointer is NULL, then there's no point even trying to
   unpack the data structure because it doesn't exist.  In that case,
	head for the exit now. */

	if (Rmats == NULL)
	{
		Status = 1 ;
		plhs[1] = mxCreateCellMatrix( 0 , 0 ) ;
		goto egress ;
	}

/* we can now use Status for the local status of this function */

	Status = 0 ;

/* get a Matlab structure mxArray for the return */

	fieldname = rmatname ;
	ReturnStruc = mxCreateStructMatrix( 1, Rmats->Nmatrix, 1, &fieldname ) ;
	if (ReturnStruc == NULL)
		goto egress ;

/* loop over entries in the RStruc */

	for (count=0 ; count<Rmats->Nmatrix ; count++)
	{

/* get a 6x6 numeric matrix mxArray */

		matrix6x6 = mxCreateDoubleMatrix(6,6,mxREAL) ;
		if (matrix6x6 == NULL)
			goto egress ;

/* get the pointer to matrix6x6's real entries */

		mxrealp = mxGetPr( matrix6x6 ) ;
		if (mxrealp == NULL)
			goto egress ;

/* fill the matrix with values from Rmats.matrix.  Bear in mind that 
   Rmats.matrix stores entries row-wise (ie, R(1,2) follows R(1,1)) but
	Matlab stores them column-wise (ie, R(2,1) follows R(1,1)).  So some
	index arithmetic is required... */

		for (i=0 ; i<6 ; i++)
		{
			for (j=0 ; j<6 ; j++)
			{
				mxrealp[j+i*6] = Rmats->matrix[36*count+i+j*6] ;
			}
		}

/* hook the resulting matrix up to the data structure mxArray */

		mxSetField( ReturnStruc, count, fieldname, matrix6x6 ) ;

	}

	plhs[1] = ReturnStruc ;
	Status = 1 ; /* if we got here we're golden */

egress:

/* since we dynamically allocated Rmats.matrix, deallocate it now. */

	if (Rmats != NULL)
		free(Rmats->matrix) ; 
	if (Status == 0)
		mexErrMsgTxt("Unable to construct return matrix:  GetRmats") ;

/* and that's it! */

	return  ;

}


/* This file contains the Matlab gateway to the Lucretia
	lattice verifier, VerifyLattice(), in LucretiaCommon.c.

   Matlab usage:

		[stat,err,warn,info] = VerifyLattice( ) ;

	where

		stat = cell array of status flag and messages
		err  = cell array of error messages from verification
		warn = cell array of warning messages from verification.
		info = cell array of informational messages from verification.

	VerifyLattice can also be called thus:

		ver = VerifyLattice('version') ;

	in which case it returns a cell array of text strings which
	contain the Lucretia version dates.

	AUTH:  PT, 13-jan-2005
	MOD:

========================================================================*/

/* include files */

#include <string.h>
#ifndef mex_h
  #define mex_h
  #include <mex.h>
#endif
#include "matrix.h"
#include "LucretiaMatlab.h"
#ifndef LUCRETIA_COMMON
  #include "LucretiaCommon.h"   /* Matlab/Octave common fns */
#endif
#include "LucretiaGlobalAccess.h" 

/* File-scoped variables */

const char* VerifyLatticeVersion = "VerifyLattice Matlab version = 18-Apr-2005" ;

/*========================================================================*/

/* Gateway procedure for lattice verification */

/* RET:    none. 
   ABORT:  never.
   FAIL:   never.                                                         */

void mexFunction( int nlhs, mxArray *plhs[],
				  int nrhs, const mxArray *prhs[] ) 
{
	int ReturnStatus = 1 ;
	int nWarn, nError, nInfo, nmsg, nmisc ;
	int nElemTot, count ;
	char prhsString[8] ;
	char** messages ;
	char** errors, **warnings, **info, **misc ;

/* There are only 2 ways this can be called: 
 
		For actual verification, nlhs = 3 and nrhs = 0 ;
		for version information, nlhs = 1 and nrhs = 1 ;

	if neither of those describe the present set of arguments, set bad
	status and goto exit */

	if ( (nrhs==1) &&
		  ( (nlhs==1) || (nlhs==0) ) ) /* possible version call */
	{
		if ( (!mxIsChar(prhs[0])) || 
			  (mxGetM(prhs[0])!=1) ||
			  (mxGetN(prhs[0])!=7)    )
		{
			ReturnStatus = 0 ;
			AddMessage( "Improper argument format for VerifyLattice",1) ;
		   goto Finish ;
		}
		mxGetString( prhs[0] , prhsString, 8 ) ;
		if ( strcmp(prhsString,"version") !=0 )
		{
			ReturnStatus = 0 ;
			AddMessage( "Improper argument format for VerifyLattice",1) ;
		   goto Finish ;
		}

/* if we got here, then it must be a proper "version" call */

		plhs[0] = LucretiaMatlabVersions( VerifyLatticeVersion ) ;
		return ;
	}

/* if we're still here then a "version" call has been ruled out.  Are
   args ok for a verification call? */

	if ( (nlhs != 4) || (nrhs!=0) )
	{
		ReturnStatus = 0 ;
		AddMessage( "Improper argument format for VerifyLattice",1) ;
		goto Finish ; 
	}

/* If the args are OK for a verification call, execute LucretiaMatlabSetup
   to set global values of various things.  If LucretiaMatlabSetup does
	not execute properly, it generates its own error messages */

	nElemTot = LucretiaMatlabSetup( ) ;
	if (nElemTot < 1)
	{
		ReturnStatus = 0 ;
		goto Finish ;
	}

/* If we got here, then we can execute the verifier... */

	VerifyLattice( ) ;

/* ...and proceed to the exit! */

Finish:

/*	Get the message stack */ 

	messages = GetAndClearMessages( &nmsg ) ;

/* if we got here with bad status, then all the messages should go into
   the status list */

	if (ReturnStatus == 0)
	{
		plhs[0] = CreateStatusCellArray( ReturnStatus, nmsg, messages ) ;
		plhs[1] = mxCreateCellMatrix( 0 , 0 ) ;
		plhs[2] = mxCreateCellMatrix( 0 , 0 ) ;
		plhs[3] = mxCreateCellMatrix( 0 , 0 ) ;
		return ;
	}

/* otherwise we need to segregate the various types of messages */

	nWarn = 0 ;
	nError = 0 ;
	nInfo = 0 ;
	for (count=0 ; count<nmsg ; count++)
	{
		if (strncmp(messages[count],"E",1)==0)
			nError++ ;
		if (strncmp(messages[count],"W",1)==0)
			nWarn++ ;
		if (strncmp(messages[count],"I",1)==0)
			nInfo++ ;
	}
	nmisc = nmsg - nError - nWarn - nInfo ;

/* allocate backbones to handle the 3 message types */

	errors   = (char**) mxMalloc( nError * sizeof(char*) ) ;
	warnings = (char**) mxMalloc( nWarn  * sizeof(char*) ) ;
	misc     = (char**) mxMalloc( nmisc  * sizeof(char*) ) ;
	info     = (char**) mxMalloc( nInfo  * sizeof(char*) ) ;

/* move the messages to the 3 new backbones */

	nWarn = 0 ;
	nError = 0 ;
	nmisc = 0 ;
	nInfo = 0 ;
	for (count=0 ; count<nmsg ; count++)
	{
		if (strncmp(messages[count],"E",1)==0)
		{
			errors[nError] = messages[count] ;
			nError++ ;
		}
		else if (strncmp(messages[count],"W",1)==0)
		{
			warnings[nWarn] = messages[count] ;
			nWarn++ ;
		}
		else if (strncmp(messages[count],"I",1)==0)
		{
			info[nInfo] = messages[count] ;
			nInfo++ ;
		}
		else
		{
			misc[nmisc] = messages[count] ;
			nmisc++ ;
		}
	}

/* Set the backbones into the return variables */

	plhs[0] = CreateStatusCellArray( ReturnStatus, nmisc, misc ) ;
	plhs[1] = CreateStatusCellArray( nError, nError, errors ) ;
	plhs[2] = CreateStatusCellArray( nWarn, nWarn, warnings ) ;
	plhs[3] = CreateStatusCellArray( nInfo, nInfo, info ) ;

	return ;

	}

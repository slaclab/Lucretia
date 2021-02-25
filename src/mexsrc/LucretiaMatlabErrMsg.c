/* The functions in this file are all related to managing the Lucretia
   error message tree for the Matlab implementation. Contents:

	AddMessage 
	GetAndClearMessages 

   AUTH:  PT, 19-aug-2004
	MOD:

*/

#include <string.h>
//#ifndef mex_h
//  #define mex_h
  #include <mex.h>
//#endif
#include "LucretiaGlobalAccess.h"

/* file-scoped variables */

int Nmsg=0 ;                  /* number of messages */
char** MsgList ;              /* backbone for the messages themselves */

/*==================================================================*/

/* add a message to the present list */

/* RET:    none.
   ABORT:  failure of mxMalloc will lead to abort via mexErrMsgTxt
           function. 
   FAIL:   none. */

void AddMessage( const char* NewMessage, int display )
{

	char** NewMsgList ; 
	char*  NewMsgLcl ; 
	int i ;

/* Create a new backbone with one additional slot */

	NewMsgList = (char**)mxMalloc( (Nmsg+1) * sizeof(char*) ) ;
	if (NewMsgList == NULL)
		mexErrMsgTxt("Unable to allocate status message table!") ;

/* if there are old messages copy them over and release the old
	backbone */

	if (Nmsg > 0)
	{
		for (i=0 ; i<Nmsg ; i++)
			NewMsgList[i] = MsgList[i] ;
		mxFree(MsgList) ;

	}

/* point the global backbone at the new backbone */

	MsgList = NewMsgList ;

/* add the new message */ 

	NewMsgLcl = (char*)mxMalloc( (1+strlen(NewMessage)) * sizeof(char) ) ;
	if (NewMsgLcl == NULL)
		mexErrMsgTxt("Unable to allocate status message table!") ;
	NewMsgLcl = strcpy( NewMsgLcl, NewMessage ) ;
	MsgList[Nmsg] = NewMsgLcl ;

/* if the message is to be displayed in real time, do that now */

	if (display==1)
		mexWarnMsgTxt(MsgList[Nmsg]) ;
	Nmsg++ ;

/* and that's it. */

	return ;

}

/*==================================================================*/

/* Return the present message stack and clear same.  Since the stack
   was allocated using mxMalloc, we can let Matlab deallocate everything
	and just reset the message counter to zero. */

/* RET:    char**, backbone of the message stack.
   FAIL:   never. */

char** GetAndClearMessages( int* MsgCount )
{
	*MsgCount = Nmsg ;
	Nmsg = 0 ;

	return MsgList ;
}

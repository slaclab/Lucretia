#include "fintrf.h"

      SUBROUTINE PutElemParsToMatlab( XSIF_INDX, NPARAM, DPARAM, pDICT, 
     &                                FIELDNAME                        )
C
C     Puts the strings in DPARAM into a cell array which is then put into 
C     the matlab data structure pDICT.  XSIF_INDX tells which entry of
C     pDICT to use, while FIELDNAME tells which field in pDICT.  NPARAM
C     is the total number of entries in DPARAM.
C
C     AUTH:  PT, 25-FEB-2004
C
C     MOD:
C       19-FEB-2012, M. Woodley
C         Change type declarations of mx-functions from INTEGER*4
C         to mwPointer

      IMPLICIT NONE

C     Argument declarations

      INTEGER*4 XSIF_INDX, NPARAM
	CHARACTER*(*) DPARAM(NPARAM)
	mwPointer pDICT               ! POINTER to matlab data structure
	CHARACTER*(*) FIELDNAME

C     local declarations

        mwPointer pCELL               ! pointer to a cell array
	mwPointer pSTRING             ! pointer to a string
	INTEGER*4 LOOP_COUNT

C     argument declarations

      mwPointer mxCreateCellMatrix ! Create a Matlab cell data structure
	    mwPointer mxCreateString     ! create a Matlab string

C========1=========2=========3=========4=========5=========6=========7=C
C                                                                      C
C                  C  O  D  E                                          C
C                                                                      C
C========1=========2=========3=========4=========5=========6=========7=C

C
C     create a cell array with dimensions of NPARAM x 1
C
      pCELL = mxCreateCellMatrix( NPARAM, 1 )
	IF ( pCELL .EQ. 0 ) THEN
	    CALL mexErrMsgTxt(
     &'ERROR> Cannot create cell matrix in PutElemParsToMatlab')
	ENDIF

C
C     loop over the entries in DPARAM and copy them to the cell
C     array
C
      DO LOOP_COUNT = 1,NPARAM

	    pSTRING = mxCreateString( DPARAM(LOOP_COUNT) )
	    IF (pSTRING .EQ. 0) THEN
	        call mexErrMsgTxt(
     &'ERROR> Cannot create string in PutElemParsToMatlab')
	    ENDIF
	    CALL mxSetCell( pCELL, LOOP_COUNT, pSTRING )

	ENDDO
C
C     now set the correct field in the data structure to the values
C     in the cell array
C
      CALL mxSetField( pDICT, XSIF_INDX, FIELDNAME, pCELL )
      
      RETURN
      END       

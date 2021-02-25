      SUBROUTINE NEWLST(IHEAD)                                           
C
C     member of MAD INPUT PARSER
C
C---- GENERATE A NEW EMPTY LIST                                          
C----------------------------------------------------------------------- 
C
C	MOD:
C		 22-MAY-2003, PT:
C			support for dynamic allocation of ILDAT.
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_ELEMENTS
	USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N) 
      SAVE

	logical*4 alloc_ok, more_links
C
C----------------------------------------------------------------------- 
C---- RESERVE A "HEAD" CELL                                              
C      IF (IUSED .GE. MAXLST) CALL OVFLOW(3,MAXLST)                       
	IF (IUSED.GE.MAXLST) THEN
		ALLOC_OK = MORE_LINKS( MAXLST_STEP )  
	    IF (.NOT.ALLOC_OK) THEN
		  ERROR = .TRUE.
		  GOTO 9999
		ENDIF
	ENDIF           
      IUSED = IUSED + 1                                                  
      IHEAD = IUSED                                                      
      ILDAT(IHEAD,1) = 1                                                 
C---- STORE LINKS FOR EMPTY LIST                                         
      ILDAT(IHEAD,2) = IHEAD                                             
      ILDAT(IHEAD,3) = IHEAD                                             
C---- CLEAR REMAINING CELL WORDS                                         
      ILDAT(IHEAD,4) = 0                                                 
      ILDAT(IHEAD,5) = 0                                                 
      ILDAT(IHEAD,6) = 0                                                 
9999  RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                

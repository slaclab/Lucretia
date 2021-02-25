      SUBROUTINE NEWRGT(IOLD,INEW)                                       
C
C     member of MAD INPUT PARSER
C
C---- INSERT NEW CELL "INEW" TO THE RIGHT OF CELL "IOLD"                 
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

	LOGICAL*4 ALLOC_OK, MORE_LINKS
C
C----------------------------------------------------------------------- 
C---- RESERVE A "CALL" CELL                                              
      IPRE = IOLD                                                        
      ISUC = ILDAT(IOLD,3)                                               
c      IF (IUSED .GE. MAXLST) CALL OVFLOW(3,MAXLST)                       
	IF (IUSED.GE.MAXLST) THEN
		ALLOC_OK = MORE_LINKS( MAXLST_STEP )  
	    IF (.NOT.ALLOC_OK) THEN
		  ERROR = .TRUE.
		  GOTO 9999
		ENDIF
	ENDIF           
      IUSED = IUSED + 1                                                  
      INEW = IUSED                                                       
      ILDAT(INEW,1) = 2                                                  
C---- SET LINKS                                                          
      ILDAT(IPRE,3) = INEW                                               
      ILDAT(INEW,2) = IPRE                                               
      ILDAT(INEW,3) = ISUC                                               
      ILDAT(ISUC,2) = INEW                                               
C---- CLEAR REMAINING CELL WORDS                                         
      ILDAT(INEW,4) = 0                                                  
      ILDAT(INEW,5) = 0                                                  
      ILDAT(INEW,6) = 0                                                  
9999  RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                

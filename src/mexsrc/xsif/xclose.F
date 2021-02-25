      SUBROUTINE XCLOSE                                                  
C---- CLOSE A FILE                                                       
C----------------------------------------------------------------------- 
C
C     MOD:
C          14-MAY-2003, PT:
C             support use of the xsif open stack.
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
C----------------------------------------------------------------------- 
      CALL RDFILE(ICLOSE,ERROR)                                          
      IF (SCAN .OR. ERROR) RETURN                                        
c      CLOSE (UNIT = ICLOSE)     
      CALL XCLOSE_STACK_MANAGE( ICLOSE )                                         
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                

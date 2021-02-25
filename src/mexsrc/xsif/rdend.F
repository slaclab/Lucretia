      SUBROUTINE RDEND(IHERR)
C
C     member of MAD INPUT PARSER
C
C---- PRINT NUMBER OF ERROR MESSAGES GENERATED                           
C----------------------------------------------------------------------- 
C
C	MOD:
c          31-jan-2001, PT:
C             set FATAL_READ_ERROR to TRUE on error # 102.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
C----------------------------------------------------------------------- 
      IF (NWARN .NE. 0 .OR. NFAIL .NE. 0) THEN                           
        WRITE (IECHO,910)                                                
        WRITE (IECHO,920) NWARN                                          
        WRITE (IECHO,930) NFAIL                                          
        WRITE (IECHO,940)                                                
        WRITE (ISCRN,910)                                                
        WRITE (ISCRN,920) NWARN                                          
        WRITE (ISCRN,930) NFAIL                                          
        WRITE (ISCRN,940)                                                
      ENDIF                                                              
c      IF (SCAN) CALL HALT(IHERR) 
c     PT, 31-jan-2001
      IF (IHERR.EQ.102) FATAL_READ_ERROR = .TRUE.                                        
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('1**************************************')                  
  920 FORMAT(' * NUMBER OF WARNING MESSAGES =',I6,' *')                  
  930 FORMAT(' * NUMBER OF FATAL ERRORS     =',I6,' *')                  
  940 FORMAT(' **************************************')                  
C----------------------------------------------------------------------- 
      END                                                                

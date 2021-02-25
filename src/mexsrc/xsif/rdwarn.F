      SUBROUTINE RDWARN
C
C     member of MAD INPUT PARSER
C
C---- MARK PLACE OF WARNING LEVEL ERROR                                  
C----------------------------------------------------------------------- 
C
C	MOD:
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
      WRITE (IECHO,910) ILINE,KTEXT                                      
      WRITE (ISCRN,910) ILINE,KTEXT                                      
      WRITE (IECHO,920) IMARK,(' ',I=1,IMARK),'?'                        
      WRITE (ISCRN,920) IMARK,(' ',I=1,IMARK),'?'                        
      NWARN = NWARN + 1                                                  
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('0* LINE',I5,' * ',A80)                                     
  920 FORMAT(' * COLUMN',I3,' *',82A1)                                   
C----------------------------------------------------------------------- 
      END                                                                

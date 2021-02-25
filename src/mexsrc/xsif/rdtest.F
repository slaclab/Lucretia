      SUBROUTINE RDTEST(STRING,FLAG)
C
C     member of MAD INPUT PARSER
C
C---- NEXT INPUT CHARACTER MUST BE CONTAINED IN "STRING"                 
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
      CHARACTER*(*)    STRING                                            
      LOGICAL           FLAG                                             
C----------------------------------------------------------------------- 
      FLAG = .FALSE.                                                     
      IF (INDEX(STRING,KLINE(ICOL)) .EQ. 0) THEN                         
        CALL RDFAIL                                                      
        IF (LEN(STRING) .EQ. 1) THEN                                     
          WRITE (IECHO,910) STRING                                       
          WRITE (ISCRN,910) STRING                                       
        ELSE                                                             
          WRITE (IECHO,920) STRING                                       
          WRITE (ISCRN,920) STRING                                       
        ENDIF                                                            
        FLAG = .TRUE.                                                    
      ENDIF                                                              
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** "',A1,'" EXPECTED'/' ')                     
  920 FORMAT(' *** ERROR *** ONE OF "',A,'" EXPECTED'/' ')               
C----------------------------------------------------------------------- 
      END                                                                

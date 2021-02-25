      SUBROUTINE RDINT(IVALUE,FLAG)
C
C     member of MAD INPUT PARSER
C
C---- DECODE UNSIGNED INTEGER                                            
C----------------------------------------------------------------------- 
C
C	MOD:
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
CC
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      LOGICAL           FLAG                                             
C----------------------------------------------------------------------- 
      FLAG = .TRUE.                                                      
      IVALUE = 0                                                           
   10 IDIG = INDEX('0123456789',KLINE(ICOL)) - 1                         
      IF (IDIG .GE. 0) THEN                                              
        IVALUE = 10 * IVALUE + IDIG                                          
        FLAG = .FALSE.                                                   
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        GO TO 10                                                         
      ENDIF                                                              
      IF (FLAG) THEN                                                     
        CALL RDFAIL                                                      
        WRITE (IECHO,910)                                                
        WRITE (ISCRN,910)                                                
      ELSE IF (INDEX('.DE',KLINE(ICOL)) .NE. 0) THEN                     
        CALL RDSKIP('0123456789.E')                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        CALL RDFAIL                                                      
        WRITE (IECHO,920)                                                
        WRITE (ISCRN,920)                                                
        FLAG = .TRUE.                                                    
        IVALUE = 0                                                         
      ENDIF

9999  CONTINUE

                                                              
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** UNSIGNED INTEGER EXPECTED'/' ')             
  920 FORMAT(' *** ERROR *** REAL VALUE NOT PERMITTED'/' ')              
C----------------------------------------------------------------------- 
      END                                                                

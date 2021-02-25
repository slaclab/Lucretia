      SUBROUTINE RDSKIP(STRING)
C
C     member of MAD INPUT PARSER
C
C---- SKIP ANY CHARACTER(S) OCCURRING IN "STRING"                        
C
C----------------------------------------------------------------------- 
C
C	MOD:
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C
C----------------------------------------------------------------------- 
C
C     modules
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      CHARACTER*(*)     STRING                                           
C----------------------------------------------------------------------- 
   10 IF ((.NOT.ENDFIL) .AND. (INDEX(STRING,KLINE(ICOL)).NE.0)) THEN     
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        GO TO 10                                                         
      ENDIF       

9999  CONTINUE

                                                       
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                

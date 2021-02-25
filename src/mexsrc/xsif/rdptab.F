      SUBROUTINE RDPTAB(RVAL,NMAX,N,FAIL)
C
C     member of MAD INPUT PARSER
C
C---- READ A SET OF REAL VALUES                                          
C----------------------------------------------------------------------- 
C
C	MOD:
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
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
      DIMENSION         RVAL(NMAX)                                       
      LOGICAL           FAIL                                             
C----------------------------------------------------------------------- 
      N    = 0                                                           
      FAIL = .FALSE.                                                     
C---- MAIN LOOP                                                          
  100 CONTINUE                                                           
        CALL RDNUMB(RTEMP,FAIL)                                          
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (FAIL) RETURN                                                 
        CALL RDTEST('(/,;',FAIL)                                         
        IF (FAIL) RETURN                                                 
        IF (KLINE(ICOL) .EQ. '(') THEN                                   
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDNUMB(RSTEP,FAIL)                                        
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (FAIL) RETURN                                               
          IF (RSTEP .EQ. 0.0) THEN                                       
            CALL RDFAIL                                                  
            WRITE (IECHO,910)                                            
            WRITE (ISCRN,910)                                            
            FAIL = .TRUE.                                                
            RETURN                                                       
          ENDIF                                                          
          CALL RDTEST(')',FAIL)                                          
          IF (FAIL) RETURN                                               
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDNUMB(RLAST,FAIL)                                        
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (FAIL) RETURN                                               
          CALL RDTEST('/,;',FAIL)                                        
          IF (FAIL) RETURN                                               
          IMAX = NINT((RLAST - RTEMP) / RSTEP) + 1                       
        ELSE                                                             
          IMAX = 1                                                       
          RSTEP = 0.0                                                    
        ENDIF                                                            
C---- STORE REAL VALUE(S)                                                
        IF (N + IMAX .GT. NMAX) THEN                                     
          CALL RDFAIL                                                    
          WRITE (IECHO,920)                                              
          WRITE (ISCRN,920)                                              
          FAIL = .TRUE.                                                  
          RETURN                                                         
        ENDIF                                                            
        DO 190 I = 1, IMAX                                               
          N = N + 1                                                      
          RVAL(N) = RTEMP                                                
          RTEMP = RTEMP + RSTEP                                          
  190   CONTINUE                                                         
C---- ANOTHER VALUE?                                                     
      IF (KLINE(ICOL) .EQ. '/') THEN                                     
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
      ENDIF              

9999  CONTINUE

                                                
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** ZERO STEP NOT ALLOWED'/' ')                 
  920 FORMAT(' *** ERROR *** TOO MANY VALUES'/' ')                       
C----------------------------------------------------------------------- 
      END                                                                

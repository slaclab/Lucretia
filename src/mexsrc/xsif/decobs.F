      SUBROUTINE DECOBS(IELEM,IR1,IR2,ERROR1)                                    
C
C     part of MAD INPUT PARSER
C
C---- DECODE OBSERVATION POINT(S) OR RANGE                                      
C
C     MOD:
C           31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C           13-july-1998, PT:
C              replaced ERROR with ERROR1 as error flag for routine.
C
C----
C
C     modules
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
C
C-----------------------------------------------------------------------        
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      LOGICAL*4         ERROR1                                                   
C-----------------------------------------------------------------------        
      CHARACTER*8       KNAME                                                   
      LOGICAL           FLAG                                                    
C-----------------------------------------------------------------------        
      ERROR1 = .FALSE.                                                           
C---- NUMBERED POINT(S)                                                         
      IF (KLINE(ICOL) .EQ. '#') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        IELEM = 0                                                               
        IF (KLINE(ICOL) .EQ. 'S') THEN                                          
          CALL RDNEXT                                                           
        IF (FATAL_READ_ERROR) GOTO 9999
          IR1 = 0                                                               
        ELSE IF (KLINE(ICOL) .EQ. 'E') THEN                                     
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          IR1 = NELM                                                            
        ELSE                                                                    
          CALL RDINT(IR1,FLAG)                                                  
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (FLAG) GO TO 800                                                   
        ENDIF                                                                   
        IR2 = IR1                                                               
        IF (KLINE(ICOL) .EQ. '/') THEN                                          
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (KLINE(ICOL) .EQ. 'S') THEN                                        
            CALL RDNEXT                                                         
            IF (FATAL_READ_ERROR) GOTO 9999
            IR2 = 0                                                             
          ELSE IF (KLINE(ICOL) .EQ. 'E') THEN                                   
            CALL RDNEXT                                                         
            IF (FATAL_READ_ERROR) GOTO 9999
            IR2 = NELM                                                          
          ELSE                                                                  
            CALL RDINT(IR2,FLAG)                                                
            IF (FATAL_READ_ERROR) GOTO 9999
            IF (FLAG) GO TO 800                                                 
          ENDIF                                                                 
        ENDIF                                                                   
C---- NAMED POINT(S) OR RANGE                                                   
      ELSE                                                                      
        CALL RDWORD(KNAME,LNAME)                                                
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LNAME .EQ. 0) THEN                                                  
          CALL RDFAIL                                                           
          WRITE (IECHO,910)                                                     
          WRITE (ISCRN,910)                                                     
          GO TO 800                                                             
        ENDIF                                                                   
        CALL RDLOOK(KNAME,8,KELEM,1,IELEM1,IELEM)                               
        IF (IELEM .EQ. 0) THEN                                                  
          CALL RDFAIL                                                           
          WRITE (IECHO,920) KNAME(1:LNAME)                                      
          WRITE (ISCRN,920) KNAME(1:LNAME)                                      
          GO TO 800                                                             
        ENDIF                                                                   
        IR1 = 1                                                                 
        IR2 = NELM                                                              
        IF (KLINE(ICOL) .EQ. '[') THEN                                          
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDINT(IR1,FLAG)                                                  
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (FLAG) GO TO 800                                                   
          IR2 = IR1                                                             
          IF (KLINE(ICOL) .EQ. '/') THEN                                        
            CALL RDNEXT                                                         
            IF (FATAL_READ_ERROR) GOTO 9999
            CALL RDINT(IR2,FLAG)                                                
            IF (FATAL_READ_ERROR) GOTO 9999
            IF (FLAG) GO TO 800                                                 
          ENDIF                                                                 
          CALL RDTEST(']',FLAG)                                                 
          IF (FLAG) GO TO 800                                                   
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
        ENDIF                                                                   
      ENDIF                                                                     
C---- CHECK ORDER OF INDICES                                                    
      IF (IR1 .GT. IR2) THEN                                                    
        CALL RDFAIL                                                             
        WRITE (IECHO,930)                                                       
        WRITE (ISCRN,930)                                                       
        GO TO 800                                                               
      ENDIF                                                                     
      RETURN                                                                    
C---- ERROR EXIT --- RANGE UNDEFINED                                            
  800 ERROR1 = .TRUE.                                                            
      IELEM = 0                                                                 
      IR1 = 0                                                                   
      IR2 = 0    

9999  IF (FATAL_READ_ERROR) ERROR1=.TRUE.
                                                               
      RETURN                                                                    
C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** "#" OR NAME EXPECTED'/' ')                         
  920 FORMAT(' *** ERROR *** UNKNOWN BEAM LINE OR ELEMENT "',A,'"'/' ')         
  930 FORMAT(' *** ERROR *** BAD INDEX ORDER'/' ')                              
C-----------------------------------------------------------------------        
      END                                                                       

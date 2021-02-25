      SUBROUTINE DECPNT(IELEM,IR,IPOS,SCAN1,ERROR1)                               
C
C     part of MAD INPUT PARSER
C
C---- DECODE SINGLE OBSERVATION POINT                                           
C
C     MOD:
C          15-DEC-2003, PT:
C             expand element names to 16 characters.
C         02-MAR-2001, PT:
C             replaced NELM with NELM_XSIF
C           31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C           13-july-1998, PT:
C              replaced ERROR with ERROR1 as error flag for routine.
C              replaced SCAN with SCAN1.
C
C----
C
C     modules
C
      USE XSIF_SIZE_PARS
      USE XSIF_ELEMENTS
      USE XSIF_INOUT
C
C-----------------------------------------------------------------------        
      IMPLICIT REAL*8(A-H,O-Z), INTEGER*4 (I-N) 
      SAVE
C
      LOGICAL*4         SCAN1,ERROR1                                              
C-----------------------------------------------------------------------        
      CHARACTER(ENAME_LENGTH) KNAME                                                   
      LOGICAL*4         FLAG                                                    
C-----------------------------------------------------------------------        
      ERROR1 = .FALSE.                                                           
C---- NUMBERED POINT                                                            
      IF (KLINE(ICOL) .EQ. '#') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        IELEM = 0                                                               
        IF (KLINE(ICOL) .EQ. 'S') THEN                                          
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          IR = 0                                                                
        ELSE IF (KLINE(ICOL) .EQ. 'E') THEN                                     
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          IR = NELM_XSIF                                                             
        ELSE                                                                    
          CALL RDINT(IR,FLAG)                                                   
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (FLAG) GO TO 800                                                   
        ENDIF                                                                   
C---- NAMED POINT                                                               
      ELSE                                                                      
        CALL RDWORD(KNAME,LNAME)                                                
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LNAME .EQ. 0) THEN                                                  
          CALL RDFAIL                                                           
          WRITE (IECHO,910)                                                     
          WRITE (ISCRN,910)                                                     
          GO TO 800                                                             
        ENDIF                                                                   
        CALL RDLOOK(KNAME,ENAME_LENGTH,KELEM,1,IELEM1,IELEM)                               
        IF (IELEM .EQ. 0) THEN                                                  
          CALL RDFAIL                                                           
          WRITE (IECHO,920) KNAME(1:LNAME)                                      
          WRITE (ISCRN,920) KNAME(1:LNAME)                                      
          GO TO 800                                                             
        ENDIF                                                                   
        IF (IETYP(IELEM) .LE. 0) THEN                                           
          CALL RDFAIL                                                           
          WRITE (IECHO,930) KNAME(1:LNAME)                                      
          WRITE (ISCRN,930) KNAME(1:LNAME)                                      
          GO TO 800                                                             
        ENDIF                                                                   
        IR = 1                                                                  
        IF (KLINE(ICOL) .EQ. '[') THEN                                          
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDINT(IR,FLAG)                                                   
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (FLAG) GO TO 800                                                   
          CALL RDTEST(']',FLAG)                                                 
          IF (FLAG) GO TO 800                                                   
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
        ENDIF                                                                   
      ENDIF                                                                     
C---- FIND ACTUAL POSITION                                                      
      IF (SCAN1 .OR. ERROR1 .OR. (.NOT. PERI)) RETURN                             
      IF (IELEM .NE. 0) THEN                                                    
        IEPOS = 0                                                               
        DO 10 JPOS = NPOS1, NPOS2                                               
          IF (ITEM(JPOS) .EQ. IELEM) THEN                                       
            IEPOS = IEPOS + 1                                                   
            IF (IEPOS .EQ. IR) THEN                                             
              IPOS = JPOS                                                       
              RETURN                                                            
            ENDIF                                                               
          ENDIF                                                                 
   10   CONTINUE                                                                
      ELSE IF (IR .EQ. 0) THEN                                                  
        DO 20 JPOS = NPOS1, NPOS2                                               
          IF (ITEM(JPOS) .LE. MAXELM) THEN                                      
            IPOS = JPOS - 1                                                     
            RETURN                                                              
          ENDIF                                                                 
   20   CONTINUE                                                                
      ELSE                                                                      
        IEPOS = 0                                                               
        DO 30 JPOS = NPOS1, NPOS2                                               
          IF (ITEM(JPOS) .LE. MAXELM) THEN                                      
            IEPOS = IEPOS + 1                                                   
            IF (IEPOS .EQ. IR) THEN                                             
              IPOS = JPOS                                                       
              RETURN                                                            
            ENDIF                                                               
          ENDIF                                                                 
   30   CONTINUE                                                                
      ENDIF                                                                     
      CALL RDFAIL                                                               
      WRITE (IECHO,940)                                                         
      WRITE (ISCRN,940)                                                         
C---- ERROR EXIT --- POSITION UNDEFINED                                         
  800 ERROR1 = .TRUE.                                                            
      IPOS = 0      


9999  IF (FATAL_READ_ERROR) ERROR1=.TRUE.
                                                            
      RETURN                                                                    
C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** "#" OR BEAM ELEMENT NAME EXPECTED'/' ')            
  920 FORMAT(' *** ERROR *** UNKNOWN BEAM ELEMENT "',A,'"'/' ')                 
  930 FORMAT(' *** ERROR *** "',A,'" IS NOT A BEAM ELEMENT'/' ')                
  940 FORMAT(' *** ERROR *** POSITION NOT FOUND'/' ')                           
C-----------------------------------------------------------------------        
      END                                                                       

      SUBROUTINE RDFILE(IFILE,FLAG)                                      
C
C     member of MAD INPUT PARSER
C
C---- DECODE LOGICAL UNIT NUMBER                                         
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
      FLAG = .FALSE.                                                     
      IFILE = 0                                                          
      CALL RDTEST(',',FLAG)                                              
      IF (FLAG) RETURN                                                   
      CALL RDNEXT                                                        
      IF (FATAL_READ_ERROR) GOTO 9999
      CALL RDINT(JFILE,FLAG)                                             
      IF (FATAL_READ_ERROR) GOTO 9999
      IF (FLAG) RETURN                                                   
      CALL RDTEST(';',FLAG)                                              
      IF (FLAG) RETURN                                                   
      IF (JFILE .LT. 1 .OR. JFILE .GT. 99) THEN                          
        CALL RDFAIL                                                      
        WRITE (IECHO,910)                                                
        WRITE (ISCRN,910)                                                
        FLAG = .TRUE.                                                    
        RETURN                                                           
      ENDIF                                                              
      IFILE = JFILE  

9999  CONTINUE

                                                    
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** LOGICAL UNIT NUMBER MUST BE IN ',           
     +       'THE RANGE 1...99'/' ')                                     
C----------------------------------------------------------------------- 
      END                                                                

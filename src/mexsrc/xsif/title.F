      SUBROUTINE TITLE
C
C     member of MAD INPUT PARSER
C
C---- PERFORM "TITLE" COMMAND                                            
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
      CALL RDTEST(';',ERROR)                                             
      IF (ERROR) RETURN                                                  
      IF (ICOL .NE. 81) THEN                                             
        CALL RDWARN                                                      
        WRITE (IECHO,910)                                                
        WRITE (ISCRN,910)                                                
      ENDIF                                                              
      CALL RDLINE                                                        
      IF (ENDFIL) THEN                                                   
        KTIT = ' '                                                       
      ELSE                                                               
        KTIT = KTEXT                                                     
      ENDIF                                                              
      ICOL = 81                                                          
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' ** WARNING ** TEXT AFTER "TITLE" SKIPPED'/' ')            
C----------------------------------------------------------------------- 
      END                                                                

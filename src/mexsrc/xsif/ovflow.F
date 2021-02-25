      SUBROUTINE OVFLOW(ISWI,ISIZE)                                      
C
C     member of MAD INPUT PARSER
C
C---- PRINT OVERFLOW MESSAGE AND QUIT                                    
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
      GO TO (10, 20, 30, 40, 50), ISWI                                   
   10   WRITE (IECHO,910)                                                
        WRITE (ISCRN,910)                                                
      IHERR=5                                                            
      GO TO 100                                                          
   20   WRITE (IECHO,920)                                                
        WRITE (ISCRN,920)                                                
      IHERR=9                                                            
      GO TO 100                                                          
   30   WRITE (IECHO,930)                                                
        WRITE (ISCRN,930)                                                
      IHERR=10                                                           
      GO TO 100                                                          
   40   WRITE (IECHO,940)                                                
        WRITE (ISCRN,940)                                                
      IHERR=11                                                           
      GO TO 100                                                          
   50   WRITE (IECHO,950)                                                
        WRITE (ISCRN,950)                                                
      IHERR=12                                                           
  100 CONTINUE                                                           
      WRITE (IECHO,990) ISIZE                                            
      WRITE (ISCRN,990) ISIZE                                            
      NFAIL = NFAIL + 1                                                  
c      CALL PLEND                                                         
      CALL RDEND(IHERR)                                                  
C----------------------------------------------------------------------- 
  910 FORMAT('0*** ERROR STOP *** ELEMENT SPACE OVERFLOW')               
  920 FORMAT('0*** ERROR STOP *** PARAMETER SPACE OVERFLOW')             
  930 FORMAT('0*** ERROR STOP *** LIST SPACE OVERFLOW')                  
  940 FORMAT('0*** ERROR STOP *** ELEMENT SEQUENCE OVERFLOW')            
  950 FORMAT('0*** ERROR STOP *** HARMON ELEMENT TABLE OVERFLOW')        
  990 FORMAT(20X,'THIS VERSION OF "XSIF" ACCEPTS ',I10,' ENTRIES')        
C----------------------------------------------------------------------- 
      END                                                                

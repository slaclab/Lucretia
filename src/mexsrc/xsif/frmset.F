      SUBROUTINE FRMSET(IBEAM,LACT,ERROR1)                                
C
C     member of MAD INPUT PARSER
C
C---- REPLACE FORMAL ARGUMENTS                                           
C----------------------------------------------------------------------- 
C
C     MOD:
C
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C          03-aug-1998, PT:
C             replaced ERROR with ERROR1 to avoid conflict with global
C             variable defined in DIMAD_INOUT module.
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      LOGICAL           ERROR1                                            
C----------------------------------------------------------------------- 
      ERROR1 = .FALSE.                                                    
      LB = LEN_TRIM(KELEM(IBEAM))                                          
C---- VALID REPLACEMENT?                                                 
      IFORM1 = IEDAT(IBEAM,1)                                            
      IFORM2 = IEDAT(IBEAM,2)                                            
      IF (IFORM1 .EQ. 0 .AND. LACT .EQ. 0) RETURN                        
      IF (IFORM1 .EQ. 0) THEN                                            
        WRITE (IECHO,910) KELEM(IBEAM)(1:LB)                             
        WRITE (ISCRN,910) KELEM(IBEAM)(1:LB)                             
        GO TO 700                                                        
      ENDIF                                                              
      IF (LACT .EQ. 0) THEN                                              
        WRITE (IECHO,920) KELEM(IBEAM)(1:LB)                             
        WRITE (ISCRN,920) KELEM(IBEAM)(1:LB)                             
        GO TO 700                                                        
      ENDIF                                                              
C---- REPLACE FORMAL ARGUMENTS BY ACTUAL ARGUMENTS                       
      IACT = ILDAT(LACT,3)                                               
      DO 10 IFORM = IFORM2, IFORM1, -1                                   
        IF (ILDAT(IACT,1) .EQ. 1) THEN                                   
          WRITE (IECHO,930) KELEM(IBEAM)(1:LB)                           
          WRITE (ISCRN,930) KELEM(IBEAM)(1:LB)                           
          GO TO 700                                                      
        ENDIF                                                            
        IHEAD = IEDAT(IFORM,3)                                           
        ICELL = ILDAT(IHEAD,3)                                           
        ILDAT(ICELL,1) = ILDAT(IACT,1)                                   
        ILDAT(ICELL,4) = ILDAT(IACT,4)                                   
        ILDAT(ICELL,5) = ILDAT(IACT,5)                                   
        ILDAT(ICELL,6) = ILDAT(IACT,6)                                   
        IACT = ILDAT(IACT,3)                                             
   10 CONTINUE                                                           
      IF (ILDAT(IACT,1) .NE. 1) THEN                                     
        WRITE (IECHO,940) KELEM(IBEAM)(1:LB)                             
        WRITE (ISCRN,940) KELEM(IBEAM)(1:LB)                             
        GO TO 700                                                        
      ENDIF                                                              
      RETURN                                                             
C---- ERROR EXIT                                                         
  700 NFAIL = NFAIL + 1                                                  
      ERROR1 = .TRUE.                                                     
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('0*** ERROR *** ACTUAL ARGUMENT LIST IS REDUNDANT ',        
     +       'FOR LINE "',A,'"'/' ')                                     
  920 FORMAT('0*** ERROR *** ACTUAL ARGUMENT LIST IS MISSING ',          
     +       'FOR LINE "',A,'"'/' ')                                     
  930 FORMAT('0*** ERROR *** TOO FEW ACTUAL ARGUMENTS FOUND ',           
     +       'FOR LINE "',A,'"'/' ')                                     
  940 FORMAT('0*** ERROR *** TOO MANY ACTUAL ARGUMENTS FOUND ',          
     +       'FOR LINE "',A,'"'/' ')                                     
C----------------------------------------------------------------------- 
      END                                                                

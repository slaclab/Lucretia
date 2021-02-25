      SUBROUTINE VALUE
C
C     member of MAD INPUT PARSER
C
C---- PRINT VALUE OF AN EXPRESSION                                       
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
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
C----------------------------------------------------------------------- 
C---- COMMA?                                                             
      CALL RDTEST(',',ERROR)                                             
      IF (ERROR) RETURN                                                  
      CALL RDNEXT                                                        
      IF (FATAL_READ_ERROR) GOTO 9999
C---- ALLOCATE PARAMETER SPACE                                           
      IPARM2 = IPARM2 - 1                                                
      IF (IPARM1 .GE. IPARM2) CALL OVFLOW(2,MAXPAR)                      
      IPARM = IPARM2                                                     
C---- DECODE EXPRESSION                                                  
      CALL DECEXP(IPARM,ERROR)                                           
      IF (ERROR) RETURN                                                  
C---- SEMICOLON?                                                         
      CALL RDTEST(';',ERROR)                                             
      IF (ERROR) RETURN                                                  
C---- EVALUATE EXPRESSION                                                
      IF (NEWPAR) THEN                                                   
        NEWPAR = .FALSE.                                                 
        CALL PARORD(ERROR)                                               
        IF (ERROR) RETURN                                                
        CALL PAREVL                                                      
      ENDIF                                                              
C---- PRINT AND DISCARD EXPRESSION                                       
      WRITE (IECHO,910) PDATA(IPARM)                                     
      WRITE (ISCRN,910) PDATA(IPARM)                                     
      IPARM2 = IPARM + 1

9999  IF (FATAL_READ_ERROR) ERROR=.TRUE.

                                                 
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('0... "VALUE" --- VALUE OF EXPRESSION IS ',F16.8/' ')       
C----------------------------------------------------------------------- 
      END                                                                

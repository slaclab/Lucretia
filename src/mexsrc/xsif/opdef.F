      SUBROUTINE OPDEF(ILCOMM)                                            
C
C     member of MAD INPUT PARSER
C
C---- CONSTRUCT OPERATION ON PARAMETERS                                  
C----------------------------------------------------------------------- 
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
C---- ALLOCATE AN UPPER PARAMETER CELL                                   
      IPARM = IPARM2 - 1                                                 
      IF (IPARM1 .GE. IPARM) CALL OVFLOW(2,MAXPAR)                       
      IPARM2 = IPARM                                                     
C---- FILL IN OPERATION DATA                                             
      IPTYP(IPARM) = IOP(LEV)                                            
      IF (IOP(LEV) .LE. 10) THEN                                         
        IPDAT(IPARM,1) = IVAL(LEV-1)                                     
      ELSE                                                               
        IPDAT(IPARM,1) = 0                                               
      ENDIF                                                              
      IPDAT(IPARM,2) = IVAL(LEV)                                         
      PDATA(IPARM) = 0.0                                                 
      IPLIN(IPARM) = ILCOMM                                               
      WRITE (KPARM(IPARM),910) IPARM                                     
      LEV = LEV - 1                                                      
      IVAL(LEV) = IPARM                                                  
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('*T',I5.5,'*')                                              
C----------------------------------------------------------------------- 
      END                                                                

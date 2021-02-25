      SUBROUTINE DEFPAR(NDICT,DICT,IEP1,IEP2,ILCOM)                             
C
C     part of MAD INPUT PARSER
C
C---- ALLOCATE PARAMETER SPACE                                                  
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_ELEMENTS
C
C-----------------------------------------------------------------------        
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      CHARACTER*8       DICT(NDICT)                                             
C-----------------------------------------------------------------------        
      IEP1 = IPARM2 - NDICT                                                     
      IEP2 = IPARM2 - 1                                                         
      IF (IPARM1 .GE. IEP1) CALL OVFLOW(2,MAXPAR)                               
      IPARM2 = IEP1                                                             
      IDICT = 0                                                                 
      DO 10 IEP = IEP1, IEP2                                                    
        IDICT = IDICT + 1                                                       
        KPARM(IEP) = DICT(IDICT)                                                
        IPTYP(IEP) = -1                                                         
        IPDAT(IEP,1) = 0                                                        
        IPDAT(IEP,2) = 0                                                        
        PDATA(IEP) = 0.0                                                        
        IPLIN(IEP) = ILCOM                                                      
   10 CONTINUE                                                                  
      RETURN                                                                    
C-----------------------------------------------------------------------        
      END                                                                       

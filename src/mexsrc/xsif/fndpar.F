      SUBROUTINE FNDPAR(ILCOM,KNAME,IPARM)                               
C
C     member of MAD INPUT PARSER
C
C---- DEAL WITH PARAMETER NAMELIST      
C
C     MOD:
C          15-DEC-2003, PT:
C             expand parameter names to 16 characters.                                 
C----------------------------------------------------------------------- 
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_ELEMENTS
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      CHARACTER(PNAME_LENGTH) KNAME                                            
C----------------------------------------------------------------------- 
C---- PREVIOUS DEFINITION?                                               
      CALL RDLOOK(KNAME,PNAME_LENGTH,KPARM(1),1,IPARM1,IPARM)                          
      IF (IPARM .NE. 0) RETURN                                           
C---- NEW DEFINITION --- ALLOCATE PARAMETER CELL                         
      IPARM = IPARM1 + 1                                                 
      IF (IPARM .GE. IPARM2) CALL OVFLOW(2,MAXPAR)                       
      IPARM1 = IPARM                                                     
C---- FILL IN DEFAULT DATA                                               
      IPTYP(IPARM) = -1                                                  
      IPDAT(IPARM,1) = 0                                                 
      IPDAT(IPARM,2) = 0                                                 
      PDATA(IPARM) = 0.0                                                 
      IPLIN(IPARM) = ILCOM                                               
      KPARM(IPARM) = KNAME                                               
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                

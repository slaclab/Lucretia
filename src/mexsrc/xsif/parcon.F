      SUBROUTINE PARCON(ILCOM,IPARM,VALUE)                               
C
C     member of MAD INPUT PARSER
C
C---- ALLOCATE CONSTANT CELL                                             
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
C----------------------------------------------------------------------- 
C---- ALLOCATE AN UPPER PARAMETER CELL                                   
      IPARM = IPARM2 - 1                                                 
      IF (IPARM1 .GE. IPARM) CALL OVFLOW(2,MAXPAR)                       
      IPARM2 = IPARM                                                     
C---- FILL IN CONSTANT DATA                                              
      IPTYP(IPARM) = 0                                                   
      IPDAT(IPARM,1) = 0                                                 
      IPDAT(IPARM,2) = 0                                                 
      PDATA(IPARM) = VALUE                                               
      IPLIN(IPARM) = ILCOM                                               
C      IDFL=0                                                            
C      RPARM=IPARM                                                       
C      RPARM=RPARM*1.0D-04                                               
C      DO 1 ID=1,5                                                       
C      IDD=RPARM                                                         
C      IF(IDD.EQ.0) THEN                                                 
C           IF(IDFL.EQ.0) THEN                                           
C                 UPARM(ID+2)=IBLANK                                     
C                         ELSE                                           
C                 UPARM(ID+2)=IDIGIT(IDD+1)                              
C            ENDIF                                                       
C                   ELSE                                                 
C            IDFL=1                                                      
C            UPARM(ID+2)=IDIGIT(IDD+1)                                   
C      ENDIF                                                             
C      RPARM=RPARM-IDD                                                   
C      RPARM=RPARM*10.0D0                                                
C    1 CONTINUE                                                          
C      KPARM(IPARM)=SPARM                                                
      WRITE(KPARM(IPARM),910) IPARM                                      
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT('*C',I5.5,'*')                                              
C----------------------------------------------------------------------- 
      END                                                                

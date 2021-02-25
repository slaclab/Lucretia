      SUBROUTINE FNDELM(ILCOM,KNAME,IELEM)                               
C     
C     member of MAD INPUT PARSER
C
C---- DEAL WITH ELEMENT NAMELIST        
C
C     MOD:
C          15-DEC-2003, PT:
C             expand element names to 16 characters.                                 
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
      CHARACTER(ENAME_LENGTH) KNAME                                            
C----------------------------------------------------------------------- 
C---- PREVIOUS DEFINITION?                                               
      CALL RDLOOK(KNAME,ENAME_LENGTH,KELEM(1),1,IELEM1,IELEM)                          
      IF (IELEM .NE. 0) RETURN                                           
C---- NEW DEFINITION --- ALLOCATE ELEMENT CELL                           
      IELEM = IELEM1 + 1                                                 
      IF (IELEM .GE. IELEM2) CALL OVFLOW(1,MAXELM)                       
      IELEM1 = IELEM                                                     
C---- FILL IN DEFAULT DATA                                               
      IETYP(IELEM) = -1                                                  
      IEDAT(IELEM,1) = 0                                                 
      IEDAT(IELEM,2) = 0                                                 
      IEDAT(IELEM,3) = 0                                                 
      IELIN(IELEM) = ILCOM                                               
      KELEM(IELEM) = KNAME                                               
      KETYP(IELEM) = '    '                                              
      RETURN                                                             
C----------------------------------------------------------------------- 
      END   


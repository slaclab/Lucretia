      SUBROUTINE RDLOOK(KWORD,LWORD,KDICT,IDICT1,IDICT2,IDICT)
C
C     member of MAD INPUT PARSER
C
C---- FIND WORD "KWORD" OF LENGTH "LWORD" IN DICTIONARY "KDICT"      
C     
C     MOD:
C          15-DEC-2003, PT:
C             expand element and parameter names to 16 characters. 
C             Permit abbreviations shorter than 4 characters, per
C             MAD 8 standards.   
C----------------------------------------------------------------------- 
C
      USE XSIF_SIZE_PARS

      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N) 
      SAVE
C
      CHARACTER*(*)       KDICT(*),KWORD   
      INTEGER*4 DLEN,WLEN                                
C----------------------------------------------------------------------- 
      CHARACTER(MAXCHARLEN) KTEMP                                            
C----------------------------------------------------------------------- 
      DLEN = LEN(KDICT(1))
	WLEN = LEN(KWORD)
      IF (IDICT1 .EQ. 0) GO TO 20                                        
      IF (IDICT1 .GT. IDICT2) GO TO 20                                   
C      L = MAX0(4,LWORD) 
C
C     if the dictionary is composed of CHARACTER variables which are
C     shorter than LWORD, take care of that now
C
      L = MIN(LWORD,DLEN,WLEN)                                                 
      DO 10 IDICT = IDICT1, IDICT2                                       
        KTEMP = KDICT(IDICT)(1:L)                                        
        IF (KWORD .EQ. KTEMP) RETURN                                     
   10 CONTINUE                                                           
   20 IDICT = 0                                                          
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                

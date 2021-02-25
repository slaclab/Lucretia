      SUBROUTINE XUSE                                                     
C
C     member of MAD INPUT PARSER
C
C*******************************                                         
C---- SET BEAM LINE TO BE USED                                           
C----------------------------------------------------------------------- 
C
C     MOD:
C		 05-DEC-2003, PT:
C			initialization for ERRFLG, ERRPTR structures.
C          16-MAY-2003, PT:
C             support for USE to select BEAM/BETA0 definition.
C          02-MAR-2001, PT:
C             replaced NELM with NELM_XSIF.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C          05-JAN-2001, PT:
C             added success/failure indicator LINE_EXPANDED.  If
C             USE statement was not in command line (ie, XUSE called
C             by subroutine XUSE2), skip the read statements at top
C             of routine.
C          11-JAN-2000, PT:
C             renamed routine and file to XUSE to eliminate conflict
C             with F90 USE statement.
C          20-aug-1998, PT:
C             commented out TIMER call to eliminate misleading
C             statement in echo file about 0 seconds elapsed CPU time.
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
      PARAMETER         (NDICT = 2)                                      
      CHARACTER*8       DICT(NDICT)                                      
      LOGICAL           SEEN(NDICT)                                      
      INTEGER           ITYPE(NDICT),IVALUE(NDICT)                         
      DIMENSION         RVAL(NDICT)                                      
C----------------------------------------------------------------------- 
      DATA DICT(1)      / 'SYMM    ' /                                   
      DATA DICT(2)      / 'SUPER   ' /                                   
C----------------------------------------------------------------------- 

C     PT, 05-jan-2001
      IF ( XUSE_FROM_FILE ) THEN
C---- COMMA?                                                             
          CALL RDTEST(',',ERROR)                                             
          IF (ERROR) GO TO 800                                               
          CALL RDNEXT  
          IF (FATAL_READ_ERROR) GOTO 9999
      ENDIF 
c     end of changed block 05-jan-2001
                                                     
C---- BEAM LINE REFERENCE                                                
      CALL DECUSE(IPERI,IACT,ERROR)                                      
      IF (FATAL_READ_ERROR) GOTO 9999
      IF (USE_BEAM_OR_BETA0) GOTO 9999
      IF (ERROR) GO TO 800                                               
C---- DEFAULT SYMMETRY FLAG AND SUPER-PERIOD                             
      ITYPE(1) = 3                                                       
      ITYPE(2) = 1                                                       
      IVALUE(2) = 1                                                        
C---- DECODE SYMMETRY FLAG AND SUPER-PERIOD COUNT                        
      CALL RDPARA(NDICT,DICT,ITYPE,SEEN,IVALUE,RVAL,ERROR)                 
      IF (FATAL_READ_ERROR) GOTO 9999
      IF (SCAN .OR. ERROR) GO TO 800                                     
      SYMM = SEEN(1)                                                     
      NSUP = MAX(IVALUE(2),1)                                              
C---- EXPAND THE BEAM LINE                                               
      NFREE = 0                                                          
      CALL EXPAND(IPERI,IACT,NELM_XSIF,NFREE,NPOS1,NPOS2,ERROR)               
      IF (ERROR) GO TO 800                                               
      PERI = .TRUE.                                                      
C---- CLEAR PRINT FLAGS                                                  
      DO 30 IPOS = NPOS1, NPOS2                                          
        PRTFLG(IPOS) = .FALSE.     
	  ERRFLG(IPOS) = .FALSE.                                      
   30 CONTINUE                                                           
      PRTFLG(NPOS1) = .TRUE.                                             
      PRTFLG(NPOS2) = .TRUE.  
	ERRPTR = 0                                           
C      CALL TIMER(' ','USE')
C     PT, 05-jan-2001:                                              
      LINE_EXPANDED = .TRUE.
      RETURN                                                             
C---- ERROR EXIT --- CLEAR LINE DATA                                     
  800 ERROR = .TRUE.                                                     
      PERI  = .FALSE.                                                    
      SYMM  = .FALSE.                                                    
      NSUP  = 1   
      LINE_EXPANDED = .FALSE.
C     end changed block 05-jan-2001   

9999  CONTINUE

                                                    
      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                

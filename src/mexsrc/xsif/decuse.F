      SUBROUTINE DECUSE(INAME,IACT1,ERROR1)                                       
C
C     part of MAD INPUT PARSER
C
C---- DECODE REFERENCE TO BEAM LINE                                             
C
C     MOD:
C		 23-dec-2003, PT:
C			if a BEAM or BETA0 is selected by USE, set flags to
C			that effect.
C          15-DEC-2003, PT:
C             expand element names to 16 characters.
C          16-may-2003, PT:
C             support for USE to point at BEAM/BETA0 definitions.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C           13-july-1998, PT:
C              replaced ERROR with ERROR1 as error flag for routine.
C              replaced IACT with IACT1.
C
C----
C
C     modules:
C     
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
      USE XSIF_ELEM_PARS
C
C-----------------------------------------------------------------------        
      IMPLICIT REAL*8(A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      LOGICAL*4         ERROR1                                                   
C-----------------------------------------------------------------------        
      CHARACTER(ENAME_LENGTH) KNAME                                                   
C-----------------------------------------------------------------------        
      ERROR1 = .FALSE.    
      USE_BEAM_OR_BETA0 = .FALSE.                                                       
C---- BEAM LINE NAME                                                            
      CALL RDWORD(KNAME,LNAME)                                                  
      IF (LNAME .EQ. 0) THEN                                                    
        CALL RDFAIL                                                             
        WRITE (IECHO,910)                                                       
        WRITE (ISCRN,910)                                                       
        ERROR1 = .TRUE.                                                          
        RETURN                                                                  
      ENDIF                                                                     
      CALL RDLOOK(KNAME,ENAME_LENGTH,KELEM,1,IELEM1,INAME)                                 
      IF (INAME .EQ. 0) THEN                                                    
        CALL RDFAIL                                                             
        WRITE (IECHO,920) KNAME(1:LNAME)                                        
        WRITE (ISCRN,920) KNAME(1:LNAME)                                        
        ERROR1 = .TRUE.                                                          
        RETURN                                                                  
      ENDIF                                                                     
      IF ( (IETYP(INAME) .NE. 0)
     &            .and.
     &     (IETYP(INAME) .NE. MAD_BET0)
     &            .and.
     &     (IETYP(INAME) .NE. MAD_BEAM) ) THEN                                             
        CALL RDFAIL                                                             
        WRITE (IECHO,930) KNAME(1:LNAME)                                        
        WRITE (ISCRN,930) KNAME(1:LNAME)                                        
        ERROR1 = .TRUE.                                                          
        RETURN                                                                  
      ENDIF        
      IF (IETYP(INAME).EQ.MAD_BET0) THEN
        IBETA0_PTR = INAME
        USE_BEAM_OR_BETA0 = .TRUE.
	  BETA0_FROM_USE = .TRUE.
      ENDIF                                                             
      IF (IETYP(INAME).EQ.MAD_BEAM) THEN
        IBEAM_PTR = INAME
        USE_BEAM_OR_BETA0 = .TRUE.
	  BEAM_FROM_USE = .TRUE.
      ENDIF     
      IF (USE_BEAM_OR_BETA0) GOTO 9999                                                        
C---- ACTUAL PARAMETER LIST                                                     
      IF (KLINE(ICOL) .EQ. '(') THEN                                            
        CALL DECLST(IACT1,0,0,ERROR1)                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (ERROR1) RETURN                                                       
      ELSE                                                                      
        IACT1 = 0                                                                
      ENDIF               

9999  IF (FATAL_READ_ERROR) ERROR1 = .TRUE.

                                                      
      RETURN                                                                    
C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** BEAM LINE NAME EXPECTED'/' ')                      
  920 FORMAT(' *** ERROR *** UNKNOWN BEAM LINE NAME "',A,'"'/' ')               
  930 FORMAT(' *** ERROR *** "',A,'" IS NOT A BEAM LINE, '
     &       'A BETA0, OR A BEAM DEFINITION.'/' ')                   
C-----------------------------------------------------------------------        
      END                                                                       

      SUBROUTINE LINE(KNAME,LNAME)                                       
C
C     member of MAD INPUT PARSER
C
C---- DEFINE A BEAM LINE                                                 
C----------------------------------------------------------------------- 
C
C	MOD:
C          15-dec-2003, PT:
C             expand element names to 16 characters.
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
      CHARACTER(ENAME_LENGTH) KNAME                                            
C----------------------------------------------------------------------- 
      LOGICAL*4           FLAG                                             
C----------------------------------------------------------------------- 
C---- IF OLD FORMAT, READ BEAM LINE NAME                                 
      IF (LNAME .EQ. 0) THEN                                             
        CALL RDTEST(',',FLAG)                                            
        IF (FLAG) GO TO 900                                              
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        CALL RDWORD(KNAME,LNAME)                                         
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LNAME .EQ. 0) THEN                                           
          CALL RDFAIL                                                    
          WRITE (IECHO,910)                                              
          WRITE (ISCRN,910)                                              
          GO TO 900                                                      
        ENDIF                                                            
      ENDIF                                                              
C---- ALLOCATE ELEMENT CELL AND TEST FOR REDEFINITION                    
      CALL FNDELM(ILCOM,KNAME,IELEM)                                     
      IF (IETYP(IELEM) .GE. 0) THEN                                      
        CALL RDWARN                                                      
        WRITE (IECHO,920) IELIN(IELEM)                                   
        WRITE (ISCRN,920) IELIN(IELEM)                                   
        IETYP(IELEM) = -1                                                
      ENDIF                                                              
C---- FORMAL ARGUMENT LIST                                               
      CALL DECFRM(IFORM1,IFORM2,FLAG)                                    
      IF (FATAL_READ_ERROR) GOTO 9999
      IF (FLAG) GO TO 900                                                
C---- IF THERE WERE ARGUMENTS, SKIP TO =                                 
      IF(IFORM1.NE.0 .AND. IFORM2.NE.0) CALL RDSKIP(':LINE')             
      IF (FATAL_READ_ERROR) GOTO 9999
C---- EQUALS SIGN?                                                       
      CALL RDTEST('=',FLAG)                                              
      IF (FLAG) GO TO 900                                                
      CALL RDNEXT                                                        
      IF (FATAL_READ_ERROR) GOTO 9999
C---- BEAM LINE LIST                                                     
      CALL DECLST(IBEAM,IFORM1,IFORM2,FLAG)                              
      IF (FATAL_READ_ERROR) GOTO 9999
      IF (FLAG) GO TO 900                                                
C---- END OF COMMAND?                                                    
      CALL RDTEST(';',FLAG)                                              
      IF (FLAG) GO TO 900                                                
C---- STORE DEFINITION                                                   
      IETYP(IELEM) = 0                                                   
      IEDAT(IELEM,1) = IFORM1                                            
      IEDAT(IELEM,2) = IFORM2                                            
      IEDAT(IELEM,3) = IBEAM                                             
      IELIN(IELEM) = ILCOM                                               
      RETURN                                                             
C---- ERROR EXIT --- LEAVE BEAM LINE UNDEFINED                           
  900 ERROR = .TRUE.                  

9999  IF (FATAL_READ_ERROR) ERROR = .TRUE.

                                   
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** BEAM LINE NAME EXPECTED'/' ')               
  920 FORMAT(' ** WARNING ** THE ABOVE NAME WAS DEFINED IN LINE ',I5,    
     +       ', IT WILL BE REDEFINED'/' ')                               
C----------------------------------------------------------------------- 
      END                                                                

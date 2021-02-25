      SUBROUTINE DECFRM(IFORM1,IFORM2,ERROR1)                                    
C
C     part of MAD INPUT PARSER
C
C---- DECODE FORMAL ARGUMENT LIST                                               
C
C     MOD:
C           15-DEC-2003, PT:
C             expand parameter and element names to 16 characters.
C		  22-may-2003, PT:
C			if FATAL_ALLOC_ERROR occurs, go directly to egress.
C           31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	      08-SEP-1998, PT:
C	         added write to ISCRN where IECHO is presently written.
C           13-july-1998, PT:
C              replaced ERROR with ERROR1 as error flag for routine.
C
C----
C
C     modules:
C
      USE XSIF_SIZE_PARS
      USE XSIF_INOUT
      USE XSIF_ELEMENTS
C
C-----------------------------------------------------------------------        
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N) 
      SAVE
C
      LOGICAL*4         ERROR1                                                   
C-----------------------------------------------------------------------        
      CHARACTER(ENAME_LENGTH) KNAME                                                   
      LOGICAL           FLAG                                                    
C-----------------------------------------------------------------------        
      ERROR1 = .FALSE.                                                           
C---- ANY ARGUMENT LIST?                                                        
      IF (KLINE(ICOL) .EQ. '(') THEN                                            
        IFORM1 = IELEM2                                                         
        IFORM2 = IELEM2 - 1                                                     
C---- ARGUMENT NAME                                                             
  100   CONTINUE                                                                
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDWORD(KNAME,LNAME)                                              
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (LNAME .EQ. 0) THEN                                                
            CALL RDFAIL                                                         
            WRITE (IECHO,910)                                                   
            WRITE (ISCRN,910)                                                   
            GO TO 200                                                           
          ENDIF                                                                 
          CALL RDLOOK(KNAME,ENAME_LENGTH,KELEM,IFORM1,IFORM2,IFORM)                        
          IF (IFORM .NE. 0) THEN                                                
            CALL RDFAIL                                                         
            WRITE (IECHO,920) KNAME(1:LNAME)                                    
            WRITE (ISCRN,920) KNAME(1:LNAME)                                    
            GO TO 200                                                           
          ENDIF                                                                 
C---- SEPARATOR?                                                                
          CALL RDTEST(',)',FLAG)                                                
          IF (FLAG) GO TO 200                                                   
C---- ALLOCATE CELL TO ARGUMENT                                                 
          IFORM1 = IELEM2 - 1                                                   
          IF (IELEM1 .GE. IFORM1) CALL OVFLOW(1,MAXELM)                         
          IELEM2 = IFORM1                                                       
C---- SET UP FORMAL BEAM LINE                                                   
          CALL NEWLST(IHEAD)  
		IF (FATAL_ALLOC_ERROR) GOTO 9999                                                  
          CALL NEWRGT(IHEAD,ICELL)        
		IF (FATAL_ALLOC_ERROR) GOTO 9999                                      
          KELEM(IFORM1) = KNAME                                                 
          KETYP(IFORM1) = '    '                                                
          IETYP(IFORM1) = 0                                                     
          IEDAT(IFORM1,1) = 0                                                   
          IEDAT(IFORM1,2) = 0                                                   
          IEDAT(IFORM1,3) = IHEAD                                               
          IELIN(IFORM1) = ILCOM                                                 
          GO TO 300                                                             
C---- ERROR RECOVERY                                                            
  200     CALL RDFIND(',);')                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          ERROR1 = .TRUE.                                                        
  300   IF (KLINE(ICOL) .EQ. ',') GO TO 100                                     
        IF (KLINE(ICOL) .EQ. ')') CALL RDNEXT                                   
        IF (FATAL_READ_ERROR) GOTO 9999
C---- NO ARGUMENT LIST                                                          
      ELSE                                                                      
        IFORM1 = 0                                                              
        IFORM2 = 0                                                              
      ENDIF           

9999  IF (FATAL_READ_ERROR) ERROR1=.TRUE.
	IF (FATAL_ALLOC_ERROR) ERROR1 = .TRUE.
                                                          
      RETURN                                                                    
C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** FORMAL ARGUMENT NAME EXPECTED'/' ')                
  920 FORMAT(' *** ERROR *** DUPLICATE FORMAL ARGUMENT "',A,'"'/' ')            
C-----------------------------------------------------------------------        
      END                                                                       


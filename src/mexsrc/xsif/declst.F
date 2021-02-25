      SUBROUTINE DECLST(IBEAM,IFORM1,IFORM2,ERROR1)                              
C
C     member of MAD INPUT PARSER
C
C---- DECODE A BEAM LIST                                                        
C
C     MOD:
C          15-DEC-2003, PT:
C             expand element/parameter names to 16 characters.
C           22-MAY-2003, PT:
C             if FATAL_ALLOC_ERROR occurs, go directly to egress.
C           31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C          13-july-1998, PT:
C             changed error flag from ERROR to ERROR1.
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
      IMPLICIT REAL*8(A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
      LOGICAL           ERROR1                                                   
C-----------------------------------------------------------------------        
      LOGICAL           FLAG                                                    
      CHARACTER(ENAME_LENGTH) KNAME                                                   
C-----------------------------------------------------------------------        
      ERROR1 = .FALSE.                                                           
C---- INITIALIZE                                                                
      ICELL = 0                                                                 
      ICALL = 1                                                                 
C---- OPENING PARENTHESIS                                                       
      CALL RDTEST('(',ERROR1)                                                    
      IF (ERROR1) GO TO 800                                                      
      CALL RDNEXT                                                               
      IF (FATAL_READ_ERROR) GOTO 9999
C---- PROCEDURE "DECODE LIST" ------------------------------------------        
  100 CALL NEWLST(IHEAD)    
	IF (FATAL_ALLOC_ERROR) GOTO 9999                                                    
      ILDAT(IHEAD,4) = ICELL                                                    
      ILDAT(IHEAD,5) = ICALL                                                    
      ICELL = IHEAD                                                             
C---- APPEND A NEW CALL CELL                                                    
  200 CALL NEWRGT(ICELL,ICELL) 
	IF (FATAL_ALLOC_ERROR) GOTO 9999                                                 
C---- REFLEXION?                                                                
      IF (KLINE(ICOL) .EQ. '-') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        IDIREP = -1                                                             
      ELSE                                                                      
        IDIREP = 1                                                              
      ENDIF                                                                     
C---- REPETITION?                                                               
      IF (INDEX('0123456789',KLINE(ICOL)) .NE. 0) THEN                          
        CALL RDINT(IREP,FLAG)                                                   
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (FLAG) GO TO 600                                                     
        CALL RDTEST('*',FLAG)                                                   
        IF (FLAG) GO TO 600                                                     
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        IDIREP = IDIREP * IREP                                                  
      ENDIF                                                                     
      ILDAT(ICELL,4) = IDIREP                                                   
C---- SUBLIST?                                                                  
      IF (KLINE(ICOL) .NE. '(') GO TO 300                                       
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        ICALL = 2                                                               
        GO TO 100                                                               
  250   ILDAT(ICELL,1) = 2                                                      
        ILDAT(ICELL,5) = IHEAD                                                  
      GO TO 500                                                                 
C---- DECODE IDENTIFIER                                                         
  300   CALL RDWORD(KNAME,LNAME)                                                
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LNAME .EQ. 0) THEN                                                  
          CALL RDFAIL                                                           
          WRITE (IECHO,910)                                                     
          WRITE (ISCRN,910)                                                     
          GO TO 600                                                             
        ELSE                                                                    
          CALL RDLOOK(KNAME,ENAME_LENGTH,KELEM(1),IFORM1,IFORM2,INAME)                        
          IF (INAME .EQ. 0) CALL FNDELM(ILCOM,KNAME,INAME)                      
        ENDIF                                                                   
        ILDAT(ICELL,1) = 3                                                      
        ILDAT(ICELL,5) = INAME                                                  
C---- ACTUAL ARGUMENT LIST?                                                     
        IF (KLINE(ICOL) .NE. '(') GO TO 400                                     
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          ICALL = 3                                                             
          GO TO 100                                                             
  350     ILDAT(ICELL,6) = IHEAD                                                
  400   CONTINUE                                                                
C---- COMMA OR RIGHT PARENTHESIS?                                               
  500 CALL RDTEST(',)',FLAG)                                                    
      IF (.NOT. FLAG) GO TO 700                                                 
C---- ERROR RECOVERY                                                            
  600 CALL RDFIND('(),;')                                                       
      IF (FATAL_READ_ERROR) GOTO 9999
      ERROR1 = .TRUE.                                                            
C---- ANOTHER MEMBER?                                                           
  700 IF (KLINE(ICOL) .EQ. ',') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        GO TO 200                                                               
      ENDIF                                                                     
C---- END OF LIST?                                                              
      IF (KLINE(ICOL) .EQ. ')') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        IHEAD = ILDAT(ICELL,3)                                                  
        ICELL = ILDAT(IHEAD,4)                                                  
        ICALL = ILDAT(IHEAD,5)                                                  
        ILDAT(IHEAD,4) = 0                                                      
        ILDAT(IHEAD,5) = 0                                                      
        ILDAT(IHEAD,6) = 0                                                      
        GO TO (800,250,350,500), ICALL                                          
      ENDIF                                                                     
C---- ANOTHER LIST WITHOUT A COMMA?                                             
      IF (KLINE(ICOL) .EQ. '(') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        ICALL = 4                                                               
        GO TO 100                                                               
      ENDIF                                                                     
C---- END OF BEAM LINE LIST                                                     
  800 IF (ERROR1) THEN                                                           
        IBEAM = 0                                                               
      ELSE                                                                      
        IBEAM = IHEAD                                                           
      ENDIF                 

9999  IF (FATAL_READ_ERROR) ERROR1=.TRUE.
	IF (FATAL_ALLOC_ERROR) ERROR1 = .TRUE.

                                                    
      RETURN                                                                    
C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** BEAM LINE MEMBER MUST BE BEAM ELEMENT',            
     +       ' NAME, BEAM LINE NAME, OR LIST IN "()"'/' ')                      
C-----------------------------------------------------------------------        
      END                                                                       

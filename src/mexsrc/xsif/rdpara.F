      SUBROUTINE RDPARA(NDICT,DICT,ITYPE,SEEN,IVALUE,RVAL,FLAG)
C
C     member of MAD INPUT PARSER
C
C---- READ PARAMETER LIST                                                
C----------------------------------------------------------------------- 
C
C	MOD:
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C	     08-SEP-1998, PT:
C	        added write to ISCRN wherever IECHO presently written.
C
C     modules:
C
      USE XSIF_INOUT
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      SAVE
C
C----------------------------------------------------------------------- 
      CHARACTER*8       DICT(NDICT)                                      
      INTEGER           ITYPE(NDICT),IVALUE(NDICT)                         
      LOGICAL           SEEN(NDICT),FLAG                                 
      DIMENSION         RVAL(NDICT)                                      
C----------------------------------------------------------------------- 
      CHARACTER*8       KNAME                                            
      LOGICAL           FAIL                                             
C----------------------------------------------------------------------- 
      FLAG = .FALSE.                                                     
      DO 10 I = 1, NDICT                                                 
        SEEN(I) = .FALSE.                                                
   10 CONTINUE                                                           
C---- SEPARATOR?                                                         
      CALL RDTEST(',;',FLAG)                                             
      IF (FLAG) RETURN                                                   
C---- ANOTHER PARAMETER?                                                 
  100 IF (KLINE(ICOL) .EQ. ',') THEN                                     
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        CALL RDWORD(KNAME,LNAME)                                         
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LNAME .EQ. 0) THEN                                           
          CALL RDFAIL                                                    
          WRITE (ISCRN,910)                                              
          WRITE (IECHO,910)                                              
          GO TO 300                                                      
        ENDIF                                                            
        CALL RDLOOK(KNAME,LNAME,DICT,1,NDICT,IDICT)                      
        IF (IDICT .EQ. 0) THEN                                           
          CALL RDFAIL                                                    
          WRITE (IECHO,920) KNAME(1:LNAME)                               
          WRITE (ISCRN,920) KNAME(1:LNAME)                               
          GO TO 300                                                      
        ENDIF                                                            
        IF (SEEN(IDICT)) THEN                                            
          CALL RDFAIL                                                    
          WRITE (IECHO,930) KNAME(1:LNAME)                               
          WRITE (ISCRN,930) KNAME(1:LNAME)                               
          GO TO 300                                                      
        ENDIF                                                            
C---- INTEGER VALUE                                                      
        IF (ITYPE(IDICT) .EQ. 1) THEN                                    
          CALL RDTEST('=',FAIL)                                          
          IF (FAIL) GO TO 300                                            
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDINT(IV,FAIL)                                            
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (FAIL) GO TO 300                                            
          IVALUE(IDICT) = IV                                               
C---- REAL VALUE                                                         
        ELSE IF (ITYPE(IDICT) .EQ. 2) THEN                               
          CALL RDTEST('=',FAIL)                                          
          IF (FAIL) GO TO 300                                            
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDNUMB(RV,FAIL)                                           
          IF (FAIL) GO TO 300                                            
          RVAL(IDICT) = RV                                               
C---- KEYWORD VALUE                                                      
        ELSE IF (ITYPE(IDICT) .GT. 100) THEN                             
          CALL RDTEST('=',FAIL)                                          
          IF (FAIL) GO TO 300                                            
          CALL RDNEXT                                                    
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDWORD(KNAME,LNAME)                                       
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (LNAME .EQ. 0) THEN                                         
            CALL RDFAIL                                                  
            WRITE (IECHO,940)                                            
            WRITE (ISCRN,940)                                            
            GO TO 300                                                    
          ENDIF                                                          
          JBASE = ITYPE(IDICT) / 100                                     
          JDICT = MOD(ITYPE(IDICT),100)                                  
          CALL RDLOOK(KNAME,LNAME,DICT(JBASE+1),1,JDICT,IV)              
          IF (IV .EQ. 0) THEN                                            
            CALL RDFAIL                                                  
            WRITE (IECHO,950) KNAME(1:LNAME)                             
            WRITE (ISCRN,950) KNAME(1:LNAME)                             
            GO TO 300                                                    
          ENDIF                                                          
          IVALUE(IDICT) = IV                                               
        ENDIF                                                            
C---- END OF FIELD                                                       
        CALL RDTEST(',;',FAIL)                                           
        IF (FAIL) GO TO 300                                              
        SEEN(IDICT) = .TRUE.                                             
        GO TO 100                                                        
C---- ERROR RECOVERY                                                     
  300   CALL RDFIND(',;')                                                
        FLAG = .TRUE.                                                    
        GO TO 100                                                        
      ENDIF               

9999  CONTINUE

                                               
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** PARAMETER KEYWORD EXPECTED'/' ')            
  920 FORMAT(' *** ERROR *** UNKNOWN PARAMETER KEYWORD "',A,'"'/' ')     
  930 FORMAT(' *** ERROR *** DUPLICATE PARAMETER "',A,'"'/' ')           
  940 FORMAT(' *** ERROR *** VALUE KEYWORD EXPECTED'/' ')                
  950 FORMAT(' *** ERROR *** UNKNOWN VALUE KEYWORD "',A,'"'/' ')         
C----------------------------------------------------------------------- 
      END                                                                

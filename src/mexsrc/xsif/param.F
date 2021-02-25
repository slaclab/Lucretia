      SUBROUTINE PARAM(KNAME,LNAME,PFLAG)                                
C
C     member of MAD INPUT PARSER
C
C---- DEFINE A PARAMETER                                                 
C----------------------------------------------------------------------- 
C
C	MOD:
C          15-dec-2003, PT:
C             expand parameter names to 16 characters.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C		 28-sep-1998, PT:
C			add message if user changes one of DIMAD's built-in
C			parameters which is accessible from the MAD input parser.
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
C----------------------------------------------------------------------- 
      CHARACTER*8       KPARA
      CHARACTER(PNAME_LENGTH) KNAME                                      
      LOGICAL           FLAG                                             
      LOGICAL           PFLAG                                            
C----------------------------------------------------------------------- 
      IF (PFLAG) THEN                                                    
C---- COMMA?                                                             
        CALL RDTEST(',',FLAG)                                            
        IF (FLAG) GO TO 900                                              
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
C---- PARAMETER NAME                                                     
        CALL RDWORD(KNAME,LNAME)                                         
        IF (FATAL_READ_ERROR) GOTO 9999
      ENDIF                                                              
      IF (LNAME .EQ. 0) THEN                                             
        CALL RDFAIL                                                      
        WRITE (IECHO,910)                                                
        WRITE (ISCRN,910)                                                
        GO TO 900                                                        
      ENDIF                                                              
      IF (KLINE(ICOL) .EQ. '[') THEN                                     
        CALL RDLOOK(KNAME,PNAME_LENGTH,KELEM,1,IELEM1,IELEM)                        
        IF (IELEM .NE. 0) THEN                                           
          IF (IETYP(IELEM) .LE. 0) IELEM = 0                             
        ENDIF                                                            
        IF (IELEM .EQ. 0) THEN                                           
          CALL RDFAIL                                                    
          WRITE (IECHO,920) KNAME(1:LNAME)                               
          WRITE (ISCRN,920) KNAME(1:LNAME)                               
          GO TO 900                                                      
        ENDIF                                                            
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
        CALL RDWORD(KPARA,LPARA)                                         
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (LPARA .EQ. 0) THEN                                           
          CALL RDFAIL                                                    
          WRITE (IECHO,930)                                              
          WRITE (ISCRN,930)                                              
          GO TO 900                                                      
        ENDIF                                                            
        CALL RDTEST(']',FLAG)                                            
        IF (FLAG) GO TO 900                                              
        IEP1 = IEDAT(IELEM,1)                                            
        IEP2 = IEDAT(IELEM,2)                                            
        CALL RDLOOK(KPARA,LPARA,KPARM,IEP1,IEP2,IPARM)                   
        IF (IPARM .EQ. 0) THEN                                           
          CALL RDFAIL                                                    
           WRITE (IECHO,940) KNAME(1:LNAME),KPARA(1:LPARA)                
           WRITE (ISCRN,940) KNAME(1:LNAME),KPARA(1:LPARA)                
          GO TO 900                                                      
        ENDIF                                                            
        CALL RDNEXT                                                      
        IF (FATAL_READ_ERROR) GOTO 9999
      ELSE                                                               
        CALL FNDPAR(ILCOM,KNAME,IPARM)                                   
      ENDIF                                                              
C---- TEST FOR REDEFINITION                                              
      IF (IPTYP(IPARM) .GE. 0) THEN                                      
        CALL RDWARN
	  IF ( IPLIN(IPARM) .EQ. 0 ) THEN    ! built-in parameter
		WRITE(IECHO,960)
		WRITE(ISCRN,960)
	  ELSE                                                      
		WRITE (IECHO,950) IPLIN(IPARM)                                   
		WRITE (ISCRN,950) IPLIN(IPARM)
	  ENDIF                                   
      ENDIF                                                              
C---- EQUALS SIGN?                                                       
      CALL RDTEST('=',FLAG)                                              
      IF (FLAG) GO TO 800                                                
      CALL RDNEXT                                                        
      IF (FATAL_READ_ERROR) GOTO 9999
C---- PARAMETER EXPRESSION                                               
      CALL DECEXP(IPARM,FLAG)                                            
      IF (FATAL_READ_ERROR) GOTO 9999
      IF (FLAG) GO TO 800                                                
C---- END OF COMMAND?                                                    
      CALL RDTEST(';',FLAG)                                              
      IF (FLAG) GO TO 800                                                
      NEWPAR = .TRUE.                                                    
      RETURN                                                             
C---- ERROR EXIT                                                         
  800 IPTYP(IPARM) = -1                                                  
      PDATA(IPARM) = 0.0                                                 
  900 ERROR = .TRUE.  

9999  IF (FATAL_READ_ERROR) ERROR = .TRUE.

                                                   
      RETURN                                                             
C----------------------------------------------------------------------- 
  910 FORMAT(' *** ERROR *** PARAMETER NAME EXPECTED'/' ')               
  920 FORMAT(' *** ERROR *** UNKNOWN BEAM ELEMENT "',A,'"'/' ')          
  930 FORMAT(' *** ERROR *** PARAMETER KEYWORD EXPECTED'/' ')            
  940 FORMAT(' *** ERROR *** UNKNOWN ELEMENT PARAMETER "',               
     +       A,'[',A,']"'/' ')                                           
  950 FORMAT(' ** WARNING ** THE ABOVE NAME WAS DEFINED IN LINE ',I5,    
     +       ', IT WILL BE REDEFINED'/' ')                               
  960 FORMAT(' ** WARNING ** THE ABOVE NAME IS AN INTRINSIC PARAMETER',    
     +       ', IT WILL BE REDEFINED'/' ')                               
C----------------------------------------------------------------------- 
      END                                                                

      SUBROUTINE DECEXP(IPARM,ERROR1)                                            
C
C     part of MAD INPUT PARSER
C
C---- DECODE A PARAMETER EXPRESSION                                             
C
C     MOD:
C          15-DEC-2003, PT:
C             expand element and parameter names to 16 characters
C           31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C		 29-SEP-1998, PT:
C			added ASIN and ABS to functions list.
C	     08-SEP-1998, PT:
C	        added write to ISCRN where IECHO is presently written.
C     
C          13-July-1998, PT:
C             replaced ERROR with ERROR1 to avoid conflict with
C             ERROR in IOFLAG common block
C
C---- Modified by PT on 14-Feb-1996 to recognize ATAN function.
C     To add more functions, PAREVL in dimad11.f must also be modified.
C
C----
C
C     Modules:
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
C-----------------------------------------------------------------------        
c      PARAMETER         (NFUN = 9)                                              
      PARAMETER         (NFUN = 10)                                              
C-----------------------------------------------------------------------        
      LOGICAL           FLAG                                                    
      CHARACTER*8       KFUN(NFUN),KPARA
      CHARACTER(ENAME_LENGTH) KNAME                                  
C-----------------------------------------------------------------------        
      DATA KFUN(1)      / 'NEG     ' /                                          
      DATA KFUN(2)      / 'SQRT    ' /                                          
      DATA KFUN(3)      / 'LOG     ' /                                          
      DATA KFUN(4)      / 'EXP     ' /                                          
      DATA KFUN(5)      / 'SIN     ' /                                          
      DATA KFUN(6)      / 'COS     ' /                                          
      DATA KFUN(7)      / 'ATAN    ' /
	DATA KFUN(8)      / 'ASIN    ' /
	DATA KFUN(9)      / 'ABS     ' /
      DATA KFUN(10)     / 'TAN     ' /
C-----------------------------------------------------------------------        
      ERROR1 = .FALSE.                                                           
C---- CLEAR STACK                                                               
      LEV = 1                                                                   
      IOP(1) = 0                                                                
C---- EXPRESSION -------------------------------------------------------        
C---- LEFT PARENTHESIS?                                                         
      IF (KLINE(ICOL) .EQ. '(') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        LEV = LEV + 1                                                           
        IOP(LEV) = 10                                                           
      ENDIF                                                                     
C---- UNARY "+" OR "-"?                                                         
  100 IF (KLINE(ICOL) .EQ. '+') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
      ELSE IF (KLINE(ICOL) .EQ. '-') THEN                                       
        CALL RDNEXT                                                             
        LEV = LEV + 1                                                           
        IOP(LEV) = 12                                                           
      ENDIF                                                                     
C---- FACTOR OR TERM ---------------------------------------------------        
C---- EXPRESSION IN BRACKETS?                                                   
  200 IF (KLINE(ICOL) .EQ. '(') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        LEV = LEV + 1                                                           
        IOP(LEV) = 10                                                           
        GO TO 100                                                               
C---- FUNCTION OR PARAMETER NAME?                                               
      ELSE IF (INDEX('ABCDEFGHIJKLMNOPQRSTUVWXYZ',                              
     +               KLINE(ICOL)) .NE. 0) THEN                                  
        CALL RDWORD(KNAME,LNAME)                                                
        IF (FATAL_READ_ERROR) GOTO 9999
C---- FUNCTION?                                                                 
        IF (KLINE(ICOL) .EQ. '(') THEN                                          
          CALL RDLOOK(KNAME,LNAME,KFUN,1,NFUN,IFUN)                             
          IF (IFUN .EQ. 0) THEN                                                 
            CALL RDFAIL                                                         
            WRITE (IECHO,910) KNAME(1:LNAME)                                    
            WRITE (ISCRN,910) KNAME(1:LNAME)                                    
            GO TO 800                                                           
          ENDIF                                                                 
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          LEV = LEV + 1                                                         
          IOP(LEV) = IFUN + 11                                                  
          LEV = LEV + 1                                                         
          IOP(LEV) = 10                                                         
          GO TO 100                                                             
C---- ELEMENT PARAMETER?                                                        
        ELSE IF (KLINE(ICOL) .EQ. '[') THEN                                     
          CALL RDLOOK(KNAME,ENAME_LENGTH,KELEM(1),1,IELEM1,IELEM)                             
          IF (IELEM .NE. 0) THEN                                                
            IF (IETYP(IELEM) .LE. 0) IELEM = 0                                  
          ENDIF                                                                 
          IF (IELEM .EQ. 0) THEN                                                
            CALL RDFAIL                                                         
            WRITE (IECHO,920) KNAME(1:LNAME)                                    
            WRITE (ISCRN,920) KNAME(1:LNAME)                                    
            GO TO 800                                                           
          ENDIF                                                                 
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          CALL RDWORD(KPARA,LPARA)                                              
          IF (FATAL_READ_ERROR) GOTO 9999
          IF (LPARA .EQ. 0) THEN                                                
            CALL RDFAIL                                                         
            WRITE (IECHO,930)
		  WRITE (ISCRN,930)                                                   
            GO TO 800                                                           
          ENDIF                                                                 
          CALL RDTEST(']',FLAG)                                                 
          IF (FLAG) GO TO 800                                                   
          IEP1 = IEDAT(IELEM,1)                                                 
          IEP2 = IEDAT(IELEM,2)                                                 
          CALL RDLOOK(KPARA,LPARA,KPARM(1),IEP1,IEP2,IEP)                          
          IF (IEP .EQ. 0) THEN                                                  
            CALL RDFAIL                                                         
            WRITE (IECHO,940) KNAME(1:LNAME),KPARA(1:LPARA)                     
            WRITE (ISCRN,940) KNAME(1:LNAME),KPARA(1:LPARA)                     
            GO TO 800                                                           
          ENDIF                                                                 
          IVAL(LEV) = IEP                                                       
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
C---- GLOBAL PARAMETER                                                          
        ELSE                                                                    
          CALL FNDPAR(ILCOM,KNAME,IVAL(LEV))                                    
        ENDIF                                                                   
C---- NUMERIC VALUE?                                                            
      ELSE IF (INDEX('0123456789.',KLINE(ICOL)) .NE. 0) THEN                    
        CALL RDNUMB(VALUE,FLAG)                                                 
        IF (FATAL_READ_ERROR) GOTO 9999
        IF (FLAG) GO TO 800                                                     
        IF (IOP(LEV) .EQ. 12) THEN                                              
          VALUE = -VALUE                                                        
          LEV = LEV - 1                                                         
        ENDIF                                                                   
        CALL PARCON(ILCOM,IVAL(LEV),VALUE)                                      
C---- ANYTHING ELSE                                                             
      ELSE                                                                      
        CALL RDFAIL                                                             
        WRITE (IECHO,950)  
	  WRITE (ISCRN,950)                                                     
        GO TO 800                                                               
      ENDIF                                                                     
C---- UNSTACK UNARY OPERATORS                                                   
  300 IF (IOP(LEV) .GT. 10) CALL OPDEF(ILCOM)                                   
C---- UNSTACK MULTIPLY OPERATORS                                                
      IF (IOP(LEV) .EQ. 3 .OR. IOP(LEV) .EQ. 4) CALL OPDEF(ILCOM)               
C---- TEST FOR MULTIPLY OPERATORS                                               
      IF (KLINE(ICOL) .EQ. '*') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        LEV = LEV + 1                                                           
        IOP(LEV) = 3                                                            
        GO TO 200                                                               
      ELSE IF (KLINE(ICOL) .EQ. '/') THEN                                       
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        LEV = LEV + 1                                                           
        IOP(LEV) = 4                                                            
        GO TO 200                                                               
      ENDIF                                                                     
C---- UNSTACK ADDING OPERATORS                                                  
      IF (IOP(LEV) .EQ. 1 .OR. IOP(LEV) .EQ. 2) CALL OPDEF(ILCOM)               
C---- TEST FOR ADDING OPERATORS                                                 
      IF (KLINE(ICOL) .EQ. '+') THEN                                            
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        LEV = LEV + 1                                                           
        IOP(LEV) = 1                                                            
        GO TO 200                                                               
      ELSE IF (KLINE(ICOL) .EQ. '-') THEN                                       
        CALL RDNEXT                                                             
        IF (FATAL_READ_ERROR) GOTO 9999
        LEV = LEV + 1                                                           
        IOP(LEV) = 2                                                            
        GO TO 200                                                               
      ENDIF                                                                     
C---- UNSTACK PARENTHESES                                                       
      IF (LEV .NE. 1) THEN                                                      
        IF (KLINE(ICOL) .EQ. ')') THEN                                          
          CALL RDNEXT                                                           
          IF (FATAL_READ_ERROR) GOTO 9999
          LEV = LEV - 1                                                         
          IVAL(LEV) = IVAL(LEV+1)                                               
          GO TO 300                                                             
        ELSE                                                                    
          CALL RDFAIL                                                           
          WRITE (IECHO,960)
		WRITE (ISCRN,960)                                                     
          GO TO 800                                                             
        ENDIF                                                                   
      ELSE IF (KLINE(ICOL) .EQ. ')') THEN                                       
        CALL RDFAIL                                                             
        WRITE (IECHO,970)
	  WRITE (ISCRN,970)                                                       
        GO TO 800                                                               
      ENDIF                                                                     
C---- DISCARD UNNEEDED TEMPORARY                                                
      IF (IVAL(1) .EQ. IPARM2 .AND. KPARM(IPARM2)(1:1) .EQ. '*') THEN           
        IPTYP(IPARM) = IPTYP(IPARM2)                                            
        IPDAT(IPARM,1) = IPDAT(IPARM2,1)                                        
        IPDAT(IPARM,2) = IPDAT(IPARM2,2)                                        
        PDATA(IPARM) = PDATA(IPARM2)                                            
        IPARM2 = IPARM2 + 1                                                     
      ELSE                                                                      
        IPTYP(IPARM) = 11                                                       
        IPDAT(IPARM,1) = 0                                                      
        IPDAT(IPARM,2) = IVAL(1)                                                
        PDATA(IPARM) = 0.0                                                      
      ENDIF                                                                     
      IF (IPTYP(IPARM) .GT. 0) NEWPAR = .TRUE.                                  
      IPLIN(IPARM) = ILCOM                                                      
      RETURN                                                                    
C---- ERROR EXIT --- LEAVE PARAMETER UNDEFINED                                  
  800 IPTYP(IPARM) = -1                                                         
      IPDAT(IPARM,1) = 0                                                        
      IPDAT(IPARM,2) = 0                                                        
      PDATA(IPARM) = 0.0                                                        
      IPLIN(IPARM) = ILCOM                                                      
      ERROR1 = .TRUE.            

9999  IF (FATAL_READ_ERROR) ERROR1=.TRUE.
                                                
      RETURN                                                                    
C-----------------------------------------------------------------------        
  910 FORMAT(' *** ERROR *** UNKNOWN FUNCTION "',A,'"'/' ')                     
  920 FORMAT(' *** ERROR *** UNKNOWN BEAM ELEMENT "',A,'"'/' ')                 
  930 FORMAT(' *** ERROR *** PARAMETER KEYWORD EXPECTED'/' ')                   
  940 FORMAT(' *** ERROR *** UNKNOWN ELEMENT PARAMETER "',A,'[',A,']"'/         
     +       ' ')                                                               
  950 FORMAT(' *** ERROR *** OPERAND MUST BE NUMBER, PARAMETER NAME,',          
     +       ' FUNCTION CALL, OR EXPRESSION IN "()"'/' ')                       
  960 FORMAT(' *** ERROR *** RIGHT PARENTHESIS MISSING'/' ')                    
  970 FORMAT(' *** ERROR *** UNBALANCED RIGHT PARENTHESIS'/' ')                 
C-----------------------------------------------------------------------        
      END                                                                       

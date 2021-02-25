      SUBROUTINE RDLABL(KWORD,LWORD)
C
C     member of MAD INPUT PARSER
C
C---- READ AN IDENTIFIER OR KEYWORD OF UP TO 16 CHARACTERS; 
C     unlike RDWORD, RDLABL allows strings starting with a number,
C     and will allow decimal points or colons in the string.
C
C----------------------------------------------------------------------- 
c
c	MOD:
C          18-NOV-2002, PT:
C             allow dashes ("-") as non-first character to support
C             ANTI-PROTON particle for BEAM element.
C          31-JAN-2001, PT:
C             if FATAL_READ_ERROR occurs, go directly to egress.
C		 23-nov-1998, PT:
C			label or type terminates on a single or double quote.
C		 19-OCT-1998, PT:
C			allow _.$ to be part of the label; variable word length.
C
C     modules
C
      USE XSIF_INOUT
C
C      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      IMPLICIT NONE
      SAVE
C
      CHARACTER*(*)      KWORD
C
      INTEGER*4          LWORD, LKWORD
C----------------------------------------------------------------------- 
      LKWORD = LEN(KWORD)
      KWORD = ' '        
      DO LWORD = 2,LKWORD
         KWORD = KWORD // ' '
      ENDDO
      LWORD = 0
	
	IF (INDEX('"''',KLINE(ICOL)) .NE. 0) THEN ! termination
		call RDNEXT
          IF (FATAL_READ_ERROR) GOTO 9999
		GOTO 9999
	ENDIF
	                                                          
      IF (INDEX('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',                            
     +          KLINE(ICOL)) .NE. 0) THEN                                
   10   CONTINUE                                                         
          IF (LWORD .LT. LKWORD) THEN                                        
            LWORD = LWORD + 1                                            
            KWORD(LWORD:LWORD) = KLINE(ICOL)                             
          ENDIF                                                          
          CALL RDNEXT                                                    	
          IF (FATAL_READ_ERROR) GOTO 9999
	
		IF (INDEX('"''',KLINE(ICOL)) .NE. 0) THEN ! termination
			call RDNEXT
              IF (FATAL_READ_ERROR) GOTO 9999
			GOTO 9999
		ENDIF
	                                                          
        IF (INDEX('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:_$-''',              
     +            KLINE(ICOL)) .NE. 0) GO TO 10                          
      ENDIF 

9999	CONTINUE

      RETURN                                                             
C----------------------------------------------------------------------- 
      END                                                                
